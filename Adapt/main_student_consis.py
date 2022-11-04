import os
import torch
import numpy as np
import wandb
import logging
from config import config
import torch.nn as nn
import time
from dataloader import MDataModule
from function import *
import torch.nn.functional as F
import shutil
import random
import sys
sys.path.append('../utilis/')
from resnet import resnet50, resnet34
from randomizations import MAS
import losses, utlis

LOG = logging.getLogger('main')

class DCG_PTM():
    def __init__(self, config):
        super().__init__()

        self.config = config
        # setup logging & wandb
        self.init_setup(config)
        self.set_wandb()


        # load model
        self.model_setup()

        # define optimizers and loss function
        self.hyperpara()
        self.loss_func()

        # load dataset
        self.load_data(self.config)

        # memory awareness synapse (pre-compute with source dataset )
        self.knowledge_preservation()

    def init_setup(self, config):
        self.global_step = 0
        self.val_step = 0
        self.mini_loss = 1000
        self.num_classes = config['num_class']
        self.lambda_mas = config['mas']
        self.center_alpha = config['center_alpha']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.meters = utlis.AverageMeterSet()

        # logging
        MODEL_CKPT = self.config['log_string_dir']
        # create log/weight folder
        if not os.path.exists(MODEL_CKPT):
            os.system("mkdir " + MODEL_CKPT)
        if self.config['is_train'] == True:
            self.SAVE_MODEL = MODEL_CKPT

    def set_wandb(self):
        # set wandb
        self.wandb = wandb
        self.wandb.init(project=config['project_name'])
        self.wandb.run.name = config['run_name']
        self.wandb.config.patch_size = config['patch_size']
        self.wandb.config.stride_size = config['stride_size']
        self.wandb.config.train_batch_size = config['train_batch_size']
        self.wandb.config.val_batch_size = config['val_batch_size']
        self.wandb.config.save_path = config['log_string_dir']
        self.wandb.config.te_lr = config['te_lr']
        self.wandb.config.lr = config['lr']
        self.wandb.config.mas = config['mas']
        self.wandb.config.confidence_thresh = config['confidence_thresh']
        self.wandb.config.pc_weight = config['pc_weight']
        self.wandb.config.center_alpha = config['center_alpha']


    def create_model(self, model_name, pretrain=False):
        model = model_name(num_classes=self.num_classes)
        if pretrain:
            checkpoint = torch.load(self.config['pretrain_path'])
            model.load_state_dict(checkpoint['model_state_dict'])  # load pretrain weights

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(self.device)
        return model

    def model_setup(self):
        self.model_student = self.create_model(model_name = resnet50, pretrain=True)
        self.model_teacher = self.create_model(model_name = resnet50, pretrain=True)  # Teacher model

        # freeze teacher last layer
        # self.model_teacher.fc.requires_grad = False

        # class-wise center from source
        checkpoint = torch.load(self.config['pretrain_path'])
        self.fea_center = checkpoint['feature_center']
        self.Pseudo_label_center = pseudo_label(self.fea_center, self.config['num_class'], dis_type= self.config['dis_type'])

    def hyperpara(self):
        # optimizer for feature center
        self.optimizer_centerloss = torch.optim.SGD(self.Pseudo_label_center.parameters(), lr= 0.001)

        # optimizer for student
        self.optimizer = torch.optim.Adam(self.model_student.parameters(), lr=config['lr'])
        decay_rate = 0.75
        decay_steps = 7500
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_steps,
                                                         gamma=decay_rate)  # update every decay_steps steps

        # optimizer for teacher
        self.optimizer_teacher = torch.optim.SGD(self.model_teacher.parameters(), lr=config['te_lr'])
        decay_rate = 0.75
        decay_steps = 7500
        self.scheduler_teacher = torch.optim.lr_scheduler.StepLR(self.optimizer_teacher, step_size=decay_steps,
                                                                 gamma=decay_rate)  # update every decay_steps steps

    def adjust_lr(self, optimizer):
        if self.global_step < 5000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        elif 5000 <= self.global_step < 4 * 5000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0005
        elif 4 * 5000 <= self.global_step < 5 * 5000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        elif 5 * 5000 <= self.global_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001

    def loss_func(self):
        # consistency criterion (feature space or probability )
        self.consistency_criterion = losses.softmax_kl_loss
        self.class_criterion = nn.CrossEntropyLoss()
#         self.class_criterion = losses.SCELoss(alpha=0.5, beta= 0.5)

    def load_data(self, config):
        # load dataset
        dataset = MDataModule(config)
        self.train_source_loader = dataset.train_source_dataloader()
        self.train_target_loader = dataset.train_target_dataloader()
        self.eval_loader = dataset.val_dataloader()

    def knowledge_preservation(self):
        train_source_iterator = iter(self.train_source_loader)
        # ewc = EWC(self.model_teacher, train_source_iterator, self.device)
        self.mas = MAS(self.model_teacher, train_source_iterator, self.device)
        del train_source_iterator, self.train_source_loader

    @torch.no_grad()
    def val_forward_step(self, batch, model):  # for val
        img_high, label, _ = batch
        img_high, label = img_high.to(self.device), label.to(self.device)
        label = np.squeeze(label)
        x_flatten, prob = model(img_high)
        return x_flatten, prob, label

    def teacher_forward_step(self, batch, model):  # for teacher
        img_high, label, _ = batch
        img_high, label = img_high.to(self.device), label.to(self.device)
        label = np.squeeze(label)
        x_flatten, prob = model(img_high)
        return x_flatten, prob, label

    def forward_step_student(self, batch, teacher_model, student_model):
        [batch1_t, batch2_t] = batch

        # Teacher Model
        # batch1_t for teacher model to generate pseudo label
        img_high_t1, label_t, _ = batch1_t
        img_high_t1, label_t = img_high_t1.to(self.device), label_t.to(self.device)
        label_t = np.squeeze(label_t)
        x_flatten_teacher, logits_teacher = teacher_model(img_high_t1)

        # Student Model
        img_high_t2, _, _ = batch2_t
        img_high_t2 = img_high_t2.to(self.device)
        x_flatten1_student, logits1_student = student_model(img_high_t1)
        x_flatten2_student, logits2_student = student_model(img_high_t2)

        # Prediction consistency loss
        # pc_loss = self.consistency_criterion(logits1_student, logits2_student)
        pc_loss = torch.nn.MSELoss()(logits1_student, logits2_student)

        # Conditional entropy (Not been used in this paper)
        conditionEntropy_loss = conditionEntropy(logits_teacher)

        # Confidence-selected knowledge distillation (CS-KD)
        consistency_criterion = loss_fn_kd
        temperature = 20
        confidence_thresh = config['confidence_thresh']
        consistency_loss, conf_mask, label_tea, disagree_id = consistency_criterion_loss(consistency_criterion, logits2_student,
                                                                                             logits_teacher.detach(),confidence_thresh,
                                                                                             temperature)

        # number of accepted pseudo labels
        conf_mask_count = conf_mask.sum()
        self.wandb.log({"Numbers of reliable pseudo labels": conf_mask_count, "step": self.global_step})
        disagree_nums = len(disagree_id)
        self.wandb.log({"Dissent nums": disagree_nums, "step": self.global_step})


        # Self-labeling loss : assign hard label for teacher
        conf_mask = torch.where(conf_mask == 1)[0]
        Self_labeling_loss = self.class_criterion(logits_teacher[conf_mask], label_tea[conf_mask])

        # Domain-guided center loss
        if self.config['fet_dim'] == self.config['num_class']:
            positive_loss, domain_center_loss = self.Pseudo_label_center.center_loss(logits_teacher, label_tea, conf_mask, self.global_step)
        else:
            positive_loss, domain_center_loss = self.Pseudo_label_center.center_loss(x_flatten_teacher, label_tea, conf_mask, self.global_step)

        if consistency_loss is not None and len(conf_mask) != 0:
            # accuracy to assign pseudo label for target data
            pseudo_label = label_tea[conf_mask]
            acc_pseudo_label = sum(x == y for x, y in zip(label_t[conf_mask], pseudo_label)) / len(pseudo_label)
        else:
            consistency_loss, acc_pseudo_label = 0, 0

        return Self_labeling_loss, pc_loss, conditionEntropy_loss, consistency_loss, logits2_student, logits_teacher, positive_loss, domain_center_loss, label_t, acc_pseudo_label

    def trainer(self):
        print("Start Training")
        config = self.config

        # dataset iterator
        valid_dataloader_iterator = iter(self.eval_loader)
        train_target_iterator = iter(self.train_target_loader)

        for iterator_num in range(0, self.config['max_iterator_num']):
            start = time.time()
            # switch to train mode
            self.model_student.train()
            self.model_teacher.train()
            self.Pseudo_label_center.train()
            self.optimizer.zero_grad()
            self.optimizer_teacher.zero_grad()
            self.optimizer_centerloss.zero_grad()

            try:
                item1 = next(train_target_iterator)
            except StopIteration:
                train_target_iterator = iter(self.train_target_loader)
                item1 = next(train_target_iterator)

            target_batch_1, target_batch_2 = item1  # target_image_20x, target_label, target_domain_label

            # feed data into techer and student model
            Self_labeling_loss, pc_loss, conditionEntropy_loss, consistency_loss, \
            logits_student, logits_teacher, positive_loss, domain_center_loss, label_t, acc_pseudo_label \
                = self.forward_step_student([target_batch_1, target_batch_2], self.model_teacher, self.model_student)

            '''
                 SUM UP ALL LOSS
            '''

            # Class prediction for source data by student model
            minibatch_size = label_t.size()[0]
            self.wandb.log({"pseudo_label_acc/teacher model": acc_pseudo_label, "step": self.global_step})

            # knowledge preservation
            mas_loss = self.mas.penalty(self.model_teacher) * self.lambda_mas
            self.wandb.log({"ewc_loss": mas_loss.item(), "step": self.global_step})

            if consistency_loss != 0:
                # student model
                loss_student = consistency_loss + pc_loss * config['pc_weight']
                loss_student.backward()

                # progressive teacher model
                loss_teacher = mas_loss + domain_center_loss * self.center_alpha + Self_labeling_loss
                loss_teacher.backward()
                for param in self.Pseudo_label_center.parameters():
                    param.grad.data *= (1. / self.center_alpha)

                self.optimizer.step()
                self.optimizer_centerloss.step()
                self.optimizer_teacher.step()
                self.scheduler_teacher.step()
                self.scheduler.step()
                self.adjust_lr(self.optimizer_centerloss)

                # record
                self.meters.update('cons_loss', consistency_loss.item())
                self.wandb.log({"cons_loss": consistency_loss.item(), "step": self.global_step})
                self.wandb.log({"prediction_consistency_loss": pc_loss.item(), "step": self.global_step})
                self.wandb.log({"domain_center_loss": domain_center_loss.item(), "step": self.global_step})
                self.wandb.log({"conditionEntropy_loss": conditionEntropy_loss.item(), "step": self.global_step})
                self.wandb.log({"Self_labeling_loss_teacher": Self_labeling_loss.item(), "step": self.global_step})
                self.wandb.log({"loss_student": loss_student.item(), "step": self.global_step})

            self.global_step += 1

            '''
            Update step results
            '''
            # student predict top1 on target domain
            prec1 = accuracy(logits_student.data, label_t.data)
            self.meters.update('top1', prec1, minibatch_size)
            self.wandb.log({"student_top1/target": prec1, "step": self.global_step})

            # teacher predict top1 on target domain
            prec1 = accuracy(logits_teacher.data, label_t.data)
            self.wandb.log({"teacher_top1/target": prec1, "step": self.global_step})

            # compute gradient and do Adam step
            self.meters.update('lr', self.optimizer.param_groups[0]['lr'])
            self.wandb.log({"lr": self.optimizer.param_groups[0]['lr'], "step": self.global_step})

            self.wandb.log({"teacher lr": self.optimizer_teacher.param_groups[0]['lr'], "step": self.global_step})
            self.wandb.log({"optimizer_centerloss lr": self.optimizer_centerloss.param_groups[0]['lr'], "step": self.global_step})
            # measure elapsed time
            self.meters.update('batch_time', time.time() - start)

            '''
             Evaluation
            '''
            if config['evaluation_steps'] and iterator_num % config['evaluation_steps'] == 0:
                start_time = time.time()

                student_prec1, student_loss, Entropy_loss_acc, true_loss = [], [], [], []

                for i in range(16):
                    try:
                        batch = next(valid_dataloader_iterator)
                    except StopIteration:
                        valid_dataloader_iterator = iter(self.eval_loader)
                        batch = next(valid_dataloader_iterator)

                    # Testing batch data(Validation).
                    Entropy_loss, class_loss, t_prec1, t_loss = self.validate(batch, self.model_student)

                    Entropy_loss_acc.append(Entropy_loss)
                    student_prec1.append(t_prec1)
                    student_loss.append(t_loss)
                    true_loss.append(class_loss)

                LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
                self.wandb.log(
                    {"Validation_target_top1": sum(student_prec1) / len(student_prec1), "step": self.val_step})
                self.wandb.log(
                    {"Validation_target_class": sum(student_loss) / len(student_loss), "step": self.val_step})
                self.wandb.log(
                    {"Entropy_loss_acc": sum(Entropy_loss_acc) / len(Entropy_loss_acc), "step": self.val_step})
                self.wandb.log({"true_loss": sum(true_loss) / len(true_loss), "step": self.val_step})

                self.val_step += 1

                # LOG.info(
                #     'Test: [{0}]\t'
                #     'Teacher Class {1}\t'
                #     'Teacher Prec@1 {2}\t'
                #         .format(
                #         self.val_step,
                #         sum(student_loss) / len(student_loss),
                #         sum(student_prec1) / len(student_prec1),
                #     ))

                if Entropy_loss_acc != 0:
                    is_best = (sum(Entropy_loss_acc) / len(Entropy_loss_acc)) < self.mini_loss
                    self.mini_loss = min(sum(Entropy_loss_acc) / len(Entropy_loss_acc), self.mini_loss)
                else:
                    is_best = False
            else:
                is_best = False

            # save model weight every iteration
            if is_best or iterator_num == 0 or iterator_num % 10000 == 0:
                self.save_checkpoint({
                    'state_dict': self.model_student.state_dict(),
                    'teacher_state_dict': self.model_teacher.state_dict(),
                    'center_feature': self.Pseudo_label_center.fea_center,
                }, self.SAVE_MODEL, self.global_step)

                self.best_weight_name = self.config["log_string_dir"] + self.config["checkpoint_path"]
                train_batch_size = self.config['train_batch_size']
                with open(self.best_weight_name, 'w') as f:
                    info = [str((iterator_num + int(self.global_step / train_batch_size)) * train_batch_size),
                            '\t',
                            str(self.mini_loss)]
                    f.writelines(info)


    def validate(self, batch, model):
        class_criterion = self.class_criterion
        meters = utlis.AverageMeterSet()
        # switch to evaluate mode
        model.eval()
        end = time.time()

        _, output, target_label = self.val_forward_step(batch, model)
        minibatch_size = len(batch)

        assert minibatch_size > 0
        meters.update('labeled_minibatch_size', minibatch_size)

        # compute output
        Entropy_loss = ValEntropy(output)

        softmax = F.softmax(output, dim=1)
        class_loss = class_criterion(softmax, target_label) / minibatch_size

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_label.data)
        meters.update('class_loss', class_loss.data, minibatch_size)
        meters.update('top1', prec1, minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)

        return Entropy_loss, class_loss, meters['top1'].avg, meters['class_loss'].avg

    def save_checkpoint(self, state, dirpath, step):
        filename = 'checkpoint.{}.ckpt'.format(step)
        checkpoint_path = os.path.join(dirpath, filename)
        best_path = os.path.join(dirpath, 'best.ckpt')
        torch.save(state, checkpoint_path)
        LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
        print("--- checkpoint saved to %s ---" % checkpoint_path)

        shutil.copyfile(checkpoint_path, best_path)
        # LOG.info("--- checkpoint copied to %s ---" % best_path)
        # print("--- checkpoint copied to %s ---" % best_path)


if __name__ == '__main__':
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logging.basicConfig(level=logging.INFO)
    t_Start = time.time()
    DCG_PTM(config).trainer()
    print("Finish Training: ", time.time() - t_Start)


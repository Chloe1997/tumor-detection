# -*- coding: utf-8 -*-
import numpy as np
import torch
from config import config
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('../utilis/')
import ramps


def L2_loss(predictor, z):
    z = F.normalize(z, dim=-1, p=2)
    predictor = F.normalize(predictor, dim=-1, p=2)

    return 2-2*(z*predictor).sum(dim=-1)


def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def mse(pred1, pred2):
    return (pred1-pred2) * (pred1-pred2)

def loss_fn_kd(input_logits, target_logits, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = temperature
    KD_loss = nn.KLDivLoss(reduce=False, size_average=False)(F.log_softmax(input_logits/T, dim=1),
                             F.softmax(target_logits/T, dim=1))* T * T
    return KD_loss

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
    return loss_fn(input_softmax, target_softmax)

def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    return num / denom

def cal_score(score):
        return 1 - score

def cosine_embedding_loss(x1, x2, state='train'):
    batch_size, hidden_size = x1.size()
    scores = torch.zeros(batch_size)
    for i in range(batch_size):
        score = cosine_similarity(x1[i], x2[i])
        scores[i] = cal_score(score)
    if state == 'train':
        return scores
    if state == 'val':
        return scores.mean()

def Diff(li1, li2):
    different_label = []
    for i1, i2 in zip(li1,li2):
        if i1 == i2:
            different_label.append(0)
        else:
            different_label.append(1)
    different_label = torch.Tensor(different_label)
    return different_label

def consistency_criterion_loss(consistency_criterion,con_logit_student, con_logit_teacher,confidence_thresh, temperature):
    '''
    Confidence-selected knowledge distillation (CS-KD)
    select high confidence pseudo labels and check the disagreement between teahcer and student model
    :param consistency_criterion: define the loss function for KD
    :param con_logit_student: the prediction of student model
    :param con_logit_teacher: the prediction of teacher model (as pseudo labels without computing gradient)
    :param confidence_thresh: pseudo label confidenc threshold
    :param step:
    :return:
    CS_KD_loss: resulting loss with confidence thresholding
    conf_mask: pseudo label mask
    label_tea: the class prediction of the teacher model
    disagree_id: the index of pseudo labels which have different prediction with high confidence
    '''

    softmax = nn.Softmax(dim=1)
    aug_loss  = consistency_criterion(con_logit_student, con_logit_teacher, temperature)
    aug_loss = aug_loss.mean(dim=1)

    # select by confidence score
    con_logit_teacher = softmax(con_logit_teacher)
    conf_tea = torch.max(con_logit_teacher,1)[0]
    label_tea = torch.max(con_logit_teacher, 1)[1]

    con_logit_student = softmax(con_logit_student)
    conf_stu = torch.max(con_logit_student,1)[0]
    label_stu = torch.max(con_logit_student, 1)[1]

    conf_mask  = torch.where(conf_tea > confidence_thresh, 1, 0)

    # check disagreement of teahcer and student model
    disagree = Diff(label_tea,label_stu)
    conf_stu_mask = torch.where(conf_stu > 0.8, 1, 0)
    disagree_mask = disagree * conf_stu_mask.cpu()
    disagree_id = torch.nonzero(disagree_mask)
    # if ignore disagree pseudo labels
    # conf_mask[disagree_id] = 0

    # resulting pseudo label mask
    conf_mask_count  = conf_mask.sum()
    CS_KD_loss = sum(aug_loss * conf_mask)/conf_mask_count
    return CS_KD_loss, conf_mask, label_tea, disagree_id


def get_current_consistency_weight(global_step):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    consistency = 1
    return consistency * ramps.sigmoid_rampup(global_step, config['consistency_rampup'])

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    total = np.shape(labels)[0]
    return torch.sum(pred==labels).item()/total

#   adjust_learning_rate(optimizer, global_step, i, len(train_target_loader))

def adjust_learning_rate(optimizer, step_in_epoch):
    global_step = step_in_epoch
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(global_step, config['lr_rampup']) * config['lr']

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def center_loss(feature, center):
    # out1 = out1.cpu() - self.center.cpu()
    # print(np.shape(out1),np.shape(center))
    out1 = feature - center
    # print(np.shape(out1))
    center_loss = (out1**2).sum(dim=1).mean()
    # print(np.shape(center_loss),center_loss,"&&&")
    return center_loss


class pseudo_label(nn.Module):
    '''
    fea_center: class-wise feature center pre-trained in source dataset
    dis_type: the distance measurement (cosine or l2 norm)
    '''
    def __init__(self, fea_center,num_class, dis_type='l2'):
        super().__init__()
        self.fea_center = nn.Parameter(fea_center.cuda()) # torch.Size([2048])
        self.conf_mask = None
        self.num_classes = num_class
        self.dis_type = dis_type
        self.use_gpu = True

    def center_loss(self, fea, label, conf_mask, step):  # 被assign 是tumor的data 和center的 cosine distance 要越小越好
        self.conf_mask = conf_mask
        fea = fea[self.conf_mask]
        label = label[self.conf_mask]
        batch_size = len(self.conf_mask)

        if self.dis_type == 'cosine':
            cosine_dist = F.cosine_similarity(self.fea_center[label], fea)
            loss = (batch_size * 1 - sum(cosine_dist)) / batch_size

        elif self.dis_type == 'l2':
            distmat = torch.pow(fea, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                      torch.pow(self.fea_center, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
            distmat.addmm_(fea, self.fea_center.t(),beta = 1, alpha = -2)

            # mat1: Tensor, mat2: Tensor, *, beta: Number=1, alpha: Number=1
            classes = torch.arange(self.num_classes).long()
            if self.use_gpu: classes = classes.cuda()
            labels = label.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = labels.eq(classes.expand(batch_size, self.num_classes))

            dist = distmat * mask.float()
            loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size # center loss

        if 1 in label.cpu().numpy():
            N, C  = fea.shape
            # instance level
            positive_loss = self.inter_tumor_loss(fea,N)
        else:
            positive_loss = 0
        
        return positive_loss, loss

    def cat_fea(self, output, step, alpha = 0.5):
        # update tumor center
        fea = torch.unsqueeze(torch.mean(output.float().detach().cpu(),0), 0)
        # print(self.tumor_center.shape, fea.shape)
        self.tumor_feature = torch.cat((self.tumor_feature, fea))

        if step % config['center_freq'] == 0:
            # alpha = min(1 - 1 / (step + 1), alpha)
            self.tumor_center = torch.cat((self.tumor_feature, self.tumor_center))
            self.tumor_center = torch.unsqueeze(torch.mean(self.tumor_center, 0),0)

    def inter_tumor_loss(self,features,N): # output= features  # 被assign 是tumor的datas 彼此的cosine distance也要越小越好
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(N, dtype=torch.bool).cuda()
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # select positives (positives)
        logits_loss = 1 - similarity_matrix
        positive_loss = torch.mean(logits_loss)
        return positive_loss


def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(target,log_likelihood)) / batch
    else:
        loss = torch.sum(torch.mul(target,log_likelihood, target))
    return loss

def ValEntropy(inputs, reduction='average'):
    softmax = nn.Softmax(dim=1)
    conf = softmax(inputs)
    N, _ = conf.size()
    pred_value = torch.max(conf, 1)[0]

    log_likelihood = -torch.log(pred_value)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(pred_value,log_likelihood)) / batch
    else:
        loss = torch.sum(torch.mul(pred_value,log_likelihood))
    return loss


def conditionEntropy(inputs, reduction='average'):
    # softmax = nn.Softmax(dim=1)
    # conf = softmax(inputs)
    # N, _ = conf.size()
    # pred_value = torch.max(conf, 1)[0]

    # log_likelihood = -torch.log(pred_value)
    # batch = inputs.shape[0]
    # if reduction == 'average':
    #     loss = torch.sum(torch.mul(pred_value,log_likelihood)) / batch
    # else:
    #     loss = torch.sum(torch.mul(pred_value,log_likelihood))
    # return loss
    batch = inputs.shape[0]
    softmax = nn.Softmax(dim=1)
    conf = softmax(inputs)
    loss =  - conf * torch.log2(conf)
    loss = torch.sum(loss) / batch
    return loss

# model regularization 
class Regularization(torch.nn.Module):
    def __init__(self,ini_model,weight_decay=100,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=ini_model

        # for param in self.model.parameters():
        #     param.requires_grad = False
        #     param.detach_()

        self.weight_decay=weight_decay
        self.p=p
        self.weight_list_ini=self.get_weight(ini_model)
        self.num = len(self.weight_list_ini)
        print(self.num)
        # self.weight_info(self.weight_list_ini)
 
    def forward(self, model):
        reg_loss = 0
        # reg_loss = sum((x - y).abs().sum() for x, y in zip(model.state_dict().values(), self.model.state_dict().values()))
        # reg_loss = reg_loss/self.num

        # self.weight_list=self.get_weight(model)#获得最新的权重
        # reg_loss = self.regularization_loss(self.weight_list_ini, self.weight_list, self.weight_decay, p=self.p)/self.num
        
        state_dict_ema = self.model.state_dict()
        state_dict_student = model.state_dict()
        num = 0
        for (ema_k, ema_v),(stu_k, stu_v) in zip(state_dict_ema.items(), state_dict_student.items()):
            assert stu_k == ema_k, "state_dict names are different!"
            num += 1
            reg_loss += (ema_v - stu_v).abs().sum()
        reg_loss = reg_loss / num
        return reg_loss

        # for ema_param, param in zip(self.model.parameters(), model.parameters()):
        #     num += 1
        #     reg_loss += (ema_param.data - param.data).abs().sum()
        # reg_loss = reg_loss / num
        # return reg_loss

 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self, weight_list_ini, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for w1, w2 in zip(weight_list_ini, weight_list):
            l2_reg_ini = torch.norm(w1[1], p=p)
            # l2_reg_ini = w1[1]
            l2_reg = torch.norm(w2[1],p=p)
            # l2_reg = w2[1]
            reg_para = torch.abs(l2_reg_ini-l2_reg)
            reg_loss = reg_loss + reg_para
 
        reg_loss= reg_loss / self.num
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")

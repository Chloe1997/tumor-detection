import os, sys
import torch
import numpy as np
from dataloader_pyvips import TumorDataModule
from config_test import config
from tqdm import tqdm
import torch.nn as nn
from evaluation import *
sys.path.append('../utilis/')
from resnet import resnet50, resnet34
import time

class ClassifyModel():
    def __init__(self, config):
        super().__init__()

        self.config = config
        # setup logging
        self.init_setting()

        # load model and weights
        self.num_class = self.config['num_class']
        self.model = resnet50(num_classes=2)
#         self.model = resnet34(num_classes=2)
        if torch.cuda.device_count() > 1:
            print("Using Multiple GPU", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        # load trained model
        inference_model_list = ['model_state_dict','teacher_state_dict','state_dict']
        inference_model = inference_model_list[self.config['inference_model']]
        self.model.load_state_dict(self.checkpoint[inference_model])
        self.model.to(self.device)


    def data_loader(self,case_name):
        dataloader = TumorDataModule(self.config,case=case_name)
        self.test_dataloader = dataloader.test_dataloader()

    def init_setting(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = torch.load(self.config['log_string_dir'] + self.config['best_weights'] )

    def forward_step(self, batch):
        img_high, x, y = batch
        img_high = img_high.to(self.device)
        with torch.cuda.amp.autocast():
            _,prob = self.model(img_high)
        return prob, x, y

    def test(self,case_name):
        print("START INFERENCE {}".format(case_name) )
        init_para(case_name)
        # load dataset
        self.data_loader(case_name)


        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.test_dataloader)):
                prob, x, y = self.forward_step(batch)

                show_testing_img(prob, x, y)

        save_evaluation(case_name)
        print("INFERENCE {} DONE".format(case_name))



# inference
if __name__ == '__main__':
    for case in config['test_list']:
        t_start = time.time()
        ClassifyModel(config).test(case)
        t_end = time.time()
        print("Cost time: {} second".format(t_end-t_start))




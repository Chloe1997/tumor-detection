# -*- coding: utf-8 -*-
import os
from os.path import join
import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile
import pickle
import pyvips
import time


class MDataset(Dataset):
    def __init__(self, config, transform, is_3D = False, type_str='train_source'):
        super().__init__()
        self.config = config
        self.transform = transform
        self.type_str = type_str

        self.domain_label = 0
        if self.type_str == 'train_source':
            self.domain_label = 0
        elif self.type_str == 'train_target':
            self.domain_label = 1

        self.is_3D = is_3D
        self.data_list = config[type_str + '_list']
        self.datas = self.load_data_pkl(self.data_list)

        self.total_len = len(self.datas)
        print("patch num: ", self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        img_high, label = self.get_data(idx)

        if self.type_str != 'train_target':
            if self.transform is not None:
                img_high = self.transform(img_high)
            domain_label = torch.tensor(self.domain_label)
            label = torch.tensor(label)
            # return (img_high,img_low, label, domain_label)
            return (img_high, label, domain_label)

        elif self.type_str == 'train_target':
            if self.transform is not None:
                img_high_1 = self.transform[0](img_high)
                img_high_2 = self.transform[1](img_high)
            domain_label = torch.tensor(self.domain_label)
            label = torch.tensor(label)
            return (img_high_1, label, domain_label), (img_high_2, label, domain_label)

    # -
    def load_data_pkl(self, case_list):
        data = []
        for case in case_list:
            # print(self.config['data_pkl_path'] + f'/{case}/{case}.pkl')
            if not self.is_3D:
                data_pkl = self.config['data_pkl_path'] + f'/{case}/{case}.pkl'
            else:
                data_pkl = self.config['3D_data_pkl_path'] + f'/{case}/{case}.pkl'
            # print('Load data pkl from: ', data_pkl)
            with open(data_pkl, 'rb') as f:
                while True:
                    try:
                        data.append(pickle.load(f))
                    except EOFError:
                        break

        mix_data = np.concatenate(data)
        np.random.shuffle(mix_data)
        print(np.shape(mix_data))
        return mix_data

    def geometric_series_sum(self, a, r, n):
        return a * (1.0 - pow(r, n)) / (1.0 - r)

    def multi_scale(self, x, y, level):
        patch_size = self.config['patch_size']

        offset = (patch_size / 2) * self.geometric_series_sum(1.0, 2.0, float(level))
        x = x - offset
        y = y - offset

        # 需確認是否倍率是2倍遞減
        x = int(x / pow(2, level))
        y = int(y / pow(2, level))

        x = 0 if x < 0 else x
        y = 0 if y < 0 else y

        return x, y

    def read2patch(self, filename, x, y, level, type=None):
        level = int(level)
        # print(filename)
        if type == None:
            slide_region = pyvips.Region.new(self.slide)
            x = int(x)
            y = int(y)
            x1, y1 = self.multi_scale(x, y, level)
            slide_fetch = slide_region.fetch(int(x1), int(y1), self.config['patch_size'], self.config['patch_size'])

            img = np.ndarray(buffer=slide_fetch,
                             dtype=np.uint8,
                             shape=[self.config['patch_size'], self.config['patch_size'], self.slide.bands])
            # print("----------------slide-----------------")
            return img
        elif type == 'mask':
            slide_region = pyvips.Region.new(self.mask)
            x = int(x)
            y = int(y)
            x1, y1 = self.multi_scale(x, y, level)
            slide_fetch = slide_region.fetch(int(x1), int(y1), self.config['patch_size'], self.config['patch_size'])
            img = np.ndarray(buffer=slide_fetch,
                             dtype=np.uint8,
                             shape=[self.config['patch_size'], self.config['patch_size'], self.mask.bands])
            return img

    def read_wsi(self, case, tif_path):
        wsi_path = tif_path
        
        if not self.is_3D:
            mask_path = self.config['data_pkl_path'] + f'/{case}/{case}_mask.tiff'
        else:
            mask_path = self.config['3D_data_pkl_path'] + f'/{case}/{case}_mask.tiff'

        if not self.is_3D:
            self.slide = pyvips.Image.new_from_file(wsi_path, page=0)
        else:
            self.slide = pyvips.Image.new_from_file(wsi_path, level=0)

        self.mask = pyvips.Image.new_from_file(mask_path, page=0)


    def get_data(self, idx):
        case_name = os.path.basename(self.datas[idx][0]).split('.')[0]

        self.read_wsi(case_name, self.datas[idx][0])

        # data = [wsi_path, level, sx, sy]
        img_high = self.read2patch(self.datas[idx][0], self.datas[idx][2], self.datas[idx][3], level=0)
        if img_high.shape[2] == 4:
            img_high = img_high[:, :, 0:3]

        case_name = os.path.basename(self.datas[idx][0]).split('.')[0]
        gt_mask_path = self.config['data_pkl_path'] + f'/{case_name}/{case_name}_mask.tiff'
        gt_mask = self.read2patch(gt_mask_path, self.datas[idx][2], self.datas[idx][3], level=0, type='mask')
        # print(gt_mask.shape)

        if 255 in gt_mask:
            label = 1
        else:
            label = 0
        return img_high, label


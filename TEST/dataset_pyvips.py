# -*- coding: utf-8 -*-
import os
from os.path import join
import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from PIL import ImageFile
import pickle
import pyvips
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TumorDataset(Dataset):
    def __init__(self, case, config, transform):
        super().__init__()
        self.config = config
        self.transform = transform
        # read wsi and mask
        self.read_wsi(case)

        self.patch_list = self.save_all_position()

        self.total_len = len(self.patch_list)
        print("patch num: ", self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        img_high, x, y = self.get_data(idx)

        if self.transform is not None:
            img_high = self.transform(img_high)
        return img_high, x, y


    def read2patch(self,x, y, level):
        level = int(level)

        x = int(x)
        y = int(y)
        # x1, y1 = self.multi_scale(x, y, level)
        x1, y1 = x,y
        slide_fetch = self.slide_region.fetch(int(x1), int(y1),self.patch_size, self.patch_size)

        img = np.ndarray(buffer=slide_fetch,
                         dtype=np.uint8,
                         shape=[self.patch_size, self.patch_size, self.slide.bands])

        return img

    def get_data(self, idx):
        # data = [wsi_path, level, sx, sy]
        img_high = self.read2patch(self.patch_list[idx][2], self.patch_list[idx][3], level=0)

        x =  self.patch_list[idx][2]
        y = self.patch_list[idx][3]

        if img_high.shape[2] == 4:
            img_high = img_high[:, :, 0:3]

        return img_high, x, y

    def read_wsi(self,case):
        self.patch_size = self.config['patch_size']
        self.stride_size = self.config['stride_size']
        self.wsi_path = self.config['wsi_root_path'] + f'{case}.tif'
        self.mask_path = self.config["mask_path"]+f"{case}/{case}_mask.tiff"
        self.slide =pyvips.Image.new_from_file(self.wsi_path, page = 0) # high level
        self.mask = pyvips.Image.new_from_file(self.mask_path, page = 0)
        self.slide_region = pyvips.Region.new(self.slide)


    def save_all_position(self):
        start_pos = int(self.config['patch_size'] * pow(2, 2))
        data_list = []
        level = 0
        for sy in range(start_pos, self.slide.height - start_pos, self.stride_size):
            for sx in range(start_pos, self.slide.width - start_pos, self.stride_size):
                data_list.append([self.wsi_path, level, sx, sy])

        return data_list

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

        return x, y



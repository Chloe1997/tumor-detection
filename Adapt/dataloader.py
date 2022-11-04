from dataset import *
from torch.utils.data import DataLoader, random_split, Dataset
from dataset import MDataset
import torchvision.transforms as transforms


class MDataModule():
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.train_batch_size = config['train_batch_size']
        self.val_batch_size = config['val_batch_size']
        self.domain = config['domain']

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, saturation=0.15, hue=0.05, contrast=0.15)
            ], p=0.5),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        self.setup()

    def setup(self):
        if self.config["is_train"]:
            # Train-Sourse
            datasets = MDataset(self.config, self.train_transform, self.domain[0], 'train_source')
            total_len = len(datasets)
            print("LEN",total_len)
            self.train_dataset_source = datasets

            # Train-Target
            datasets = MDataset(self.config, [self.val_transform,self.train_transform], self.domain[1], 'train_target')
            # total_len = len(datasets)
            self.train_dataset_target = datasets

            datasets = MDataset(self.config, self.val_transform, self.domain[1], 'val')
            # total_len = len(datasets)
            self.valid_dataset = datasets


    def train_source_dataloader(self):
        X_sourse = DataLoader(self.train_dataset_source, batch_size=self.train_batch_size, num_workers=2,
                              shuffle=False, drop_last=True)
        return X_sourse

    def train_target_dataloader(self):
        X_target = DataLoader(self.train_dataset_target, batch_size=self.train_batch_size, num_workers=3,
                              shuffle=False, drop_last=True)
        return X_target

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.val_batch_size, num_workers= 3, drop_last=True, shuffle=False)







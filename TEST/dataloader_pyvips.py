from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset_pyvips import TumorDataset



class TumorDataModule():
    def __init__(self, config, case):
        super().__init__()
        self.config = config
        self.test_batch_size = config['test_batch_size']

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        datasets = TumorDataset(case, self.config, self.test_transform)
        self.test_dataset = datasets


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, pin_memory=True, num_workers=4, drop_last=False)







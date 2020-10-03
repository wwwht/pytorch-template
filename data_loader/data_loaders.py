from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from totch.utils.data import Dataset 
import cv2
import pandas as pd

class KonDataset(Dataset):
    """Some Information about KonDataset"""
    def __init__(self, csv_path, data_path, train_transform, val_transform, image_col_number = 1, mos_col_number = 2, val=False):
        self.is_val = val
        self.data_path = data_path
        self.csv_path = csv_path
        self.train_transform = train_transform
        self.val_transform = val_transform

        data_csv = pd.read_csv(csv_path) # data info
        self.data_dict = pd.DataFrame(data_csv,columns=['image_name','c1','c2','c3','c4','c5','c_total'])
        self.image_name_list = np.asarray(self.data_dict.iloc[:,image_col_number]) #image+name
        self.mos = np.asarray(self.data_info.iloc[:, mos_col_number]) # mos label
        self.data_len = len(self.data_info.index) # len of datasets
        # super(KonDataset, self).__init__()

    def __getitem__(self, index):
        image_name = self.image_name_list[index]
        img_path = os.path.join(self.data_path, image_name)
        img = cv2.imread(img_path)
        if not self.is_val:
            img_tfms = self.train_transform(img)
        else:
            img_tfms = self.val_transform(img)
        
        label = self.mos[index]
        return img_tfms, label

    def __len__(self):
        return self.data_len

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def get_DataLoader(data_dir, csv_path, batch_size, shuffle=True, validation_split=0.2, num_workers=4, training=True):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        normalize
    ])
    val_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        normalize
    ])
    dataset = KonDataset(csv_path, data_dir, train_transform, val_transform,image_col_number=1, mos_col_number=2, val=False)
    batch_size = batch_size
    random_seed = 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=True, 
        sampler=train_sampler,
        pin_memory=True,num_workers=2
    )
    val_loader = DataLoader(
        KonDataset(csv_path, data_dir, train_transform, val_transform,image_col_number=1, mos_col_number=2, val=True),
        batch_size = batch_size,
        shuffle = False,
        sampler = valid_sampler,
        pin_memory=True,num_workers=2
    )

    return train_loader, val_loader



    




if __name__ == "__main__":
    pass
import torch 
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import sampler

class MyDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self, df, transform, ):
        self.df = df
        self.transform = transform
        # super(MyDataset, self).__init__()

    def __getitem__(self, index):

        return

    def __len__(self):
        return self.df
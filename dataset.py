import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import *

from PIL import Image
import pandas as pd


transforms = Compose([
        RandomResizedCrop(size = 256, scale = (0.6, 1.0)),
        RandomHorizontalFlip(p = 0.5),
        RandomAffine(degrees = (-20, 20), translate = (0.1, 0.1), scale = (0.95, 1.05)),
        ColorJitter(brightness = (0.6, 1.4), contrast = (0.6, 1.4)),
        GaussianBlur(kernel_size = 3, sigma = (0.1, 3.0)),
        Resize(size = 224),
        ToTensor(),
        Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ])


class CheXpert(Dataset):

  def __init__(self, img_dir, dataframe, transform = False):
    self.img_dir = img_dir
    self.transform = transform
    self.dataframe = pd.read_csv(dataframe)

  
  def __len__(self):
    return len(self.dataframe)
  
  def __getitem__(self, idx):
    # Get image
    root_dir = 'ConVIRT'
    img = os.path.join(root_dir, self.dataframe.loc[idx, "Path"])
    img = Image.open(img)
    
    if img.mode != "RGB":
      img = img.convert("RGB")
      
    if self.transform is not None:
      return self.transform(img)
      
    # Get text label
    label = self.dataframe.loc[idx, "Report"]

    return img, label
  
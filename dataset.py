import os
os.sys.path.append('..')
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


class MyDataset(Dataset):
  
  def __init__(self, opt, is_dev=False, is_test=False):
    super().__init__()
    self.opt = opt
    self.is_test = is_test

    if is_dev:
      self.path = opt.dev_dataset
    elif is_test:
      self.path = opt.test_dataset
    else:
      self.path = opt.train_dataset

    self.data = sorted(glob.glob(self.path + '/data/*'))
    self.gt = sorted(glob.glob(self.path + '/gt/*'))

    print(self.data)
    print('\n\n==============================\n\n')
    print(self.gt)


  def __len__(self):
    return len(self.data)

  def increase_contrast(self, img):
    mean = np.mean(img)
    alpha = 1.4
    beta = 0.8
    output = img - mean
    output = output*alpha + mean*beta
    output = output / 255.
    output[np.where(output<0)] = 0
    output[np.where(output>1)] = 1
    return torch.FloatTensor(output)
 
  def __getitem__(self, index):
    img = cv2.imread(self.data[index], -1)
    img_gt = cv2.imread(self.gt[index], -1)

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img_gt = cv2.resize(img_gt, (224, 224), interpolation=cv2.INTER_AREA)

    # if self.is_test:
    #   img = self.increase_contrast(img)
    # else:
    img = torch.FloatTensor(img) / 255
    img_gt = torch.FloatTensor(img_gt) / 255

    img = img.permute(2, 0, 1)
    img_gt = img_gt.permute(2, 0, 1)

    return img, img_gt

  

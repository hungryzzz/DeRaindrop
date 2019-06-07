import os
os.sys.path.append('..')
import torch
import argparse

class Options():
  
  def __init__(self):
    self.parser = argparse.ArgumentParser()


  def initialize(self):
    self.parser.add_argument('--train_dataset', type=str, default='./dataset/train', help='path to training dataset')
    self.parser.add_argument('--dev_dataset', type=str, default='./dataset/test_a', help='path to development dataset')
    self.parser.add_argument('--test_dataset', type=str, default='./dataset/test_b', help='path to testing dataset')
    self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--iterator_number', type=int, default=10, help='number of iterator')
    self.parser.add_argument('--checkpoint_path', type=str, default='./', help='path to checkpoint')
    self.parser.add_argument('--load', type=int, default=1, help='whether to load the last model')
    self.parser.add_argument('--save', type=int, default=0, help='path to save results')    

  def parse(self):
    self.initialize()
    opt = self.parser.parse_args()
    return opt

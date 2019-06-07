import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
import cv2

import numpy as np

class GANLoss(nn.Module):

  def __init__(self, real_label=1.0, fake_label=0.0):
    super(GANLoss, self).__init__()
    self.real_label = real_label
    self.fake_label = fake_label
    self.loss = nn.BCELoss().cuda()

  def convert_tensor(self, prob, is_real):
    if is_real:
      return Variable(torch.FloatTensor(prob.shape).fill_(self.real_label)).cuda()
    else:
      return Variable(torch.FloatTensor(prob.shape).fill_(self.fake_label)).cuda()

  def __call__(self, prob, is_real):
    return self.loss(prob, self.convert_tensor(prob, is_real))

class AttentionLoss(nn.Module):

  def __init__(self, N=4, theta=0.8):
    super(AttentionLoss, self).__init__()
    self.N = N
    self.theta = theta
    self.loss = nn.MSELoss().cuda()

  def __call__(self, A, M):
    attention_loss = None
    for i in range(1, self.N+1):
      if  i == 1:
        attention_loss = pow(self.theta, float(self.N-i)) * self.loss(A[i-1], M)
      else:
        attention_loss += pow(self.theta, float(self.N-i)) * self.loss(A[i-1], M)
    return attention_loss


class MultiScaleLoss(nn.Module):

  def __init__(self, landa=[0.6, 0.8, 1.0]):
    super(MultiScaleLoss, self).__init__()
    self.landa = landa
    self.loss = nn.MSELoss().cuda()

  def __call__(self, S, gt):
    batch_s = gt.shape[0]
    T = []
    for i in range(batch_s):
      temp_list = []
      ground_truth = gt[i].permute(1, 2, 0).cpu().detach().numpy()
      ground_truth = (ground_truth*255).astype(np.uint8)
      t = cv2.resize(ground_truth, None, fx=1.0/4.0,fy=1.0/4.0, interpolation=cv2.INTER_AREA)
      t = (torch.FloatTensor(t)/255).permute(2, 0, 1).cuda()
      temp_list.append(t)
      t = cv2.resize(ground_truth, None, fx=1.0/2.0,fy=1.0/2.0, interpolation=cv2.INTER_AREA)
      t = (torch.FloatTensor(t)/255).permute(2, 0, 1).cuda()
      temp_list.append(t)
      temp_list.append(gt[i])
      T.append(temp_list)

    temp_T = []
    for i in range(3):
      temp_list = []
      for j in range(batch_s):
        temp_list.append(T[j][i])
      temp_T.append(Variable(torch.stack(temp_list)).cuda())
    T = temp_T

    multi_scale_loss = self.landa[0] * self.loss(S[0], T[0])
    for i in range(1, 3):
      multi_scale_loss += self.landa[i] * self.loss(S[i], T[i])
    return multi_scale_loss

    
class PerceptualLoss(nn.Module):

  def __init__(self):
    super(PerceptualLoss, self).__init__()
    self.model = (vgg16(pretrained=True)).cuda()
    self.trainable(self.model, False)
    self.vgg_layers = self.model.features
    self.layer_name_mapping = {
      '1': "relu1_1",
      '3': "relu1_2",
      '6': "relu2_1",
      '8': "relu2_2"
    }
    self.loss = nn.MSELoss().cuda()

  def trainable(self, net, trainable):
    for param in net.parameters():
      param.requires_grad = trainable

  def vgg_output(self, x):
    output = []
    for name, module in self.vgg_layers._modules.items():
      x = module(x)
      if name in self.layer_name_mapping:
        output.append(x)
    return output

  def __call__(self, O, T):
    vgg_O = self.vgg_output(O)
    vgg_T = self.vgg_output(T)
    output_len = len(vgg_T)

    perceptual_loss = None
    for i in range(output_len):
      if i == 0:
        perceptual_loss = self.loss(vgg_O[i], vgg_T[i]) / float(output_len)
      else:
        perceptual_loss += self.loss(vgg_O[i], vgg_T[i]) / float(output_len)
    return perceptual_loss


class MapLoss(nn.Module):

  def __init__(self, gamma=0.05):
    super(MapLoss, self).__init__()
    self.gamma = gamma
    self.loss = nn.MSELoss().cuda()

  def __call__(self, D_O, D_R, A_N):
    zeros = Variable(torch.zeros(D_R.shape)).cuda()
    return self.gamma * (self.loss(D_O, A_N) + self.loss(D_R, zeros))


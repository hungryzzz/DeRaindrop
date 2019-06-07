import os
os.sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np

from models import *
from dataset import MyDataset
from loss import *

from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

class Trainer(object):

  def __init__(self, opt):
    self.generator = Generator().cuda()
    self.discriminator = Discriminator().cuda()
    self.optimizer1 = optim.Adam(filter(lambda p : p.requires_grad, self.generator.parameters()), lr=opt.lr, betas=(0.5, 0.99)) 
    self.optimizer2 = optim.Adam(filter(lambda p : p.requires_grad, self.discriminator.parameters()), lr=opt.lr, betas=(0.5, 0.99))

    self.gan_loss_func = GANLoss()
    self.attention_loss_func = AttentionLoss()
    self.multi_scale_loss_func = MultiScaleLoss()
    self.perceptual_loss_func = PerceptualLoss()
    self.map_loss_func = MapLoss()
    self.mse_func = nn.MSELoss().cuda()

    self.iter_num = opt.iterator_number
    self.batch_size = opt.batch_size
    self.opt = opt
    self.checkpoint_path = os.path.join(opt.checkpoint_path + 'model.pkl')
    self.start = 0
    self.load_model()
    print("# iterator number : {}".format(opt.iterator_number))
    print("# batch size : {}".format(opt.batch_size))

    train_dataset = MyDataset(opt)
    dev_dataset = MyDataset(opt, is_dev=True)
    self.train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    self.dev_loader = DataLoader(dev_dataset, batch_size=opt.batch_size)
    print("# train dataset : {}".format(len(train_dataset)))
    print("# development dataset : {}".format(len(dev_dataset)))

  def load_model(self):
    if self.opt.load == 0:
      return
    try:
      checkpoint = torch.load(self.checkpoint_path)
    except FileNotFoundError:
      return
    else:
      self.start = checkpoint["epoch"]
      self.generator.load_state_dict(checkpoint["generator"])
      self.discriminator.load_state_dict(checkpoint["discriminator"])
      self.optimizer1.load_state_dict(checkpoint["optimizer1"])
      self.optimizer2.load_state_dict(checkpoint["optimizer2"])
      print("Model Loaded! From the {} epoch".format(self.start))    

  def variable_grad(self, x, is_train):
    if is_train:
      return Variable(x, requires_grad=True).cuda()
    else:
      return Variable(x).cuda()

  def get_binary_mask(self, gt, input_data):
    temp = torch.abs(gt - input_data)
    binary_mask = torch.where(temp > (30.0/255.0), torch.full_like(temp, 1), torch.full_like(temp, 0))
    binary_mask = torch.max(binary_mask, dim=0)[0].unsqueeze(0)
    return binary_mask

  def handle_tensor(self, tensor):
    output = tensor.permute(1, 2, 0).cpu().detach().numpy()
    return (output*255).astype(np.uint8)

  def forward(self, input_data, gt, is_train=True, is_test=False):
    '''
      input_data: raindrops image
      gt: ground truth
      attention_map: attention map
      frame1: autoencoder last 5th
      frame2: autoencoder last 3rd
      encoder_output: autoencoder last 1st
      O_map_2d: D(encoder_output)
      R_map_2d: D(ground truth)
    '''
    input_data = self.variable_grad(input_data, is_train)
    gt = self.variable_grad(gt, is_train)
    attention_map, frame1, frame2, encoder_output = self.generator(input_data)

    # error
    batch_size = input_data.shape[0]
    error = self.mse_func(encoder_output, gt).item()
    PSNR = 0
    SSIM = 0
    for i in range(batch_size):
      a = self.handle_tensor(encoder_output[i])
      b = self.handle_tensor(gt[i])
      PSNR += psnr(a, b)
      SSIM += ssim(a, b, multichannel=True)
    PSNR /= batch_size
    SSIM /= batch_size

    if is_train:
      # calculate loss of generator
      binary_mask = []
      for i in range(batch_size):
        binary_mask.append(self.get_binary_mask(gt[i], input_data[i]))
      binary_mask = torch.stack(binary_mask)
      binary_mask = self.variable_grad(binary_mask, is_train)

      attention_loss = self.attention_loss_func(attention_map, binary_mask)
      S = [frame1, frame2, encoder_output]
      multi_scale_loss = self.multi_scale_loss_func(S, gt)
      perceptual_loss = self.perceptual_loss_func(encoder_output, gt)

      O_map_2d, O_prob = self.discriminator(encoder_output)
      gan_loss = self.gan_loss_func(O_prob, is_real=False)
      generator_loss = -0.01*gan_loss + attention_loss + multi_scale_loss + perceptual_loss

      # calculate loss of discriminator
      R_map_2d, R_prob = self.discriminator(gt)
      map_loss = self.map_loss_func(O_map_2d, R_map_2d, attention_map[-1])
      gt_gan_loss = self.gan_loss_func(R_prob, is_real=True)
      discriminator_loss = gt_gan_loss + gan_loss + map_loss

      return generator_loss, discriminator_loss, error, PSNR, SSIM, map_loss, attention_loss, multi_scale_loss, perceptual_loss, attention_map[-1]
    elif is_test:
      return encoder_output, error, PSNR, SSIM
    else:
      return encoder_output, error, PSNR, SSIM

  def update_eval(self, eval, error, PSNR, SSIM):
    eval['error'] += error
    eval['psnr'] += PSNR
    eval['ssim'] += SSIM
    return eval

  def train(self):
    for epoch in range(self.start, self.iter_num):

      # train
      self.generator.train()
      self.generator.zero_grad()
      self.discriminator.train()
      self.discriminator.zero_grad()
      train_eval = {'error': 0, 'psnr': 0, 'ssim': 0}
      train_loader_len = len(self.train_loader)
      for i, data in enumerate(self.train_loader):
        generator_loss, discriminator_loss, error, PSNR, SSIM, map_loss, attention_loss, multi_scale_loss, perceptual_loss, attention_map = self.forward(data[0].cuda(), data[1].cuda())

        self.optimizer1.zero_grad()
        generator_loss.backward(retain_graph=True)
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        self.optimizer2.step()

        cv2.imwrite('attention/{}_{}.png'.format(epoch+1, i+1), self.get_output_img(attention_map[0]))

        train_eval = self.update_eval(train_eval, error, PSNR, SSIM)
        if (i+1) % 50 == 0:
          print('epoch : {} '.format(epoch+1) + '\titerator : {}'.format(i+1))
          # print(type(generator_loss), type(discriminator_loss))
          print("# error : {}".format(error))
          print("# PSNR : {}".format(PSNR))
          print("# SSIM : {}".format(SSIM))
          print("# attention loss : {}".format(attention_loss))
          print("# multi-scale loss : {}".format(multi_scale_loss))
          print("# perceptual loss : {}".format(perceptual_loss))
          print("# map loss : {}".format(map_loss))
          print('# generator loss : {}'.format(generator_loss))
          print('# discriminator loss : {}'.format(discriminator_loss) + "\n\n")

      print('epoch : {}'.format(epoch+1) + '\n# train dataset error : {}'.format(train_eval['error']/train_loader_len))
      print('# train dataset PSNR : {}'.format(train_eval['psnr']/train_loader_len))
      print('# train dataset PSNR : {}'.format(train_eval['ssim']/train_loader_len))

      # evaluate
      self.generator.eval()
      self.discriminator.eval()
      dev_eval = {'error': 0, 'psnr': 0, 'ssim': 0}
      dev_loader_len = len(self.dev_loader)
      for i, data in enumerate(self.dev_loader):
        output, error, PSNR, SSIM = self.forward(data[0].cuda(), data[1].cuda(), is_train=False)
        dev_eval = self.update_eval(dev_eval, error, PSNR, SSIM)
        if i < 10:
          for j in range(output.shape[0]):
            cv2.imwrite('./dev_output_{}/{}_rain_{}.png'.format(self.opt.save, 2*i+j, epoch+1), self.get_output_img(data[0][j]))
            cv2.imwrite('./dev_output_{}/{}_gt_{}.png'.format(self.opt.save, 2*i+j, epoch+1), self.get_output_img(data[1][j]))
            cv2.imwrite('./dev_output_{}/{}_predict_{}.png'.format(self.opt.save, 2*i+j, epoch+1), self.get_output_img(output[j]))

      print('epoch : {}'.format(epoch+1) + '\n# development dataset error : {}'.format(dev_eval['error']/dev_loader_len))
      print('# development dataset PSNR : {}'.format(dev_eval['psnr']/dev_loader_len))
      print('# development dataset PSNR : {}'.format(dev_eval['ssim']/dev_loader_len), "\n\n")

    #  save model
    if self.start < self.iter_num:
      state_dict = {"generator": self.generator.state_dict(), "discriminator": self.discriminator.state_dict(),
                    "optimizer1": self.optimizer1.state_dict(), "optimizer2": self.optimizer2.state_dict(),
                    "epoch": self.iter_num}
      torch.save(state_dict, self.checkpoint_path)
      print("model saved to {}".format(self.checkpoint_path) + '\n')

  def get_output_img(self, x):
    output = x.permute(1, 2, 0).cpu().detach().numpy()
    output[np.where(output<0)] = 0
    output[np.where(output>1)] = 1
    output = (output*255).astype(np.uint8)
    return output

  def test(self):
    test_dataset = MyDataset(self.opt, is_test=True)
    print('# test dataset : {}'.format(len(test_dataset)))
    test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
    self.generator.eval()
    self.discriminator.eval()
    test_eval = {'error': 0, 'psnr': 0, 'ssim': 0}
    test_loader_len = len(test_loader)
    for i, data in enumerate(test_loader):
      output, error, PSNR, SSIM = self.forward(data[0].cuda(), data[1].cuda(), is_train=False, is_test=True)
      test_eval = self.update_eval(test_eval, error, PSNR, SSIM)
      if i < 20:
        for j in range(output.shape[0]):
          # print("# output shape : {}".format(self.get_output_img(output[j]).shape))
          cv2.imwrite('./output_{}/{}_rain.png'.format(self.opt.save, 2*i+j), self.get_output_img(data[0][j]))
          cv2.imwrite('./output_{}/{}_gt.png'.format(self.opt.save, 2*i+j), self.get_output_img(data[1][j]))
          cv2.imwrite('./output_{}/{}_predict.png'.format(self.opt.save, 2*i+j), self.get_output_img(output[j]))
      
    print('# test dataset error : {}'.format(test_eval['error']/test_loader_len))
    print('# test dataset PSNR : {}'.format(test_eval['psnr']/test_loader_len))
    print('# test dataset PSNR : {}'.format(test_eval['ssim']/test_loader_len) + "\n\n")

 

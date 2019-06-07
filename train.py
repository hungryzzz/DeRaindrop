import os
os.sys.path.append('..')

from options import Options
from train_module import Trainer

opt = Options().parse()
# trainer = Trainer(opt)
# trainer.train()
# trainer.test()











from dataset import MyDataset
# from torch.utils.data import DataLoader
# # from models import * 

train_dataset = MyDataset(opt)

# train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

# print(len(train_loader))

# for i, data in enumerate(train_loader):
#   if i == 0 or i == 1:
#     print(data)
#     print("\n\n=============================\n\n")
#   else:
#     break

# input_data = train_dataset[0][0].unsqueeze(0)
# print(input_data.shape)
# print(train_dataset[0][1].shape)
# generator = Generator()
# discriminator = Discriminator()

# mask_list, frame1, frame2, x = generator(input_data)
# print("========== Mask list:")
# print(len(mask_list))
# for i in range(len(mask_list)):
#   print(i, mask_list[i].shape)
# print("\n\n=========== frame1:")
# print(type(frame1), "\t\t", frame1.shape)
# print("\n\n=========== frame2:")
# print(type(frame2), "\t\t", frame2.shape)
# print("\n\n=========== x:")
# print(type(x), "\t\t", x.shape)

# mask, prob = discriminator(x)
# print("\n\n\n=========== mask:")
# print(type(mask), "\t\t", mask.shape)
# print("\n\n\n============ prob:")
# print(prob.item(), type(prob))

# train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

# loader_len = len(train_loader)

# print("# train set : {}".format(loader_len))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
from torch.autograd import Variable
from configutils import load_config
from torchvision import datasets, transforms, utils
from mnist_outlier.read_outlier import MnistOutlier
from util34 import DeterministicConditional, GaussianConditional, WALI
from torchsummary import summary
# k = 0
# l = 0
# train_true ={}
# train_outlier = {}
# for i in range(10):
#     train_true[i] = []
#     train_outlier[i] = []
# print(o.if_outlier.shape)
# for i in range(o.if_outlier.shape[0]):
#     # print(o.train_raw[i])
#     if o.if_outlier[i] == 0:
#         train_true[o.train_raw[i]].append(i)
#     else:
#         train_outlier[o.train_raw[i]].append(i)
#     # if i==1:
#     #     print(k)
#     #     print(o.train_raw[k])
#     #     l+=1
#     # k+=1
# print(len(train_true[0]))
# print(len(train_outlier[0]))
# # for i in len()train[0]:
# #     if o.if_outlier[i] == 1:
# #         print(i)
# #         l+=1
# # print(l)
# import matplotlib.pyplot as plt
# first_img = o.train_images[train_outlier[0][0]].reshape((28,28))
# plt.imshow(first_img, cmap='gray')
# plt.show()
# print("the label: ", o.train_labels[train_outlier[0][0]])
# print("if it's outlier: ", o.if_outlier[train_outlier[0][0]]) # it's an outlier
# # from configutils import load_config,save_config
# # save_config(train_true,'true_01.json')
# # save_config(train_outlier,'outlier_01.json')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils import data
import os
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
# from util2 import DeterministicConditional, GaussianConditional, JointCritic, WALI,MMD_NET
# from util3 import DeterministicConditional, GaussianConditional, JointCritic, WALI,MMD_NET
# from util4 import DeterministicConditional, GaussianConditional, WALI

from torchvision import datasets, transforms, utils
import tensorflow as tf
cudnn.benchmark = True
crops = 10
from torch.utils.data.distributed import DistributedSampler
data_path = '/home/huaqin/VOCdevkit/VOC2007'

save_path = "/data1/JCST/results"

IMAGE_SIZE = 227
NUM_CHANNELS = 3
# DIM = 64
# NLAT = 100
# LEAK = 0.2
import tensorflow as tf

# training hyperparameters
BATCH_SIZE = 128
num_gpu = 1
BETAS = 0.01
# BETA3 = 0.00335
# BETA4 = 0.004
ITER = 20000
unit_iter = 50
# NUM_CHANNELS = 1
# 64x64
DIM = 64
# 256x256
# DIM = 256
NLAT = 256
LEAK = 0.2
RECON_lamb = 11.25  # 13.34
RECON_lamb_z = 0.36
C_ITERS = 5  # critic iterations
MMD_ITERS = 3  # mmd iterations!
EG_ITERS = 1  # encoder / generator iterations
LAMBDA = 10  # strength of gradient penalty
LEARNING_RATE = 1.3e-4
BETA1 = 0.5
BETA2 = 0.9
num_exp = 29

TANH = 0
EPOCH = 300


'''
num_exp=4,loss=rgp_gp, MMD: 1.0 0.12 EG:0.16 0.09
num_exp=5 EG update
'''
from configutils import load_config
from torchvision import datasets
import matplotlib.pyplot as plt
import skimage.io as io
from torchsummary import summary
# import cv2
from PIL import Image
import numpy as np
from torch.autograd import Variable
import torch
'''
num_exp=4,loss=rgp_gp, MMD: 1.0 0.12 EG:0.16 0.09
num_exp=5 EG update
'''

from PascalLoader import  DataLoader
root_dir = '/home/huaqin/B/'

attrs = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
attr_list = attrs.split()
cuda_list = [i.strip() for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
device_ids = [i for i in range(len(cuda_list))]
att_index = {}
index_att = {}
for i in range(len(attr_list)):
    att_index[attr_list[i]] = i+1

for key in att_index.keys():
    index_att[att_index[key]] = key
print(index_att[21].lower()+'.json')



def create_encoder():
    # 64
    mapping = nn.Sequential(
        # 64
        # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        # Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        # Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        # Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, NLAT, 1, 1))
        # 128
        # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        # Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        # Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        # Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 16, 4, 2, 1, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        # Conv2d(DIM * 16, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        # Conv2d(DIM * 16, NLAT, 1, 1)
        # 512
        # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        # Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        # Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        # Conv2d(DIM * 4, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        # Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 16, 4, 2, 1, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        # Conv2d(DIM * 16, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        # Conv2d(DIM * 16, NLAT, 1, 1)

        # 224
        Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        Conv2d(DIM * 8, DIM * 8, 5, 2, 2, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        Conv2d(DIM * 8, DIM * 16, 3, 2, 1, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        Conv2d(DIM * 16, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        Conv2d(DIM * 16, NLAT, 1, 1)
    )
    # 256
    # mapping = nn.Sequential(
    #     Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
    #     Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
    #     Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    #     # Conv2d(DIM * 4, DIM * 4, 3, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    #     Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
    #     Conv2d(DIM * 8, DIM * 8, 5, 2, 2, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
    #     # Conv2d(DIM * 8, DIM * 8, 5, 2, 2, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
    #     Conv2d(DIM * 8, DIM * 16, 3, 2, 1, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
    #     # Conv2d(DIM * 8, DIM * 8, 5, 2, 2, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
    #     Conv2d(DIM * 16, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
    #     Conv2d(DIM * 16, NLAT, 1, 1))

    return DeterministicConditional(mapping)


def create_generator():
    mapping = nn.Sequential(
        # 64
        # ConvTranspose2d(NLAT, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8), LeakyReLU(inplace=True,negative_slope=LEAK),
        # ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), LeakyReLU(inplace=True,negative_slope=LEAK),
        # ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), LeakyReLU(inplace=True,negative_slope=LEAK),
        # ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), LeakyReLU(inplace=True,negative_slope=LEAK),
        # ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh()
        # #128
        # ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM*16, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM * 8, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh()
        # 256
        ConvTranspose2d(NLAT, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8),
        LeakyReLU(inplace=True, negative_slope=LEAK),
        ConvTranspose2d(DIM * 8, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8),
        LeakyReLU(inplace=True, negative_slope=LEAK),
        ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4),
        LeakyReLU(inplace=True, negative_slope=LEAK),
        ConvTranspose2d(DIM * 4, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4),
        LeakyReLU(inplace=True, negative_slope=LEAK),
        ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2),
        LeakyReLU(inplace=True, negative_slope=LEAK),
        ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM),
        LeakyReLU(inplace=True, negative_slope=LEAK),
        ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh()
    )
    return DeterministicConditional(mapping)


class Encoder_with_drop(nn.Module):
    def __init__(self, gpu_mode, shift=None):
        super(Encoder_with_drop, self).__init__()

        self.Conv1 = Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False)
        self.B1 = BatchNorm2d(DIM)
        self.ReLU1 = ReLU(inplace=True)

        self.Conv2 = Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False)
        self.B2 = BatchNorm2d(DIM * 2)
        self.ReLU2 = ReLU(inplace=True)

        self.Conv3 = Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False)
        self.B3 = BatchNorm2d(DIM * 4)
        self.ReLU3 = ReLU(inplace=True)

        self.Conv4 = Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False)
        self.B4 = BatchNorm2d(DIM * 8)
        self.ReLU4 = ReLU(inplace=True)

        self.Conv5 = Conv2d(DIM * 8, DIM * 8, 5, 2, 2, bias=False)
        self.B5 = BatchNorm2d(DIM * 8)
        self.ReLU5 = ReLU(inplace=True)

        self.Conv6 = Conv2d(DIM * 8, DIM * 16, 3, 2, 1, bias=False)
        self.B6 = BatchNorm2d(DIM * 16)
        self.ReLU6 = ReLU(inplace=True)

        self.Conv7 = Conv2d(DIM * 16, DIM * 16, 4, 1, 0, bias=False)
        self.B7 = BatchNorm2d(DIM * 16)
        self.ReLU7 = ReLU(inplace=True)

        self.Out = Conv2d(DIM * 16, NLAT, 1, 1)

        self.gpu_mode = gpu_mode

        self.shift = shift

    def set_shift(self, value):
        if self.shift is None:
            return
        assert list(self.shift.data.size()) == list(value.size())
        self.shift.data = value

    def forward(self, input, var_beta=0.1, clip_beta=0.2):
        out1 = self.ReLU1(self.B1(self.Conv1(input)))
        # print(out1.shape)
        out2 = self.ReLU2(self.B2(self.Conv2(out1)))
        # print(out2.shape)
        out3 = self.ReLU3(self.B3(self.Conv3(out2)))
        # print(out3.shape)
        out4 = self.ReLU4(self.B4(self.Conv4(out3)))
        # print(out4.shape)
        out5 = self.ReLU5(self.B5(self.Conv5(out4)))
        # print(out5.shape)
        out6 = self.ReLU6(self.B6(self.Conv6(out5)))
        # print(out6.shape)
        out7 = self.Conv7(out6)
        # print(out7.shape)
        output = self.Out(out7)
        if self.shift is not None:
            output = output + self.shift
        return output


class Generator_with_drop(nn.Module):
    def __init__(self, gpu_mode, shift=None, tanh=1):
        super(Generator_with_drop, self).__init__()
        self.tanh = tanh
        self.ConvT1 = ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False)
        self.B1 = BatchNorm2d(DIM * 16)
        self.ReLU1 = LeakyReLU(inplace=True, negative_slope=LEAK)
        self.ConvT2 = ConvTranspose2d(DIM * 16, DIM * 8, 3, 2, 1, bias=False)
        self.B2 = BatchNorm2d(DIM * 8)
        self.ReLU2 = LeakyReLU(inplace=True, negative_slope=LEAK)
        self.ConvT3 = ConvTranspose2d(DIM * 8, DIM * 8, 4, 2, 1, bias=False)
        self.B3 = BatchNorm2d(DIM * 8)
        self.ReLU3 = LeakyReLU(inplace=True, negative_slope=LEAK)
        self.ConvT4 = ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False)
        self.B4 = BatchNorm2d(DIM * 4)
        self.ReLU4 = LeakyReLU(inplace=True, negative_slope=LEAK)
        self.ConvT5 = ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False)
        self.B5 = BatchNorm2d(DIM * 2)
        self.ReLU5 = LeakyReLU(LEAK, inplace=True)
        self.ConvT6 = ConvTranspose2d(DIM * 2, DIM, 5, 2, 1, bias=False)
        self.B6 = BatchNorm2d(DIM)
        self.ReLU6 = LeakyReLU(LEAK, inplace=True)
        self.ConvT7 = ConvTranspose2d(DIM, NUM_CHANNELS, 3, 2, 0, bias=False)

        self.act = Tanh()

        self.shift = shift

        self.gpu = gpu_mode

    def set_shift(self, value):
        if self.shift is None:
            return
        assert list(self.shift.data.size()) == list(value.size())
        self.shift.data = value

    def forward(self, input, var_beta=0.1, clip_beta=0.2):

        out1 = self.ReLU1(self.B1(self.ConvT1(input)))
        # print(out1.shape)
        out2 = self.ReLU2(self.B2(self.ConvT2(out1)))
        # print(out2.shape)
        out3 = self.ReLU3(self.B3(self.ConvT3(out2)))
        # print(out3.shape)
        out4 = self.ReLU4(self.B4(self.ConvT4(out3)))
        # print(out4.shape)
        out5 = self.ReLU5(self.B5(self.ConvT5(out4)))
        # print(out5.shape)
        out6 = self.ReLU6(self.B6(self.ConvT6(out5)))
        # print(out6.shape)
        out7 = self.ConvT7(out6)
        # print(out7.shape)
        if self.tanh:
            output = self.act(out7)
        else:
            output = out7

        if self.shift is not None:
            output = output + self.shift
        return output


def create_z_disc():
    mapping = nn.Sequential(


    Conv2d(NLAT, 1024, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(512, 256, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(256, 128, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(128, 1, 1, 1, 0)

    )
    return mapping

device = torch.device("cuda", device_ids[0])

if torch.cuda.device_count() > 1:
    num_gpu = torch.cuda.device_count()
print("use %d GPUs" % num_gpu)
if torch.cuda.device_count() < 1:
    num_gpu = 1
if not TANH:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
else:
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

train_transform = transforms.Compose(
    [transforms.RandomResizedCrop(227), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize, ])

val_transform = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(227),
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
# DataLoader initialize
train_data = DataLoader(data_path, 'trainval', transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE * num_gpu, shuffle=True,
                                           num_workers=2)

val_data = DataLoader(data_path, 'test', transform=val_transform, random_crops=crops)
val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                         batch_size=BATCH_SIZE * num_gpu,
                                         shuffle=False,
                                         num_workers=2)

noise = torch.randn(BATCH_SIZE, NLAT, 1, 1, device=device)

def create_WALI(tanh=TANH):
  # E = create_encoder()
  if torch.cuda.is_available():
      G = Generator_with_drop(gpu_mode=True,tanh=tanh)
  else:
      G = Generator_with_drop(gpu_mode=False,tanh=tanh)
  if torch.cuda.is_available():
      E = Encoder_with_drop(gpu_mode=True)
  else:
      E = Encoder_with_drop(gpu_mode=False)

  C = create_z_disc()
  # MMDX,MMDZ = create_mmds()
  wali = WALI(E,G,C,loss_type='mse')

  # wali = WALI(E, G, C,MMDX,MMDZ,channel=channel,pads=pads)
  return wali


import os
# 1600 -》 36
#
def main():
    # if not os.path.exists
    mmds = os.path.join(save_path,'aaes-norm%d' % IMAGE_SIZE)
    if not os.path.exists(os.path.join(save_path,'aaes-norm%d' % IMAGE_SIZE)):
        os.makedirs(os.path.join(save_path,'aaes-norm%d' % IMAGE_SIZE))
        print("目录创建成功！")
    if not os.path.exists(mmds+"/%d" % num_exp):
        os.makedirs(mmds+"/%d" % num_exp)
        print("目录创建成功！")
    if not os.path.exists(mmds+"/voc"):
        os.makedirs(mmds+"/voc")
        print("目录创建成功！")
    if not os.path.exists(mmds+"/voc"+"/%d" % num_exp):
        os.makedirs(mmds+"/voc"+"/%d" % num_exp)
        print("目录创建成功！")

#
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    wali = create_WALI(tanh=TANH).cuda(device=device_ids[0])

    summary(wali.get_G(), (NLAT, 1, 1))
    summary(wali.get_E(), (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        wali = torch.nn.parallel.DataParallel(wali,device_ids=device_ids)
        optimizerEG = Adam(list(wali._modules['module'].get_encoder_parameters()) + list(wali._modules['module'].get_generator_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerEG2 = Adam(list(wali._modules['module'].get_encoder_parameters()) + list(wali._modules['module'].get_generator_parameters()),
                            lr=LEARNING_RATE * 0.1, betas=(BETA1, BETA2))
        optimizerC = Adam(wali._modules['module'].get_critic_parameters(),
                          lr=LEARNING_RATE * 0.5, betas=(BETA1, BETA2))
        optimizerE2 = Adam(list(wali._modules['module'].get_encoder_parameters()),
                           lr=LEARNING_RATE * 0.5, betas=(BETA1, BETA2))
    else:
        optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerEG2 = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
                            lr=LEARNING_RATE * 0.1, betas=(BETA1, BETA2))
        optimizerC = Adam(wali.get_critic_parameters(),
                          lr=LEARNING_RATE * 0.5, betas=(BETA1, BETA2))
        optimizerE2 = Adam(list(wali.get_encoder_parameters()),
                           lr=LEARNING_RATE * 0.5, betas=(BETA1, BETA2))


    var_beta = -1
    clip_beta = -1
    threshold = 0.06
    _beta0 = 0.15
    decay = 0.9
    num_dec = 0

    C_update, EG_update, E_update = False, True, False
    EG_losses, MMD_C_losses, MMD_EG_losses, = [], [], []
    recon_losses = []
    # mss_losses, mss_x_losses, l1s = [], [], []
    # EG_losses22 = []
    curr_iter = C_iter = EG_iter1 = EG_iter2 = MMD_iter = 0
    print('Training starts...')
#     # sign_xs = []
#     # sign_
    var_beta = -1
    clip_beta = -1
    threshold = 0.06
    _beta0 = 0.15
    decay = 0.9
    num_dec = 0
    # 26_19306
    epoch = 0
    # epoch = 27
    # curr_iter = 19307
    #
    # init_x = np.load(mmds+"/%d/sample.npy" % 28)
    # init_x = torch.from_numpy(init_x)
    # init_x = init_x.type(torch.FloatTensor)
    # init_x = init_x.cuda(device=device_ids[0])
    while curr_iter < ITER and epoch <= EPOCH:
        # loader = data.DataLoader(svhn, BATCH_SIZE * num_gpu, shuffle=True,
        #                          num_workers=2)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE * num_gpu, shuffle=True,
                                                   num_workers=2)
        for batch_idx, (x, labels) in enumerate(train_loader):
            x = x.cuda(device=device_ids[0])
            if curr_iter == 0:
                init_x = x
                # if local_rank == 0:
                np.save(mmds+"/%d/sample.npy" % num_exp,init_x.cpu().numpy())
                curr_iter += 1

            z = torch.randn(x.size(0), NLAT, 1, 1).cuda(device=device_ids[0])

            if EG_update:
                recon_loss, EG_loss, C_loss = wali.forward(x, z, loss_type='mse')
                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerE2.zero_grad()

                recon_loss = recon_loss.mean()
                EG_loss = EG_loss.mean()
                C_loss = C_loss.mean()

                recon_loss.backward()
                optimizerEG.step()

                EG_iter1 += 1
                if EG_iter1 == EG_ITERS:
                    EG_iter1 = 0
                    C_update, EG_update, E_update = True, False, False
                continue

            if C_update:
                recon_loss, EG_loss, C_loss = wali.forward(x, z, loss_type='mse')
                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerE2.zero_grad()

                recon_loss = recon_loss.mean()
                EG_loss = EG_loss.mean()
                C_loss = C_loss.mean()

                C_loss.backward()
                optimizerC.step()

                C_iter += 1
                if C_iter == C_ITERS:
                    C_iter = 0
                    C_update, EG_update, E_update = False, False, True
                continue

            if E_update:
                recon_loss, EG_loss, C_loss = wali.forward(x, z, loss_type='mse')

                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerE2.zero_grad()

                recon_loss = recon_loss.mean()
                EG_loss = EG_loss.mean()
                C_loss = C_loss.mean()

                EG_loss.backward()
                optimizerE2.step()



                EG_iter2 += 1

                if EG_iter2 == EG_ITERS:
                    EG_iter2 = 0
                    C_update, EG_update, E_update = False, True, False
                    # if local_rank == 0:
                    recon_losses.append(recon_loss.item())
                    MMD_EG_losses.append(EG_loss.item())
                    MMD_C_losses.append(C_loss.item())
                    curr_iter += 1

            if curr_iter % 10 == 0:
                print(
                    '[%d/%d]\tW-distance: %.4f\tC-loss: %.4f\trecon-loss: %.4f\t'
                    % (curr_iter, ITER, MMD_EG_losses[-1], MMD_C_losses[-1], recon_losses[-1]))
                print("input max: ", x.max().item(), "input min: ", x.min().item())
            if curr_iter % 200 == 0:
                wali.eval()
                if torch.cuda.device_count() > 1:
                    real_x, rect_x = init_x[:BATCH_SIZE], wali._modules['module'].reconstruct(
                        init_x[:BATCH_SIZE]).detach_()
                else:
                    real_x, rect_x = init_x[:BATCH_SIZE], wali.reconstruct(init_x[:BATCH_SIZE]).detach_()
                rect_imgs = torch.cat((real_x.unsqueeze(1), rect_x.unsqueeze(1)), dim=1)
                rect_imgs = rect_imgs.view((BATCH_SIZE) * 2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
                if torch.cuda.device_count() > 1:
                    genr_imgs = wali._modules['module'].generate(noise).detach_().cpu()
                else:
                    genr_imgs = wali.generate(noise).detach_().cpu()

                if not TANH:
                    rect_imgs[:,0,:,:] = rect_imgs[:,0,:,:]*0.229+0.485
                    rect_imgs[:, 1, :, :] = rect_imgs[:, 1, :, :] * 0.224 + 0.456
                    rect_imgs[:, 2, :, :] = rect_imgs[:, 2, :, :] * 0.225 + 0.406

                    genr_imgs[:, 0, :, :] = genr_imgs[:, 0, :, :] * 0.229 + 0.485
                    genr_imgs[:, 1, :, :] = genr_imgs[:, 1, :, :] * 0.224 + 0.456
                    genr_imgs[:, 2, :, :] = genr_imgs[:, 2, :, :] * 0.225 + 0.406
                    utils.save_image(rect_imgs, mmds + '/%d/rect%d.png' % (num_exp, curr_iter))
                    utils.save_image(genr_imgs, mmds + '/%d/genr%d.png' % (num_exp, curr_iter))
                else:
                    utils.save_image(rect_imgs * 0.5 + 0.5, mmds + '/%d/rect%d.png' % (num_exp, curr_iter))
                    utils.save_image(genr_imgs * 0.5 + 0.5, mmds + '/%d/genr%d.png' % (num_exp, curr_iter))
                wali.train()

            # save model
            if curr_iter % (ITER // 40) == 0:
                if torch.cuda.device_count() > 1:
                    torch.save(wali.module, mmds + '/voc/%d/save_model_%d.ckpt' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].state_dict(), mmds + '/%d/total_%d.pth' % (num_exp, curr_iter))

                else:
                    torch.save(wali, mmds + '/voc/%d/save_model_%d.ckpt' % (num_exp, curr_iter))
                    torch.save(wali.state_dict(), mmds + '/%d/total_%d.pth' % (num_exp, curr_iter))


            if curr_iter % (ITER // 80) == 0:
                np.save(mmds + "/%d/EGloss.npy" % num_exp, recon_losses)
                np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
                np.save(mmds + "/%d/MMDCloss.npy" % num_exp, MMD_C_losses)


            #break
        if torch.cuda.device_count() > 1:
            torch.save(wali.module, mmds + '/voc/%d/save_model_epoch_%d_%d.ckpt' % (num_exp, epoch,curr_iter))
            torch.save(wali._modules['module'].state_dict(), mmds + '/%d/total_epoch_%d.pth' % (num_exp, epoch))
        else:
            torch.save(wali, mmds + '/voc/%d/save_model_epoch_%d_%d.ckpt' % (num_exp, epoch,curr_iter))
            torch.save(wali.state_dict(), mmds + '/%d/total_epoch_%d.pth' % (num_exp, epoch))
        epoch += 1
       # break
    # plot training loss curve
    if torch.cuda.device_count() >= 1:
        np.save(mmds + "/%d/EGloss.npy" % num_exp, recon_losses)
        np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
        np.save(mmds + "/%d/MMDCloss.npy" % num_exp, MMD_C_losses)


#
if __name__ == "__main__":
    main()


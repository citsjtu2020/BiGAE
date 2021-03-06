# from mnist_outlier.read_outlier import MnistOutlier
#
import numpy as np
import math
import math
import random
# o = MnistOutlier(0.1)
#
# # print(outlier.train_images.shape)
# print('train images:', o.train_images.shape)
# print('train labels:', o.train_labels.shape)
# print("train raw labels",o.train_raw.shape)
# print('train if outlier', o.if_outlier.shape)
# print('test images:', o.test_images.shape)
# print('test labels:', o.test_labels.shape)
# print('validation images:', o.validation_images.shape)
# print('validation labels:', o.validation_labels.shape)
# print(o.if_outlier)
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
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
# from util2 import DeterministicConditional, GaussianConditional, JointCritic, WALI,MMD_NET
# from util3 import DeterministicConditional, GaussianConditional, JointCritic, WALI,MMD_NET
from util9 import DeterministicConditional, GaussianConditional, JointCritic, WALI,MMD_NET

from torchvision import datasets, transforms, utils
# os.environ['MASTER_ADDR'] = 'localhost'
cudnn.benchmark = True
import argparse

# torch.distributed.init_process_group(backend="nccl")
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)

# 2??? ?????????????????????gpu
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)

data_path = '/home/huaqin/celeba'

save_path = "/data1/JCST/results"



IMAGE_SIZE = 256
NUM_CHANNELS = 3
# DIM = 64
# NLAT = 100
# LEAK = 0.2

# training hyperparameters
BATCH_SIZE = 40
num_gpu = 2
BETAS = 0.01
# BETA3 = 0.00335
# BETA4 = 0.004
ITER = 20000
unit_iter = 50
#NUM_CHANNELS = 1
# 64x64
DIM = 64
# 256x256
# DIM = 256
NLAT = 256
LEAK = 0.2
RECON_lamb = 10.05
RECON_lamb_z = 0.36
C_ITERS = 5       # critic iterations
MMD_ITERS = 3       # mmd iterations!
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 10       # strength of gradient penalty
LEARNING_RATE = 1.3e-4
BETA1 = 0.5
BETA2 = 0.9
# num_exp = 23
num_exp = 24
'''
num_exp=4,loss=rgp_gp, MMD: 1.0 0.12 EG:0.16 0.09
num_exp=5 EG update
'''
cuda_list = [i.strip() for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
device_ids = [i for i in range(len(cuda_list))]
from configutils import load_config
from torchvision import datasets
import matplotlib.pyplot as plt
import skimage.io as io
from torchsummary import summary
# import cv2
# from PIL import Image
import numpy as np
from torch.autograd import Variable
import torch
from torch.utils.data.distributed import DistributedSampler

# # 1) ?????????
# torch.distributed.init_process_group(backend="nccl")

root_dir = '/home/huaqin/B/'


attrs = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
attr_list = attrs.split()
att_index = {}
index_att = {}
for i in range(len(attr_list)):
    att_index[attr_list[i]] = i+1

for key in att_index.keys():
    index_att[att_index[key]] = key
print(index_att[21].lower()+'.json')


# 2800 => 54 3600 => 71(3630) 70(3580) 4200 => 82(4180) 83(4230) 4400 => 86
class CustomDataset(data.Dataset):
    def __init__(self,aims,mode='train',pos=1):
        super(CustomDataset, self).__init__()
        # self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        # self.file = pd.read_csv(csv_file,header=None,iterator=True)
        # self.subsize = 25000
        # self.max_x = 128690
        # self.min_x = -390
        self.aims = aims
        self.true_config = {}
        self.out_config = {}
        if aims < 0:
            aim_file = 'total_hq3.json'
            aim_data = load_config(aim_file)[mode][:]
        else:
            aim_file = index_att[self.aims].lower()+'_hq.json'
            aim_data = load_config(aim_file)[mode][str(int(pos*aims))]
        self.train_data = aim_data


        # # true_aimss = load_config("true_01.json")
        # # outlier_aimss = load_config("outlier_01.json")
        # keyss = list(true_aimss.keys())
        # for k in keyss:
        #     self.true_config[int(k)] = []
        #     tmps = true_aimss[k][:]
        #     self.true_config[int(k)] = tmps
        #
        #     self.out_config[int(k)] = []
        #     tmps = outlier_aimss[k][:]
        #     self.out_config[int(k)] = tmps
        # # self.aims = aims
        # self.o = MnistOutlier(0.1)
        # self.train_data = self.o.train_images
        # self.train_label = self.o.train_labels
        # self.train_raw = self.o.train_raw
        # self.if_out = self.o.if_outlier

        # self.lens = lens
        # self.max_y = 83070
        # self.mu = 12.503158925964646
        # self.std = 76.2139849775572


    def __len__(self):
        return len(self.train_data[:])
        # return 5562245
        # return 180

    def __getitem__(self, item):
        # trac = self.cs.get_chunk(128).as_matrix().astype('float')
        # .as_matrix().astype('float')
        item = item % (self.__len__())
        aim_image = self.train_data[item]
        aim_path = os.path.join(data_path,aim_image)
        item_image = io.imread(aim_path)
        item_image = np.transpose(item_image, (2, 0, 1))
        item_image = item_image / 255.0
        item_image = (item_image - 0.5)/0.5

        item_image = torch.from_numpy(item_image)
        item_image = item_image.type(torch.FloatTensor)
        # if torch.cuda.is_available():
        #     item_image = item_image.to("cuda")


        # raw_index = self.true_config[self.aims][item]
        #
        # item_data = self.train_data[raw_index].transpose(2,0,1)
        # item_label = self.train_label[raw_index]
        # item_raw = self.train_raw[raw_index]
        # item_if = self.if_out[raw_index]


        return item_image



#
def create_encoder():
    #64
    mapping = nn.Sequential(
        #64
        # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        # Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        # Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        # Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, NLAT, 1, 1))
        #128
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

    # 256
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
        #64
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

# def create_generator_with_loss():
#     # mapping = nn.Sequential(
#     #     ConvTranspose2d(NLAT, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
#     #     ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
#     #     ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
#     #     ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
#     #     ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh())
#     self.ConvT1 = ConvTranspose2d(NLAT,DIM*8,4,1,0,bias=False)
#     se
#     return DeterministicConditional(mapping)
#
'''
  mapping = nn.Sequential(
         Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        # Conv2d(DIM * 4, DIM * 4, 3, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        Conv2d(DIM * 8, DIM * 8, 5, 2, 2, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 8, 5, 2, 2, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        Conv2d(DIM * 8, DIM * 16, 3, 2, 1, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 8, 5, 2, 2, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        Conv2d(DIM * 16, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        Conv2d(DIM * 16, NLAT, 1, 1))
'''

class Encoder_with_drop(nn.Module):
    def __init__(self,gpu_mode,shift=None):
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

    def forward(self, input,var_beta=0.1,clip_beta=0.2):
        out1 = self.ReLU1(self.B1(self.Conv1(input)))
        print(out1.shape)
        out2 = self.ReLU2(self.B2(self.Conv2(out1)))
        print(out2.shape)
        out3 = self.ReLU3(self.B3(self.Conv3(out2)))
        print(out3.shape)
        out4 = self.ReLU4(self.B4(self.Conv4(out3)))
        print(out4.shape)
        out5 = self.ReLU5(self.B5(self.Conv5(out4)))
        print(out5.shape)
        out6 = self.ReLU6(self.B6(self.Conv6(out5)))
        print(out6.shape)
        out7 = self.Conv7(out6)
        print(out7.shape)
        output = self.Out(out7)
        if self.shift is not None:
            output = output + self.shift
        return output




class Generator_with_drop(nn.Module):
    def __init__(self,gpu_mode,shift=None):
        super(Generator_with_drop, self).__init__()
        #64
        # self.ConvT1 = ConvTranspose2d(NLAT,DIM*8,4,1,0,bias=False)
        # self.B1 = BatchNorm2d(DIM*8)
        # self.ReLU1 = LeakyReLU(inplace=True,negative_slope=LEAK)
        # self.ConvT2 = ConvTranspose2d(DIM*8,DIM*4,4,2,1,bias=False)
        # self.B2 = BatchNorm2d(DIM*4)
        # self.ReLU2 = LeakyReLU(inplace=True,negative_slope=LEAK)
        # self.ConvT3 = ConvTranspose2d(DIM*4,DIM*2,4,2,1,bias=False)
        # self.B3 = BatchNorm2d(DIM*2)
        # self.ReLU3 = LeakyReLU(inplace=True,negative_slope=LEAK)
        # self.ConvT4 = ConvTranspose2d(DIM*2,DIM,4,2,1,bias=False)
        # self.B4 = BatchNorm2d(DIM)
        # self.ReLU4 = LeakyReLU(LEAK,inplace=True)
        # self.ConvT5 = ConvTranspose2d(DIM,NUM_CHANNELS,4,2,1,bias=False)
        #128
        # self.ConvT1 = ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False)
        # self.B1 = BatchNorm2d(DIM * 16)
        # self.ReLU1 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT2 = ConvTranspose2d(DIM * 16, DIM * 8, 4, 2, 1, bias=False)
        # self.B2 = BatchNorm2d(DIM * 8)
        # self.ReLU2 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT3 = ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False)
        # self.B3 = BatchNorm2d(DIM * 4)
        # self.ReLU3 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT4 = ConvTranspose2d(DIM * 4, DIM*2, 4, 2, 1, bias=False)
        # self.B4 = BatchNorm2d(DIM*2)
        # self.ReLU4 = LeakyReLU(LEAK, inplace=True)
        # self.ConvT5 = ConvTranspose2d(DIM*2, DIM, 4, 2, 1, bias=False)
        # self.B5 = BatchNorm2d(DIM)
        # self.ReLU5 = LeakyReLU(LEAK,inplace=True)
        # self.ConvT6 = ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False)

        # # 224 x 224
        # self.ConvT1 = ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False)
        # self.B1 = BatchNorm2d(DIM * 16)
        # self.ReLU1 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT2 = ConvTranspose2d(DIM * 16, DIM * 8, 3, 2, 1, bias=False)
        # self.B2 = BatchNorm2d(DIM * 8)
        # self.ReLU2 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT3 = ConvTranspose2d(DIM * 8, DIM * 8, 4, 2, 1, bias=False)
        # self.B3 = BatchNorm2d(DIM * 8)
        # self.ReLU3 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT4 = ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False)
        # self.B4 = BatchNorm2d(DIM * 4)
        # self.ReLU4 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT5 = ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False)
        # self.B5 = BatchNorm2d(DIM * 2)
        # self.ReLU5 = LeakyReLU(LEAK, inplace=True)
        # self.ConvT6 = ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False)
        # self.B6 = BatchNorm2d(DIM)
        # self.ReLU6 = LeakyReLU(LEAK, inplace=True)
        # self.ConvT7 = ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False)

        #256
        # 256
        self.ConvT1 = ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False)
        self.B1 = BatchNorm2d(DIM * 16)
        self.ReLU1 = LeakyReLU(inplace=True, negative_slope=LEAK)
        self.ConvT2 = ConvTranspose2d(DIM * 16, DIM * 8, 4, 2, 1, bias=False)
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
        self.ConvT6 = ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False)
        self.B6 = BatchNorm2d(DIM)
        self.ReLU6 = LeakyReLU(LEAK, inplace=True)
        self.ConvT7 = ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False)

        # # 512
        # self.ConvT1 = ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False)
        # self.B1 = BatchNorm2d(DIM * 16)
        # self.ReLU1 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT2 = ConvTranspose2d(DIM * 16, DIM * 8, 4, 2, 1, bias=False)
        # self.B2 = BatchNorm2d(DIM * 8)
        # self.ReLU2 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT3 = ConvTranspose2d(DIM * 8, DIM * 8, 4, 2, 1, bias=False)
        # self.B3 = BatchNorm2d(DIM * 8)
        # self.ReLU3 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT4 = ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False)
        # self.B4 = BatchNorm2d(DIM * 4)
        # self.ReLU4 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT5 = ConvTranspose2d(DIM * 4, DIM * 4, 4, 2, 1, bias=False)
        # self.B5 = BatchNorm2d(DIM * 4)
        # self.ReLU5 = LeakyReLU(LEAK, inplace=True)
        # self.ConvT6 = ConvTranspose2d(DIM * 4, DIM*2, 4, 2, 1, bias=False)
        # self.B6 = BatchNorm2d(DIM*2)
        # self.ReLU6 = LeakyReLU(LEAK, inplace=True)
        # self.ConvT7 = ConvTranspose2d(DIM *2, DIM, 4, 2, 1, bias=False)
        # self.B7 = BatchNorm2d(DIM)
        # self.ReLU7 = LeakyReLU(LEAK, inplace=True)
        # self.ConvT8 = ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False)

        # ConvTranspose2d()

        # 256
        # self.ConvT1 = ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False)
        # self.B1 = BatchNorm2d(DIM*16)
        # self.ReLU1 = LeakyReLU(inplace=True,negative_slope=LEAK)
        # self.ConvT2 = ConvTranspose2d(DIM*16,DIM*8,3,2,1,output_padding=1,bias=False)
        # self.B2 = BatchNorm2d(DIM*8)
        # self.ReLU2 = LeakyReLU(inplace=True,negative_slope=LEAK)
        # self.ConvT3 = ConvTranspose2d(DIM*8,DIM*8,5,2,2,output_padding=1,bias=False)
        # self.B3 = BatchNorm2d(DIM*8)
        # self.ReLU3 = LeakyReLU(inplace=True,negative_slope=LEAK)
        # # self.ConvT4 = ConvTranspose2d(DIM * 8, DIM * 8, 3, 2, 1, bias=False)
        # # self.B4 = BatchNorm2d(DIM * 8)
        # # self.ReLU4 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT4 = ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False)
        # self.B4 = BatchNorm2d(DIM * 4)
        # self.ReLU4 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT5 = ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False)
        # self.B5 = BatchNorm2d(DIM * 2)
        # self.ReLU5 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT6 = ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False)
        # self.B6 = BatchNorm2d(DIM)
        # self.ReLU6 = LeakyReLU(inplace=True, negative_slope=LEAK)
        # self.ConvT7 = ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False)
        self.act = Tanh()

        self.shift = shift

        self.gpu = gpu_mode
        # selLeakyReLU(inplace=True,negative_slope=0.2)
        # if gpu_mode:
        #     self.noise1 = Variable(torch.FloatTensor(BATCH_SIZE, 64, 4, 4)).cuda()
        # else:
        #     self.noise1 = Variable(torch.FloatTensor(BATCH_SIZE, 64, 4, 4)).cuda()

# class Generator_with_drop(nn.Module):

    def set_shift(self, value):
        if self.shift is None:
            return
        assert list(self.shift.data.size()) == list(value.size())
        self.shift.data = value

    def forward(self, input,var_beta=0.1,clip_beta=0.2):
        # print("input:")
        # print(input.size())
        # print(self.mapping)
        out1 = self.ReLU1(self.B1(self.ConvT1(input)))
        # tmp_var1 = torch.std(out1)
        # if var_beta > 0:
        #     var1 = var_beta*(tmp_var1.item())
        #
        #     # min_out1 = torch.min(out1).item()
        #     # max_out1 = torch.max(out1).item()
        #     if self.gpu:
        #         noise1 = torch.normal(0, std=var1, size=out1.size()).cuda()
        #         # noise1 = noise1.cuda()
        #         # noise1 = torch.clamp(noise1,min=min_out1*clip_beta,max=max_out1*clip_beta)
        #         # noise1 = noise1.cuda()
        #     else:
        #         noise1 = torch.normal(0, std=var1, size=out1.size())
        #     out1 = out1+noise1
        out2 = self.ReLU2(self.B2(self.ConvT2(out1)))
        # tmp_var2 = torch.std(out2)
        # if var_beta > 0:
        #     var2 = var_beta*(tmp_var2.item())
        #
        #
        #     # min_out2 = torch.min(out2).item()
        #     # max_out2 = torch.max(out2).item()
        #     if self.gpu:
        #         noise2 = torch.normal(0, std=var2, size=out2.size()).cuda()
        #         # noise2 = torch.clamp(noise2, min=min_out2 * clip_beta, max=max_out2 * clip_beta)
        #         # noise2 = noise2.cuda()
        #     else:
        #         noise2 = torch.normal(0, std=var2, size=out2.size())
        #     out2 = out2+noise2
        out3 = self.ReLU3(self.B3(self.ConvT3(out2)))
        # tmp_var3 = torch.std(out3)
        # if var_beta > 0:
        #     var3 = var_beta*(tmp_var3.item())
        #
        #     # min_out3 = torch.min(out3).item()
        #     # max_out3 = torch.max(out3).item()
        #     if self.gpu:
        #         noise3 = torch.normal(0, std=var3, size=out3.size()).cuda()
        #
        #         # noise3 = torch.clamp(noise3, min=min_out3 * clip_beta, max=max_out3 * clip_beta)
        #         # noise3 = noise3.cuda()
        #     else:
        #         noise3 = torch.normal(0, std=var3, size=out3.size())
        #     out3 = out3+noise3
        out4 = self.ReLU4(self.B4(self.ConvT4(out3)))
        out5 = self.ReLU5(self.B5(self.ConvT5(out4)))
        out6 = self.ReLU6(self.B6(self.ConvT6(out5)))
        # out7 = self.ReLU7(self.B7(self.ConvT7(out6)))
        # tmp_var4 = torch.std(out4)
        # if var_beta > 0:
        #     var4 = var_beta*(tmp_var4.item())
        #     # min_out4 = torch.min(out4).item()
        #     # max_out4 = torch.max(out4).item()
        #     if self.gpu:
        #         noise4 = torch.normal(0, std=var4, size=out4.size()).cuda()
        #         # noise4 = torch.clamp(noise4, min=min_out4 * clip_beta, max=max_out4 * clip_beta)
        #         # noise4 = noise4.cuda()
        #     else:
        #         noise4 = torch.normal(0, std=var4, size=out4.size())
        #     out4 = out4+noise4
        #64
        # out5 = self.ConvT5(out4)
        # output = self.act(out5)
        #128
        # out6 = self.ConvT6(out5)
        # output = self.act(out6)
        #256
        out7 = self.ConvT7(out6)
        output = self.act(out7)
        #512
        # out7 = self.ReLU7(self.B7(self.ConvT7(out6)))
        # out8 = self.ConvT8(out7)
        # output = self.act(out8)
        # tmp_var5 = torch.std(out5)
        # if var_beta > 0:
        #     var5 = var_beta*tmp_var5.item()
        #     # min_out5 = torch.min(out5).item()
        #     # max_out5 = torch.max(out5).item()
        #     if self.gpu:
        #         noise5 = torch.normal(0, std=var5, size=out5.size()).cuda()
        #         # noise5 = torch.clamp(noise5, min=min_out5 * clip_beta, max=max_out5 * clip_beta)
        #         # noise5 = noise5.cuda()
        #     else:
        #         noise5 = torch.normal(0, std=var5, size=out5.size())
        #     out5 = out5 + noise5
        # output = self.act(out7)

        # noise1 = torch.clamp()
        # out1 = var1 + noise1
        # ???

        # output = self.mapping(input)
        if self.shift is not None:
            output = output + self.shift
        return output
#
#
def create_critic():
  x_mapping = nn.Sequential(
      #64
      # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 8, DIM * 8, 4, 1, 0), LeakyReLU(LEAK))
      #128
      # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 8, DIM * 16, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 16, DIM * 16, 4, 1, 0), LeakyReLU(LEAK)

  # 256
  # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
      #   # Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
      #   # Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      #   # Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      #   # Conv2d(DIM * 8, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      #   # Conv2d(DIM * 8, DIM * 16, 4, 2, 1), LeakyReLU(LEAK),
      #   # Conv2d(DIM * 16, DIM * 16, 4, 1, 0), LeakyReLU(LEAK)

      Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM * 8, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM * 8, DIM * 16, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM * 16, DIM * 16, 4, 1, 0), LeakyReLU(LEAK)


    #512
      # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 4, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 8, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 8, DIM * 16, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 16, DIM * 16, 4, 1, 0), LeakyReLU(LEAK)
  )

  z_mapping = nn.Sequential(
      #64
      # Conv2d(NLAT, 256, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(256, 256, 1, 1, 0), LeakyReLU(LEAK),
      #128
      # Conv2d(NLAT, 256, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(256, 512, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK)

      #256
      Conv2d(NLAT, 256, 1, 1, 0), LeakyReLU(LEAK),
      Conv2d(256, 1024, 1, 1, 0), LeakyReLU(LEAK),
      Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK)

      # Conv2d(NLAT, 256, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(256, 1024, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK)
    )

  joint_mapping = nn.Sequential(
      #64
      # Conv2d(DIM * 8 + 256, 1024, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(1024, 1024, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(512, 1, 1, 1, 0))
      # #128
      # Conv2d(DIM * 16 + 512, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 1024, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(512, 1, 1, 1, 0)

      # 256

      Conv2d(DIM * 16 + 512, 2048, 1, 1, 0), LeakyReLU(LEAK),
      Conv2d(2048, 2048, 1, 1, 0), LeakyReLU(LEAK),
      Conv2d(2048, 1024, 1, 1, 0), LeakyReLU(LEAK),
      Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
      Conv2d(512, 1, 1, 1, 0)

      # Conv2d(DIM * 16 + 256, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 1024, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(512, 1, 1, 1, 0)
  )


  return JointCritic(x_mapping, z_mapping, joint_mapping)
#
def create_mmds():
    mmd_x = nn.Sequential(
        #64
        # Conv2d(DIM * 8,DIM*16,1,1,0),LeakyReLU(LEAK),
        # Conv2d(DIM*16,DIM*8,1,1,0),LeakyReLU(LEAK),
        # Conv2d(DIM * 8, DIM * 4, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(DIM*4,16,1,1,0)
        #128
        # Conv2d(DIM * 16, DIM * 16, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(DIM * 16, DIM * 8, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(DIM * 8, DIM * 4, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(DIM * 4, 16, 1, 1, 0)

        ##256
        Conv2d(DIM * 16, DIM * 16, 1, 1, 0), LeakyReLU(LEAK),
        Conv2d(DIM * 16, DIM * 8, 1, 1, 0), LeakyReLU(LEAK),
        Conv2d(DIM * 8, DIM * 4, 1, 1, 0), LeakyReLU(LEAK),
        Conv2d(DIM * 4, 16, 1, 1, 0)
    )

    mmd_z = nn.Sequential(
        #64
        # Conv2d(256,512,1,1,0),LeakyReLU(LEAK),
        # Conv2d(512,128,1,1,0),LeakyReLU(LEAK),
        # Conv2d(128,16,1,1,0)
        # #128
        # Conv2d(512, 1024, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(1024, 256, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(256, 128, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(128, 32, 1, 1, 0)

        # 256
        Conv2d(512, 2048, 1, 1, 0), LeakyReLU(LEAK),
        Conv2d(2048, 1024, 1, 1, 0), LeakyReLU(LEAK),
        Conv2d(1024, 256, 1, 1, 0), LeakyReLU(LEAK),
        Conv2d(256, 32, 1, 1, 0)
    )
    return MMD_NET(mmd_x),MMD_NET(mmd_z)
#
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
device = torch.device("cuda", device_ids[0])

if torch.cuda.device_count() > 1:
    num_gpu = torch.cuda.device_count()
print("use %d GPUs" % num_gpu)
if torch.cuda.device_count() < 1:
    num_gpu = 1
svhn = CustomDataset(aims=-1)
#     # svhn = datasets.CIFAR10('data/CIFAR10', train=True, transform=transform)
# loader = data.DataLoader(svhn, BATCH_SIZE//num_gpu,sampler=DistributedSampler(svhn))
loader = data.DataLoader(svhn,BATCH_SIZE*num_gpu,shuffle=True,
                                                num_workers=2)
# # ??????batch size * ??????
#                                                 batch_size=BATCH_SIZE * len(device_ids),
# loader2 = data.DataLoader(svhn, BATCH_SIZE//num_gpu,sampler=DistributedSampler(svhn))
# //num_gpu
noise = torch.randn(BATCH_SIZE, NLAT, 1, 1, device=device)
def create_WALI():
    if torch.cuda.is_available():
        G = Generator_with_drop(gpu_mode=True)
    else:
        G = Generator_with_drop(gpu_mode=False)
    E = create_encoder()
    C = create_critic()
    MMDX,MMDZ = create_mmds()

    wali = WALI(E, G, C,MMDX,MMDZ,loss_type="per",window_size=11,size_average=True,val_range=2,l1=False,l2=True,pads=False)
    return wali
import os

def main():
    # if not os.path.exists
    mmds = os.path.join(save_path,'mmds-pertucal%d' % IMAGE_SIZE)
    if not os.path.exists(os.path.join(save_path,'mmds-pertucals%d' % IMAGE_SIZE)):
        os.makedirs(os.path.join(save_path,'mmds-pertucals%d' % IMAGE_SIZE))
        print("?????????????????????")
    if not os.path.exists(mmds+"/%d" % num_exp):
        os.makedirs(mmds+"/%d" % num_exp)
        print("?????????????????????")
    if not os.path.exists(mmds+"/celeba"):
        os.makedirs(mmds+"/celeba")
        print("?????????????????????")
    if not os.path.exists(mmds+"/celeba"+"/%d" % num_exp):
        os.makedirs(mmds+"/celeba"+"/%d" % num_exp)
        print("?????????????????????")
#
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 4) ???????????????????????????????????????gpu
    # model.to(device)
    wali = create_WALI().cuda(device=device_ids[0])
    # total_epoch_192.pth
    # /data1/JCST/results/mmds-pertucal256/celeba/22
    # save_model_epoch_239_12122.ckpt
    # now_model = torch.load("/data1/JCST/results/mmds-pertucal256/celeba/22/total_epoch_215.pth",map_location="cuda:0")
    # now_model = torch.load("/data1/JCST/results/mmds-pertucal256/celeba/23/total_epoch_239.pth",map_location="cuda:0")
    # save_model_epoch_261_13234.ckpt
    now_model = torch.load("/data1/JCST/results/mmds-pertucal256/celeba/24/total_epoch_261.pth",map_location="cuda:0")
    # save_model_epoch_276_13992.ckpt
    now_model = torch.load("/data1/JCST/results/mmds-pertucal256/celeba/24/total_epoch_276.pth", map_location="cuda:0")
    # save_model_epoch_326_16518.ckpt
    now_model = torch.load("/data1/JCST/results/mmds-pertucal256/celeba/24/total_epoch_326.pth", map_location="cuda:0")
    wali.load_state_dict(now_model)
    # if local_rank == 0:
    summary(wali.get_G(),(NLAT,1,1))
    summary(wali.get_E(),(NUM_CHANNELS,IMAGE_SIZE,IMAGE_SIZE))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) ??????
        wali = torch.nn.parallel.DataParallel(wali,device_ids=device_ids)
        # wali= torch.nn.parallel.DistributedDataParallel(wali,
        #                                                   device_ids=[local_rank],
        #                                                   output_device=local_rank,find_unused_parameters=True)

        optimizerEG = Adam(list(wali._modules['module'].get_encoder_parameters()) + list(
            wali._modules['module'].get_generator_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        # optimizerEG2 = Adam(list(wali._modules['module'].get_encoder_parameters()) + list(wali._modules['module'].get_generator_parameters()),
        #                    lr=LEARNING_RATE*0.25, betas=(BETA1, BETA2))
        optimizerEG2 = Adam(list(wali._modules['module'].get_encoder_parameters()),
                            lr=LEARNING_RATE * 0.25, betas=(BETA1, BETA2))
        optimizerEG3 = Adam(list(wali._modules['module'].get_encoder_parameters()) + list(
            wali._modules['module'].get_generator_parameters()),
                            lr=LEARNING_RATE*0.952, betas=(BETA1, BETA2))
        # * 1.5
        optimizerE2 = Adam(list(wali._modules['module'].get_encoder_parameters()),
                           lr=LEARNING_RATE * 0.37, betas=(BETA1, BETA2))
        optimizerC = Adam(wali._modules['module'].get_critic_parameters(),
                          lr=LEARNING_RATE, betas=(BETA1, BETA2))

        optimizerXM = Adam(list(wali._modules['module'].get_C().get_x_net_parameters()) + list(
            wali._modules['module'].get_mmdx_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerZM = Adam(list(wali._modules['module'].get_C().get_z_net_parameters()) + list(
            wali._modules['module'].get_mmdz_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerXZM = Adam(list(wali._modules['module'].get_C().get_x_net_parameters()) + list(
            wali._modules['module'].get_C().get_z_net_parameters()) + list(
            wali._modules['module'].get_mmdx_parameters()))
        optimizerMox = Adam(list(wali._modules['module'].get_mmdx_parameters()), lr=LEARNING_RATE * 0.5,
                            betas=(BETA1, BETA2))
        optimizerMoz = Adam(list(wali._modules['module'].get_mmdz_parameters()), lr=LEARNING_RATE * 0.5,
                            betas=(BETA1, BETA2))
#
    else:
        optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        # optimizerEG2 = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
        #                    lr=LEARNING_RATE*0.25, betas=(BETA1, BETA2))
        optimizerEG2 = Adam(list(wali.get_encoder_parameters()),
                            lr=LEARNING_RATE * 0.25, betas=(BETA1, BETA2))
        optimizerEG3 = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
                            lr=LEARNING_RATE, betas=(BETA1, BETA2))
        # * 1.5
        optimizerE2 = Adam(list(wali.get_encoder_parameters()),
                           lr=LEARNING_RATE * 0.34, betas=(BETA1, BETA2))
        optimizerC = Adam(wali.get_critic_parameters(),
                          lr=LEARNING_RATE, betas=(BETA1, BETA2))

        optimizerXM = Adam(list(wali.get_C().get_x_net_parameters()) + list(wali.get_mmdx_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerZM = Adam(list(wali.get_C().get_z_net_parameters()) + list(wali.get_mmdz_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerXZM = Adam(
            list(wali.get_C().get_x_net_parameters()) + list(wali.get_C().get_z_net_parameters()) + list(
                wali.get_mmdx_parameters()))
        optimizerMox = Adam(list(wali.get_mmdx_parameters()), lr=LEARNING_RATE * 0.5, betas=(BETA1, BETA2))
        optimizerMoz = Adam(list(wali.get_mmdz_parameters()), lr=LEARNING_RATE * 0.5, betas=(BETA1, BETA2))
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))])

#
#     # EG_losses, C_losses, Recon_x_losses, Recon_z_losses, EG_losses2 = [], [], [], [], []
#     EG_losses, C_losses, MMD_C_losses,MMD_EG_losses,Recon_z_losses, EG_losses2,mmd_penaltys = [], [], [], [], [],[],[]
    EG_losses, C_losses, MMD_C_losses,MMD_EG_losses,Recon_z_losses, EG_losses2,mmd_penaltys = [], [], [], [], [],[],[]
    per_losses,feature_losses,style_losses = [],[],[]
    Reconxs = []
    Reconzs = []
    EG_losses22 = []
    curr_iter = C_iter = EG_iter1=EG_iter2=MMD_iter= 0

#     alphas = []
    total_block = 0
#
    min_total_x = float('inf')
    min_total_z = float('inf')
    C_update, MMD_UPDATE,EG_update1,EG_update2 = True, False,False,False
    block_step = C_ITERS+EG_ITERS+MMD_ITERS+EG_ITERS
    print('Training starts...')
#     # sign_xs = []
#     # sign_
    var_beta = -1
    clip_beta = -1
    threshold = 0.06
    _beta0 = 0.15
    decay = 0.9
    num_dec = 0

    # 23:
    # epoch = 216
    # curr_iter = 10910

    # 24:
    # save_model_epoch_239_12122.ckpt
    # epoch = 240
    # curr_iter = 12123

    # save_model_epoch_276_13992.ckpt

    # epoch = 262
    # curr_iter = 13235

    # epoch = 277
    # curr_iter = 13993

    # save_model_epoch_326_16518.ckpt
    epoch = 327
    curr_iter = 16519
    # num_exp
    init_x = np.load(mmds+"/%d/sample.npy" % 22)
    init_x = torch.from_numpy(init_x)
    init_x = init_x.type(torch.FloatTensor)
    init_x = init_x.cuda(device=device_ids[0])

    while curr_iter < ITER:
        # loader.sampler.set_epoch(epoch)
        # loader2.sampler.set_epoch(epoch)
        loader = data.DataLoader(svhn, BATCH_SIZE * num_gpu, shuffle=True,
                                 num_workers=2)
        for batch_idx, x in enumerate(loader, 1):
            # iter(loader2).
            x = x.cuda(device=device_ids[0])
            # print(x.size())
#
            if curr_iter == 0:
                init_x = x
                # if local_rank == 0:
                np.save(mmds+"/%d/sample.npy" % num_exp,init_x.cpu().numpy())
                curr_iter += 1
#
            z = torch.randn(x.size(0), NLAT, 1, 1).cuda(device=device_ids[0])
            '''
                def forward(self, x, z, lamb=10,beta1=0.1,beta2=0.1):
              z_hat, x_tilde = self.encode(x), self.generate(z)
              x_tilde_copy = x_tilde.clone().detach()
              z_recon = self.encode(x_tilde_copy)
              z_hat_copy = z_hat.clone().detach()
              x_recon = self.generate(z_hat_copy)


              data_preds, sample_preds = self.criticize(x, z_hat, x_tilde, z)
              EG_loss = torch.mean(data_preds - sample_preds)
              C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, z_hat.data, x_tilde.data, z.data)
              RECON_X_loss = F.mse_loss(input=x_recon,target=x,reduction='mean')
              RECON_Z_loss = F.mse_loss(input=z_recon,target=z,reduction='mean')
              EG_loss2 = EG_loss+beta1*RECON_X_loss+beta2*RECON_Z_loss
              return C_loss, EG_loss,RECON_X_loss,RECON_Z_loss,EG_loss2
            '''
            if C_update:
                #C_loss,EG_loss,mss_loss,l1_conv,mss_x_loss,EG_loss2
                # wali.forward(x=,z=,lamb=,beta1=,beta2=,gan=,loss_type=)
                C_loss, EG_loss = wali.forward(x=x,z=z,lamb=LAMBDA,gan=0,loss_type='raw',beta1=0.2,beta2=0.28,beta3=0.72,methods=0,l1=False,l2=True,val_range=2,normalize="relu", pads=False,ssm_alpha=0.675)
                # print(C_loss)
                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()
                # print(C_loss)
                # C_loss, EG_loss = results[0],results[1]
                # EG_loss2 = beta1*(mss_loss*beta3+RECON_Z_loss*beta2)+EG_loss
                #                 if l1:
                #                     return C_loss,EG_loss,mss_loss,l1_conv,mss_x_loss,EG_loss2,RECON_Z_loss
                # def forward(self, x, z, lamb=10, beta1=0.01, beta2=0.01, beta3=0.03, gan=0, loss_type='raw',
                #             var_beta=-1, clip_beta=-1, methods=0, l1=True, var_lange=2, normalize="relu", pads=False,
                #             ssm_alpha=0.84):

                # C_loss, EG_loss = wali.forward(x=x,z=z,lamb=LAMBDA,gan=0,loss_type='raw',methods=1,var_beta=var_beta,clip_beta=clip_beta)
                    # wali(x, z, lamb=LAMBDA,gan=0,loss_type='raw')



                optimizerC.zero_grad()
                optimizerEG2.zero_grad()
                optimizerEG.zero_grad()
                optimizerMoz.zero_grad()
                optimizerEG3.zero_grad()
                optimizerE2.zero_grad()

                C_loss.backward()
                optimizerC.step()

                if MMD_iter < MMD_ITERS:
                    C_lossk, EG_lossk, mmd_penalty = wali.forward(x=x, z=z, lamb=LAMBDA, beta1=0.8, beta2=0.62, beta3=1.0,
                                                                  gan=1, loss_type='rep_gp', methods=0,
                                                                  var_beta=var_beta,
                                                                  clip_beta=clip_beta)
                    # C_lossk, EG_lossk, mmd_penalty = results[0],results[1],results[2]
                    C_lossk = C_lossk.mean()
                    EG_lossk = EG_lossk.mean()
                    mmd_penalty = mmd_penalty.mean()
                    optimizerC.zero_grad()
                    optimizerEG2.zero_grad()
                    optimizerEG.zero_grad()
                    optimizerMoz.zero_grad()
                    optimizerEG3.zero_grad()
                    optimizerE2.zero_grad()

                    C_lossk.backward()
                    optimizerMoz.step()

                # C_lossk, EG_lossk, mmd_penalty = wali.forward(x=x, z=z, lamb=LAMBDA, beta1=1, beta2=0.62, beta3=1.0,
                #                                               gan=1, loss_type='rep_gp', methods=0, var_beta=var_beta,
                #                                               clip_beta=clip_beta)
                # optimizerMox.zero_grad()
                # C_lossk.backward()
                # optimizerMox.step()



                # print("C_update")
                C_iter += 1
                MMD_iter+=1

                if C_iter == C_ITERS:
                    C_iter = 0
                    MMD_iter = 0
                    C_update,MMD_UPDATE,EG_update1,EG_update2 = False, False,True,False
                continue

            if EG_update1:
                optimizerC.zero_grad()
                optimizerEG2.zero_grad()
                optimizerEG.zero_grad()
                optimizerMoz.zero_grad()
                optimizerEG3.zero_grad()
                optimizerE2.zero_grad()

                # C_loss,EG_loss,per_loss,feature_loss,style_loss,EG_loss2,RECON_Z_loss

                C_loss, EG_loss, per_loss,feature_loss,style_loss, EG_loss2, RECON_Z_loss, RECON_X_loss = wali.forward(x=x,z=z,lamb=LAMBDA,gan=2,loss_type='per',beta1=0.2,beta2=0.28,beta3=0.72,methods=0,l1=True,l2=False,val_range=2,normalize="relu", pads=False,ssm_alpha=0.675)
                # C_loss, EG_loss, mss_loss, l1_conv, mss_x_loss, EG_loss2, RECON_Z_loss, RECON_X_loss = results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7]
                # C_loss, EG_loss = wali.forward(x=x, z=z, lamb=LAMBDA, gan=0, loss_type='raw',methods=1,var_beta=var_beta,clip_beta=clip_beta)
                # EG_loss.backward()
                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()
                per_loss = per_loss.mean()
                feature_loss = feature_loss.mean()
                style_loss = style_loss.mean()
                EG_loss2 = EG_loss2.mean()
                RECON_Z_loss = RECON_Z_loss.mean()
                RECON_X_loss = RECON_X_loss.mean()

                EG_loss2.backward()
                optimizerEG.step()

                optimizerC.zero_grad()
                optimizerEG2.zero_grad()
                optimizerEG.zero_grad()
                optimizerMoz.zero_grad()
                optimizerEG3.zero_grad()

                # optimizerEG2.zero_grad()
                # C_loss,EG_loss,per_loss,feature_loss,style_loss,EG_loss2,RECON_Z_loss
                C_loss, EG_loss, per_loss,feature_loss,style_loss, EG_loss2, RECON_Z_loss, RECON_X_loss = wali.forward(x=x,z=z,lamb=LAMBDA,gan=2,loss_type='per',beta1=0.2,beta2=0.28,beta3=0.72,methods=0,l1=True,l2=False,val_range=2,normalize="relu", pads=False,ssm_alpha=0.675)
                # C_loss, EG_loss, mss_loss, l1_conv, mss_x_loss, EG_loss2, RECON_Z_loss, RECON_X_loss = results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7]
                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()
                per_loss = per_loss.mean()
                feature_loss = feature_loss.mean()
                style_loss = style_loss.mean()
                EG_loss2 = EG_loss2.mean()
                RECON_Z_loss = RECON_Z_loss.mean()
                RECON_X_loss = RECON_X_loss.mean()

                RECON_X_loss2 = RECON_lamb*RECON_X_loss
                # RECON_Z_loss2 = RECON_Z_loss*RECON_lamb_z
                # RECON_Loss = RECON_X_loss2 + RECON_Z_loss2
                # RECON_Z_loss.backward()

                # optimizerE2.zero_grad()
                RECON_X_loss2.backward()
                optimizerEG3.step()

                optimizerC.zero_grad()
                optimizerEG2.zero_grad()
                optimizerEG.zero_grad()
                optimizerMoz.zero_grad()
                optimizerEG3.zero_grad()
                optimizerE2.zero_grad()

                # C_loss,EG_loss,per_loss,feature_loss,style_loss,EG_loss2,RECON_Z_loss
                C_loss, EG_loss, per_loss,feature_loss,style_loss, EG_loss2, RECON_Z_loss, RECON_X_loss = wali.forward(x=x,z=z,lamb=LAMBDA,gan=2,loss_type='per',beta1=0.2,beta2=0.3,beta3=0.7,methods=0,l1=True,l2=False,val_range=2,normalize="relu", pads=False,ssm_alpha=0.675)
                # C_loss, EG_loss, mss_loss, l1_conv, mss_x_loss, EG_loss2, RECON_Z_loss, RECON_X_loss = results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7]

                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()
                per_loss = per_loss.mean()
                feature_loss = feature_loss.mean()
                style_loss = style_loss.mean()
                EG_loss2 = EG_loss2.mean()
                RECON_Z_loss = RECON_Z_loss.mean()
                RECON_X_loss = RECON_X_loss.mean()
                RECON_Z_loss2 = RECON_Z_loss*RECON_lamb_z

                RECON_Z_loss2.backward()
                optimizerE2.step()

                # if local_rank == 0:
                C_losses.append(C_loss.item())
                EG_losses.append(EG_loss.item())
                per_losses.append(per_loss.item())
                feature_losses.append(feature_loss.item())
                Reconxs.append(RECON_X_loss2)
                Reconzs.append(RECON_Z_loss)
                style_losses.append(style_loss.item())
                EG_losses2.append(EG_loss2.item())

                optimizerC.zero_grad()
                optimizerEG2.zero_grad()
                optimizerEG3.zero_grad()
                optimizerEG.zero_grad()
                optimizerMoz.zero_grad()
                optimizerE2.zero_grad()

                # RECON_Z_loss,EG_loss2,
                C_loss, EG_loss, mmd_penalty = wali.forward(x=x,z=z,lamb=LAMBDA,beta1=0.34,beta2=0.12,gan=1,loss_type='reg-gp',methods=1,var_beta=var_beta,clip_beta=clip_beta)
                # C_loss, EG_loss, mmd_penalty = results[0],results[1],results[2]
                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()
                mmd_penalty = mmd_penalty.mean()


                EG_loss.backward()
                optimizerEG2.step()
                # if local_rank == 0:
                MMD_C_losses.append(C_loss.item())
                MMD_EG_losses.append(EG_loss.item())
                Recon_z_losses.append(RECON_Z_loss.item())
                # EG_losses22.append(EG_loss2.item())
                mmd_penaltys.append(mmd_penalty.item())



                # print("EG update1")
                EG_iter1 += 1
                if EG_iter1 == EG_ITERS:
                    EG_iter1 = 0
                    C_update, MMD_UPDATE, EG_update1, EG_update2 = True, False, False, False
                    curr_iter+=1
                    # C_update,MMD_UPDATE,EG_update1,EG_update2 = False,True,False,False
                # continue
            #     and local_rank == 0
            if curr_iter % 10 == 0:
                # print(EG_loss2)
                print("Outside: input size", x.size(),
                "z_size", z.size())
                print(
                    '[%d/%d]\tW-distance: %.4f\tC-loss: %.4f\tperceptual loss:%.4f\tfeature loss:%.4f\tstyle loss:%.4f\tMMD-EG-loss: %.4f\tMMD-C-loss: %.4f\treconz-loss: %.4f\tmmd_penalty:%.4f\ttotal_gen-loss:%.4f\tRECON_x:%.4f'
                    % (curr_iter, ITER, EG_losses[-1], C_losses[-1],per_losses[-1],feature_losses[-1],style_losses[-1],MMD_EG_losses[-1],MMD_C_losses[-1],Reconzs[-1],mmd_penaltys[-1],
                       EG_losses2[-1],Reconxs[-1]))

                # plot reconstructed images and samples
            #     and local_rank == 0
            if curr_iter % 200 == 0 :
                wali.eval()
                # //num_gpu
                # //num_gpu
                if torch.cuda.device_count() > 1:
                    real_x, rect_x = init_x[:BATCH_SIZE], wali._modules['module'].reconstruct(init_x[:BATCH_SIZE]).detach_()
                else:
                    real_x, rect_x = init_x[:BATCH_SIZE], wali.reconstruct(init_x[:BATCH_SIZE]).detach_()
                rect_imgs = torch.cat((real_x.unsqueeze(1), rect_x.unsqueeze(1)), dim=1)
                rect_imgs = rect_imgs.view((BATCH_SIZE)*2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
                if torch.cuda.device_count() > 1:
                    genr_imgs = wali._modules['module'].generate(noise).detach_().cpu()
                else:
                    genr_imgs = wali.generate(noise).detach_().cpu()
                utils.save_image(rect_imgs * 0.5 + 0.5, mmds+'/celeba/%d/rect%d.png' % (num_exp,curr_iter))
                utils.save_image(genr_imgs * 0.5 + 0.5, mmds+'/celeba/%d/genr%d.png' % (num_exp,curr_iter))
                wali.train()
#
            # save model
            # and local_rank==0
            if curr_iter % (ITER // 20) == 0 :
                if torch.cuda.device_count() > 1:
                    torch.save(wali.module, mmds + '/celeba/%d/save_model_%d.ckpt' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].state_dict(),
                               mmds + '/celeba/%d/total_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_G().state_dict(),
                               mmds + '/celeba/%d/G_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_C().state_dict(),
                               mmds + '/celeba/%d/C_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_E().state_dict(),
                               mmds + '/celeba/%d/E_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_MMDX().state_dict(),
                               mmds + '/celeba/%d/MMDX_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_MMDZ().state_dict(),
                               mmds + '/celeba/%d/MMDZ_%d.pth' % (num_exp, curr_iter))
                else:
                    torch.save(wali.module, mmds + '/celeba/%d/save_model_%d.ckpt' % (num_exp, curr_iter))
                    torch.save(wali.state_dict(), mmds + '/celeba/%d/total_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_G().state_dict(), mmds + '/celeba/%d/G_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_C().state_dict(), mmds + '/celeba/%d/C_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_E().state_dict(), mmds + '/celeba/%d/E_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_MMDX().state_dict(), mmds + '/celeba/%d/MMDX_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_MMDZ().state_dict(), mmds + '/celeba/%d/MMDZ_%d.pth' % (num_exp, curr_iter))
            # and local_rank==0
            if curr_iter % (ITER // 40) == 0:
                np.save(mmds + "/%d/EGloss.npy" % num_exp, EG_losses)
                np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
                np.save(mmds + "/%d/MMDCloss.npy" % num_exp, MMD_C_losses)
                np.save(mmds + "/%d/mmdpenalty.npy" % num_exp, mmd_penaltys)
                np.save(mmds + "/%d/EGloss_reg.npy" % num_exp, EG_losses2)
                np.save(mmds + "/%d/MMDEGloss_reg.npy" % num_exp, EG_losses22)
                np.save(mmds + "/%d/style_loss.npy" % num_exp, style_losses)
                np.save(mmds + "/%d/per_loss.npy" % num_exp, per_losses)
                np.save(mmds + "/%d/feature_loss.npy" % num_exp, feature_losses)
                np.save(mmds + "/%d/reconxs.npy" % num_exp, Reconxs)
                np.save(mmds + "/%d/Closs.npy" % num_exp, C_losses)

                np.save(mmds + "/%d/RECONZ.npy" % num_exp, Recon_z_losses)

            #break
       # break
        if torch.cuda.device_count() > 1:
            torch.save(wali.module, mmds + '/celeba/%d/save_model_epoch_%d_%d.ckpt' % (num_exp, epoch, curr_iter))
            torch.save(wali._modules['module'].state_dict(), mmds + '/celeba/%d/total_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_G().state_dict(), mmds + '/celeba/%d/G_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_C().state_dict(), mmds + '/celeba/%d/C_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_E().state_dict(), mmds + '/celeba/%d/E_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_MMDX().state_dict(), mmds + '/celeba/%d/MMDX_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_MMDZ().state_dict(), mmds + '/celeba/%d/MMDZ_epoch_%d.pth' % (num_exp, epoch))
        else:
            torch.save(wali.module, mmds + '/celeba/%d/save_model_epoch_%d_%d.ckpt' % (num_exp, epoch, curr_iter))
            torch.save(wali.state_dict(), mmds + '/celeba/%d/total_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_G().state_dict(), mmds + '/celeba/%d/G_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_C().state_dict(), mmds + '/celeba/%d/C_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_E().state_dict(), mmds + '/celeba/%d/E_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_MMDX().state_dict(), mmds + '/celeba/%d/MMDX_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_MMDZ().state_dict(), mmds + '/celeba/%d/MMDZ_epoch_%d.pth' % (num_exp, epoch))

        epoch = epoch+1
    # plot training loss curve
    if torch.cuda.device_count() >= 1:
        np.save(mmds + "/%d/EGloss.npy" % num_exp, EG_losses)
        np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
        np.save(mmds + "/%d/MMDCloss.npy" % num_exp, MMD_C_losses)
        np.save(mmds + "/%d/mmdpenalty.npy" % num_exp, mmd_penaltys)
        np.save(mmds + "/%d/EGloss_reg.npy" % num_exp, EG_losses2)
        np.save(mmds + "/%d/MMDEGloss_reg.npy" % num_exp, EG_losses22)
        np.save(mmds + "/%d/style_loss.npy" % num_exp, style_losses)
        np.save(mmds + "/%d/per_loss.npy" % num_exp, per_losses)
        np.save(mmds + "/%d/feature_loss.npy" % num_exp, feature_losses)
        np.save(mmds + "/%d/reconxs.npy" % num_exp, Reconxs)
        np.save(mmds + "/%d/Closs.npy" % num_exp, C_losses)
        np.save(mmds + "/%d/RECONZ.npy" % num_exp, Recon_z_losses)
        EG_losses = np.load(mmds + "/%d/EGloss.npy" % num_exp)
        C_losses = np.load(mmds + "/%d/Closs.npy" % num_exp)
    # plt.show()
#
#
#
#
# if __name__ == "__main__":
#     main()
#   #   ceshi = create_encoder()
#   #   k = torch.randn([2,80,1,1])
#   #   G = create_generator()
#   #   kk = G(k)
#   #   print(kk.size())
# # print(ceshi(k).size())

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
from torch.utils.data.distributed import DistributedSampler
from mnist_outlier.read_outlier import MnistOutlier
from util_wgan_base import DeterministicConditional, GaussianConditional, WALI,JointCritic
from torchsummary import summary
import torch.backends.cudnn as cudnn
import torch.distributed as dist
cudnn.benchmark = True
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
import os


data_path = '/home/huaqin/celeba'
save_path = "/data1/JCST/results"

import os
cuda_list = [i.strip() for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
device_ids = [i for i in range(len(cuda_list))]

IMAGE_SIZE = 128
NUM_CHANNELS = 3

# training hyperparameters
BATCH_SIZE = 128
BETAS = 0.01
# BETA3 = 0.00335
# BETA4 = 0.004
unit_iter = 50
#NUM_CHANNELS = 1
# 64x64
DIM = 64
# 256x256
# DIM = 256
# NLAT = 256
NLAT = 128
LEAK = 0.2
MMD_ITERS = 3       # mmd iterations!
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 10       # strength of gradient penalty
LEARNING_RATE = 1.2e-4
BETA1 = 0.5
BETA2 = 0.9
num_exp = 23

num_gpu = 1

ITER = 20000




C_ITERS = 5       # critic iterations


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

class CustomDataset(data.Dataset):
    def __init__(self,aims,mode='train',pos=1):
        super(CustomDataset, self).__init__()

        self.aims = aims
        self.true_config = {}
        self.out_config = {}
        if aims < 0:
            aim_file = 'total_hq2.json'
            aim_data = load_config(aim_file)[mode][:]
        else:
            aim_file = index_att[self.aims].lower()+'_hq.json'
            aim_data = load_config(aim_file)[mode][str(int(pos*aims))]
        self.train_data = aim_data


    def __len__(self):
        return len(self.train_data[:])

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

        return item_image

def create_encoder():
    #64
    mapping = nn.Sequential(
        # 64
        # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        # Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        # Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        # Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, NLAT, 1, 1)
        #128
        Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        Conv2d(DIM * 4, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        Conv2d(DIM * 8, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        Conv2d(DIM * 8, NLAT, 1, 1)
    # 256
    # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
    # Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
    # Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    # Conv2d(DIM * 4, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    # Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
    # Conv2d(DIM * 8, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
    # Conv2d(DIM * 8, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
    # Conv2d(DIM * 16, NLAT, 1, 1)
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
    ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16),
    LeakyReLU(inplace=True, negative_slope=LEAK),
    ConvTranspose2d(DIM * 16, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8),
    LeakyReLU(inplace=True, negative_slope=LEAK),
    ConvTranspose2d(DIM * 8, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8),
    LeakyReLU(inplace=True, negative_slope=LEAK),
    ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4),
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
        self.ConvT1 = ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False)
        self.B1 = BatchNorm2d(DIM * 16)
        self.ReLU1 = LeakyReLU(inplace=True, negative_slope=LEAK)
        self.ConvT2 = ConvTranspose2d(DIM * 16, DIM * 8, 4, 2, 1, bias=False)
        self.B2 = BatchNorm2d(DIM * 8)
        self.ReLU2 = LeakyReLU(inplace=True, negative_slope=LEAK)
        self.ConvT3 = ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False)
        self.B3 = BatchNorm2d(DIM * 4)
        self.ReLU3 = LeakyReLU(inplace=True, negative_slope=LEAK)
        self.ConvT4 = ConvTranspose2d(DIM * 4, DIM*2, 4, 2, 1, bias=False)
        self.B4 = BatchNorm2d(DIM*2)
        self.ReLU4 = LeakyReLU(LEAK, inplace=True)
        self.ConvT5 = ConvTranspose2d(DIM*2, DIM, 4, 2, 1, bias=False)
        self.B5 = BatchNorm2d(DIM)
        self.ReLU5 = LeakyReLU(LEAK,inplace=True)
        self.ConvT6 = ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False)
        #256
        # 256
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
        # self.ConvT5 = ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False)
        # self.B5 = BatchNorm2d(DIM * 2)
        # self.ReLU5 = LeakyReLU(LEAK, inplace=True)
        # self.ConvT6 = ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False)
        # self.B6 = BatchNorm2d(DIM)
        # self.ReLU6 = LeakyReLU(LEAK, inplace=True)
        # self.ConvT7 = ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False)

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

    def set_shift(self, value):
        if self.shift is None:
            return
        assert list(self.shift.data.size()) == list(value.size())
        self.shift.data = value

    def forward(self, input,var_beta=0.1,clip_beta=0.2):

        out1 = self.ReLU1(self.B1(self.ConvT1(input)))

        out2 = self.ReLU2(self.B2(self.ConvT2(out1)))
        out3 = self.ReLU3(self.B3(self.ConvT3(out2)))
        out4 = self.ReLU4(self.B4(self.ConvT4(out3)))
        out5 = self.ReLU5(self.B5(self.ConvT5(out4)))
        # out6 = self.ReLU6(self.B6(self.ConvT6(out5)))

        #64
        # out5 = self.ConvT5(out4)
        # output = self.act(out5)
        #128
        out6 = self.ConvT6(out5)
        output = self.act(out6)
        #256
        # out7 = self.ConvT7(out6)
        # output = self.act(out7)
        #512
        # out7 = self.ReLU7(self.B7(self.ConvT7(out6)))
        # out8 = self.ConvT8(out7)
        # output = self.act(out8)

        # output = self.mapping(input)

        if self.shift is not None:
            output = output + self.shift
        return output



def create_critic():
  x_mapping = nn.Sequential(
      #64
      # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 8, DIM * 8, 4, 1, 0), LeakyReLU(LEAK)
      # # 128
      Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM * 8, DIM * 16, 4, 2, 1), LeakyReLU(LEAK),
      Conv2d(DIM * 16, DIM * 16, 4, 1, 0), LeakyReLU(LEAK)
      # #256
      # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 8, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 8, DIM * 16, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 16, DIM * 16, 4, 1, 0), LeakyReLU(LEAK)

      # 512
      # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 4, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 8, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 8, DIM * 16, 4, 2, 1), LeakyReLU(LEAK),
      # Conv2d(DIM * 16, DIM * 16, 4, 1, 0), LeakyReLU(LEAK)
  )

  # z_mapping = nn.Sequential(
  #     #64
  #     # Conv2d(NLAT, 256, 1, 1, 0), LeakyReLU(LEAK),
  #     # Conv2d(256, 256, 1, 1, 0), LeakyReLU(LEAK),
  #     # 128
  #     # Conv2d(NLAT, 256, 1, 1, 0), LeakyReLU(LEAK),
  #     # Conv2d(256, 512, 1, 1, 0), LeakyReLU(LEAK),
  #     # Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK)
  #     # 256
  #     # Conv2d(NLAT, 256, 1, 1, 0), LeakyReLU(LEAK),
  #     # Conv2d(256, 512, 1, 1, 0), LeakyReLU(LEAK),
  #     # Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK)
  #
  #     # 512
  #     Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
  #     Conv2d(512, 1024, 1, 1, 0), LeakyReLU(LEAK),
  #     Conv2d(1024, 1024, 1, 1, 0), LeakyReLU(LEAK),
  #     Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK)
  #   )

  joint_mapping = nn.Sequential(
      #64
      # Conv2d(DIM * 8 + 256, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 1024, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(1024, 1, 1, 1, 0)
      #128
      # Conv2d(DIM * 16 + 512, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 1024, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(512, 1, 1, 1, 0)

      Conv2d(DIM * 16, 2048, 1, 1, 0), LeakyReLU(LEAK),
      Conv2d(2048, 2048, 1, 1, 0), LeakyReLU(LEAK),
      Conv2d(2048, 1024, 1, 1, 0), LeakyReLU(LEAK),
      Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
      Conv2d(512, 1, 1, 1, 0)
      # 256
      #  + 512
      # Conv2d(DIM * 16, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 1024, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(512, 1, 1, 1, 0)

      # 512
      #  + 512
      # Conv2d(DIM * 16, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 2048, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(2048, 1024, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
      # Conv2d(512, 1, 1, 1, 0)
  )

  return JointCritic(x_mapping, joint_mapping)

device = torch.device("cuda", device_ids[0])

if torch.cuda.device_count() > 1:
    num_gpu = torch.cuda.device_count()
if torch.cuda.device_count() < 1:
    num_gpu = 1

print("use %d GPUs" % num_gpu)

svhn = CustomDataset(aims=-1)
#     # svhn = datasets.CIFAR10('data/CIFAR10', train=True, transform=transform)
# loader = data.DataLoader(svhn, BATCH_SIZE//num_gpu,sampler=DistributedSampler(svhn))
loader = data.DataLoader(svhn,BATCH_SIZE*num_gpu,shuffle=True,
                                                num_workers=2)
noise = torch.randn(BATCH_SIZE, NLAT, 1, 1, device=device)

def create_WALI(channel=3,pads=False):
  # E = create_encoder()
  if torch.cuda.is_available():
      G = Generator_with_drop(gpu_mode=True)
  else:
      G = Generator_with_drop(gpu_mode=False)
  C = create_critic()
  # MMDX,MMDZ = create_mmds()

  wali = WALI(G, C,channel=channel,pads=pads)
  return wali

Recon_lamb = 6.172
import os

def main():
    # if not os.path.exists
    # if local_rank == 0:
    mmds = os.path.join(save_path,'wgan%d' % IMAGE_SIZE)
    if not os.path.exists(os.path.join(save_path,'wgan%d' % IMAGE_SIZE)):
        os.makedirs(os.path.join(save_path,'wgan%d' % IMAGE_SIZE))
        print("目录创建成功！")
    if not os.path.exists(mmds+"/%d" % num_exp):
        os.makedirs(mmds+"/%d" % num_exp)
        print("目录创建成功！")
    if not os.path.exists(mmds+"/celeba"):
        os.makedirs(mmds+"/celeba")
        print("目录创建成功！")
    if not os.path.exists(mmds+"/celeba"+"/%d" % num_exp):
        os.makedirs(mmds+"/celeba"+"/%d" % num_exp)
        print("目录创建成功！")
#
    wali = create_WALI().cuda(device=device_ids[0])
    # if local_rank == 0:
    summary(wali.get_G(),(NLAT,1,1))
    # summary(wali.get_E(),(NUM_CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
    # summary(wali.get_G(),(100,1,1))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        wali = torch.nn.parallel.DataParallel(wali,device_ids=device_ids)
        # list(wali._modules['module'].get_encoder_parameters()) +
        optimizerEG = Adam(list(wali._modules['module'].get_generator_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        # list(wali._modules['module'].get_encoder_parameters()) +
        optimizerEG2 = Adam(list(wali._modules['module'].get_generator_parameters()),
                            lr=LEARNING_RATE * 0.83, betas=(BETA1, BETA2))
        optimizerC = Adam(wali._modules['module'].get_critic_parameters(),
                          lr=LEARNING_RATE, betas=(BETA1, BETA2))
        # optimizerE2 = Adam(list(wali._modules['module'].get_encoder_parameters()),
        #                    lr=LEARNING_RATE * 0.2, betas=(BETA1, BETA2))
    else:
        optimizerEG = Adam(list(wali.get_generator_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerEG2 = Adam(list(wali.get_generator_parameters()),
                            lr=LEARNING_RATE * 0.83, betas=(BETA1, BETA2))
        optimizerC = Adam(wali.get_critic_parameters(),
                          lr=LEARNING_RATE, betas=(BETA1, BETA2))



    # svhn = CustomDataset(aims=-1)
    print(svhn.__len__())

    print('Training starts...')
    var_beta = -1
    clip_beta = -1
    threshold = 0.06
    _beta0 = 0.15
    decay = 0.9
    num_dec = 0

    EG_losses, C_losses = [], []
    mss_losses, mss_x_losses, l1s = [], [], []
    EG_losses22 = []
    curr_iter = C_iter = EG_iter1 = EG_iter2 = MMD_iter = 0

    C_update, EG_update = True, False

    epoch = 0
    while curr_iter < ITER:
        loader = data.DataLoader(svhn, BATCH_SIZE * num_gpu, shuffle=True,
                                 num_workers=2)
        for batch_idx, x in enumerate(loader, 1):
            x = x.cuda(device=device_ids[0])

            if curr_iter == 0:
                init_x = x
                # if local_rank == 0:
                np.save(mmds + "/%d/samples.npy" % num_exp, init_x.cpu())
                curr_iter += 1

            z = torch.randn(x.size(0), NLAT, 1, 1).cuda(device=device_ids[0])
            if C_update:
                #     def forward(self, x, z, lamb=10,beta1=0.01,beta2=0.01,beta3=0.03,gan=0,loss_type='raw',var_beta=-1,clip_beta=-1,methods=0,l1=True,l2=False,val_range=2,normalize="relu",pads=False,ssm_alpha=0.84):
                C_loss, EG_loss = wali.forward(x=x, z=z, lamb=LAMBDA, gan=0, loss_type='raw', beta1=1.25, beta2=0.3,
                                               beta3=0.7, methods=0,
                                               l1=True,
                                               val_range=2,
                                               normalize="relu",
                                               pads=True,
                                               ssm_alpha=0.84)

                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()

                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerEG2.zero_grad()

                C_loss.backward()
                optimizerC.step()

                C_iter += 1
                if C_iter == C_ITERS:
                    C_iter = 0
                    C_update, EG_update = False, True
                continue

            if EG_update:
                C_loss, EG_loss = wali.forward(x=x, z=z, lamb=LAMBDA, gan=0, loss_type='raw', beta1=1.25, beta2=0.3,
                                               beta3=0.7, methods=0,
                                               l1=True,
                                               val_range=2,
                                               normalize="relu",
                                               pads=True,
                                               ssm_alpha=0.84)

                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()

                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerEG2.zero_grad()

                EG_loss.backward()
                optimizerEG.step()

                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerEG2.zero_grad()

                C_loss, EG_loss = wali.forward(x=x, z=z, lamb=LAMBDA, gan=0, loss_type='raw', beta1=1.25,
                                                       beta2=0.3,
                                                       beta3=0.7, methods=0,
                                                       l1=True,
                                                       val_range=2,
                                                       normalize="relu",
                                                       pads=True,
                                                       ssm_alpha=0.84)

                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()

                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerEG2.zero_grad()

                EG_iter2 += 1

                if EG_iter2 == EG_ITERS:
                    EG_iter2 = 0
                    C_update, EG_update = True, False
                    EG_losses.append(EG_loss.item())
                    C_losses.append(C_loss.item())
                    curr_iter += 1

            if curr_iter % 10==0:
                # Reconx:%.4f
                print(
                    '[%d/%d]\tW-distance: %.4f\tC-loss: %.4f'
                    % (curr_iter, ITER, EG_losses[-1], C_losses[-1]))

            if curr_iter % 200 == 0:
                wali.eval()
                if torch.cuda.device_count() > 1:
                    genr_imgs = wali._modules['module'].generate(noise).detach_().cpu()
                else:
                    genr_imgs = wali.generate(noise).detach_().cpu()
                utils.save_image(genr_imgs * 0.5 + 0.5, mmds + '/%d/genr%d.png' % (num_exp, curr_iter))
                wali.train()

            if curr_iter % (ITER // 20) == 0:
                if torch.cuda.device_count() > 1:
                    torch.save(wali.module, mmds + '/celeba/%d/save_model_%d.ckpt' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].state_dict(), mmds + '/%d/total_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_G().state_dict(), mmds + '/%d/G_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_C().state_dict(), mmds + '/%d/C_%d.pth' % (num_exp, curr_iter))
                else:
                    torch.save(wali.state_dict(), mmds + '/%d/total_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_G().state_dict(), mmds + '/%d/G_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_C().state_dict(), mmds + '/%d/C_%d.pth' % (num_exp, curr_iter))

            if curr_iter % (ITER // 40) == 0:
                np.save(mmds + "/%d/EGloss.npy" % num_exp, EG_losses)
                np.save(mmds + "/%d/Closs.npy" % num_exp, C_losses)

        if torch.cuda.device_count() > 1:
            torch.save(wali.module, mmds + '/celeba/%d/save_model_epoch_%d_%d.ckpt' % (num_exp, epoch, curr_iter))
            torch.save(wali._modules['module'].state_dict(), mmds + '/%d/total_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_G().state_dict(), mmds + '/%d/G_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_C().state_dict(), mmds + '/%d/C_epoch_%d.pth' % (num_exp, epoch))
        else:
            torch.save(wali, mmds + '/celeba/%d/save_model_epoch_%d_%d.ckpt' % (num_exp, epoch, curr_iter))
            torch.save(wali.state_dict(), mmds + '/%d/total_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_G().state_dict(), mmds + '/%d/G_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_C().state_dict(), mmds + '/%d/C_%d.pth' % (num_exp, epoch))

        epoch += 1


    if torch.cuda.device_count() >= 1:
        np.save(mmds + "/%d/EGloss.npy" % num_exp, EG_losses)
        # np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
        np.save(mmds + "/%d/Closs.npy" % num_exp, C_losses)

        EG_losses = np.load(mmds + "/%d/EGloss.npy" % num_exp)
        C_losses = np.load(mmds + "/%d/Closs.npy" % num_exp)

# if __name__ == '__main__':
#     main()
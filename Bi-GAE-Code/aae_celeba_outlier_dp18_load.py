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

cudnn.benchmark = True
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
# RANK = int(os.environ['SLURM_PROCID'])  # ????????????????????????????????????
# LOCAL_RANK = int(os.environ['SLURM_LOCALID']) # ???????????????????????????????????????.
# GPU_NUM = int(os.environ['SLURM_NTASKS'])     # ????????? GPU ??????.
# print(RANK)
# print(GPU_NUM)
# port = 25557
# IP = '127.0.0.1'
# host_addr_full = 'tcp://' + IP + ':' + str(port)
# # torch.distributed.init_process_group(backend="nccl",init_method='env://')
from torch.utils.data.distributed import DistributedSampler
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)

# 2??? ?????????????????????gpu
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)

data_path = '/home/huaqin/celeba'
save_path = "/data1/JCST/results"



IMAGE_SIZE = 512
NUM_CHANNELS = 3
# DIM = 64
# NLAT = 100
# LEAK = 0.2

# training hyperparameters
BATCH_SIZE = 20
BETAS = 0.01
# DIM = 64
# NLAT = 100
# LEAK = 0.2

# BETA3 = 0.00335
# BETA4 = 0.004
ITER = 20000
# num_gpu = 2
unit_iter = 50
#NUM_CHANNELS = 1
# 64x64
DIM = 64
# 256x256
# DIM = 256
NLAT = 512
LEAK = 0.2

C_ITERS = 5       # critic iterations
MMD_ITERS = 3       # mmd iterations!
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 10       # strength of gradient penalty
LEARNING_RATE = 1.6e-4
BETA1 = 0.5
BETA2 = 0.9
num_exp = 21
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

root_dir = '/home/huaqin/B/'

# # ??????skimage????????????
# img_skimage = io.imread('mmds/cifar/0/rect2400.png')        # skimage.io imread()-----np.ndarray,  (H x W x C), [0, 255],RGB
# print(img_skimage)
#
# # ??????opencv????????????
# img_cv = cv2.imread('mmds/cifar/0/rect2400.png')            # cv2.imread()------np.array, (H x W xC), [0, 255], BGR
# print(img_cv)
#
# # ??????PIL??????
# img_pil = Image.open('mmds/cifar/0/rect2400.png')         # PIL.Image.Image??????
# img_pil_1 = np.array(img_pil)           # (H x W x C), [0, 255], RGB
# print(img_pil_1)
#
# plt.figure()
# for i, im in enumerate([img_skimage, img_cv, img_pil_1]):
#     ax = plt.subplot(1, 3, i + 1)
#     ax.imshow(im)
#     plt.pause(0.01)
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
            aim_file = 'total_hq5.json'
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


        # aim_file = self.csv_path
        # aims = pd.read_csv(aim_file, header=None, iterator=True, skiprows=item)
        # datas = aims.get_chunk(1)
        # data = np.array(datas.iloc[:,0:-1]).astype(float)
        # label = np.array(datas.iloc[:,-1]).astype(int)
        # data = (data - self.mu)/self.std

        # data = torch.Tensor(item_data)
        # # data = torch.squeeze(data,dim=0)
        # label = torch.Tensor(item_label)
        # raw = torch.Tensor(item_raw)
        # out = torch.Tensor(item_if)
        return item_image

        # return data,label,raw,out
        # input_item = item
        # while True:
        #     ok,trac_output, condition, condition2,abs_data = self.get_data(input_item)
        #     if not ok:
        #         bias = random.randint(0,10)
        #         input_item = (bias+self.gap+input_item) % self.__len__()
        #     else:
        #         return trac_output, condition, condition2,abs_data


        # condition3 = torch.Tensor(trac_t)
        # condition3 = torch.Tensor(condition3)

        # ,trac_out2,transs


        # return trac_output, condition, condition2,trac_out2,transs

        # trac_dest_x = trac_y[-1]
        # trac_source_y = trac_y[-1]




        # # ????????????????????????
        # return landmarks

def create_encoder():
    #64
    mapping = nn.Sequential(
        #64
        # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        # Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        # Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        # Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, NLAT, 1, 1)
        # 128
        # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        # Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        # Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        # Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 16, 4, 2, 1, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        # Conv2d(DIM * 16, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        # Conv2d(DIM * 16, NLAT, 1, 1)
        # #256
        # Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        # Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        # Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        # Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        # Conv2d(DIM * 8, DIM * 16, 4, 2, 1, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        # Conv2d(DIM * 16, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
        # Conv2d(DIM * 16, NLAT, 1, 1)

        # 512
        Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        Conv2d(DIM * 4, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        Conv2d(DIM * 8, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
        Conv2d(DIM * 8, DIM * 16, 4, 2, 1, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
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

# svhn = CustomDataset(aims=-1)
# # svhn = datasets.CIFAR10('data/CIFAR10', train=True, transform=transform)
# loader = data.DataLoader(svhn, BATCH_SIZE, shuffle=True, num_workers=4)
# curr_iter = 0
# while curr_iter < ITER:
#     for batch_idx, x in enumerate(loader, 1):
#         print(x.size())
#         break
#     break
#
# # ceshi = create_encoder()
# # k = torch.randn([2,1,28,28])
# # print(ceshi(k).size())
#
# # mm = nn.Sequential(
# #     Conv2d(1, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
# #     Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
# #     Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
# #     Conv2d(DIM * 4, DIM * 4, 3, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
# #     Conv2d(DIM * 4, NLAT, 1, 1, 0)
# # # Conv2d(DIM * 4, NLAT, 1, 1, 0)
# # )
# # print(mm(k).size())
#
def create_generator():
    mapping = nn.Sequential(
        #64
        # ConvTranspose2d(NLAT, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8), LeakyReLU(inplace=True,negative_slope=LEAK),
        # ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), LeakyReLU(inplace=True,negative_slope=LEAK),
        # ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), LeakyReLU(inplace=True,negative_slope=LEAK),
        # ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        # ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh()
        # 128
        # ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM * 16, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM),
        # LeakyReLU(inplace=True, negative_slope=LEAK),
        # ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh()
        #256
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
        self.ConvT5 = ConvTranspose2d(DIM * 4, DIM * 4, 4, 2, 1, bias=False)
        self.B5 = BatchNorm2d(DIM * 4)
        self.ReLU5 = LeakyReLU(LEAK, inplace=True)
        self.ConvT6 = ConvTranspose2d(DIM * 4, DIM*2, 4, 2, 1, bias=False)
        self.B6 = BatchNorm2d(DIM*2)
        self.ReLU6 = LeakyReLU(LEAK, inplace=True)
        self.ConvT7 = ConvTranspose2d(DIM *2, DIM, 4, 2, 1, bias=False)
        self.B7 = BatchNorm2d(DIM)
        self.ReLU7 = LeakyReLU(LEAK, inplace=True)
        self.ConvT8 = ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False)
        # ConvTranspose2d()
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
        # out7 = self.ConvT7(out6)
        # output = self.act(out7)
        #512
        out7 = self.ReLU7(self.B7(self.ConvT7(out6)))
        out8 = self.ConvT8(out7)
        output = self.act(out8)
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

def create_z_disc():
    mapping = nn.Sequential(
    # Conv2d(NLAT, 256, 1, 1, 0), LeakyReLU(LEAK),
    # Conv2d(256, 256, 1, 1, 0), LeakyReLU(LEAK),
    # Conv2d(256, 128, 1, 1, 0), LeakyReLU(LEAK),
    # Conv2d(128, 1, 1, 1, 0)
        #128
        # Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(512, 256, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(256, 128, 1, 1, 0), LeakyReLU(LEAK),
        # Conv2d(128, 1, 1, 1, 0)
    # 256
    # Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
    # Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK),
    # Conv2d(512, 256, 1, 1, 0), LeakyReLU(LEAK),
    # Conv2d(256, 128, 1, 1, 0), LeakyReLU(LEAK),
    # Conv2d(128, 1, 1, 1, 0)

    # 512
    Conv2d(NLAT, 1024, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(1024, 1024, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(1024, 512, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(512, 256, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(256, 128, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(128, 1, 1, 1, 0)
    )
    return mapping
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

device = torch.device("cuda", device_ids[0])

num_gpu = 1

if torch.cuda.device_count() > 1:
    num_gpu = torch.cuda.device_count()

if torch.cuda.device_count() <= 1:
    num_gpu = 1

print("use %d GPUs" % num_gpu)
svhn = CustomDataset(aims=-1)
#     # svhn = datasets.CIFAR10('data/CIFAR10', train=True, transform=transform)
# loader = data.DataLoader(svhn, BATCH_SIZE//num_gpu,sampler=DistributedSampler(svhn))
loader = data.DataLoader(svhn,BATCH_SIZE*num_gpu,shuffle=True,
                                                num_workers=2)

noise = torch.randn(BATCH_SIZE, NLAT, 1, 1, device=device)

def create_WALI():
  E = create_encoder()
  if torch.cuda.is_available():
      G = Generator_with_drop(gpu_mode=True)
  else:
      G = Generator_with_drop(gpu_mode=False)
  C = create_z_disc()
  # MMDX,MMDZ = create_mmds()
  wali = WALI(E,G,C,loss_type='mse')

  # wali = WALI(E, G, C,MMDX,MMDZ,channel=channel,pads=pads)
  return wali
import os
# 1600 -??? 36
#
def main():
    # if not os.path.exists
    mmds = os.path.join(save_path,'aaes%d' % IMAGE_SIZE)
    if not os.path.exists(os.path.join(save_path,'aaes%d' % IMAGE_SIZE)):
        os.makedirs(os.path.join(save_path,'aaes%d' % IMAGE_SIZE))
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
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    wali = create_WALI().cuda(device=device_ids[0])
    # summary(wali.get_G(),(100,1,1))
    summary(wali.get_G(), (NLAT, 1, 1))
    summary(wali.get_E(), (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
#

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) ??????
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



    # if local_rank == 0:
    #     summary(wali.get_G(),(NLAT,1,1))
    #     summary(wali.get_E(),(NUM_CHANNELS,IMAGE_SIZE,IMAGE_SIZE))

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))])
#     svhn = CustomDataset(aims=-1)
# #     # svhn = datasets.CIFAR10('data/CIFAR10', train=True, transform=transform)
#     loader = data.DataLoader(svhn, BATCH_SIZE, shuffle=True, num_workers=4)
#     noise = torch.randn(BATCH_SIZE, NLAT, 1, 1, device=device)
#
#     # EG_losses, C_losses, Recon_x_losses, Recon_z_losses, EG_losses2 = [], [], [], [], []
#     EG_losses, C_losses, MMD_C_losses,MMD_EG_losses,Recon_z_losses, EG_losses2,mmd_penaltys = [], [], [], [], [],[],[]
#     print('Training starts...')
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

    epoch = 0
    while curr_iter < ITER:
        loader = data.DataLoader(svhn, BATCH_SIZE * num_gpu, shuffle=True,
                                 num_workers=2)
        for batch_idx, x in enumerate(loader, 1):
            # x = x.to(device)
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

                utils.save_image(rect_imgs * 0.5 + 0.5, mmds + '/%d/rect%d.png' % (num_exp, curr_iter))
                utils.save_image(genr_imgs * 0.5 + 0.5, mmds + '/%d/genr%d.png' % (num_exp, curr_iter))
                wali.train()

            # save model
            if curr_iter % (ITER // 20) == 0:
                if torch.cuda.device_count() > 1:
                    torch.save(wali.module, mmds + '/celeba/%d/save_model_%d.ckpt' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].state_dict(), mmds + '/%d/total_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_G().state_dict(), mmds + '/%d/G_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_C().state_dict(), mmds + '/%d/C_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_E().state_dict(), mmds + '/%d/E_%d.pth' % (num_exp, curr_iter))
                else:
                    torch.save(wali.state_dict(), mmds + '/%d/total_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_G().state_dict(), mmds + '/%d/G_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_C().state_dict(), mmds + '/%d/C_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_E().state_dict(), mmds + '/%d/E_%d.pth' % (num_exp, curr_iter))

            if curr_iter % (ITER // 40) == 0:
                np.save(mmds + "/%d/EGloss.npy" % num_exp, recon_losses)
                np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
                np.save(mmds + "/%d/MMDCloss.npy" % num_exp, MMD_C_losses)


            #break
        if torch.cuda.device_count() > 1:
            torch.save(wali.module, mmds + '/celeba/%d/save_model_epoch_%d_%d.ckpt' % (num_exp, epoch,curr_iter))
            torch.save(wali._modules['module'].state_dict(), mmds + '/%d/total_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_G().state_dict(), mmds + '/%d/G_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_C().state_dict(), mmds + '/%d/C_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_E().state_dict(), mmds + '/%d/E_epoch_%d.pth' % (num_exp, epoch))
        else:
            torch.save(wali.state_dict(), mmds + '/%d/total_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_G().state_dict(), mmds + '/%d/G_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_C().state_dict(), mmds + '/%d/C_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_E().state_dict(), mmds + '/%d/E_epoch_%d.pth' % (num_exp, epoch))
        epoch += 1
       # break
    # plot training loss curve
    if torch.cuda.device_count() >= 1:
        np.save(mmds + "/%d/EGloss.npy" % num_exp, recon_losses)
        np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
        np.save(mmds + "/%d/MMDCloss.npy" % num_exp, MMD_C_losses)

        EG_losses = np.load(mmds + "/%d/MMDEloss.npy" % num_exp)
        C_losses = np.load(mmds + "/%d/MMDCloss.npy" % num_exp)

        plt.figure(0, figsize=(10, 7.5))
        plt.title('Main Training loss curve')
        plt.plot(recon_losses, label='Encoder + Generator')
        # plt.plot(C_losses, label='Criic')
        # plt.plot(C_losses, label='Critic')
        #
        # plt.plot(EG_losses2,label='After Regularization')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(mmds + '/%d/loss_curve%d.png' % (num_exp, num_exp))

        plt.figure(1, figsize=(10, 7.5))
        plt.title('Reconstruction of z loss curve')
        plt.plot(EG_losses, label='Encoder + Generator')
        plt.plot(C_losses, label='Criic')
        # plt.plot(C_losses, label='Critic')
        #
        # plt.plot(EG_losses2,label='After Regularization')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(mmds + '/%d/recon_total%d.png' % (num_exp, num_exp))
        # plt.show()



#
# if __name__ == "__main__":
#     main()
# #     ceshi = create_encoder()
# #     k = torch.randn([2,80,1,1])
# #     G = create_generator()
# #     kk = G(k)
# #     print(kk.size())
# # print(ceshi(k).size())

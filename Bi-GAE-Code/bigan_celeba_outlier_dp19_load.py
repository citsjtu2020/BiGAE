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
from util11 import DeterministicConditional, GaussianConditional, WALI,JointCritic
from torchsummary import summary
import torch.backends.cudnn as cudnn
import torch.distributed as dist
cudnn.benchmark = True
import tensorflow as tf
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
import os
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '162.128.0.22'
    os.environ['MASTER_PORT'] = '29555'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.manual_seed(1)
    fn(rank, size)
    print("MM")
    print(dist.get_rank())
    print(dist.get_world_size())
    print(dist.is_available())

data_path = '/home/huaqin/celeba'

save_path = "/data1/JCST/results"


import os
cuda_list = [i.strip() for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
device_ids = [i for i in range(len(cuda_list))]

IMAGE_SIZE = 256
NUM_CHANNELS = 3

imageWidth = 224
imageHeight = 224
imageDepth = 3
resize_min = 256

def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature([], tf.string, default_value=""),
                "height": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "width": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "channels": tf.FixedLenFeature([1], tf.int64, default_value=[3]),
                "colorspace": tf.FixedLenFeature([], tf.string, default_value=""),
                "img_format": tf.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "bbox_xmin": tf.VarLenFeature(tf.float32),
                "bbox_xmax": tf.VarLenFeature(tf.float32),
                "bbox_ymin": tf.VarLenFeature(tf.float32),
                "bbox_ymax": tf.VarLenFeature(tf.float32),
                "text": tf.FixedLenFeature([], tf.string, default_value=""),
                "filename": tf.FixedLenFeature([], tf.string, default_value="")
               }
    parsed_features = tf.parse_single_example(example_proto, features)
    xmin = tf.expand_dims(parsed_features["bbox_xmin"].values, 0)
    xmax = tf.expand_dims(parsed_features["bbox_xmax"].values, 0)
    ymin = tf.expand_dims(parsed_features["bbox_ymin"].values, 0)
    ymax = tf.expand_dims(parsed_features["bbox_ymax"].values, 0)
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])
    height = parsed_features["height"]
    width = parsed_features["width"]
    channels = parsed_features["channels"]
    bbox_begin, bbox_size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.concat(axis=0, values=[height, width, channels]),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        use_image_if_no_bounding_boxes=True)
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.cast(tf.stack([offset_y, offset_x, target_height, target_width]), tf.int32)
    cropped = tf.image.decode_and_crop_jpeg(parsed_features["image"], crop_window, channels=3)
    image_train = tf.image.resize_images(cropped, [imageHeight, imageWidth],
                                         method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    image_train = tf.cast(image_train, tf.uint8)
    image_train = tf.image.convert_image_dtype(image_train, tf.float32)
    return image_train, parsed_features["label"][0], parsed_features["text"], parsed_features["filename"]

# training hyperparameters
BATCH_SIZE = 40
BETAS = 0.01
# BETA3 = 0.00335
# BETA4 = 0.004
ITER = 20000
num_gpu = 2
unit_iter = 50

# training hyperparameters
# num_gpu = 2
# BETA3 = 0.00335
# BETA4 = 0.004
#NUM_CHANNELS = 1
# 64x64
DIM = 64
# 256x256
# DIM = 256
NLAT = 256
LEAK = 0.2

C_ITERS = 5       # critic iterations
MMD_ITERS = 3       # mmd iterations!
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 10       # strength of gradient penalty
LEARNING_RATE = 1.3e-4
BETA1 = 0.5
BETA2 = 0.9
num_exp = 22

from configutils import load_config
from torchvision import datasets
import matplotlib.pyplot as plt
import skimage.io as io
from torchsummary import summary
import cv2
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
            aim_file = 'total_hq3.json'
            aim_data = load_config(aim_file)[mode][:]
        else:
            aim_file = index_att[self.aims].lower()+'_hq.json'
            aim_data = load_config(aim_file)[mode][str(int(pos*aims))]
        self.train_data = aim_data


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
noise = torch.randn(BATCH_SIZE, NLAT, 1, 1, device=device)

def create_WALI(channel=3,pads=False):
    if torch.cuda.is_available():
        G = Generator_with_drop(gpu_mode=True)
    else:
        G = Generator_with_drop(gpu_mode=False)
    E = create_encoder()
    C = create_critic()
    wali = WALI(E, G, C,channel=channel,pads=pads)
    return wali

Recon_lamb = 7.19
import os

def main():
    # if not os.path.exists
    # if local_rank == 0:
    mmds = os.path.join(save_path,'bigan%d' % IMAGE_SIZE)
    if not os.path.exists(os.path.join(save_path,'bigan%d' % IMAGE_SIZE)):
        os.makedirs(os.path.join(save_path,'bigan%d' % IMAGE_SIZE))
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
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    wali = create_WALI().cuda(device=device_ids[0])
    # save_model_epoch_309_15656.ckpt
    now_model = torch.load("/data1/JCST/results/bigan256/22/total_epoch_309.pth", map_location="cuda:0")
    wali.load_state_dict(now_model)
    # if local_rank == 0:
    summary(wali.get_G(),(NLAT,1,1))
    summary(wali.get_E(),(NUM_CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
    # summary(wali.get_G(),(100,1,1))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) ??????
        wali = torch.nn.parallel.DataParallel(wali,device_ids=device_ids)
        optimizerEG = Adam(list(wali._modules['module'].get_encoder_parameters()) + list(wali._modules['module'].get_generator_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerEG2 = Adam(list(wali._modules['module'].get_encoder_parameters()) + list(wali._modules['module'].get_generator_parameters()),
                            lr=LEARNING_RATE * 0.83, betas=(BETA1, BETA2))
        optimizerC = Adam(wali._modules['module'].get_critic_parameters(),
                          lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerE2 = Adam(list(wali._modules['module'].get_encoder_parameters()),
                           lr=LEARNING_RATE * 0.2, betas=(BETA1, BETA2))
    else:
        optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
                           lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerEG2 = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
                            lr=LEARNING_RATE * 0.83, betas=(BETA1, BETA2))
        optimizerC = Adam(wali.get_critic_parameters(),
                          lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizerE2 = Adam(list(wali.get_encoder_parameters()),
                           lr=LEARNING_RATE * 0.2, betas=(BETA1, BETA2))



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

    epoch = 310
    # save_model_epoch_309_15656.ckpt
    curr_iter = 15657

    init_x = np.load(mmds + "/%d/samples.npy" % 22)
    init_x = torch.from_numpy(init_x)
    init_x = init_x.type(torch.FloatTensor)
    init_x = init_x.cuda(device=device_ids[0])

    loads = True

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

                C_loss, EG_loss,Reconx = wali.forward(x=x, z=z, lamb=LAMBDA, gan=0, loss_type='raw', beta1=1.25, beta2=0.3,
                                               beta3=0.7, methods=0,
                                               l1=True,
                                               val_range=2,
                                               normalize="relu",
                                               pads=True,
                                               ssm_alpha=0.84)

                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()
                Reconx = Reconx.mean()

                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerEG2.zero_grad()
                optimizerE2.zero_grad()

                C_loss.backward()
                optimizerC.step()

                C_iter += 1
                if C_iter == C_ITERS:
                    C_iter = 0
                    C_update, EG_update = False, True
                continue

            if EG_update:
                C_loss, EG_loss,Reconx = wali.forward(x=x, z=z, lamb=LAMBDA, gan=0, loss_type='raw', beta1=1.25, beta2=0.3,
                                               beta3=0.7, methods=0,
                                               l1=True,
                                               val_range=2,
                                               normalize="relu",
                                               pads=True,
                                               ssm_alpha=0.84)

                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()
                Reconx = Reconx.mean()

                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerEG2.zero_grad()
                optimizerE2.zero_grad()

                EG_loss.backward()
                optimizerEG.step()

                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerEG2.zero_grad()
                optimizerE2.zero_grad()

                C_loss, EG_loss, Reconx = wali.forward(x=x, z=z, lamb=LAMBDA, gan=0, loss_type='raw', beta1=1.25,
                                                       beta2=0.3,
                                                       beta3=0.7, methods=0,
                                                       l1=True,
                                                       val_range=2,
                                                       normalize="relu",
                                                       pads=True,
                                                       ssm_alpha=0.84)

                C_loss = C_loss.mean()
                EG_loss = EG_loss.mean()
                Reconx = Reconx.mean()

                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerEG2.zero_grad()
                optimizerE2.zero_grad()

                Reconx = Reconx*Recon_lamb
                Reconx.backward()
                optimizerEG2.step()

                optimizerC.zero_grad()
                optimizerEG.zero_grad()
                optimizerEG2.zero_grad()
                optimizerE2.zero_grad()

                EG_iter2 += 1

                if EG_iter2 == EG_ITERS:
                    EG_iter2 = 0
                    C_update, EG_update = True, False
                    # if local_rank == 0:
                    EG_losses.append(EG_loss.item())
                    C_losses.append(C_loss.item())
                    curr_iter += 1

            if curr_iter % 10==0:
                print(
                    '[%d/%d]\tW-distance: %.4f\tC-loss: %.4f\tReconx:%.4f'
                    % (curr_iter, ITER, EG_losses[-1], C_losses[-1],Reconx))

            if curr_iter % 200 == 0:
                wali.eval()
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
                utils.save_image(rect_imgs * 0.5 + 0.5, mmds + '/%d/rect%d.png' % (num_exp, curr_iter))
                utils.save_image(genr_imgs * 0.5 + 0.5, mmds + '/%d/genr%d.png' % (num_exp, curr_iter))
                wali.train()

            # save model
            if curr_iter % (ITER // 40) == 0:
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

            if curr_iter % (ITER // 80) == 0:
                np.save(mmds + "/%d/EGloss.npy" % num_exp, EG_losses)
                # np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
                np.save(mmds + "/%d/Closs.npy" % num_exp, C_losses)

                # EG_losses = np.load(mmds + "/%d/EGloss.npy" % num_exp)
                # C_losses = np.load(mmds + "/%d/Closs.npy" % num_exp)

        if torch.cuda.device_count() > 1:
            torch.save(wali.module, mmds + '/celeba/%d/save_model_epoch_%d_%d.ckpt' % (num_exp, epoch, curr_iter))
            torch.save(wali._modules['module'].state_dict(), mmds + '/%d/total_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_G().state_dict(), mmds + '/%d/G_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_C().state_dict(), mmds + '/%d/C_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_E().state_dict(), mmds + '/%d/E_epoch_%d.pth' % (num_exp, epoch))
        else:
            torch.save(wali.state_dict(), mmds + '/%d/total_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_G().state_dict(), mmds + '/%d/G_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_C().state_dict(), mmds + '/%d/C_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_E().state_dict(), mmds + '/%d/E_%d.pth' % (num_exp, epoch))

        epoch += 1


    if torch.cuda.device_count() >= 1:
        np.save(mmds + "/%d/EGloss.npy" % num_exp, EG_losses)
        # np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
        np.save(mmds + "/%d/Closs.npy" % num_exp, C_losses)

        EG_losses = np.load(mmds + "/%d/EGloss.npy" % num_exp)
        C_losses = np.load(mmds + "/%d/Closs.npy" % num_exp)
        plt.figure(0, figsize=(10, 7.5))
        plt.title('Main Training loss curve')
        plt.plot(EG_losses, label='Encoder + Generator')
        plt.plot(C_losses, label='Criic')
        # plt.plot(C_losses, label='Critic')
        #
        # plt.plot(EG_losses2,label='After Regularization')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(mmds + '/%d/loss_curve%d.png' % (num_exp, num_exp))


# if __name__ == '__main__':
#     main()
from mnist_outlier.read_outlier import MnistOutlier
#
import numpy as np
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
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
from util4 import DeterministicConditional, GaussianConditional, JointCritic, WALI,MMD_NET
from torchvision import datasets, transforms, utils

cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

data_path = '/home/qian/huaqin/tfdata/mmds'




# training hyperparameters
BATCH_SIZE = 64
BETAS = 0.01
# BETA3 = 0.00335
# BETA4 = 0.004
ITER = 20000
unit_iter = 50
IMAGE_SIZE = 28
NUM_CHANNELS = 1
DIM = 64
NLAT = 80
LEAK = 0.2

C_ITERS = 5       # critic iterations
MMD_ITERS = 3       # mmd iterations!
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 10       # strength of gradient penalty
LEARNING_RATE = 5e-5
BETA1 = 0.5
BETA2 = 0.9
num_exp = 12
from configutils import load_config

class CustomDataset(data.Dataset):
    def __init__(self,aims):
        super(CustomDataset, self).__init__()
        # self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        # self.file = pd.read_csv(csv_file,header=None,iterator=True)
        # self.subsize = 25000
        # self.max_x = 128690
        # self.min_x = -390

        self.aims = aims
        self.true_config = {}
        self.out_config = {}
        # true_aimss = load_config("true_01.json")
        # outlier_aimss = load_config("outlier_01.json")
        true_aimss = load_config("true_00.json")
        outlier_aimss = load_config("outlier_00.json")
        keyss = list(true_aimss.keys())
        for k in keyss:
            self.true_config[int(k)] = []
            tmps = true_aimss[k][:]
            self.true_config[int(k)] = tmps

            self.out_config[int(k)] = []
            tmps = outlier_aimss[k][:]
            self.out_config[int(k)] = tmps
        self.true_configs = []
        self.out_configs = []
        if aims < 0:
            for k in keyss:
                for c in self.true_config[int(k)][:]:
                    self.true_configs.append(c)
                for c in self.out_config[int(k)][:]:
                    self.out_configs.append(c)
        # self.aims = aims
        # self.o = MnistOutlier(0.1)
        self.o = MnistOutlier(0.0)
        self.train_data = self.o.train_images
        self.train_label = self.o.train_labels
        # print(self.train_data.shape)
        # print(self.train_label.shape)
        self.train_raw = self.o.train_raw
        self.if_out = self.o.if_outlier


        # self.lens = lens
        # self.max_y = 83070
        # self.mu = 12.503158925964646
        # self.std = 76.2139849775572


    def __len__(self):

        if self.aims < 0:
            return len(self.true_configs[:])
        else:
            return len(self.true_config[self.aims][:])
        # return 5562245
        # return 180

    def __getitem__(self, item):
        # trac = self.cs.get_chunk(128).as_matrix().astype('float')
        # .as_matrix().astype('float')
        item = item % (self.__len__())
        if self.aims < 0:
            raw_index = self.true_configs[item]
        else:
            raw_index = self.true_config[self.aims][item]
        # print(raw_index)

        item_data = self.train_data[raw_index].transpose(2,0,1)
        #item_data = (item_data - 0.5) * 2
        item_label = self.train_label[raw_index]
        item_raw = self.train_raw[raw_index]
        item_if = self.if_out[raw_index]

        # print(item_data.shape)
        # print(item_label.shape)
        # print(item_if)
        # print(item_raw)


        # aim_file = self.csv_path
        # aims = pd.read_csv(aim_file, header=None, iterator=True, skiprows=item)
        # datas = aims.get_chunk(1)
        # data = np.array(datas.iloc[:,0:-1]).astype(float)
        # label = np.array(datas.iloc[:,-1]).astype(int)
        # data = (data - self.mu)/self.std

        data = torch.Tensor(item_data)
        # data = torch.squeeze(data,dim=0)
        label = torch.Tensor(item_label)
        raw = torch.Tensor(item_raw)
        out = torch.Tensor(item_if)

        # return data, label, raw, out
        return data, label

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




        # # 采用这个，不错。
        # return landmarks

def create_encoder():
  mapping = nn.Sequential(
    Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
    Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
    Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    Conv2d(DIM * 4, DIM * 4, 3, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    Conv2d(DIM * 4, NLAT, 1, 1, 0))
  return DeterministicConditional(mapping)

# ceshi = create_encoder()
# k = torch.randn([2,1,28,28])
# print(ceshi(k).size())

# mm = nn.Sequential(
#     Conv2d(1, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
#     Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
#     Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
#     Conv2d(DIM * 4, DIM * 4, 3, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
#     Conv2d(DIM * 4, NLAT, 1, 1, 0)
# # Conv2d(DIM * 4, NLAT, 1, 1, 0)
# )
# print(mm(k).size())

def create_generator():
    mapping = nn.Sequential(
        ConvTranspose2d(NLAT, DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
        ConvTranspose2d(DIM * 4, DIM * 2, 3, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
        ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
        ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh())
    return DeterministicConditional(mapping)


def create_critic():
  x_mapping = nn.Sequential(
    Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
    Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
    Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
    Conv2d(DIM * 4, DIM * 4, 3, 1, 0), LeakyReLU(LEAK))

  z_mapping = nn.Sequential(
    Conv2d(NLAT, 256, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(256, 256, 1, 1, 0), LeakyReLU(LEAK))

  joint_mapping = nn.Sequential(
    Conv2d(DIM * 4 + 256, 512, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(512, 1, 1, 1, 0))

  return JointCritic(x_mapping, z_mapping, joint_mapping)

def create_mmds():
    mmd_x = nn.Sequential(
        Conv2d(DIM*4,512,1,1,0),LeakyReLU(LEAK),
        Conv2d(512,256,1,1,0),LeakyReLU(LEAK),
        Conv2d(256,16,1,1,0)
    )

    mmd_z = nn.Sequential(
        Conv2d(256,256,1,1,0),LeakyReLU(LEAK),
        Conv2d(256,128,1,1,0),LeakyReLU(LEAK),
        Conv2d(128,16,1,1,0)
    )
    return MMD_NET(mmd_x),MMD_NET(mmd_z)

def create_WALI(channel=3,pads=False):
  E = create_encoder()
  G = create_generator()
  C = create_critic()
  MMDX,MMDZ = create_mmds()

  wali = WALI(E, G, C,MMDX,MMDZ,channel=channel,pads=pads)
  return wali
import os

def main():
    mmds = os.path.join(data_path,'mnist')
    # if not os.path.exists
    if not os.path.exists(mmds):
        os.makedirs(mmds)
        print("目录创建成功！")
    if not os.path.exists(mmds+"/%d" % num_exp):
        os.makedirs(mmds+"/%d" % num_exp)
        print("目录创建成功！")
    # if not os.path.exists("mmds/mnist"):
    #     os.makedirs("mmds/mnist")
    #     print("目录创建成功！")
    # if not os.path.exists("mmds/mnist/%d" % num_exp):
    #     os.makedirs("mmds/mnist/%d" % num_exp)
    #     print("目录创建成功！")

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    wali = create_WALI(channel=1,pads=True).to(device)
    # wali = create_WALI().cuda()

    optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
                       lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizerEG2 = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
                       lr=LEARNING_RATE*0.1, betas=(BETA1, BETA2))
    optimizerC = Adam(wali.get_critic_parameters(),
                      lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizerE2 = Adam(list(wali.get_encoder_parameters()),
                        lr=LEARNING_RATE * 0.2, betas=(BETA1, BETA2))

    optimizerXM = Adam(list(wali.get_C().get_x_net_parameters())+list(wali.get_mmdx_parameters()),lr=LEARNING_RATE,betas=(BETA1,BETA2))
    optimizerZM = Adam(list(wali.get_C().get_z_net_parameters())+list(wali.get_mmdz_parameters()),lr=LEARNING_RATE,betas=(BETA1,BETA2))
    optimizerXZM = Adam(list(wali.get_C().get_x_net_parameters())+list(wali.get_C().get_z_net_parameters())+list(wali.get_mmdx_parameters()))
    optimizerMox = Adam(list(wali.get_mmdx_parameters()),lr=LEARNING_RATE,betas=(BETA1,BETA2))
    optimizerMoz = Adam(list(wali.get_mmdz_parameters()),lr=LEARNING_RATE,betas=(BETA1,BETA2))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    svhn = CustomDataset(aims=-1)
    print(svhn.__len__())
    # svhn = datasets.CIFAR10('data/CIFAR10', train=True, transform=transform)
    loader = data.DataLoader(svhn, BATCH_SIZE, shuffle=True, num_workers=4)
    noise = torch.randn(BATCH_SIZE, NLAT, 1, 1, device=device)

    print('Training starts...')
    #     # sign_xs = []
    #     # sign_
    var_beta = -1
    clip_beta = -1
    threshold = 0.06
    _beta0 = 0.15
    decay = 0.9
    num_dec = 0

    # EG_losses, C_losses, Recon_x_losses, Recon_z_losses, EG_losses2 = [], [], [], [], []
    EG_losses, C_losses, MMD_C_losses,MMD_EG_losses,Recon_z_losses, EG_losses2,mmd_penaltys = [], [], [], [], [],[],[]
    curr_iter = C_iter = EG_iter1=EG_iter2=MMD_iter= 0
    alphas = []
    total_block = 0
     
    min_total_x = float('inf')
    min_total_z = float('inf')
    C_update, MMD_UPDATE,EG_update1,EG_update2 = True, False,False,False
    block_step = C_ITERS+EG_ITERS+MMD_ITERS+EG_ITERS
    print('Training starts...')
    # sign_xs = []
    # sign_
    EG_losses, C_losses, MMD_C_losses, MMD_EG_losses, Recon_z_losses, EG_losses2, mmd_penaltys = [], [], [], [], [], [], []
    mss_losses, mss_x_losses, l1s = [], [], []
    EG_losses22 = []
    curr_iter = C_iter = EG_iter1 = EG_iter2 = MMD_iter = 0
    while curr_iter < ITER:
        for batch_idx, (x,_) in enumerate(loader, 1):
            x = x.to(device)

            if curr_iter == 0:
                init_x = x
                curr_iter += 1

            z = torch.randn(x.size(0), NLAT, 1, 1).to(device)
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
            # if curr_iter < 1000:
            #     BETA3 = 0.5*BETAS
            #     BETA4 = BETAS - BETA3
            #     alphas.append(0.5)
            #     if curr_iter > unit_iter:
            #         tmpkx = Recon_x_losses[-unit_iter:]
            #         mean_tmpkx = np.mean(tmpkx)
            #         tmpkz = Recon_z_losses[-unit_iter:]
            #         mean_tmpkz = np.mean(tmpkz)
            #         if mean_tmpkx < min_total_x:
            #             min_total_x = mean_tmpkx
            #         if mean_tmpkz < min_total_z:
            #             min_total_z = mean_tmpkz
            # else:
            #     tmp_now_x_2 = Recon_x_losses[-unit_iter:]
            #     tmp_now_x_1 = Recon_x_losses[-2*unit_iter:-unit_iter]
            #     tmp_now_x_0 = Recon_x_losses[-3*unit_iter:-2*unit_iter]
            #
            #     mean_now_x_2 = np.mean(tmp_now_x_2)
            #     mean_now_x_1 = np.mean(tmp_now_x_1)
            #     mean_now_x_0 = np.mean(tmp_now_x_0)
            #
            #     tmp_now_z_2 = Recon_z_losses[-unit_iter:]
            #     tmp_now_z_1 = Recon_z_losses[-2 * unit_iter:-unit_iter]
            #     tmp_now_z_0 = Recon_z_losses[-3 * unit_iter:-2 * unit_iter]
            #
            #     mean_now_z_2 = np.mean(tmp_now_z_2)
            #     mean_now_z_1 = np.mean(tmp_now_z_1)
            #     mean_now_z_0 = np.mean(tmp_now_z_0)
            #
            #     sign_x1 = mean_now_x_2/mean_now_x_1
            #     sign_z1 = mean_now_z_2/mean_now_z_1
            #
            #     sign_x2 = mean_now_x_2/min_total_x
            #     sign_z2 = mean_now_z_2/min_total_z
            #
            #     if mean_now_x_2 < min_total_x:
            #         min_total_x = mean_now_x_2
            #     if mean_now_z_2 < min_total_z:
            #         min_total_z = mean_now_z_2
            #
            #     sign_x = max([sign_x1,sign_x2])
            #     sign_z = max([sign_z1,sign_z2])
            #
            # #
            #     signs = [sign_x,sign_z]
            #     tmp_max = max(signs)
            #     tmp_min = min(signs)
            #     tmp_alpha = 0.5
            #     if tmp_max > 1:
            #         if tmp_min > 1:
            #             tmp_alpha = 0.5 + random.uniform(0.2,0.4)*(1/(1+math.exp(1-(tmp_max/tmp_min))))
            #         else:
            #             tmp_alpha = 0.5 + random.uniform(0.2,0.4)*(1/(1+math.exp(1-(tmp_max))))
            #     else:
            #         tmp_alpha = random.uniform(0,1)
            #
            #     if sign_z > sign_x:
            #         BETA4 = tmp_alpha*BETAS
            #         BETA3 = BETAS - BETA4
            #     else:
            #         BETA3 = tmp_alpha*BETAS
            #         BETA4 = BETAS - BETA3
            #     tmp_alpha = random.uniform(0, 1)
            #     BETA4 = tmp_alpha * BETAS
            #     BETA3 = BETAS - BETA4
            #     alphas.append(tmp_alpha)

            if C_update:
                # C_loss,EG_loss,mss_loss,l1_conv,mss_x_loss,EG_loss2
                # wali.forward(x=,z=,lamb=,beta1=,beta2=,gan=,loss_type=)
                C_loss, EG_loss, mss_loss, l1_conv, mss_x_loss, EG_loss2, RECON_Z_loss = wali.forward(x=x, z=z,
                                                                                                      lamb=LAMBDA,
                                                                                                      gan=0,
                                                                                                      loss_type='msssim',
                                                                                                      beta1=1.25,
                                                                                                      beta2=0.3,
                                                                                                      beta3=0.7,
                                                                                                      methods=0,
                                                                                                      l1=True,
                                                                                                      val_range=2,
                                                                                                      normalize="relu",
                                                                                                      pads=True,
                                                                                                      ssm_alpha=0.84)
                # EG_loss2 = beta1*(mss_loss*beta3+RECON_Z_loss*beta2)+EG_loss
                #                 if l1:
                #                     return C_loss,EG_loss,mss_loss,l1_conv,mss_x_loss,EG_loss2,RECON_Z_loss

                # def forward(self, x, z, lamb=10, beta1=0.01, beta2=0.01, beta3=0.03, gan=0, loss_type='raw',
                #             var_beta=-1, clip_beta=-1, methods=0, l1=True, var_lange=2, normalize="relu", pads=False,
                #             ssm_alpha=0.84):

                # C_loss, EG_loss = wali.forward(x=x,z=z,lamb=LAMBDA,gan=0,loss_type='raw',methods=1,var_beta=var_beta,clip_beta=clip_beta)
                # wali(x, z, lamb=LAMBDA,gan=0,loss_type='raw')

                optimizerC.zero_grad()
                C_loss.backward()
                optimizerC.step()

                C_lossk, EG_lossk, mmd_penalty = wali.forward(x=x, z=z, lamb=LAMBDA, beta1=1, beta2=0.62, beta3=1.0,
                                                              gan=1, loss_type='rep_gp', methods=0, var_beta=var_beta,
                                                              clip_beta=clip_beta)
                optimizerMoz.zero_grad()
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

                if C_iter == C_ITERS:
                    C_iter = 0
                    C_update, MMD_UPDATE, EG_update1, EG_update2 = False, False, True, False
                continue

            if EG_update1:
                optimizerEG.zero_grad()
                C_loss, EG_loss, mss_loss, l1_conv, mss_x_loss, EG_loss2, RECON_Z_loss = wali.forward(x=x, z=z,
                                                                                                      lamb=LAMBDA,
                                                                                                      gan=0,
                                                                                                      loss_type='msssim',
                                                                                                      beta1=1.25,
                                                                                                      beta2=0.3,
                                                                                                      beta3=0.7,
                                                                                                      methods=0,
                                                                                                      l1=True,
                                                                                                      pads=wali.pads,
                                                                                                      val_range=2,
                                                                                                      normalize="relu",
                                                                                                      ssm_alpha=0.84)
                # C_loss, EG_loss = wali.forward(x=x, z=z, lamb=LAMBDA, gan=0, loss_type='raw',methods=1,var_beta=var_beta,clip_beta=clip_beta)
                # EG_loss.backward()
                EG_loss2.backward()
                optimizerEG.step()

                optimizerEG2.zero_grad()
                C_loss, EG_loss, mss_loss, l1_conv, mss_x_loss, EG_loss2, RECON_Z_loss = wali.forward(x=x, z=z,
                                                                                                      lamb=LAMBDA,
                                                                                                      gan=0,
                                                                                                      loss_type='msssim',
                                                                                                      beta1=1.25,
                                                                                                      beta2=0.3,
                                                                                                      beta3=0.7,
                                                                                                      methods=0,
                                                                                                      l1=True,
                                                                                                      val_range=2,
                                                                                                      normalize="relu",
                                                                                                      pads=wali.pads,
                                                                                                      ssm_alpha=0.84)
                C_losses.append(C_loss.item())
                EG_losses.append(EG_loss.item())
                mss_losses.append(mss_loss.item())
                l1s.append(l1_conv.item())
                mss_x_losses.append(mss_x_loss.item())
                EG_losses2.append(EG_loss2.item())
                # RECON_Z_loss,EG_loss2,
                C_loss, EG_loss, mmd_penalty = wali.forward(x=x, z=z, lamb=LAMBDA, beta1=0.36, beta2=0.12, gan=1,
                                                            loss_type='reg-gp', methods=1, var_beta=var_beta,
                                                            clip_beta=clip_beta)
                EG_loss.backward()
                MMD_C_losses.append(C_loss.item())
                MMD_EG_losses.append(EG_loss.item())
                Recon_z_losses.append(RECON_Z_loss.item())
                # EG_losses22.append(EG_loss2.item())
                mmd_penaltys.append(mmd_penalty.item())
                optimizerEG2.step()

                # print("EG update1")
                EG_iter1 += 1
                if EG_iter1 == EG_ITERS:
                    EG_iter1 = 0
                    C_update, MMD_UPDATE, EG_update1, EG_update2 = True, False, False, False
                    curr_iter += 1
                    # C_update,MMD_UPDATE,EG_update1,EG_update2 = False,True,False,False
                # continue

            # if MMD_UPDATE:
            #     optimizerMoz.zero_grad()
            #     C_loss, EG_loss, RECON_Z_loss, EG_loss2, mmd_penalty = wali.forward(x=x, z=z, lamb=LAMBDA, beta1=1,
            #                                                                         beta2=0.62, beta3=1.0, gan=1,
            #                                                                         loss_type='rep_gp', methods=0,
            #                                                                         var_beta=var_beta,
            #                                                                         clip_beta=clip_beta)
            #
            #     C_loss.backward()
            #     # optimizerMox.step()
            #     optimizerMoz.step()
            #     # print("MMD update")
            #     MMD_iter += 1
            #     if MMD_iter == MMD_ITERS:
            #         MMD_iter = 0
            #         C_update, MMD_UPDATE, EG_update1, EG_update2 = False, False, False, True
            #     continue

            # if EG_update2:
            #     optimizerEG2.zero_grad()
            #     # optimizerE2.zero_grad()
            #     C_loss, EG_loss, mss_loss, l1_conv, mss_x_loss, EG_loss2 = wali.forward(x=x, z=z, lamb=LAMBDA, gan=0,
            #                                                                             loss_type='msssim', beta1=1.0,
            #                                                                             methods=0, l1=True, val_range=2,
            #                                                                             normalize="relu", pads=False,
            #                                                                             ssm_alpha=0.84)
            #     # C_loss, EG_loss = wali.forward(x=x, z=z, lamb=LAMBDA, gan=0, loss_type='raw',methods=0)
            #     C_losses.append(C_loss.item())
            #     EG_losses.append(EG_loss.item())
            #     mss_losses.append(mss_loss.item())
            #     l1s.append(l1_conv.item())
            #     mss_x_losses.append(mss_x_loss.item())
            #     EG_losses2.append(EG_loss2.item())
            #     if curr_iter > 100 and curr_iter % 200 == 0:
            #         loss_change = (np.mean(EG_losses[-100:]) / np.mean(EG_losses[-200:-100])) - 1
            #         # loss_change
            #         loss_change = abs(loss_change)
            #         if loss_change < threshold:
            #             if num_dec == 0:
            #                 var_beta = _beta0
            #                 clip_beta = 2 * _beta0
            #             else:
            #                 if num_dec >= 25:
            #                     num_dec = 25
            #                 var_beta = _beta0 * (decay ** num_dec)
            #                 clip_beta = 2 * var_beta
            #             num_dec = num_dec + 1
            #             print(var_beta)
            #             # print(num_dec+`)
            #     C_loss, EG_loss, RECON_Z_loss, EG_loss2, mmd_penalty = wali.forward(x=x, z=z, lamb=LAMBDA, beta1=0.25,
            #                                                                         beta2=0.12, gan=1,
            #                                                                         loss_type='mmd_b', methods=1,
            #                                                                         var_beta=var_beta,
            #                                                                         clip_beta=clip_beta)
            #     EG_loss2.backward()
            #     MMD_C_losses.append(C_loss.item())
            #     MMD_EG_losses.append(EG_loss.item())
            #     Recon_z_losses.append(RECON_Z_loss.item())
            #     EG_losses22.append(EG_loss2.item())
            #     mmd_penaltys.append(mmd_penalty.item())
            #     optimizerEG2.step()
            #     # optimizerE2.step()
            #     # print("EG update2")
            #     EG_iter2 += 1
            #     if EG_iter2 == EG_ITERS:
            #         EG_iter2 = 0
            #         C_update, MMD_UPDATE, EG_update1, EG_update2 = True, False, False, False
            #         curr_iter += 1

            # if EG_update:
            #
            #     optimizerEG.zero_grad()
            #     EG_loss2.backward()
            #     EG_losses2.append(EG_loss2.item())
            #     EG_losses.append(EG_loss.item())
            #     C_losses.append(C_loss.item())
            #     Recon_x_losses.append(RECON_X_loss.item())
            #     Recon_z_losses.append(RECON_Z_loss.item())
            #
            #     optimizerEG.step()
            #     EG_iter += 1
            #
            #     if EG_iter == EG_ITERS:
            #         EG_iter = 0
            #         C_update, EG_update = True, False
            #         curr_iter += 1
            #     else:
            #         continue
            # print(curr_iter)
            # print training statistics
            if curr_iter % 10 == 0:
                # print(EG_loss2)
                print(
                    '[%d/%d]\tW-distance: %.4f\tC-loss: %.4f\tMSSSIM:%.4f\tL1:%.4f\tMMD-EG-loss: %.4f\tMMD-C-loss: %.4f\treconz-loss: %.4f\tmmd_penalty:%.4f\ttotal_gen-loss:%.4f'
                    % (curr_iter, ITER, EG_losses[-1], C_losses[-1], mss_losses[-1], l1s[-1], MMD_EG_losses[-1],
                       MMD_C_losses[-1], Recon_z_losses[-1], mmd_penaltys[-1],
                       EG_losses2[-1]))
                # plot reconstructed images and samples
                if curr_iter % 100 == 0:
                    wali.eval()
                    real_x, rect_x = init_x[:32], wali.reconstruct(init_x[:32]).detach_()
                    rect_imgs = torch.cat((real_x.unsqueeze(1), rect_x.unsqueeze(1)), dim=1)
                    rect_imgs = rect_imgs.view(64, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
                    genr_imgs = wali.generate(noise).detach_().cpu()
                    utils.save_image(rect_imgs * 0.5 + 0.5, mmds+'/%d/rect%d.png' % (num_exp,curr_iter))
                    utils.save_image(genr_imgs * 0.5 + 0.5, mmds+'/%d/genr%d.png' % (num_exp,curr_iter))
                    wali.train()

            # save model
            if curr_iter % (ITER // 20) == 0:
                torch.save(wali, mmds+'/%d/total_%d.pth' % (num_exp,curr_iter))
                torch.save(wali.get_G(), mmds+'/%d/G_%d.pth' % (num_exp,curr_iter))
                torch.save(wali.get_C(), mmds+'/%d/C_%d.pth' % (num_exp,curr_iter))
                torch.save(wali.get_E(), mmds+'/%d/E_%d.pth' % (num_exp,curr_iter))
                torch.save(wali.get_MMDX(),mmds+'/%d/MMDX_%d.pth' % (num_exp,curr_iter))
                torch.save(wali.get_MMDZ(),mmds+'/%d/MMDZ_%d.pth' % (num_exp,curr_iter))
    # plot training loss curve
    '''
    np.save(mmds+"/%d/EGloss.npy" % num_exp,EG_losses)
    np.save(mmds+"/%d/MMDEloss.npy" % num_exp,MMD_EG_losses)
    np.save(mmds+"/%d/MMDCloss.npy" % num_exp,MMD_C_losses)
    np.save(mmds+"/%d/mmdpenalty.npy" % num_exp,mmd_penaltys)
    np.save(mmds + "/%d/EGloss_reg.npy" % num_exp, EG_losses2)
    np.save(mmds+"/%d/MMDEGloss_reg.npy" % num_exp,EG_losses22)
    np.save(mmds+"/%d/msssim_raw.npy" % num_exp,mss_x_losses)
    np.save(mmds + "/%d/msssim.npy" % num_exp, mss_losses)
    np.save(mmds + "/%d/l1.npy" % num_exp, l1s)
    np.save(mmds+"/%d/Closs.npy" % num_exp,C_losses)
    # np.save("mmds/RECONX9.npy",Recon_x_losses)
    np.save(mmds+"/%d/RECONZ.npy" % num_exp,Recon_z_losses)
    # np.save("mmds/alphas9.npy",alphas)
    EG_losses = np.load(mmds+"/%d/EGloss.npy" % num_exp)
    C_losses = np.load(mmds+"/%d/Closs.npy" % num_exp)

    plt.figure(0,figsize=(10, 7.5))
    plt.title('Main Training loss curve')
    plt.plot(EG_losses, label='Encoder + Generator')
    # plt.plot(C_losses, label='Criic')
    plt.plot(C_losses, label='Critic')
    #
    # plt.plot(EG_losses2,label='After Regularization')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mmds+'/celeba/loss_curve%d.png' % num_exp)
    # plt.show()

    plt.figure(1,figsize=(10, 5))
    plt.title('MMD Regularization loss curve')
    plt.plot(MMD_C_losses, label='Critic MMD of X')
    plt.plot(EG_losses2, label='MMD Regularization for Generator')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mmds+'/celeba/MMD_global%d.png' % num_exp)

    plt.figure(2, figsize=(10, 5))
    plt.title('Encoder Regularization')
    # plt.plot(MMD_C_losses, label='Critic MMD of X')
    plt.plot(Recon_z_losses, label='Reconstruction of z')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mmds+'/celeba/reconz%d.png' % num_exp)

    plt.figure(3, figsize=(10, 5))
    plt.title('MSSSIM')
    # plt.plot(MMD_C_losses, label='Critic MMD of X')
    plt.plot(mss_x_losses, label='MS-SSIM of x')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mmds + '/celeba/MS-SSIM%d.png' % num_exp)
    # plt.show()

    plt.figure(4, figsize=(10, 5))
    plt.title('L1')
    # plt.plot(MMD_C_losses, label='Critic MMD of X')
    plt.plot(mss_x_losses, label='L1 of x')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mmds + '/celeba/L1-%d.png' % num_exp)
    '''
    np.save(mmds+"/%d/EGloss.npy" % num_exp,EG_losses)
    np.save(mmds+"/%d/MMDEloss.npy" % num_exp,MMD_EG_losses)
    np.save(mmds+"/%d/MMDCloss.npy" % num_exp,MMD_C_losses)
    np.save(mmds + "/%d/mmdpenalty.npy" % num_exp, mmd_penaltys)
    # np.save(mmds+"mmds/mnist/%d/mmdpenalty.npy" % num_exp,mmd_penaltys)
    np.save(mmds + "/%d/EGloss_reg.npy" % num_exp, EG_losses2)
    np.save(mmds + "/%d/MMDEGloss_reg.npy" % num_exp, EG_losses22)
    np.save(mmds + "/%d/msssim_raw.npy" % num_exp, mss_x_losses)
    np.save(mmds + "/%d/msssim.npy" % num_exp, mss_losses)
    np.save(mmds + "/%d/l1.npy" % num_exp, l1s)
    np.save(mmds + "/%d/Closs.npy" % num_exp, C_losses)
    # np.save("mmds/RECONX9.npy",Recon_x_losses)
    np.save(mmds + "/%d/RECONZ.npy" % num_exp, Recon_z_losses)
    # np.save("mmds/alphas9.npy",alphas)
    EG_losses = np.load(mmds + "/%d/EGloss.npy" % num_exp)
    C_losses = np.load(mmds + "/%d/Closs.npy" % num_exp)

    plt.figure(0,figsize=(10, 7.5))
    plt.title('Main Training loss curve')
    plt.plot(EG_losses, label='Encoder + Generator')
    # plt.plot(C_losses, label='Criic')
    plt.plot(C_losses, label='Critic')
    #
    # plt.plot(EG_losses2,label='After Regularization')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mmds+'/%d/loss_curve%d.png' % (num_exp,num_exp))
    # plt.show()

    plt.figure(1,figsize=(10, 5))
    plt.title('MMD Regularization loss curve')
    plt.plot(MMD_C_losses, label='Critic MMD of X')
    plt.plot(EG_losses2, label='MMD Regularization for Generator')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig(mmds+'/%d/MMD_global%d.png' % (num_exp,num_exp))
    #
    plt.figure(2, figsize=(10, 5))
    plt.title('Encoder Regularization')
    # plt.plot(MMD_C_losses, label='Critic MMD of X')
    plt.plot(Recon_z_losses, label='Reconstruction of z')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mmds + '/%d/reconz%d.png' % (num_exp,num_exp))

    plt.figure(3, figsize=(10, 5))
    plt.title('MSSSIM')
    # plt.plot(MMD_C_losses, label='Critic MMD of X')
    plt.plot(mss_x_losses, label='MS-SSIM of x')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mmds + '/%d/MS-SSIM%d.png' % (num_exp,num_exp))
    # plt.show()

    plt.figure(4, figsize=(10, 5))
    plt.title('L1')
    # plt.plot(MMD_C_losses, label='Critic MMD of X')
    plt.plot(l1s, label='L1 of x')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mmds + '/%d/L1-%d.png' % (num_exp,num_exp))
    plt.show()




if __name__ == "__main__":
    main()
  #   ceshi = create_encoder()
  #   k = torch.randn([2,80,1,1])
  #   G = create_generator()
  #   kk = G(k)
  #   print(kk.size())
# print(ceshi(k).size())

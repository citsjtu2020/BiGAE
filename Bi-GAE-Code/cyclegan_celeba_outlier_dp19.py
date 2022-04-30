import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision import datasets, transforms, utils
from configutils import load_config
from cycle_model.models import Generator
from cycle_model.models import Discriminator
from cycle_model.models import CycleGANFramework
from cycle_model.utils import ReplayBuffer
from cycle_model.utils import LambdaLR
from cycle_model.utils import Logger
from cycle_model.utils import weights_init_normal
from cycle_model.datasets import ImageDataset
from torch.utils import data

ITER = 40000

C_ITERS = 5       # critic iterations
MMD_ITERS = 3       # mmd iterations!
EG_ITERS = 1

NLAT = 256

LAMBDA = 10.0       # strength of gradient penalty
# loss_lamb_id=5.0,lamb_gp=10.0,loss_lamb_cycle=10.0
LAMBDA_ID = 5.0
LAMBDA_CYCLE=10.0

LEARNING_RATE = 1.3e-4

import random
data_path = '/home/huaqin/celeba'
save_path = "/data1/JCST/results"
attrs = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
attr_list = attrs.split()
att_index = {}
index_att = {}
for i in range(len(attr_list)):
    att_index[attr_list[i]] = i+1

for key in att_index.keys():
    index_att[att_index[key]] = key
print(index_att[21].lower()+'.json')
import os
import skimage.io as io
import numpy as np
from torchsummary import summary

class CustomDataset(data.Dataset):
    def __init__(self,aims,mode='train',pos=1,unaligned=True):
        super(CustomDataset, self).__init__()
        self.aims = aims
        self.unaligned = unaligned
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
    def __getitem__(self, item):
        item = item % (self.__len__())
        aim_image1 = self.train_data[item]
        aim_path1 = os.path.join(data_path,aim_image1)
        if self.unaligned:
            item2 = random.randint(0, (self.__len__() - 1))
            aim_image2 = self.train_data[item2]
        else:
            aim_image2 = self.train_data[item]
        aim_path2 = os.path.join(data_path, aim_image2)
        item_image1 = io.imread(aim_path1)
        item_image2 = io.imread(aim_path2)
        item_image1 = np.transpose(item_image1, (2, 0, 1))
        item_image2 = np.transpose(item_image2, (2, 0, 1))
        item_image1 = item_image1 / 255.0
        item_image2 = item_image2 / 255.0
        item_image1 = (item_image1 - 0.5)/0.5
        item_image2 = (item_image2 - 0.5) / 0.5
        item_image1 = torch.from_numpy(item_image1)
        item_image2 = torch.from_numpy(item_image2)
        item_image1 = item_image1.type(torch.FloatTensor)
        item_image2 = item_image2.type(torch.FloatTensor)
        return item_image1,item_image2

BATCH_SIZE=20

data_path = '/home/huaqin/celeba'

save_path = "/data1/JCST/results"

LEARNING_RATE = 1.3e-4

IMAGE_SIZE = 256
NUM_CHANNELS = 3
cuda_list = [i.strip() for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
device_ids = [i for i in range(len(cuda_list))]
device = torch.device("cuda", device_ids[0])

if torch.cuda.device_count() > 1:
    num_gpu = torch.cuda.device_count()
if torch.cuda.device_count() < 1:
    num_gpu = 1

print("use %d GPUs" % num_gpu)

svhn = CustomDataset(aims=-1, unaligned=True)




parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_feature', type=int, default=64, help='num of features')
parser.add_argument('--down', type=int, default=3, help='down sample num')
parser.add_argument('--num_exp', type=int, default=22, help='number of exp')
parser.add_argument('--loss_type', type=str, default="raw", help='type of loss')

parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)
num_exp=opt.num_exp
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

def main():
    # if not os.path.exists
    loader = data.DataLoader(svhn, BATCH_SIZE * num_gpu, shuffle=True,
                             num_workers=2)
    mmds = os.path.join(save_path, 'cycle%d' % IMAGE_SIZE)
    if not os.path.exists(os.path.join(save_path, 'cycle%d' % IMAGE_SIZE)):
        os.makedirs(os.path.join(save_path, 'cycle%d' % IMAGE_SIZE))
        print("目录创建成功！")
    if not os.path.exists(mmds + "/%d" % num_exp):
        os.makedirs(mmds + "/%d" % num_exp)
        print("目录创建成功！")
    if not os.path.exists(mmds + "/celeba"):
        os.makedirs(mmds + "/celeba")
        print("目录创建成功！")
    if not os.path.exists(mmds + "/celeba" + "/%d" % num_exp):
        os.makedirs(mmds + "/celeba" + "/%d" % num_exp)
        print("目录创建成功！")

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc, n_feature=opt.n_feature, down=opt.down,encode_feature=[],nlat=NLAT,image_size=IMAGE_SIZE,n_residual_blocks=6)
    netG_B2A = Generator(opt.output_nc, opt.input_nc, n_feature=opt.n_feature, down=opt.down,encode_feature=[],nlat=NLAT,image_size=IMAGE_SIZE,n_residual_blocks=6)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    netG_A2B_model = torch.load("/data1/JCST/results/cycle256/22/G_A2B_epoch_345.pth",map_location="cuda:0")
    netG_B2A_model = torch.load("/data1/JCST/results/cycle256/22/G_B2A_epoch_345.pth", map_location="cuda:0")
    netD_A_model = torch.load("/data1/JCST/results/cycle256/22/D_A_epoch_345.pth", map_location="cuda:0")
    netD_B_model = torch.load("/data1/JCST/results/cycle256/22/D_B_epoch_345.pth", map_location="cuda:0")

    # netG_A2B.load_state_dict(netG_A2B_model)
    # netG_B2A.load_state_dict(netG_B2A_model)
    # netD_A.load_state_dict(netD_A_model)
    # netD_B.load_state_dict(netD_B_model)
    # def __init__(self, netG_A2B, netG_B2A, netD_A, netD_B, loss_type='raw', max_size=100, initial=True):
    wali = CycleGANFramework(netG_A2B, netG_B2A, netD_A, netD_B, loss_type=opt.loss_type, max_size=BATCH_SIZE * num_gpu,
                             initial=False).cuda(device=device_ids[0])

    wali.netG_A2B.load_state_dict(netG_A2B_model)
    wali.netD_A.load_state_dict(netD_A_model)
    wali.netD_B.load_state_dict(netD_B_model)
    wali.netG_B2A.load_state_dict(netG_B2A_model)

    summary(wali.get_G_A2B(), (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    summary(wali.get_G_B2A(), (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

    summary(wali.get_D_A(), (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    summary(wali.get_D_B(), (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        wali = torch.nn.parallel.DataParallel(wali,device_ids=device_ids)

        optimizer_G = torch.optim.Adam(itertools.chain(wali._modules['module'].get_g_a2b_parameters(), wali._modules['module'].get_g_b2a_parameters()),
                                       lr=opt.lr, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(wali._modules['module'].get_d_a_parameters(), lr=opt.lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(wali._modules['module'].get_d_b_parameters(), lr=opt.lr, betas=(0.5, 0.999))

    else:
        optimizer_G = torch.optim.Adam(itertools.chain(wali.get_g_a2b_parameters(), wali.get_g_b2a_parameters()),
                                       lr=opt.lr, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(wali.get_d_a_parameters(), lr=opt.lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(wali.get_d_b_parameters(), lr=opt.lr, betas=(0.5, 0.999))

    num_batches = len(loader) // ((C_ITERS+EG_ITERS))
    print(num_batches)
    print(len(loader))
    epochess = (ITER // num_batches)+2
    epoch = 345
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                           lr_lambda=LambdaLR(epochess, epoch,
                                                                              (epochess // 2 +1)).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=LambdaLR(epochess, epoch,
                                                                            (epochess // 2 + 1)).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=LambdaLR(epochess, epoch,
                                                                            (epochess // 2 + 1)).step)

    loss_Gs, loss_identity_As, loss_identity_Bs, loss_GAN_A2Bs, loss_GAN_B2As, loss_cycle_ABAs, loss_cycle_BABs = [],[],[],[],[],[],[]

    loss_D_As, loss_D_Bs = [],[]

    curr_iter = C_iter = EG_iter1 = EG_iter2 = MMD_iter = 0

    C_update, EG_update = True, False


    curr_iter = 17476

    # init_x0 = np.load(mmds + "/%d/sample_A.npy" % num_exp)
    # init_x1 = np.load(mmds + "/%d/sample_B.npy" % num_exp)
    #
    # init_x0 = torch.from_numpy(init_x0)
    # init_x0 = init_x0.type(torch.FloatTensor)
    #
    # init_x1 = torch.from_numpy(init_x1)
    # init_x1 = init_x1.type(torch.FloatTensor)



    tmp_loss_D_As = 0.0
    tmp_loss_D_Bs = 0.0

    loads = True

    while curr_iter < ITER:
        loader = data.DataLoader(svhn, BATCH_SIZE * num_gpu, shuffle=True,
                                 num_workers=2)

        for i, x in enumerate(loader, 1):
            # Set model input
            # x = x.cuda(device=device_ids[0])
            # x = x
            real_A = x[0].cuda(device=device_ids[0])
            real_B = x[1].cuda(device=device_ids[0])
            # real_A = Variable(input_A.copy_(batch['A']))
            # real_B = Variable(input_B.copy_(batch['B']))

            # init_x1.cuda(device=device_ids[0])
            # init_x0.cuda(device=device_ids[0])

            if curr_iter == 0 or loads:
                init_x0 = x[0].cuda(device=device_ids[0])
                init_x1 = x[1].cuda(device=device_ids[0])
                if curr_iter == 0:
                # if local_rank == 0:
                    np.save(mmds+"/%d/sample_A.npy" % num_exp,init_x0.cpu().numpy())
                    np.save(mmds + "/%d/sample_B.npy" % num_exp, init_x1.cpu().numpy())
                curr_iter += 1
                loads = False

            if C_update:
                #     forward(self, x_a,x_b,disc=False,loss_lamb_id=5.0,lamb_gp=10.0,loss_lamb_cycle=10.0):
                loss_D_A, loss_D_B = wali.forward(real_A,real_B,disc=True,lamb_gp=LAMBDA,loss_lamb_cycle=LAMBDA_CYCLE,loss_lamb_id=LAMBDA_ID)

                loss_D_A = loss_D_A.mean()
                loss_D_B = loss_D_B.mean()

                optimizer_G.zero_grad()
                optimizer_D_A.zero_grad()
                optimizer_D_B.zero_grad()

                loss_D_A.backward()
                optimizer_D_A.step()

                optimizer_G.zero_grad()
                optimizer_D_A.zero_grad()
                optimizer_D_B.zero_grad()

                loss_D_B.backward()
                optimizer_D_B.step()

                optimizer_G.zero_grad()
                optimizer_D_A.zero_grad()
                optimizer_D_B.zero_grad()

                C_iter += 1
                tmp_loss_D_As = loss_D_A.item()
                tmp_loss_D_Bs = loss_D_B.item()

                if C_iter == C_ITERS:
                    C_iter = 0
                    C_update, EG_update = False, True
                continue

            if EG_update:
                optimizer_G.zero_grad()
                optimizer_D_A.zero_grad()
                optimizer_D_B.zero_grad()
                #     forward(self, x_a,x_b,disc=False,loss_lamb_id=5.0,lamb_gp=10.0,loss_lamb_cycle=10.0):

                loss_G, loss_identity_A, loss_identity_B, loss_GAN_A2B, loss_GAN_B2A, loss_cycle_ABA, loss_cycle_BAB = wali.forward(real_A,real_B,disc=False,lamb_gp=LAMBDA,loss_lamb_cycle=LAMBDA_CYCLE,loss_lamb_id=LAMBDA_ID)

                loss_G = loss_G.mean()
                loss_identity_A = loss_identity_A.mean()
                loss_identity_B = loss_identity_B.mean()
                loss_GAN_A2B = loss_GAN_A2B.mean()
                loss_GAN_B2A = loss_GAN_B2A.mean()
                loss_cycle_ABA = loss_cycle_ABA.mean()
                loss_cycle_BAB = loss_cycle_BAB.mean()

                loss_G.backward()
                optimizer_G.step()

                optimizer_G.zero_grad()
                optimizer_D_A.zero_grad()
                optimizer_D_B.zero_grad()

                EG_iter2 += 1

                if EG_iter2 == EG_ITERS:
                    EG_iter2 = 0
                    C_update, EG_update = True, False
                    # if local_rank == 0:
                    loss_Gs.append(loss_G.item())
                    loss_identity_As.append(loss_identity_A.item())
                    loss_identity_Bs.append(loss_identity_B.item())
                    loss_GAN_A2Bs.append(loss_GAN_A2B.item())
                    loss_GAN_B2As.append(loss_GAN_B2A.item())
                    loss_cycle_ABAs.append(loss_cycle_ABA.item())
                    loss_cycle_BABs.append(loss_cycle_BAB.item())

                    loss_D_As.append(tmp_loss_D_As)
                    loss_D_Bs.append(tmp_loss_D_Bs)
                    curr_iter += 1

            if curr_iter % 10 == 0:
                # Reconx:%.4f
                print(
                    '[%d/%d]\tDiscriminator A loss: %.4f\tDiscriminator B loss: %.4f\tEG loss: %.4f\tIdentify A loss: %.4f\tIdentify B loss: %.4f\tEG_GAN_A2B loss: %.4f\tEG_GAN_B2A loss: %.4f\tEG_Cycle_ABA loss: %.4f\tEG_Cycle_BAB loss: %.4f'
                    % (curr_iter, ITER, loss_D_As[-1], loss_D_Bs[-1],loss_Gs[-1],loss_identity_As[-1],loss_identity_Bs[-1],loss_GAN_A2Bs[-1],loss_GAN_B2As[-1],loss_cycle_ABAs[-1],loss_cycle_BABs[-1]))


            if curr_iter % 200 == 0 :
                wali.eval()
                # //num_gpu
                # //num_gpu
                if torch.cuda.device_count() > 1:
                    real_xA, rect_xA = init_x0[:BATCH_SIZE], wali._modules['module'].reconstruct_ABA(init_x0[:BATCH_SIZE]).detach_()
                else:
                    real_xA, rect_xA = init_x0[:BATCH_SIZE], wali.reconstruct_ABA(init_x0[:BATCH_SIZE]).detach_()
                if torch.cuda.device_count() > 1:
                    real_xB, rect_xB = init_x1[:BATCH_SIZE], wali._modules['module'].reconstruct_BAB(
                        init_x1[:BATCH_SIZE]).detach_()
                else:
                    real_xB, rect_xB = init_x1[:BATCH_SIZE], wali.reconstruct_BAB(init_x1[:BATCH_SIZE]).detach_()
                rect_imgsA = torch.cat((real_xA.unsqueeze(1), rect_xA.unsqueeze(1)), dim=1)
                rect_imgsA = rect_imgsA.view((BATCH_SIZE)*2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()

                rect_imgsB = torch.cat((real_xB.unsqueeze(1), rect_xB.unsqueeze(1)), dim=1)
                rect_imgsB = rect_imgsB.view((BATCH_SIZE) * 2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()

                real_xA = init_x0[:BATCH_SIZE]
                real_xB = init_x1[:BATCH_SIZE]
                if torch.cuda.device_count() > 1:
                    genr_imgsA = wali._modules['module'].generate_A2B(real_xB).detach_().cpu()
                else:
                    genr_imgsA = wali.generate_A2B(real_xB).detach_().cpu()

                if torch.cuda.device_count() > 1:
                    genr_imgsB = wali._modules['module'].generate_B2A(real_xA).detach_().cpu()
                else:
                    genr_imgsB = wali.generate_B2A(real_xA).detach_().cpu()

                utils.save_image(rect_imgsA * 0.5 + 0.5, mmds+'/celeba/%d/rectA%d.png' % (num_exp,curr_iter))
                utils.save_image(rect_imgsB * 0.5 + 0.5, mmds + '/celeba/%d/rectB%d.png' % (num_exp, curr_iter))
                utils.save_image(genr_imgsA * 0.5 + 0.5, mmds+'/celeba/%d/genrA%d.png' % (num_exp,curr_iter))
                utils.save_image(genr_imgsB * 0.5 + 0.5, mmds + '/celeba/%d/genrB%d.png' % (num_exp, curr_iter))
                wali.train()

            if curr_iter % (ITER // 20) == 0:
                if torch.cuda.device_count() > 1:
                    torch.save(wali.module, mmds + '/celeba/%d/save_model_%d.ckpt' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].state_dict(), mmds + '/%d/total_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_G_B2A().state_dict(), mmds + '/%d/G_B2A_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_D_A().state_dict(), mmds + '/%d/D_A_%d.pth' % (num_exp, curr_iter))

                    torch.save(wali._modules['module'].get_G_A2B().state_dict(), mmds + '/%d/G_A2B_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali._modules['module'].get_D_B().state_dict(), mmds + '/%d/D_B_%d.pth' % (num_exp, curr_iter))
                else:
                    torch.save(wali.state_dict(), mmds + '/%d/total_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_G_B2A().state_dict(), mmds + '/%d/G_B2A_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_D_A().state_dict(), mmds + '/%d/D_A_%d.pth' % (num_exp, curr_iter))

                    torch.save(wali.get_G_A2B().state_dict(), mmds + '/%d/G_A2B_%d.pth' % (num_exp, curr_iter))
                    torch.save(wali.get_D_B().state_dict(), mmds + '/%d/D_B_%d.pth' % (num_exp, curr_iter))

            if curr_iter % (ITER // 40) == 0:
                np.save(mmds + "/%d/loss_Gs.npy" % num_exp, loss_Gs)
                # np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
                np.save(mmds + "/%d/loss_D_As.npy" % num_exp, loss_D_As)
                np.save(mmds + "/%d/loss_D_Bs.npy" % num_exp, loss_D_Bs)

                np.save(mmds + "/%d/loss_GAN_A2Bs.npy" % num_exp, loss_GAN_A2Bs)
                np.save(mmds + "/%d/loss_GAN_B2As.npy" % num_exp, loss_GAN_B2As)

                np.save(mmds + "/%d/loss_cycle_ABAs.npy" % num_exp, loss_cycle_ABAs)
                np.save(mmds + "/%d/loss_cycle_BABs.npy" % num_exp, loss_cycle_BABs)

                np.save(mmds + "/%d/loss_identity_As.npy" % num_exp, loss_identity_As)
                np.save(mmds + "/%d/loss_identity_Bs" % num_exp, loss_identity_Bs)


        if torch.cuda.device_count() > 1:
            torch.save(wali.module, mmds + '/celeba/%d/save_model_epoch_%d_%d.ckpt' % (num_exp, epoch, curr_iter))

            torch.save(wali._modules['module'].state_dict(), mmds + '/%d/total_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_G_B2A().state_dict(),
                       mmds + '/%d/G_B2A_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_D_A().state_dict(), mmds + '/%d/D_A_epoch_%d.pth' % (num_exp, epoch))

            torch.save(wali._modules['module'].get_G_A2B().state_dict(),
                       mmds + '/%d/G_A2B_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali._modules['module'].get_D_B().state_dict(), mmds + '/%d/D_B_epoch_%d.pth' % (num_exp, epoch))


        else:
            torch.save(wali.state_dict(), mmds + '/%d/total_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_G_B2A().state_dict(),
                       mmds + '/%d/G_B2A_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_D_A().state_dict(), mmds + '/%d/D_A_epoch_%d.pth' % (num_exp, epoch))

            torch.save(wali.get_G_A2B().state_dict(),
                       mmds + '/%d/G_A2B_epoch_%d.pth' % (num_exp, epoch))
            torch.save(wali.get_D_B().state_dict(), mmds + '/%d/D_B_epoch_%d.pth' % (num_exp, epoch))

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        epoch += 1

    if torch.cuda.device_count() >= 1:
        np.save(mmds + "/%d/loss_Gs.npy" % num_exp, loss_Gs)
        # np.save(mmds + "/%d/MMDEloss.npy" % num_exp, MMD_EG_losses)
        np.save(mmds + "/%d/loss_D_As.npy" % num_exp, loss_D_As)
        np.save(mmds + "/%d/loss_D_Bs.npy" % num_exp, loss_D_Bs)

        np.save(mmds + "/%d/loss_GAN_A2Bs.npy" % num_exp, loss_GAN_A2Bs)
        np.save(mmds + "/%d/loss_GAN_B2As.npy" % num_exp, loss_GAN_B2As)

        np.save(mmds + "/%d/loss_cycle_ABAs.npy" % num_exp, loss_cycle_ABAs)
        np.save(mmds + "/%d/loss_cycle_BABs.npy" % num_exp, loss_cycle_BABs)

        np.save(mmds + "/%d/loss_identity_As.npy" % num_exp, loss_identity_As)
        np.save(mmds + "/%d/loss_identity_Bs" % num_exp, loss_identity_Bs)

if __name__ == "__main__":
    main()
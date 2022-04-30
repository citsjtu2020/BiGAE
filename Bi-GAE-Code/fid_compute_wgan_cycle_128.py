import torch
import skimage.io as io
import numpy as np
import argparse
from torchvision import datasets, transforms, utils
import random
from cycle_model.models import Generator
from cycle_model.models import Discriminator
from cycle_model.models import CycleGANFramework
from cycle_model.utils import ReplayBuffer
from cycle_model.utils import LambdaLR
from cycle_model.models import CycleGANFramework
# CUDA_VISIBLE_DEVICES=2 python fid_model_compute_256.py --num_exp=22 --low=5000 --high=10000 --image_size=256 --nlat=256
# CUDA_VISIBLE_DEVICES=3 python fid_model_compute_256.py --num_exp=22 --low=10000 --high=15000 --image_size=256 --nlat=256
# CUDA_VISIBLE_DEVICES=2 python fid_model_compute_256.py --num_exp=22 --low=15000 --high=18000 --image_size=256 --nlat=256
# CUDA_VISIBLE_DEVICES=3 python fid_model_compute_256.py --num_exp=22 --low=18000 --high=20000 --image_size=256 --nlat=256
# CUDA_VISIBLE_DEVICES=3 python fid_compute_wgan_cycle_128.py --num_exp=23 --low=18000 --high=20000 --image_size=128 --nlat=128
# CUDA_VISIBLE_DEVICES=3 python fid_compute_wgan_cycle.py --num_exp=21 --low=5000 --high=10000 --image_size=512 --nlat=512
parser = argparse.ArgumentParser()
parser.add_argument('--low', type=int, default=5000, help='num of features')
parser.add_argument('--high', type=int, default=10000, help='num of features')
parser.add_argument('--loss_type', type=str, default="raw", help='type of loss')
parser.add_argument('--image_size', type=int, default=128, help='size of images')
parser.add_argument("--num_exp", type=int, default=23, help="id of experiments")
parser.add_argument("--nlat", type=int, default=128, help="id of experiments")
parser.add_argument('--n_feature', type=int, default=64, help='num of features')
parser.add_argument('--down', type=int, default=3, help='down sample num')
opt = parser.parse_args()
print(opt)

mmd_path0 = '/data1/JCST/results/mmds256/celeba/22'
mmd_path_base0 = '/data1/JCST/results/mmds256/celeba/22'

data_path = '/home/huaqin/celeba/'

attrs = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
attr_list = attrs.split()
att_index = {}
index_att = {}
for i in range(len(attr_list)):
    att_index[attr_list[i]] = i + 1

for key in att_index.keys():
    index_att[att_index[key]] = key
print(index_att[21].lower() + '.json')

attrs = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
attr_list = attrs.split()
att_index = {}
index_att = {}
for i in range(len(attr_list)):
    att_index[attr_list[i]] = i + 1

for key in att_index.keys():
    index_att[att_index[key]] = key
print(index_att[21].lower() + '.json')

from torch.utils import data
import os
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
# from util2 import DeterministicConditional, GaussianConditional, JointCritic, WALI,MMD_NET
from util4 import DeterministicConditional, GaussianConditional, JointCritic, WALI, MMD_NET
from torchvision import datasets, transforms, utils
# from wmgan_celeba_outlier5 import create_WALI
from configutils import load_config
# from bigan_celeba import create_WALI
#
# from bigan_celeba2 import create_WALI
# from wmgan_celeba_outlier17 import create_WALI
# from wmgan_celeba_outlier13 import create_WALI
# import aae_celeba_outlier_dp18_load
# import biganqp_celeba_outlier_dp18_load
# import bigan_celeba_outlier_dp18_load
# import wmgan_celeba_outlier_dp18_load
from configutils import load_config
import wmgan_celeba_nommd_dp19_load
import wmgan_celeba_mmd2mse_dp19_load
import wmgan_celeba_nossim_dp19_load
import wgan_base_celeba_outlier_dp18_load
import aae_celeba_outlier_dp19_load
import bigan_celeba_outlier_dp19_load
import biganqp_celeba_outlier_dp19_load
import wmgan_celeba_outlier_dp19_load
import wmgan_celeba_percept_dp19_load
import wmgan_celeba_dp_128_load
import bigan_celeba_dp_128_load
import biganqp_celeba_dp_128_load
import wgan_base_dp_128_load
import aae_celeba_dp_128_load
# import cyclegan_celeba_outlier_dp19_load

import wmgan_celeba_outlier_dp19_load

output_root = "/data1/JCST/results/recon-gener"
os.makedirs(output_root, exist_ok=True)


def select_encoder(aim=10000, basedir=mmd_path_base0, modeldir=mmd_path0,cycle=False):
    filelists = os.listdir(basedir)
    ckpt_lists = []
    for i in filelists:
        if '.ckpt' in i and 'epoch' in i:
            ckpt_lists.append(i)

    iteration_to_epoch = {}

    for ck in ckpt_lists:
        cks = ck.split(".")
        itoes = cks[0].strip().split("_")
        # print(itoes)
        try:
            tmp_epoch = int(itoes[-2])
            tmp_iter = np.abs(int(itoes[-1]) - aim)
            iteration_to_epoch[tmp_iter] = tmp_epoch
        except:
            pass

    iters = list(iteration_to_epoch.keys())

    iters.sort()
    aim_epoch = iteration_to_epoch[iters[0]]

    if cycle:
        # aim_file = "G_A2B_epoch_%d.pth" % aim_epoch
        return aim_epoch
    else:
        aim_file = "total_epoch_%d.pth" % aim_epoch

    aim_file = os.path.join(modeldir, aim_file)
    if not os.path.exists(aim_file):
        aim_file = os.path.join(modeldir,"total_%d.pth" % aim_epoch)
    return aim_file


# opt = parser.parse_args()
print(opt)
os.makedirs(output_root, exist_ok=True)
os.makedirs(os.path.join(output_root, "exp%d" % opt.image_size), exist_ok=True)
os.makedirs(os.path.join(os.path.join(output_root, "exp%d" % opt.image_size), "%d" % opt.num_exp), exist_ok=True)

base_path = os.path.join(os.path.join(output_root, "exp%d" % opt.image_size), "%d" % opt.num_exp)


class CustomDataset(data.Dataset):
    def __init__(self, aims, mode='ceshi', pos=1):
        super(CustomDataset, self).__init__()
        self.aims = aims
        self.true_config = {}
        self.out_config = {}
        if aims < 0:
            aim_file = 'total_hq2.json'
            aim_data = load_config(aim_file)[mode][:]
        else:
            aim_file = index_att[self.aims].lower() + '_hq.json'
            # if isinstance(mode,list)
            aim_data = load_config(aim_file)[mode][str(int(pos * aims))]
        self.train_data = aim_data

    def __len__(self):
        return len(self.train_data[:])

    def __getitem__(self, item):
        item = item % (self.__len__())
        aim_image = self.train_data[item]
        aim_path = os.path.join(data_path, aim_image)
        item_image = io.imread(aim_path)
        item_image = np.transpose(item_image, (2, 0, 1))
        item_image = item_image / 255.0
        item_image = (item_image - 0.5) / 0.5

        item_image = torch.from_numpy(item_image)
        item_image = item_image.type(torch.FloatTensor)

        return item_image, aim_image


def make_generates(device, network, nlat=512, batch_size=16, name="mmds", iteration=10000):
    train_dataset = []
    encoded_train = []
    test_dataset = []
    encoded_test = []
    encoded_valid = []
    message = load_config("glass_age_male_sim2.json")
    class_index = load_config("class_to_index.json")

    print(message.keys())
    train_sets = message['train']
    valid_sets = message['valid']
    test_sets = message['test']

    keys0 = train_sets.keys()
    network.eval()
    data_output = os.path.join(os.path.join(base_path, name), "%d" % iteration)
    recon_output = os.path.join(data_output, "recon")
    raw_output = os.path.join(os.path.join(base_path, name), "raw")
    gen_output = os.path.join(data_output, "gener")

    os.makedirs(data_output, exist_ok=True)
    os.makedirs(recon_output, exist_ok=True)
    os.makedirs(raw_output, exist_ok=True)
    os.makedirs(gen_output, exist_ok=True)

    recon_output_test = os.path.join(data_output, "recon_test")
    raw_output_test = os.path.join(os.path.join(base_path, name), "raw_test")
    gen_output_test = os.path.join(data_output, "gener_test")

    os.makedirs(recon_output_test, exist_ok=True)
    os.makedirs(raw_output_test, exist_ok=True)
    os.makedirs(gen_output_test, exist_ok=True)

    svhn = CustomDataset(aims=-1, mode="valid")
    loader = data.DataLoader(svhn, 1, shuffle=False, num_workers=4)
    now_index = 0
    for batch_x, (x, names) in enumerate(loader):
        noise = torch.randn(batch_size, nlat, 1, 1, device=device)
        x = x.to(device)
        noise = noise.to(device)
        # ex = network.reconstruct(x)
        gen_ex = network.generate(noise)
        name0 = names[0]
        for k in keys0:
            # "celeba-hq/celeba-512/193492.jpg"
            # view((BATCH_SIZE)*2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
            if name0 in valid_sets[k]:
                # kex = ex.view((3, opt.image_size, opt.image_size))
                kgen = gen_ex[random.randint(0, gen_ex.size()[0] - 1)].view((3, opt.image_size, opt.image_size))
                # kex = kex.detach().cpu()
                k_raw = x.detach().cpu().view((3, opt.image_size, opt.image_size))
                kgen = kgen.detach().cpu()
                aim_name0 = name0.split("/")[-1].strip()

                # utils.save_image(kex * 0.5 + 0.5, os.path.join(recon_output, aim_name0))
                utils.save_image(kgen * 0.5 + 0.5, os.path.join(gen_output, aim_name0))
                utils.save_image(k_raw * 0.5 + 0.5, os.path.join(raw_output, aim_name0))

                break
        print(batch_x)
    svhn = CustomDataset(aims=-1, mode="test")
    loader = data.DataLoader(svhn, 1, shuffle=False, num_workers=4)
    now_index = 0

    for batch_x, (x, names) in enumerate(loader):
        noise = torch.randn(batch_size, nlat, 1, 1, device=device)
        x = x.to(device)
        noise = noise.to(device)
        # ex = network.reconstruct(x)
        gen_ex = network.generate(noise)
        name0 = names[0]
        for k in keys0:
            if name0 in test_sets[k]:
                # kex = ex.view((3, opt.image_size, opt.image_size))
                kgen = gen_ex[random.randint(0, gen_ex.size()[0] - 1)].view((3, opt.image_size, opt.image_size))
                # kex = kex.detach().cpu()
                k_raw = x.detach().cpu().view((3, opt.image_size, opt.image_size))
                kgen = kgen.detach().cpu()
                aim_name0 = name0.split("/")[-1].strip()
                # utils.save_image(kex * 0.5 + 0.5, os.path.join(recon_output, aim_name0))
                utils.save_image(kgen * 0.5 + 0.5, os.path.join(gen_output, aim_name0))
                utils.save_image(k_raw * 0.5 + 0.5, os.path.join(raw_output, aim_name0))
                break
        print(batch_x)
# parser.add_argument('--loss_type', type=str, default="raw", help='type of loss')
def make_cycle_reconstruct(device, network, nlat=512,batch_size=16 ,name="cycle", iteration=10000):
    os.makedirs(os.path.join(base_path, name), exist_ok=True)

    train_dataset = []
    encoded_train = []
    test_dataset = []
    encoded_test = []
    encoded_valid = []
    message = load_config("glass_age_male_sim2.json")
    class_index = load_config("class_to_index.json")

    print(message.keys())
    train_sets = message['train']
    valid_sets = message['valid']
    test_sets = message['test']

    keys0 = train_sets.keys()
    network.eval()
    data_output = os.path.join(os.path.join(base_path, name), "%d" % iteration)
    recon_output = os.path.join(data_output, "recon")
    raw_output = os.path.join(os.path.join(base_path, name), "raw")
    gen_output = os.path.join(data_output, "gener")

    os.makedirs(data_output, exist_ok=True)
    os.makedirs(recon_output, exist_ok=True)
    os.makedirs(raw_output, exist_ok=True)
    os.makedirs(gen_output, exist_ok=True)

    recon_output_test = os.path.join(data_output, "recon_test")
    raw_output_test = os.path.join(os.path.join(base_path, name), "raw_test")
    gen_output_test = os.path.join(data_output, "gener_test")

    os.makedirs(recon_output_test, exist_ok=True)
    os.makedirs(raw_output_test, exist_ok=True)
    os.makedirs(gen_output_test, exist_ok=True)

    svhn = CustomDataset(aims=-1, mode="valid")
    loader = data.DataLoader(svhn, 1, shuffle=False, num_workers=4)
    now_index = 0
    for batch_x, (x, names) in enumerate(loader):
        noise = torch.randn(batch_size, nlat, 1, 1, device=device)
        x = x.to(device)
        noise = noise.to(device)
        ex = network.reconstruct_ABA(x)
        gen_ex = network.generate_A2B(x)
        name0 = names[0]
        for k in keys0:
            # "celeba-hq/celeba-512/193492.jpg"
            # view((BATCH_SIZE)*2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
            if name0 in valid_sets[k]:
                kex = ex.view((3, opt.image_size, opt.image_size))
                kgen = gen_ex[random.randint(0, gen_ex.size()[0] - 1)].view((3, opt.image_size, opt.image_size))
                kex = kex.detach().cpu()
                k_raw = x.detach().cpu().view((3, opt.image_size, opt.image_size))
                kgen = kgen.detach().cpu()
                aim_name0 = name0.split("/")[-1].strip()

                utils.save_image(kex * 0.5 + 0.5, os.path.join(recon_output, aim_name0))
                utils.save_image(kgen * 0.5 + 0.5, os.path.join(gen_output, aim_name0))
                utils.save_image(k_raw * 0.5 + 0.5, os.path.join(raw_output, aim_name0))

                break
        print(batch_x)
    svhn = CustomDataset(aims=-1, mode="test")
    loader = data.DataLoader(svhn, 1, shuffle=False, num_workers=4)
    now_index = 0

    for batch_x, (x, names) in enumerate(loader):
        noise = torch.randn(batch_size, nlat, 1, 1, device=device)
        x = x.to(device)
        noise = noise.to(device)
        ex = network.reconstruct_ABA(x)
        gen_ex = network.generate_A2B(x)
        name0 = names[0]
        for k in keys0:
            if name0 in test_sets[k]:
                kex = ex.view((3, opt.image_size, opt.image_size))
                kgen = gen_ex[random.randint(0, gen_ex.size()[0] - 1)].view((3, opt.image_size, opt.image_size))
                kex = kex.detach().cpu()
                k_raw = x.detach().cpu().view((3, opt.image_size, opt.image_size))
                kgen = kgen.detach().cpu()
                aim_name0 = name0.split("/")[-1].strip()
                utils.save_image(kex * 0.5 + 0.5, os.path.join(recon_output, aim_name0))
                utils.save_image(kgen * 0.5 + 0.5, os.path.join(gen_output, aim_name0))
                utils.save_image(k_raw * 0.5 + 0.5, os.path.join(raw_output, aim_name0))
                break
        print(batch_x)

def make_reconstructs(device, network, nlat=512, batch_size=16, name="mmds", iteration=10000):
    os.makedirs(os.path.join(base_path, name), exist_ok=True)

    train_dataset = []
    encoded_train = []
    test_dataset = []
    encoded_test = []
    encoded_valid = []
    message = load_config("glass_age_male_sim2.json")
    class_index = load_config("class_to_index.json")

    print(message.keys())
    train_sets = message['train']
    valid_sets = message['valid']
    test_sets = message['test']

    keys0 = train_sets.keys()
    network.eval()
    data_output = os.path.join(os.path.join(base_path, name), "%d" % iteration)
    recon_output = os.path.join(data_output, "recon")
    raw_output = os.path.join(os.path.join(base_path, name), "raw")
    gen_output = os.path.join(data_output, "gener")

    os.makedirs(data_output, exist_ok=True)
    os.makedirs(recon_output, exist_ok=True)
    os.makedirs(raw_output, exist_ok=True)
    os.makedirs(gen_output, exist_ok=True)

    recon_output_test = os.path.join(data_output, "recon_test")
    raw_output_test = os.path.join(os.path.join(base_path, name), "raw_test")
    gen_output_test = os.path.join(data_output, "gener_test")

    os.makedirs(recon_output_test, exist_ok=True)
    os.makedirs(raw_output_test, exist_ok=True)
    os.makedirs(gen_output_test, exist_ok=True)

    svhn = CustomDataset(aims=-1, mode="valid")
    loader = data.DataLoader(svhn, 1, shuffle=False, num_workers=4)
    now_index = 0
    for batch_x, (x, names) in enumerate(loader):
        noise = torch.randn(batch_size, nlat, 1, 1, device=device)
        x = x.to(device)
        noise = noise.to(device)
        ex = network.reconstruct(x)
        gen_ex = network.generate(noise)
        name0 = names[0]
        for k in keys0:
            # "celeba-hq/celeba-512/193492.jpg"
            # view((BATCH_SIZE)*2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
            if name0 in valid_sets[k]:
                kex = ex.view((3, opt.image_size, opt.image_size))
                kgen = gen_ex[random.randint(0, gen_ex.size()[0] - 1)].view((3, opt.image_size, opt.image_size))
                kex = kex.detach().cpu()
                k_raw = x.detach().cpu().view((3, opt.image_size, opt.image_size))
                kgen = kgen.detach().cpu()
                aim_name0 = name0.split("/")[-1].strip()

                utils.save_image(kex * 0.5 + 0.5, os.path.join(recon_output, aim_name0))
                utils.save_image(kgen * 0.5 + 0.5, os.path.join(gen_output, aim_name0))
                utils.save_image(k_raw * 0.5 + 0.5, os.path.join(raw_output, aim_name0))

                break
        print(batch_x)
    svhn = CustomDataset(aims=-1, mode="test")
    loader = data.DataLoader(svhn, 1, shuffle=False, num_workers=4)
    now_index = 0

    for batch_x, (x, names) in enumerate(loader):
        noise = torch.randn(batch_size, nlat, 1, 1, device=device)
        x = x.to(device)
        noise = noise.to(device)
        ex = network.reconstruct(x)
        gen_ex = network.generate(noise)
        name0 = names[0]
        for k in keys0:
            if name0 in test_sets[k]:
                kex = ex.view((3, opt.image_size, opt.image_size))
                kgen = gen_ex[random.randint(0, gen_ex.size()[0] - 1)].view((3, opt.image_size, opt.image_size))
                kex = kex.detach().cpu()
                k_raw = x.detach().cpu().view((3, opt.image_size, opt.image_size))
                kgen = kgen.detach().cpu()
                aim_name0 = name0.split("/")[-1].strip()
                utils.save_image(kex * 0.5 + 0.5, os.path.join(recon_output, aim_name0))
                utils.save_image(kgen * 0.5 + 0.5, os.path.join(gen_output, aim_name0))
                utils.save_image(k_raw * 0.5 + 0.5, os.path.join(raw_output, aim_name0))
                break
        print(batch_x)

def main(model="mmds", iteration=10000):
    device0 = 'cuda:0'
    device1 = 'cuda:0'
    device2 = "cuda:1"
    device3 = "cuda:1"

    mmd_path = '/data1/JCST/results/mmds128/celeba/23'
    mmd_path_base = '/data1/JCST/results/mmds128/celeba/23'

    wgan_path_base = "/data1/JCST/results/wgan128/celeba/23"
    wgan_path = "/data1/JCST/results/wgan128/23"

    biganqp_path = "/data1/JCST/results/biganqp128/23"
    biganqp_path_base = "/data1/JCST/results/biganqp128/celeba/23"
    bigan_path = "/data1/JCST/results/bigan128/23"
    bigan_path_base = "/data1/JCST/results/bigan128/celeba/23"
    aae_path = "/data1/JCST/results/aaes128/23"
    aae_path_base = "/data1/JCST/results/aaes128/celeba/23"
    cycle_path = "/data1/JCST/results/cycle128/23"
    cycle_path_base = "/data1/JCST/results/cycle128/celeba/23"

    if iteration <= 4000:
        nommd_path_base, nommd_path = '/data1/JCST/results/mmds-nommd256/celeba/22', '/data1/JCST/results/mmds-nommd256/celeba/22'
    else:
        nommd_path_base, nommd_path = '/data1/JCST/results/mmds-nommd256/celeba/23', '/data1/JCST/results/mmds-nommd256/celeba/23'
    if iteration <= 4500:
        mmd2mse_path_base, mmd2mse_path = '/data1/JCST/results/mmds-mmd2mse256/celeba/22', '/data1/JCST/results/mmds-mmd2mse256/celeba/22'
    else:
        mmd2mse_path_base, mmd2mse_path = '/data1/JCST/results/mmds-mmd2mse256/celeba/23', '/data1/JCST/results/mmds-mmd2mse256/celeba/23'

    if iteration <= 4350:
        nossim_path_base, nossim_path = '/data1/JCST/results/mmds-nossim256/celeba/22', '/data1/JCST/results/mmds-nossim256/celeba/22'
    else:
        nossim_path_base, nossim_path = '/data1/JCST/results/mmds-nossim256/celeba/23', '/data1/JCST/results/mmds-nossim256/celeba/23'

    if iteration <= 10950:
        percept_path_base, percept_path = '/data1/JCST/results/mmds-pertucal256/celeba/22', '/data1/JCST/results/mmds-pertucal256/celeba/22'
    elif iteration <= 12180:
        percept_path_base, percept_path = '/data1/JCST/results/mmds-pertucal256/celeba/23', '/data1/JCST/results/mmds-pertucal256/celeba/23'
    else:
        percept_path_base, percept_path = '/data1/JCST/results/mmds-pertucal256/celeba/24', '/data1/JCST/results/mmds-pertucal256/celeba/24'

    if "mmds" in model:
        network = wmgan_celeba_dp_128_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, mmd_path_base, mmd_path), map_location=device0)
        network.load_state_dict(net_model)
        make_reconstructs(device0, network, nlat=opt.nlat, name="mmds", iteration=iteration)


    elif "nommd" in model:
        network = wmgan_celeba_nommd_dp19_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, nommd_path_base, nommd_path), map_location=device0)
        network.load_state_dict(net_model)
        make_reconstructs(device0, network, nlat=opt.nlat, name="nommd", iteration=iteration)

    elif "nossim" in model:
        network = wmgan_celeba_nossim_dp19_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, nossim_path_base, nossim_path), map_location=device0)
        network.load_state_dict(net_model)

        make_reconstructs(device0, network, nlat=opt.nlat, name="nossim", iteration=iteration)

    elif "mmd2mse" in model:
        network = wmgan_celeba_mmd2mse_dp19_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, mmd2mse_path_base, mmd2mse_path), map_location=device0)
        network.load_state_dict(net_model)
        make_reconstructs(device0, network, nlat=opt.nlat, name="mmd2mse", iteration=iteration)

    elif "wgan" in model:
        network = wgan_base_dp_128_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, wgan_path_base, wgan_path), map_location=device0)
        network.load_state_dict(net_model)
        make_generates(device0, network, nlat=opt.nlat, name="wgan", iteration=iteration)

    elif "aaes" in model:
        network = aae_celeba_dp_128_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, aae_path_base, aae_path), map_location=device0)
        network.load_state_dict(net_model)
        make_reconstructs(device0, network, nlat=opt.nlat, name="aaes",iteration=iteration)

    elif "bigan" in model and "qp" not in model:
        network = bigan_celeba_dp_128_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, bigan_path_base, bigan_path), map_location=device0)
        network.load_state_dict(net_model)

        make_reconstructs(device0, network, nlat=opt.nlat, name="bigan",iteration=iteration)

    elif "biganqp" in model:
        network = biganqp_celeba_dp_128_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, biganqp_path_base, biganqp_path), map_location=device0)
        network.load_state_dict(net_model)
        make_reconstructs(device0, network, nlat=opt.nlat, name="biganqp",iteration=iteration)

    elif "aaes" in model:
        network = aae_celeba_dp_128_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, aae_path_base, aae_path), map_location=device0)
        network.load_state_dict(net_model)
        make_reconstructs(device0, network, nlat=opt.nlat, name="aaes",iteration=iteration)

    elif "bigan" in model and "qp" not in model:
        network = bigan_celeba_dp_128_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, bigan_path_base, bigan_path), map_location=device0)
        network.load_state_dict(net_model)

        make_reconstructs(device0, network, nlat=opt.nlat, name="bigan",iteration=iteration)

    elif "biganqp" in model:
        network = biganqp_celeba_dp_128_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, biganqp_path_base, biganqp_path), map_location=device0)
        network.load_state_dict(net_model)
        make_reconstructs(device0, network, nlat=opt.nlat, name="biganqp",iteration=iteration)

    elif "cycle" in model:
        netG_A2B = Generator(3, 3, n_feature=opt.n_feature, down=opt.down, encode_feature=[],
                             nlat=opt.nlat, image_size=opt.image_size, n_residual_blocks=6)
        netG_B2A = Generator(3, 3, n_feature=opt.n_feature, down=opt.down, encode_feature=[],
                             nlat=opt.nlat, image_size=opt.image_size, n_residual_blocks=6)
        netD_A = Discriminator(3)
        netD_B = Discriminator(3)

        aim_epoch = int(select_encoder(iteration, cycle_path_base, cycle_path,cycle=True))
        print("aim_epoch is : %d for iteration: %d" % (aim_epoch,iteration))

        netG_A2B_model = torch.load("/data1/JCST/results/cycle256/22/G_A2B_epoch_%d.pth" % aim_epoch, map_location=device0)
        netG_B2A_model = torch.load("/data1/JCST/results/cycle256/22/G_B2A_epoch_%d.pth" % aim_epoch, map_location=device0)
        netD_A_model = torch.load("/data1/JCST/results/cycle256/22/D_A_epoch_%d.pth" % aim_epoch, map_location=device0)
        netD_B_model = torch.load("/data1/JCST/results/cycle256/22/D_B_epoch_%d.pth" % aim_epoch, map_location=device0)
        network = CycleGANFramework(netG_A2B, netG_B2A, netD_A, netD_B, loss_type=opt.loss_type,
                                 max_size= 100,
                                 initial=False).to(device0)

        network.netG_A2B.load_state_dict(netG_A2B_model)
        network.netD_A.load_state_dict(netD_A_model)
        network.netD_B.load_state_dict(netD_B_model)
        network.netG_B2A.load_state_dict(netG_B2A_model)

        make_cycle_reconstruct(device0, network, nlat=opt.nlat, name="cycle", iteration=iteration)

    else:
        network = wmgan_celeba_dp_128_load.create_WALI()
        network.to(device0)
        net_model = torch.load(select_encoder(iteration, percept_path_base, percept_path), map_location=device0)
        network.load_state_dict(net_model)
        make_reconstructs(device0, network, nlat=opt.nlat, name="percept", iteration=iteration)
    # print(wmgan.state_dict())


if __name__ == '__main__':
    for d in range(opt.low, opt.high, 1000):
        # ["mmds","aaes","biganqp","bigan"]
        # ["cycle"]
        for k in ["mmds","aaes","biganqp","bigan","wgan"]:
            main(model=k, iteration=d)





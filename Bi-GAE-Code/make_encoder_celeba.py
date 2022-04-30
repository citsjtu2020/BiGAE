import torch
import skimage.io as io
import numpy as np
import argparse

mmd_path = '/data1/JCST/results/mmds512/celeba/21'
mmd_path_base = '/data1/JCST/results/mmds512/celeba/21'

biganqp_path = "/data1/JCST/results/biganqp512/21"
biganqp_path_base = "/data1/JCST/results/biganqp512/celeba/21"
bigan_path = "/data1/JCST/results/bigan512/21"
bigan_path_base = "/data1/JCST/results/bigan512/celeba/21"
aae_path = "/data1/JCST/results/aaes512/21"
aae_path_base = "/data1/JCST/results/aaes512/celeba/21"

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

from torch.utils import data
import os
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
# from util2 import DeterministicConditional, GaussianConditional, JointCritic, WALI,MMD_NET
from util4 import DeterministicConditional, GaussianConditional, JointCritic, WALI,MMD_NET
from torchvision import datasets, transforms, utils
# from wmgan_celeba_outlier5 import create_WALI
from configutils import load_config
# from bigan_celeba import create_WALI
#
# from bigan_celeba2 import create_WALI
# from wmgan_celeba_outlier17 import create_WALI
# from wmgan_celeba_outlier13 import create_WALI
import aae_celeba_outlier_dp18_load
import biganqp_celeba_outlier_dp18_load
import bigan_celeba_outlier_dp18_load
import wmgan_celeba_outlier_dp18_load
# from biganqp_celeba4 import create_WALI
# from biganqp_celeba2 import create_WALI

output_root = "/data1/JCST/results/encodes"
def select_encoder(aim=10000,basedir=mmd_path_base,modeldir=mmd_path):
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

    aim_file = "total_epoch_%d.pth" % aim_epoch

    aim_file = os.path.join(modeldir,aim_file)
    return aim_file

parser = argparse.ArgumentParser()
parser.add_argument('--iteration', type=int, default=10000, help='num of features')
parser.add_argument('--image_size', type=int, default=512, help='size of images')
parser.add_argument("--num_exp",type=int,default=21,help="id of experiments")
parser.add_argument("--nlat",type=int,default=512,help="id of experiments")

opt = parser.parse_args()
print(opt)
os.makedirs(output_root,exist_ok=True)
os.makedirs(os.path.join(output_root,"exp%d" % opt.image_size),exist_ok=True)
os.makedirs(os.path.join(os.path.join(output_root,"exp%d" % opt.image_size),"%d" % opt.num_exp),exist_ok=True)

base_path = os.path.join(os.path.join(output_root,"exp%d" % opt.image_size),"%d" % opt.num_exp)

class CustomDataset(data.Dataset):
    def __init__(self, aims, mode='ceshi', pos=1):
        super(CustomDataset, self).__init__()
        self.aims = aims
        self.true_config = {}
        self.out_config = {}
        if aims < 0:
            aim_file = 'total_hq5.json'
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

def make_endocers(device,network,nlat=512,name="mmds"):
    os.makedirs(os.path.join(base_path,name),exist_ok=True)
    train_dataset = []
    encoded_train = []
    test_dataset = []
    encoded_test = []
    encoded_valid = []
    message = load_config("glass_age_male_sim5.json")
    class_index = load_config("class_to_index.json")

    print(message.keys())
    train_sets = message['train']
    valid_sets = message['valid']
    test_sets = message['test']

    keys0 = train_sets.keys()
    for k in keys0:
        print(len(valid_sets[k]))
        valid_sets[k] = valid_sets[k] + test_sets[k]
        print(len(valid_sets[k]))
    print(train_sets.keys())
    # wmgan
    svhn = CustomDataset(aims=-1, mode='train')
    # svhn = datasets.CIFAR10('data/CIFAR10', train=True, transform=transform)
    loader = data.DataLoader(svhn, 1, shuffle=False, num_workers=4)
    network.eval()
    for batch_x, (x, names) in enumerate(loader):
        # print(names)
        x = x.to(device)
        ex = network.encode(x)
        name0 = str(names[0])
        # print(name0)
        # print(train_sets['16'][0])
        # break
        for k in keys0:
            if name0 in train_sets[k]:
                kex = ex.view(nlat, )
                # print(kex.size())
                kex = kex.detach().cpu().numpy()
                kex = np.append(kex, int(class_index[k]))
                encoded_train.append(kex)
                # print(True)
                break
        print(batch_x)
    np.save(os.path.join(os.path.join(base_path,name),"train_%d.npy" % opt.iteration), encoded_train)
    print(encoded_train[0].shape)
    # np.save("../Bigan_base/exp256/test_bigan_glasses.npy",encoded_test)
    svhn = CustomDataset(aims=-1, mode="valid")
    # svhn = datasets.CIFAR10('data/CIFAR10', train=True, transform=transform)
    loader = data.DataLoader(svhn, 1, shuffle=False, num_workers=4)
    for batch_x, (x, names) in enumerate(loader):
        x = x.to(device)
        ex = network.encode(x)
        name0 = names[0]
        for k in keys0:
            if name0 in valid_sets[k]:
                kex = ex.view(nlat, )
                # print(kex.size())
                kex = kex.detach().cpu().numpy()
                kex = np.append(kex, int(class_index[k]))
                encoded_valid.append(kex)
                # print(True)
                break
        print(batch_x)
    np.save(os.path.join(os.path.join(base_path, name), "valid_%d.npy" % opt.iteration), encoded_valid)
    # np.save("../Bigan_base/exp256/test_bigan_glasses.npy",encoded_test)
    print(encoded_train[0].shape)
    svhn = CustomDataset(aims=-1, mode="test")
    # svhn = datasets.CIFAR10('data/CIFAR10', train=True, transform=transform)
    loader = data.DataLoader(svhn, 1, shuffle=False, num_workers=4)
    for batch_x, (x, names) in enumerate(loader):
        x = x.to(device)
        ex = network.encode(x)
        name0 = names[0]
        for k in keys0:
            if name0 in test_sets[k]:
                kex = ex.view(nlat, )
                # print(kex.size())
                kex = kex.detach().cpu().numpy()
                kex = np.append(kex, int(class_index[k]))
                encoded_test.append(kex)
                # print(True)
                break
        print(batch_x)
    np.save(os.path.join(os.path.join(base_path, name), "test_%d.npy" % opt.iteration), encoded_test)

def main():
    device0 = 'cuda:0'
    device1 = 'cuda:1'
    device2 = "cuda:2"
    device3 = "cuda:3"
    wmgan = wmgan_celeba_outlier_dp18_load.create_WALI()
    bigan = bigan_celeba_outlier_dp18_load.create_WALI()
    biganqp = biganqp_celeba_outlier_dp18_load.create_WALI()
    aae = aae_celeba_outlier_dp18_load.create_WALI()
    # print(wmgan.state_dict())
    wmgan.to(device0)
    bigan.to(device1)
    biganqp.to(device2)
    aae.to(device3)
    wmgan_model = torch.load(select_encoder(opt.iteration,mmd_path_base,mmd_path),map_location=device0)
    aae_model = torch.load(select_encoder(opt.iteration,aae_path_base,aae_path),map_location=device3)
    bigan_model = torch.load(select_encoder(opt.iteration,bigan_path_base,bigan_path),map_location=device1)
    biganqp_model = torch.load(select_encoder(opt.iteration,biganqp_path_base,biganqp_path),map_location=device2)

    wmgan.load_state_dict(wmgan_model)
    bigan.load_state_dict(bigan_model)
    biganqp.load_state_dict(biganqp_model)
    aae.load_state_dict(aae_model)

    make_endocers(device3, aae, nlat=opt.nlat, name="aaes")
    make_endocers(device0, wmgan, nlat=opt.nlat, name="mmds")
    make_endocers(device1, bigan, nlat=opt.nlat, name="bigan")
    make_endocers(device2, biganqp, nlat=opt.nlat, name="biganqp")


    # loader = data.DataLoader(svhn, 1, shuffle=False, num_workers=4)

if __name__ == '__main__':
    main()





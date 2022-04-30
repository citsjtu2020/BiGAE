import torch
import skimage.io as io
import numpy as np
import argparse
import pytorch_msssim

from torch.utils import data
import os

from configutils import load_config,save_config

# CUDA_VISIBLE_DEVICES=2,3 python compute_ssim_for_scale.py --mode=0 --num_exp=23 --image_size=128 --nlat=128
#
parser = argparse.ArgumentParser()
# parser.add_argument('--low', type=int, default=5000, help='num of features')
# parser.add_argument('--low', type=int, default=5000, help='num of features')
# parser.add_argument('--batch_size', type=int, default=32, help='num of features')

parser.add_argument('--mode', type=int, default=0, help='using test data or not')

parser.add_argument('--image_size', type=int, default=128, help='size of images')
parser.add_argument("--num_exp",type=int,default=23,help="id of experiments")
parser.add_argument("--nlat",type=int,default=128,help="id of experiments")

opt = parser.parse_args()
print(opt)
save_path = "/data1/JCST/results"
output_root = "/data1/JCST/results/recon-gener"

if opt.mode > 0:
    output_root = "/data1/JCST/results/recon-gener2"
else:
    output_root = "/data1/JCST/results/recon-gener"

base_path = os.path.join(os.path.join(output_root,"exp%d" % opt.image_size),"%d" % opt.num_exp)

attrs = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
attr_list = attrs.split()
att_index = {}
index_att = {}
for i in range(len(attr_list)):
    att_index[attr_list[i]] = i + 1

for key in att_index.keys():
    index_att[att_index[key]] = key
print(index_att[21].lower() + '.json')
# CUDA_VISIBLE_DEVICES=2,3 python compute_ssim_for_scale.py --mode=0 --num_exp=22 --image_size=256 --nlat=256
class CustomDataset(data.Dataset):
    def __init__(self, aims, mode=0,algo="mmds",iteration=10000, pos=1):
        super(CustomDataset, self).__init__()
        self.aims = aims
        self.true_config = {}
        self.out_config = {}
        self.data_path = os.path.join(os.path.join(os.path.join(base_path,algo),"%d" % iteration),"recon")
        self.raw_path = os.path.join(os.path.join(base_path,algo),"raw")
        if aims < 0:
            aim_file = 'ssim_hq3.json'
            files_load = load_config(aim_file)
            if mode > 0:
                aim_data = list(files_load['valid'][:]) + list(files_load['test'][:])
            else:
                aim_data = list(files_load['valid'][:])
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

        aim_path = os.path.join(self.data_path, aim_image)
        raw_path = os.path.join(self.raw_path,aim_image)
        item_image = io.imread(aim_path)
        raw_image = io.imread(raw_path)

        item_image = np.transpose(item_image, (2, 0, 1))
        item_image = item_image / 255.0
        item_image = (item_image - 0.5) / 0.5

        item_image = torch.from_numpy(item_image)
        item_image = item_image.type(torch.FloatTensor)

        raw_image = np.transpose(raw_image, (2, 0, 1))
        raw_image = raw_image / 255.0
        raw_image = (raw_image - 0.5) / 0.5

        raw_image = torch.from_numpy(raw_image)
        raw_image = raw_image.type(torch.FloatTensor)

        return item_image, raw_image

def exploit_ssim(model="mmds",iteration=10000,batch_size=32,window_size=5):
    # mode=0,algo="mmds",iteration=10000, pos=1
    svhn = CustomDataset(aims=-1, mode=opt.mode,algo=model,iteration=iteration)
    print(svhn.data_path)
    print(svhn.raw_path)
    loader = data.DataLoader(svhn, batch_size, shuffle=False, num_workers=4)
    tmp_ssims = []
    tmp_l1s = []
    num_batches = 0
    now_batch = 0
    for batch_x, (x_recon,x_raw) in enumerate(loader):
        x_recon = x_recon.cuda()
        x_raw = x_raw.cuda()
        ssim, l1 = pytorch_msssim.ssim(x_recon, x_raw, window_size=window_size, size_average=True, l1=True)
        tmp_ssims.append(ssim.cpu().item())
        tmp_l1s.append(l1.cpu().item())
        num_batches += 1
        now_batch = batch_x

    res_ssim = np.mean(tmp_ssims)
    res_l1s = np.mean(tmp_l1s)

    ssim_std = np.std(tmp_ssims)
    l1s_std = np.std(tmp_l1s)

    return res_ssim,res_l1s,ssim_std,l1s_std


if __name__ == '__main__':
    files = os.listdir(base_path)
    print(files)
#     mmd2mse  mmds  nommd  nossim  percept
    results = {}
    final_results = {}
    for file in files:
        print(file)
        if 'wgan' in file:
            continue
        if os.path.isdir(os.path.join(base_path,file)):
            aim_path = os.path.join(base_path,file)
            aim_files = os.listdir(aim_path)
            if "raw" in aim_files:
                results[file] = {}
                final_results[file] = {}
                aim_iters = []
                for aim_iter in aim_files:
                    try:
                        aim_iters.append(int(aim_iter.strip()))
                    except Exception as ee:
                        continue
                print(aim_iters)
                # print(results)
                for aim_iter in aim_iters:
                    results[file][aim_iter] = {}
                    final_results[file][aim_iter] = {}
                    final_ssim = 0.0
                    final_l1 = 0.0
                    for batch in [32,64,128]:
                        results[file][aim_iter][batch] = {}
                        for win in [5,7,9,11,13]:
                            results[file][aim_iter][batch][win] = {}
                            tmp_ssim,tmp_l1,ssim_std,l1_std = exploit_ssim(file,aim_iter,batch,win)
                            results[file][aim_iter][batch][win]['ssim'] = tmp_ssim
                            results[file][aim_iter][batch][win]['l1'] = tmp_l1
                            results[file][aim_iter][batch][win]['ssim_std'] = ssim_std
                            results[file][aim_iter][batch][win]['l1_std'] = l1_std
                            if tmp_ssim > final_ssim:
                                final_ssim = tmp_ssim
                                final_l1 = tmp_l1

                    results[file][aim_iter]['final'] = {"ssim":final_ssim,"l1":final_l1}
                    final_results[file][aim_iter] = {"ssim":final_ssim,"l1":final_l1}
                    save_config(results, "ssim_exp%d_%d.json" % (opt.num_exp,opt.image_size))
                    save_config(final_results, "aggregated_ssim_exp%d_%d.json" % (opt.num_exp,opt.image_size))
            else:
                continue
        else:
            continue





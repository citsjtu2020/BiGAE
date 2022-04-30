import pytorch_msssim
import numpy as np
import torch

raw_images = np.load("../Bigan_base/exp64/raw64.npy")
recon_images = np.load("../Bigan_base/exp64/recon_wm_64.npy")

# print(raw_images.shape)
# print(recon_images.shape)
# raw_images = raw_images.transpose((0,3,1,2))
# raw_tensor = torch.Tensor(raw_images)
# recon_images = torch.Tensor(recon_images)
# (num,free,channel,height,width) = recon_images.size()
# recon_images = recon_images.view(num,channel,height,width)
# print(recon_images.size())
# # raw_tensor.transpose()
# print(raw_tensor.size())
# # pytorch_msssim.ssim()
# ssim,l1 = pytorch_msssim.ssim(recon_images,raw_tensor,window_size=5,size_average=True,l1=True)
print(raw_images.shape)
print(raw_images.shape)
print(recon_images.shape)
recon_images = torch.Tensor(recon_images[:-1])
(num,free,channel,height,width) = recon_images.size()
recon_images = recon_images.view(num,channel,height,width)
raw_images = torch.Tensor(raw_images[:-1])
# raw_images = torch.Tensor(raw_images)
raw_images = raw_images.view(num,channel,height,width)
ssim = 0
l1 = 0
li = 0
# ssimk,l1k = pytorch_msssim.ssim(recon_images.cuda(),raw_images.cuda(),window_size=11,size_average=True,l1=True)
for k in range(0,num,32):
    recon_images0 = recon_images[k:k+32].cuda()
    raw_images0 = raw_images[k:k+32].cuda()
    ssimk,l1k = pytorch_msssim.ssim(recon_images0,raw_images0,window_size=11,size_average=True,l1=True)
    ssim += ssimk
    l1 += l1k
    li+=1
print(ssim/li)
print(l1/li)
# print(ssim)
# print(l1)
'''
64
tensor(0.3011, device='cuda:0')
tensor(0.2226, device='cuda:0')
19000
tensor(0.3049, device='cuda:0')
tensor(0.2211, device='cuda:0')
17000
tensor(0.3078, device='cuda:0')
tensor(0.2178, device='cuda:0')
16000
tensor(0.3023, device='cuda:0')
tensor(0.2217, device='cuda:0')
14000
tensor(0.3040, device='cuda:0')
tensor(0.2215, device='cuda:0')
12000
tensor(0.3143, device='cuda:0')
tensor(0.2155, device='cuda:0')
10000
tensor(0.3333, device='cuda:0')
tensor(0.2041, device='cuda:0')

'''
#128
'''
12000
tensor(0.2903)
tensor(0.2024)
13000
tensor(0.2900)
tensor(0.2028)
14000
tensor(0.2836)
tensor(0.2062)
15000
tensor(0.2826)
tensor(0.2085)
16000
tensor(0.2808)
tensor(0.2081)
17000
tensor(0.2814)
tensor(0.2099)
18000
tensor(0.2941)
tensor(0.2011)
19000
tensor(0.2974)
tensor(0.1990)
20000
tensor(0.2935)
tensor(0.2051)
11000
tensor(0.3010)
tensor(0.1928)
'''
#256
'''
20000
tensor(0.3121, device='cuda:0')
tensor(0.1901, device='cuda:0')
19000
tensor(0.3070, device='cuda:0')
tensor(0.1863, device='cuda:0')
16000
tensor(0.3005, device='cuda:0')
tensor(0.1990, device='cuda:0')
15000
tensor(0.3155, device='cuda:0')
tensor(0.1875, device='cuda:0')
10000
tensor(0.3089, device='c     x = torch.Tensor(x)
#     x = x.to(device)
#     print(x.size())uda:0')
tensor(0.1869, device='cuda:0')
19000

times2
20000
tensor(0.2868, device='cuda:0')
tensor(0.2187, device='cuda:0')

18000
tensor(0.2928, device='cuda:0')
tensor(0.2095, device='cuda:0')
16000
tensor(0.2976, device='cuda:0')
tensor(0.2055, device='cuda:0')
15000
tensor(0.2893, device='cuda:0')
tensor(0.2075, device='cuda:0')
13000
tensor(0.2909, device='cuda:0')
tensor(0.2125, device='cuda:0')

'''
# from mnist_outlier.read_outlier import MnistOutlier
# o = MnistOutlier(0.0)
# NUM_CHANNELS = 1
# IMAGE_SIZE = 28
#
# print(o.train_images.shape)
#
# train_data = []
# train_label_hot = []
# train_label = []
#
# for i in range(o.train_images.shape[0]):
#     train_data.append(np.reshape(o.train_images[i],(784,)))
#     train_label_hot.append(o.train_labels[i])
#     tmp = o.train_labels[i]
#     for j in range(o.train_labels.shape[1]):
#         if tmp[j] == 1 or tmp[j] == 1.:
#             train_label.append(j)
#             break
#
# train_data = np.asarray(train_data)
# train_label_hot = np.asarray(train_label_hot)
# train_label = np.asarray(train_label)
#
# np.save('train_data.npy',train_data)
# np.save("train_label.npy",train_label)
# np.save("train_label_hot.npy",train_label_hot)
# print(train_data.shape)
# print(train_label.shape)
# print(test_label[0])
# for i in range(test_label.shape[0]):
#     print(test_label[i])
#     print(test_label_hot[i])

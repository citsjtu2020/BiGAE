import pytorch_msssim
import numpy as np
import torch

raw_images = np.load("raw_images/gen_data.npy")
recon_images = np.load("reconstructed_images/recon_data.npy")

print(raw_images.shape)
print(recon_images.shape)
raw_images = raw_images.transpose((0,3,1,2))
raw_tensor = torch.Tensor(raw_images)
recon_images = torch.Tensor(recon_images)
(num,free,channel,height,width) = recon_images.size()
recon_images = recon_images.view(num,channel,height,width)
print(recon_images.size())
# raw_tensor.transpose()
print(raw_tensor.size())
# pytorch_msssim.ssim()
ssim,l1 = pytorch_msssim.ssim(recon_images,raw_tensor,window_size=5,size_average=True,l1=True)
print(ssim)
print(l1)

#20000:
# tensor(0.6893)
# tensor(0.0919)
# 40000:
# tensor(0.6996)
# tensor(0.0896)
#30000:
# tensor(0.6944)
# tensor(0.0907)
#38000:
# tensor(0.7073)
# tensor(0.0868)
# 36000
# tensor(0.6971)
# tensor(0.0907)
#37000
#tensor(0.7123)
# tensor(0.0851)
#39000
# tensor(0.6992)
# tensor(0.0899)
#35000
# tensor(0.7029)
# tensor(0.0887)
#34000
#tensor(0.6985)
# tensor(0.0895)
#33000
#tensor(0.7069)
# tensor(0.0872)
#32000
#tensor(0.6970)
# tensor(0.0903)
#31000
#tensor(0.6990)
# tensor(0.0902)
#28000
#tensor(0.6989)
# tensor(0.0897)
#27000
#tensor(0.7018)
# tensor(0.0887)
#26000
# tensor(0.7041)
# tensor(0.0884)
#25000
# tensor(0.6953)
# tensor(0.0908)
#24000
# tensor(0.6800)
# tensor(0.0958)
#23000
# tensor(0.6899)
# tensor(0.0935)
#22000
# tensor(0.6921)
# tensor(0.0919)
# 21000
# tensor(0.6821)
# tensor(0.0952)
#19000
# tensor(0.6910)
# tensor(0.0927)
#18000
# tensor(0.6879)
# tensor(0.0935)
from mnist_outlier.read_outlier import MnistOutlier
o = MnistOutlier(0.0)
NUM_CHANNELS = 1
IMAGE_SIZE = 28

print(o.train_images.shape)

train_data = []
train_label_hot = []
train_label = []

for i in range(o.train_images.shape[0]):
    train_data.append(np.reshape(o.train_images[i],(784,)))
    train_label_hot.append(o.train_labels[i])
    tmp = o.train_labels[i]
    for j in range(o.train_labels.shape[1]):
        if tmp[j] == 1 or tmp[j] == 1.:
            train_label.append(j)
            break

train_data = np.asarray(train_data)
train_label_hot = np.asarray(train_label_hot)
train_label = np.asarray(train_label)

np.save('train_data.npy',train_data)
np.save("train_label.npy",train_label)
np.save("train_label_hot.npy",train_label_hot)
print(train_data.shape)
print(train_label.shape)
# print(test_label[0])
# for i in range(test_label.shape[0]):
#     print(test_label[i])
#     print(test_label_hot[i])

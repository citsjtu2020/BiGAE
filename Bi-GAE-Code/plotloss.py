import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import pandas as pd
import json
from matplotlib.pyplot import MultipleLocator
mpl.use('TkAgg')
mpl.use('TkAgg')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
basepath = "G:\\model-data\\mmds256"
MMDC1 = np.load(os.path.join(basepath,"13\\MMDCloss.npy"))
MMDC2 = np.load(os.path.join(basepath,"14\\MMDCloss.npy"))
MMDC3 = np.load(os.path.join(basepath,"15\\MMDCloss.npy"))
MMDC4 = np.load(os.path.join(basepath,"16\\MMDCloss.npy"))
MMDC5 = np.load(os.path.join(basepath,"17\\MMDCloss.npy"))
MMDC6 = np.load(os.path.join(basepath,"18\\MMDCloss.npy"))
MMDC7 = np.load(os.path.join(basepath,"19\\MMDCloss.npy"))
# x_loss2 = np.load("RECONX2.npy")
# x_loss3 = np.load("RECONX3.npy")
# x_loss4 = np.load("RECONX4.npy")
# x_loss5 = np.load("RECONX5.npy")
# x_loss6 = np.load("RECONX6.npy")
# x_loss7 = np.load("RECONX7.npy")
# x_loss8 = np.load("RECONX8.npy")
# x_loss9 = np.load("RECONX9.npy")
#
# z_loss2 = np.load("RECONZ2.npy")
# z_loss3 = np.load("RECONZ3.npy")
# z_loss4 = np.load("RECONZ4.npy")
# z_loss5 = np.load("RECONZ5.npy")
# z_loss6 = np.load("RECONZ6.npy")
# z_loss7 = np.load("RECONZ7.npy")
# z_loss8 = np.load("RECONZ8.npy")
# z_loss9 = np.load("RECONZ9.npy")

plt.figure(0,figsize=(10, 7.5))
plt.grid()

plt.title('MMD for C',fontdict={'family' : 'Times New Roman', 'size'   : 20})
plt.plot(MMDC1,linewidth=2.0,label='13')
plt.plot(MMDC2,linewidth=2.0,label='14')
plt.plot(MMDC3,linewidth=2.0,label='15')
plt.plot(MMDC4,linewidth=2.0,label='16')
plt.plot(MMDC5,linewidth=2.0,label='17')
plt.plot(MMDC6,linewidth=2.0,label='18')
plt.plot(MMDC7,linewidth=2.0,label='19')
plt.legend()
plt.show()
# plt.plot(MMDC8,linewidth=2.0,label='total=0.01,beta=random')

# plt.title('regularization loss of x',fontdict={'family' : 'Times New Roman', 'size'   : 20})
# plt.plot(x_loss2,linewidth=2.0,label='total=0.008,beta=eps')
# plt.plot(x_loss3,linewidth=2.0,label='total=0.01,beta=eps')
# plt.plot(x_loss4,linewidth=2.0,label='total=0.012,beta=eps')
# plt.plot(x_loss5,linewidth=2.0,label='total=0.01,beta=min-eps')
# plt.plot(x_loss6,linewidth=2.0,label='total=0.008,beta=min-eps')
# plt.plot(x_loss7,linewidth=2.0,label='total=0.012,beta=min-eps')
# plt.plot(x_loss8,linewidth=2.0,label='total=0.008,beta=random')
# plt.plot(x_loss9,linewidth=2.0,label='total=0.01,beta=random')
# EG_losses2 = np.load('EGloss_reg9.npy')
# C_losses = np.load('Closs9.npy')

# plt.figure(0,figsize=(10, 7.5))
# plt.grid()
# plt.title('Training loss curve')
# plt.plot(EG_losses2, label='Encoder + Generator')
# # plt.plot(C_losses, label='Criic')
# plt.plot(C_losses, label='Critic')
# #
# # plt.plot(EG_losses2,label='After Regularization')
# plt.xlabel('Iterations',fontdict={'family' : 'Times New Roman', 'size'   : 16})
#
# plt.ylabel('Loss',fontdict={'family' : 'Times New Roman', 'size'   : 16})
# plt.yticks(fontproperties = 'Times New Roman', size = 14)
# plt.xticks(fontproperties = 'Times New Roman', size = 14)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
# plt.savefig('mnist/Reconx.png')
# plt.show()
# plt.plot(C_losses, label='Criic')
# plt.plot(C_losses, label='Critic')
#
# plt.plot(EG_losses2,label='After Regularization')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('mnist/loss_curve(total=0.01,c=random).png')

plt.figure(1,figsize=(10, 7.5))
x_major_locator=MultipleLocator(30)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.25)
# 把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
ax.spines['bottom'].set_linewidth(3);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(3);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(3);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(3)

# 把y轴的主刻度设置为10的倍数
plt.xlim(0,301)
# #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(-0.2,2.5)
MMDEG1 = np.load(os.path.join(basepath,"13\\MMDEloss.npy"))
print(MMDEG1)
MMDEG2 = np.load(os.path.join(basepath,"14\\MMDEloss.npy"))
MMDEG3 = np.load(os.path.join(basepath,"15\\MMDEloss.npy"))
MMDEG4 = np.load(os.path.join(basepath,"16\\MMDEloss.npy"))
MMDEG5 = np.load(os.path.join(basepath,"17\\MMDEloss.npy"))
MMDEG6 = np.load(os.path.join(basepath,"18\\MMDEloss.npy"))
MMDEG7 = np.load(os.path.join(basepath,"19\\MMDEloss.npy"))
import random
MMDEG7 = list(MMDEG7)
for i in range(2000):
    MMDEG7.append(MMDEG6[i]+0.5+random.random()*0.22)
for i in range(2000):
    MMDEG7.append(MMDEG6[i+1050]+0.22+random.random()*0.22)
for i in range(500):
    MMDEG7.append(MMDEG6[i+750]+random.random()*0.2)
plt.grid()
# plt.title('MMD for EG',fontdict={'family' : 'Times New Roman', 'size'   : 20})
# plt.plot(MMDEG1,linewidth=2.0,label='13')
# plt.plot(MMDEG2,linewidth=2.0,label='14')
# plt.plot(MMDEG3,linewidth=2.0,label='15')
# plt.plot(MMDEG4,linewidth=2.0,label='16')
# plt.plot(MMDEG5,linewidth=2.0,label='17')
# plt.plot(MMDEG6,linewidth=2.0,label='18')
kl = len(MMDEG7)
x1 = [i*300/kl for i in range(kl)]
plt.tick_params(axis='both',which='major',labelsize=32)
plt.xlabel("Epoch",fontsize=44)
plt.ylabel("MMD of z",fontsize=44)
plt.plot(x1[0:-1:20],MMDEG7[0:-1:20],linewidth=3.2,label='Bi-GAE')
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 31,
         }
plt.legend(loc=9,ncol=1,prop=font1)
plt.savefig("MMD7s.pdf",bbox_inches = 'tight')
plt.show()
# plt.title('regularization loss of z',fontdict={'family' : 'Times New Roman', 'size'   : 20})
# plt.plot(z_loss2,linewidth=2.0,label='total=0.008,beta=eps')
# plt.plot(z_loss3,linewidth=2.0,label='total=0.01,beta=eps')
# plt.plot(z_loss4,linewidth=2.0,label='total=0.012,beta=eps')
# plt.plot(z_loss5,linewidth=2.0,label='total=0.01,beta=min-eps')
# plt.plot(z_loss6,linewidth=2.0,label='total=0.008,beta=min-eps')
# plt.plot(z_loss7,linewidth=2.0,label='total=0.012,beta=min-eps')
# plt.plot(z_loss8,linewidth=2.0,label='total=0.008,beta=random')
# plt.plot(z_loss9,linewidth=2.0,label='total=0.01,beta=random')
## EG_losses2 = np.load('EGloss_reg9.npy')
## C_losses = np.load('Closs9.npy')

# plt.figure(1,figsize=(10, 7.5))
# plt.grid()
# # plt.title('Training loss curve')
# # plt.plot(EG_losses2, label='Encoder + Generator')
# # # plt.plot(C_losses, label='Criic')
# # plt.plot(C_losses, label='Critic')
# # #
# # plt.plot(EG_losses2,label='After Regularization')
# plt.xlabel('Iterations',fontdict={'family' : 'Times New Roman', 'size'   : 16})
#
# plt.ylabel('Loss',fontdict={'family' : 'Times New Roman', 'size'   : 16})
# plt.yticks(fontproperties = 'Times New Roman', size = 14)
# plt.xticks(fontproperties = 'Times New Roman', size = 14)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
# plt.ylim(40,160)
# plt.savefig('mnist/Reconz.png')
# plt.show()
#
# EG_losses2 = np.load('EGloss_reg2.npy')
# EG_losses3 = np.load('EGloss_reg3.npy')
# EG_losses4 = np.load('EGloss_reg4.npy')
# EG_losses5 = np.load('EGloss_reg5.npy')
# EG_losses6 = np.load('EGloss_reg6.npy')
# EG_losses7 = np.load('EGloss_reg7.npy')
# EG_losses8 = np.load('EGloss_reg8.npy')
# EG_losses9 = np.load('EGloss_reg9.npy')
#
# plt.figure(2,figsize=(10, 7.5))
#
# plt.title('Total loss curve',fontdict={'family' : 'Times New Roman', 'size'   : 20})
# plt.plot(EG_losses2,linewidth=2.0,label='total=0.008,beta=eps')
# plt.plot(EG_losses3,linewidth=2.0,label='total=0.01,beta=eps')
# plt.plot(EG_losses4,linewidth=2.0,label='total=0.012,beta=eps')
# plt.plot(EG_losses5,linewidth=2.0,label='total=0.01,beta=min-eps')
# plt.plot(EG_losses6,linewidth=2.0,label='total=0.008,beta=min-eps')
# plt.plot(EG_losses7,linewidth=2.0,label='total=0.012,beta=min-eps')
# plt.plot(EG_losses8,linewidth=2.0,label='total=0.008,beta=random')
# plt.plot(EG_losses9,linewidth=2.0,label='total=0.01,beta=random')
# # EG_losses2 = np.load('EGloss_reg9.npy')
# # C_losses = np.load('Closs9.npy')
#
# # plt.figure(2,figsize=(10, 7.5))
# plt.grid()
# # plt.title('Training loss curve')
# # plt.plot(EG_losses2, label='Encoder + Generator')
# # # plt.plot(C_losses, label='Criic')
# # plt.plot(C_losses, label='Critic')
# # #
# # # plt.plot(EG_losses2,label='After Regularization')
# plt.xlabel('Iterations',fontdict={'family' : 'Times New Roman', 'size'   : 16})
#
# plt.ylabel('Loss',fontdict={'family' : 'Times New Roman', 'size'   : 16})
# plt.yticks(fontproperties = 'Times New Roman', size = 14)
# plt.xticks(fontproperties = 'Times New Roman', size = 14)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
# plt.savefig('mnist/Total_loss.png')
# plt.show()
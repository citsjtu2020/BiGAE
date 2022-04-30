# MNIST classification using Support Vector algorithm with RBF kernel
# Author: Krzysztof Sopyla <krzysztofsopyla@gmail.com>
# https://ksopyla.com
# License: MIT

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
#fetch original mnist dataset
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# import custom module
from mnist_helpers import *


# it creates mldata folder in your root project folder
# mnist = fetch_openml(data_home='datastes/')
# mnist = fetch_openml("mnist_784")
# #minist object contains: data, COL_NAMES, DESCR, target fields
# #you can check it by running
# mnist.keys()
#
# #data field is 70k x 784 array, each row represents pixels from 28x28=784 image
# images = mnist.data
# targets = mnist.target
# base_images = np.load("train_data.npy")
# base_targets = np.load("train_label.npy",allow_pickle=True)
#
# test_images = np.load("test_data.npy")
# test_labels = np.load("test_label.npy")

# base_images = np.load("wmgan_19000_train_data.npy")
# base_targets = np.load("wmgan_19000_train_label.npy",allow_pickle=True)
#
# test_images = np.load("wmgan_19000_test_data.npy")
# test_labels = np.load("wmgan_19000_test_label.npy")

# base_images = np.load("data/alae_20000_train_data.npy")
# base_targets = np.load("data/alae_20000_train_label.npy",allow_pickle=True)
#
# test_images = np.load("data/alae_20000_test_data.npy")
# test_labels = np.load("data/alae_20000_test_label.npy")


# aae_64_gla_train = np.load('../Bigan_base/exp64/train_aae_glasses.npy')
# aae_64_gla_test = np.load('../Bigan_base/exp64/test_aae_glasses.npy')
# bigan_64_gla_train = np.load('../Bigan_base/exp64/train_bigan_glasses.npy')
# bigan_64_gla_test = np.load('../Bigan_base/exp64/test_bigan_glasses.npy')
# biganqp_64_gla_train = np.load('../Bigan_base/exp64/train_biganqp_glasses.npy')
# biganqp_64_gla_test = np.load('../Bigan_base/exp64/test_biganqp_glasses.npy')
# wmgan_64_gla_train = np.load('../Bigan_base/exp64/train_wmgan_glasses.npy')
# wmgan_64_gla_test = np.load('../Bigan_base/exp64/test_wmgan_glasses.npy')

# aae_64_gla_train = np.load('../Bigan_base/exp128/train_aae_glasses.npy')
# aae_64_gla_test = np.load('../Bigan_base/exp128/test_aae_glasses.npy')
# bigan_64_gla_train = np.load('../Bigan_base/exp128/train_bigan_glasses.npy')
# bigan_64_gla_test = np.load('../Bigan_base/exp128/test_bigan_glasses.npy')
# biganqp_64_gla_train = np.load('../Bigan_base/exp128/train_biganqp_glasses.npy')
# biganqp_64_gla_test = np.load('../Bigan_base/exp128/test_biganqp_glasses.npy')
# wmgan_64_gla_train = np.load('../Bigan_base/exp128/train_wmgan_glasses.npy')
# wmgan_64_gla_test = np.load('../Bigan_base/exp128/test_wmgan_glasses.npy')

# aae_64_gla_train = np.load('../Bigan_base/exp256/train_aae_glasses.npy')
# aae_64_gla_test = np.load('../Bigan_base/exp256/test_aae_glasses.npy')
# bigan_64_gla_train = np.load('../Bigan_base/exp256/train_bigan_glasses.npy')
# bigan_64_gla_test = np.load('../Bigan_base/exp256/test_bigan_glasses.npy')
# biganqp_64_gla_train = np.load('../Bigan_base/exp256/train_biganqp_glasses.npy')
# biganqp_64_gla_test = np.load('../Bigan_base/exp256/test_biganqp_glasses.npy')
# wmgan_64_gla_train = np.load('../Bigan_base/exp256/train_wmgan_glasses.npy')
# wmgan_64_gla_test = np.load('../Bigan_base/exp256/test_wmgan_glasses.npy')


# aae_64_gla_train = np.load('../Bigan_base/exp64/train_aae_age.npy')
# aae_64_gla_test = np.load('../Bigan_base/exp64/test_aae_age.npy')
# bigan_64_gla_train = np.load('../Bigan_base/exp64/train_bigan_age.npy')
# bigan_64_gla_test = np.load('../Bigan_base/exp64/test_bigan_age.npy')
# biganqp_64_gla_train = np.load('../Bigan_base/exp64/train_biganqp_age.npy')
# biganqp_64_gla_test = np.load('../Bigan_base/exp64/test_biganqp_age.npy')
# wmgan_64_gla_train = np.load('../Bigan_base/exp64/train_wmgan_age.npy')
# wmgan_64_gla_test = np.load('../Bigan_base/exp64/test_wmgan_age.npy')

aae_64_gla_train = np.load('../Bigan_base/exp128/train_aae_age.npy')
aae_64_gla_test = np.load('../Bigan_base/exp128/test_aae_age.npy')
bigan_64_gla_train = np.load('../Bigan_base/exp128/train_bigan_age.npy')
bigan_64_gla_test = np.load('../Bigan_base/exp128/test_bigan_age.npy')
biganqp_64_gla_train = np.load('../Bigan_base/exp128/train_biganqp_age.npy')
biganqp_64_gla_test = np.load('../Bigan_base/exp128/test_biganqp_age.npy')
wmgan_64_gla_train = np.load('../Bigan_base/exp128/train_wmgan_age.npy')
wmgan_64_gla_test = np.load('../Bigan_base/exp128/test_wmgan_age.npy')

# aae_64_gla_train = np.load('../Bigan_base/exp256/train_aae_age.npy')
# aae_64_gla_test = np.load('../Bigan_base/exp256/test_aae_age.npy')
# bigan_64_gla_train = np.load('../Bigan_base/exp256/train_bigan_age.npy')
# bigan_64_gla_test = np.load('../Bigan_base/exp256/test_bigan_age.npy')
# biganqp_64_gla_train = np.load('../Bigan_base/exp256/train_biganqp_age.npy')
# biganqp_64_gla_test = np.load('../Bigan_base/exp256/test_biganqp_age.npy')
# wmgan_64_gla_train = np.load('../Bigan_base/exp256/train_wmgan_age.npy')
# wmgan_64_gla_test = np.load('../Bigan_base/exp256/test_wmgan_age.npy')

X_aae_train = aae_64_gla_train[:,0:-1]
Y_aae_train = aae_64_gla_train[:,-1]
X_aae_test = aae_64_gla_test[:,0:-1]
Y_aae_test = aae_64_gla_test[:,-1]

print(X_aae_train.shape)

X_qp_train = biganqp_64_gla_train[:,0:-1]
Y_qp_train = biganqp_64_gla_train[:,-1]
X_qp_test = biganqp_64_gla_test[:,0:-1]
Y_qp_test = biganqp_64_gla_test[:,-1]

X_bi_train = bigan_64_gla_train[:,0:-1]
Y_bi_train = bigan_64_gla_train[:,-1]
X_bi_test = bigan_64_gla_test[:,0:-1]
Y_bi_test = bigan_64_gla_test[:,-1]

X_wm_train = wmgan_64_gla_train[:,0:-1]
Y_wm_train = wmgan_64_gla_train[:,-1]
X_wm_test = wmgan_64_gla_test[:,0:-1]
Y_wm_test = wmgan_64_gla_test[:,-1]

'''
indexss = {'21/40/32':0,'21/40/-32':1,'21/-40/32':2,'21/-40/-32':3,'-21/40/32':4,'-21/40/-32':5,'-21/-40/32':6,'-21/-40/-32':7}
male: 0,1,2,3
young: 0,1,4,5
samling: 0,2,4,6
'''

from sklearn.model_selection import train_test_split
x_aae_train, x_aae_valid, y_aae_train, y_aae_valid = train_test_split(X_aae_train, Y_aae_train, test_size=0.05, random_state=42)
print(x_aae_train.shape)
x_wm_train, x_wm_valid, y_wm_train, y_wm_valid = train_test_split(X_wm_train, Y_wm_train, test_size=0.05, random_state=42)
x_bi_train, x_bi_valid, y_bi_train, y_bi_valid = train_test_split(X_bi_train, Y_bi_train, test_size=0.05, random_state=42)
x_qp_train, x_qp_valid, y_qp_train, y_qp_valid = train_test_split(X_qp_train, Y_qp_train, test_size=0.05, random_state=42)
print(sum(Y_wm_test))
print(sum(Y_wm_train))

'''
male
'''

# x_train = []
# y_train = []
# sum0 = 0
# sum1 = 1
# for i in range(x_aae_train.shape[0]):
#     if int(y_aae_train[i]) == 0 or int(y_aae_train[i])==1 or int(y_aae_train[i])==2 or int(y_aae_train[i])==3:
#         x_train.append(x_aae_train[i])
#         y_train.append(0)
#         sum0+=1
#     else:
#         x_train.append(x_aae_train[i])
#         y_train.append(1)
#         sum1+=1
# print(x_aae_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)
# sums = min(sum0,sum1)


# x_train = []
# y_train = []
# sum0 = 0
# sum1 = 1
# for i in range(x_wm_train.shape[0]):
#     if int(y_wm_train[i]) == 0 or int(y_wm_train[i])==1 or int(y_wm_train[i])==2 or int(y_wm_train[i])==3:
#         x_train.append(x_wm_train[i])
#         y_train.append(0)
#         sum0+=1
#     else:
#         x_train.append(x_wm_train[i])
#         y_train.append(1)
#         sum1+=1
# print(x_wm_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)

# x_train = []
# y_train = []
# sum0 = 0
# sum1 = 1
# for i in range(x_bi_train.shape[0]):
#     if int(y_bi_train[i]) == 0 or int(y_bi_train[i])==1 or int(y_bi_train[i])==2 or int(y_bi_train[i])==3:
#         x_train.append(x_bi_train[i])
#         y_train.append(0)
#         sum0+=1
#     else:
#         x_train.append(x_bi_train[i])
#         y_train.append(1)
#         sum1+=1
# print(x_bi_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)
# #

# x_train = []
# y_train = []
# sum0 = 0
# sum1 = 1
# for i in range(x_qp_train.shape[0]):
#     if int(y_qp_train[i]) == 0 or int(y_qp_train[i])==1 or int(y_qp_train[i])==2 or int(y_qp_train[i])==3:
#         x_train.append(x_qp_train[i])
#         y_train.append(0)
#         sum0+=1
#     else:
#         x_train.append(x_qp_train[i])
#         y_train.append(1)
#         sum1+=1
# print(x_qp_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)

# for i in range(x_aae_train.shape[0]):
#     # print(y_aae_train[i])
#     if y_aae_train[i] == 1:
#         x_train.append(x_aae_train[i])
#         y_train.append(y_aae_train[i])
#         sum1 +=1
#         if sum1 >= 1.25*tmpl:
#             break
# print(len(x_train))

'''
age                                      
'''
# x_train1 = []
# y_train1 = []
# sum0 = 0
# sum1 = 1
# for i in range(x_aae_train.shape[0]):
#     if int(y_aae_train[i]) == 0 or int(y_aae_train[i])==1 or int(y_aae_train[i])==4 or int(y_aae_train[i])==5:
#         x_train1.append(x_aae_train[i])
#         y_train1.append(0)
#         sum0+=1
#     else:
#         x_train1.append(x_aae_train[i])
#         y_train1.append(1)
#         sum1+=1
# print(x_aae_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)

# x_train1 = []
# y_train1 = []
# sum0 = 0
# sum1 = 1
# for i in range(x_wm_train.shape[0]):
#     if int(y_wm_train[i]) == 0 or int(y_wm_train[i])==1 or int(y_wm_train[i])==4 or int(y_wm_train[i])==5:
#         x_train1.append(x_wm_train[i])
#         y_train1.append(0)
#         sum0+=1
#     else:
#         x_train1.append(x_wm_train[i])
#         y_train1.append(1)
#         sum1+=1
# print(x_wm_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)
#
# x_train1 = []
# y_train1 = []
# sum0 = 0
# sum1 = 1
# for i in range(x_bi_train.shape[0]):
#     if int(y_bi_train[i]) == 0 or int(y_bi_train[i])==1 or int(y_bi_train[i])==4 or int(y_bi_train[i])==5:
#         x_train1.append(x_bi_train[i])
#         y_train1.append(0)
#         sum0+=1
#     else:
#         x_train1.append(x_bi_train[i])
#         y_train1.append(1)
#         sum1+=1
# print(x_bi_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)


#
# x_train1 = []
# y_train1 = []
# sum0 = 0
# sum1 = 1
# for i in range(x_qp_train.shape[0]):
#     if int(y_qp_train[i]) == 0 or int(y_qp_train[i])==1 or int(y_qp_train[i])==4 or int(y_qp_train[i])==5:
#         x_train1.append(x_qp_train[i])
#         y_train1.append(0)
#         sum0+=1
#     else:
#         x_train1.append(x_qp_train[i])
#         y_train1.append(1)
#         sum1+=1
# print(x_bi_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)
'''
smaling
'''
# x_train1 = []
# y_train1 = []
# sum0 = 0
# sum1 = 1
# for i in range(x_aae_train.shape[0]):
#     if int(y_aae_train[i]) == 0 or int(y_aae_train[i])==2 or int(y_aae_train[i])==4 or int(y_aae_train[i])==6:
#         x_train1.append(x_aae_train[i])
#         y_train1.append(0)
#         sum0+=1
#     else:
#         x_train1.append(x_aae_train[i])
#         y_train1.append(1)
#         sum1+=1
# print(x_aae_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)
#
# # x_train1 = []
# # y_train1 = []
# # sum0 = 0
# # sum1 = 1
# # for i in range(x_wm_train.shape[0]):
# #     if int(y_wm_train[i]) == 0 or int(y_wm_train[i])==2 or int(y_wm_train[i])==4 or int(y_wm_train[i])==6:
# #         x_train1.append(x_wm_train[i])
# #         y_train1.append(0)
# #         sum0+=1
# #     else:
# #         x_train1.append(x_wm_train[i])
# #         y_train1.append(1)
# #         sum1+=1
# # print(x_wm_train.shape[0])
# # print(sum0)
# # print(sum1)
# # print(sum1+sum0)
#

# x_train1 = []
# y_train1 = []
# sum0 = 0
# sum1 = 1
# for i in range(x_wm_train.shape[0]):
#     if int(y_wm_train[i]) == 0 or int(y_wm_train[i])==2 or int(y_wm_train[i])==4 or int(y_wm_train[i])==6:
#         x_train1.append(x_wm_train[i])
#         y_train1.append(0)
#         sum0+=1
#     else:
#         x_train1.append(x_wm_train[i])
#         y_train1.append(1)
#         sum1+=1
# print(x_wm_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)

# x_train1 = []
# y_train1 = []
# sum0 = 0
# sum1 = 1
# for i in range(x_bi_train.shape[0]):
#     if int(y_bi_train[i]) == 0 or int(y_bi_train[i])==2 or int(y_bi_train[i])==4 or int(y_bi_train[i])==6:
#         x_train1.append(x_bi_train[i])
#         y_train1.append(0)
#         sum0+=1
#     else:
#         x_train1.append(x_bi_train[i])
#         y_train1.append(1)
#         sum1+=1
# print(x_bi_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)
#
# x_train1 = []
# y_train1 = []
# sum0 = 0
# sum1 = 1
# for i in range(x_qp_train.shape[0]):
#     if int(y_qp_train[i]) == 0 or int(y_qp_train[i])==2 or int(y_qp_train[i])==4 or int(y_qp_train[i])==6:
#         x_train1.append(x_qp_train[i])
#         y_train1.append(0)
#         sum0+=1
#     else:
#         x_train1.append(x_qp_train[i])
#         y_train1.append(1)
#         sum1+=1
# print(x_qp_train.shape[0])
# print(sum0)
# print(sum1)
# print(sum1+sum0)
#
#
# tmpl = min(len(y_train1) - sum(y_train1),sum(y_train1))
# print(tmpl)
# x_train = []
# y_train = []
# sum0 = 0
# sum1 = 1
# for i in range(len(x_train1)):
#     if y_train1[i] == 1:
#         x_train.append(x_train1[i])
#         y_train.append(y_train1[i])
# print(len(x_train))
# for i in range(len(x_train1)):
#     # print(y_aae_train[i])
#     if y_train1[i] == 0:
#         x_train.append(x_train1[i])
#         y_train.append(y_train1[i])
#         sum1 +=1
#         if sum1 >= 1.25*tmpl:
#             break
#
# print(len(x_train))

# lenss =
# tmpl = min(y_aae_train.shape[0] - sum(y_aae_train),sum(y_aae_train))
# tmpl = min(y_aae_train.shape[0] - sum(y_aae_train),sum(y_aae_train))
# print(tmpl)
# x_train = []
# y_train = []
# sum0 = 0
# sum1 = 1
# for i in range(x_aae_train.shape[0]):
#     if y_aae_train[i] == 0:
#         x_train.append(x_aae_train[i])
#         y_train.append(y_aae_train[i])
# print(x_aae_train.shape[0])
# for i in range(x_aae_train.shape[0]):
#     # print(y_aae_train[i])
#     if y_aae_train[i] == 1:
#         x_train.append(x_aae_train[i])
#         y_train.append(y_aae_train[i])
#         sum1 +=1
#         if sum1 >= 1.25*tmpl:
#             break
#
# print(len(x_train))

# tmpl = min(y_wm_train.shape[0] - sum(y_wm_train),sum(y_wm_train))
# print(tmpl)
# x_train = []
# y_train = []
# sum0 = 0
# sum1 = 1
# for i in range(x_wm_train.shape[0]):
#     if y_wm_train[i] == 0:
#         x_train.append(x_wm_train[i])
#         y_train.append(y_wm_train[i])
# print(x_wm_train.shape[0])
# for i in range(x_wm_train.shape[0]):
#     # print(y_aae_train[i])
#     if y_wm_train[i] == 1:
#         x_train.append(x_wm_train[i])
#         y_train.append(y_wm_train[i])
#         sum1 +=1
#         if sum1 >= 1.25*tmpl:
#             break
#
# print(len(x_train))

# tmpl = min(y_bi_train.shape[0] - sum(y_bi_train),sum(y_bi_train))
# print(tmpl)
# x_train = []
# y_train = []
# sum0 = 0
# sum1 = 1
# for i in range(x_bi_train.shape[0]):
#     if y_bi_train[i] == 0:
#         x_train.append(x_bi_train[i])
#         y_train.append(y_bi_train[i])
# print(x_bi_train.shape[0])
# for i in range(x_bi_train.shape[0]):
#     # print(y_aae_train[i])
#     if y_bi_train[i] == 1:
#         x_train.append(x_bi_train[i])
#         y_train.append(y_bi_train[i])
#         sum1 +=1
#         if sum1 >= 1.25*tmpl:
#             break
#
# print(len(x_train))

# tmpl = min(y_qp_train.shape[0] - sum(y_qp_train),sum(y_qp_train))
# print(tmpl)
# x_train = []
# y_train = []
# sum0 = 0
# sum1 = 1
# for i in range(x_qp_train.shape[0]):
#     if y_qp_train[i] == 0:
#         x_train.append(x_qp_train[i])
#         y_train.append(y_qp_train[i])
# print(x_qp_train.shape[0])
# for i in range(x_qp_train.shape[0]):
#     # print(y_aae_train[i])
#     if y_qp_train[i] == 1:
#         x_train.append(x_qp_train[i])
#         y_train.append(y_qp_train[i])
#         sum1 +=1
#         if sum1 >= 1.25*tmpl:
#             break
#
# print(len(x_train))
#Joint
# x_train = x_aae_train
# y_train = y_aae_train

# x_train = x_wm_train
# y_train = y_wm_train
#
# x_train = x_bi_train
# y_train = y_bi_train
#
x_train = x_qp_train
y_train = y_qp_train

ada = AdaBoostClassifier(learning_rate=0.25,n_estimators=480)
# ada = AdaBoostClassifier(learning_rate=0.13
#                                       ,n_estimators=400)
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
# classifier.fit(X_train, y_train)
# ada.fit(x_aae_train, y_aae_train)
# ada.fit(x_wm_train, y_wm_train)
# ada.fit(x_bi_train, y_bi_train)
ada.fit(x_train, y_train)
#We learn the digits on train part
# start_time = dt.datetime.now()
# print('Start learning at {}'.format(str(start_time)))
# # classifier.fit(X_train, y_train)
# # ada.fit(x_aae_train, y_aae_train)
# # ada.fit(x_wm_train, y_wm_train)
# # ada.fit(x_bi_train, y_bi_train)
# ada.fit(x_qp_train, y_qp_train)
#
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))

# # Now predict the value of the test
'''
male
# '''
# Y_test = []
# for i in range(Y_aae_test.shape[0]):
#     if int(Y_aae_test[i]) == 0 or int(Y_aae_test[i]) == 1 or int(Y_aae_test[i]) == 2 or int(Y_aae_test[i]) == 3:
#         Y_test.append(0)
#     else:
#         Y_test.append(1)
#
# # print(Y_test)
# # expected = Y_aae_test
# expected = Y_test
#
# predicted = ada.predict(X_aae_test)

# Y_test = []
# for i in range(Y_wm_test.shape[0]):
#     if int(Y_wm_test[i]) == 0 or int(Y_wm_test[i]) == 1 or int(Y_wm_test[i]) == 2 or int(Y_wm_test[i]) == 3:
#         Y_test.append(0)
#     else:
#         Y_test.append(1)
#
# # print(Y_test)
# # expected = Y_wm_test
# expected = Y_test
#
# predicted = ada.predict(X_wm_test)



# Y_test = []
# for i in range(Y_wm_test.shape[0]):
#     if int(Y_wm_test[i]) == 0 or int(Y_wm_test[i]) == 1 or int(Y_wm_test[i]) == 2 or int(Y_wm_test[i]) == 3:
#         Y_wm_test[i] == 0
#     else:
#         Y_wm_test[i] == 1
# expected = Y_wm_test
# predicted = ada.predict(X_wm_test)
#
# Y_test = []
# for i in range(Y_bi_test.shape[0]):
#     if int(Y_bi_test[i]) == 0 or int(Y_bi_test[i]) == 1 or int(Y_bi_test[i]) == 2 or int(Y_bi_test[i]) == 3:
#         # Y_bi_test[i] == 0
#         Y_test.append(0)
#     else:
#         # Y_bi_test[i] == 1
#         Y_test.append(1)
# # expected = Y_bi_test
# expected = Y_test
# predicted = ada.predict(X_bi_test)

# Y_test = []
# for i in range(Y_qp_test.shape[0]):
#     if int(Y_qp_test[i]) == 0 or int(Y_qp_test[i]) == 1 or int(Y_qp_test[i]) == 2 or int(Y_qp_test[i]) == 3:
#         Y_test.append(0)
#     else:
#         Y_test.append(1)
# # expected = Y_qp_test
# expected = Y_test
# predicted = ada.predict(X_qp_test)

'''
age
# '''
# Y_test = []
# for i in range(Y_aae_test.shape[0]):
#     if int(Y_aae_test[i]) == 0 or int(Y_aae_test[i]) == 1 or int(Y_aae_test[i]) == 4 or int(Y_aae_test[i]) == 5:
#         Y_test.append(0)
#     else:
#         Y_test.append(1)
#
# # print(Y_test)
# # expected = Y_aae_test
# expected = Y_test
#
# predicted = ada.predict(X_aae_test)

# Y_test = []
# for i in range(Y_wm_test.shape[0]):
#     if int(Y_wm_test[i]) == 0 or int(Y_wm_test[i]) == 1 or int(Y_wm_test[i]) == 4 or int(Y_wm_test[i]) == 5:
#         # Y_bi_test[i] == 0
#         Y_test.append(0)
#     else:
#         # Y_bi_test[i] == 1
#         Y_test.append(1)
# # expected = Y_bi_test
# expected = Y_test
# predicted = ada.predict(X_wm_test)



# Y_test = []
# for i in range(Y_wm_test.shape[0]):
#     if int(Y_wm_test[i]) == 0 or int(Y_wm_test[i]) == 1 or int(Y_wm_test[i]) == 2 or int(Y_wm_test[i]) == 3:
#         Y_wm_test[i] == 0
#     else:
#         Y_wm_test[i] == 1
# expected = Y_wm_test
# predicted = ada.predict(X_wm_test)
#
# Y_test = []
# for i in range(Y_bi_test.shape[0]):
#     if int(Y_bi_test[i]) == 0 or int(Y_bi_test[i]) == 1 or int(Y_bi_test[i]) == 4 or int(Y_bi_test[i]) == 5:
#         # Y_bi_test[i] == 0
#         Y_test.append(0)
#     else:
#         # Y_bi_test[i] == 1
#         Y_test.append(1)
# # expected = Y_bi_test
# expected = Y_test
# predicted = ada.predict(X_bi_test)
#
# Y_test = []
# for i in range(Y_bi_test.shape[0]):
#     if int(Y_qp_test[i]) == 0 or int(Y_qp_test[i]) == 1 or int(Y_qp_test[i]) == 4 or int(Y_qp_test[i]) == 5:
#         # Y_bi_test[i] == 0
#         Y_test.append(0)
#     else:
#         # Y_bi_test[i] == 1
#         Y_test.append(1)
# # expected = Y_bi_test
# expected = Y_test
# predicted = ada.predict(X_qp_test)

'''
samling
# '''
# Y_test = []
# for i in range(Y_aae_test.shape[0]):
#     if int(Y_aae_test[i]) == 0 or int(Y_aae_test[i]) == 2 or int(Y_aae_test[i]) == 4 or int(Y_aae_test[i]) == 6:
#         Y_test.append(0)
#     else:
#         Y_test.append(1)
#
# # print(Y_test)
# # expected = Y_aae_test
# expected = Y_test
# #
# predicted = ada.predict(X_aae_test)

# Y_test = []
# for i in range(Y_wm_test.shape[0]):
#     if int(Y_wm_test[i]) == 0 or int(Y_wm_test[i]) == 2 or int(Y_wm_test[i]) == 4 or int(Y_wm_test[i]) == 6:
#         # Y_bi_test[i] == 0
#         Y_test.append(0)
#     else:
#         # Y_bi_test[i] == 1
#         Y_test.append(1)
# # expected = Y_bi_test
# expected = Y_test
# predicted = ada.predict(X_wm_test)



# Y_test = []
# for i in range(Y_wm_test.shape[0]):
#     if int(Y_wm_test[i]) == 0 or int(Y_wm_test[i]) == 2 or int(Y_wm_test[i]) == 4 or int(Y_wm_test[i]) == 6:
#         Y_wm_test[i] == 0
#     else:
#         Y_wm_test[i] == 1
# expected = Y_wm_test
# predicted = ada.predict(X_wm_test)

# Y_test = []
# for i in range(Y_bi_test.shape[0]):
#     if int(Y_bi_test[i]) == 0 or int(Y_bi_test[i]) == 2 or int(Y_bi_test[i]) == 4 or int(Y_bi_test[i]) == 6:
#         # Y_bi_test[i] == 0
#         Y_test.append(0)
#     else:
#         # Y_bi_test[i] == 1
#         Y_test.append(1)
# # expected = Y_bi_test
# expected = Y_test
# predicted = ada.predict(X_bi_test)
#
# Y_test = []
# for i in range(Y_bi_test.shape[0]):
#     if int(Y_qp_test[i]) == 0 or int(Y_qp_test[i]) == 2  or int(Y_qp_test[i]) == 4 or int(Y_qp_test[i]) == 6:
#         # Y_bi_test[i] == 0
#         Y_test.append(0)
#     else:
#         # Y_bi_test[i] == 1
#         Y_test.append(1)
# # expected = Y_bi_test
# expected = Y_test
# predicted = ada.predict(X_qp_test)
'''
Joint
'''
# expected = Y_aae_test
# predicted = ada.predict(X_aae_test)

# expected = Y_wm_test
# predicted = ada.predict(X_wm_test)

# expected = Y_bi_test
# predicted = ada.predict(X_bi_test)

expected = Y_qp_test
predicted = ada.predict(X_qp_test)

#
# # show_some_digits(X_test,predicted,title_text="Predicted {}")
#
# print("Classification report for classifier %s:\n%s\n"
#       % (ada, metrics.classification_report(expected, predicted)))
#
# cm = metrics.confusion_matrix(expected, predicted)
# print("Confusion matrix:\n%s" % cm)
#
# # plot_confusion_matrix(cm)
# # metrics.precision_score(expected, predicted)
# # metrics.f1_score(expected, predicted)
# # metrics.recall_score(expected, predicted)
# # metrics.st
#
print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),metrics.precision_score(expected, predicted,average='weighted'),metrics.f1_score(expected, predicted,average='weighted'),metrics.recall_score(expected, predicted,average='weighted')))
print(metrics.precision_score(expected, predicted,average=None))
print(metrics.f1_score(expected, predicted,average=None))
print(metrics.recall_score(expected, predicted,average=None))

'''
glasses:
64:
aae:
Start learning at 2020-11-09 23:04:52.601910
Stop learning 2020-11-09 23:05:00.231133
Elapsed learning 0:00:07.629223
Accuracy=0.8583462265772736,Precision=0.9387193407176531,F1=0.8898319644330118,Recall=0.8583462265772736
[0.21444201 0.97838058]
[0.32236842 0.92090612]
[0.64900662 0.86980961]


bigan
Start learning at 2020-11-09 23:39:27.857541
Stop learning 2020-11-09 23:39:35.386675
Elapsed learning 0:00:07.529134
Accuracy=0.830668729585697,Precision=0.9332567506877752,F1=0.8712934474812896,Recall=0.830668729585697
[0.17382999 0.97484277]
[0.2698295  0.90422946]
[0.60264901 0.84315503]

wmgan
Start learning at 2020-11-09 23:41:11.056105
Stop learning 2020-11-09 23:41:18.583654
Elapsed learning 0:00:07.527549
Accuracy=0.8425305140106584,Precision=0.9342335593648968,F1=0.8789718050548159,Recall=0.8425305140106584
[0.18609407 0.97520149]
[0.284375   0.91153178]
[0.60264901 0.85566636]
biganqp
Start learning at 2020-11-09 23:42:16.184419
Stop learning 2020-11-09 23:42:23.762754
Elapsed learning 0:00:07.578335
Accuracy=0.8317001891009111,Precision=0.9442229599093748,F1=0.8736683940689571,Recall=0.8317001891009111
[0.20280948 0.98482257]
[0.32061069 0.90395369]
[0.76490066 0.83535811]

#128
aae
Start learning at 2020-11-10 00:03:26.352741
Stop learning 2020-11-10 00:03:41.128174
Elapsed learning 0:00:14.775433
Accuracy=0.841327144576242,Precision=0.9358890946139227,F1=0.8785195776730187,Recall=0.841327144576242
[0.18981019 0.97674419]
[0.29163469 0.91065725]
[0.62913907 0.85294651]

wmgan
Start learning at 2020-11-09 23:56:33.338098
Stop learning 2020-11-09 23:56:40.880908
Elapsed learning 0:00:07.542810
Accuracy=0.8710675605982465,Precision=0.9406508593358341,F1=0.8982715897637584,Recall=0.8710675605982465
[0.23522459 0.97927982]
[0.3466899  0.92847606]
[0.6589404  0.88268359]

bigan
Start learning at 2020-11-09 23:58:02.599531
Stop learning 2020-11-09 23:58:10.108273
Elapsed learning 0:00:07.508742
Accuracy=0.826542891524841,Precision=0.9347085698224014,F1=0.8689125694704284,Recall=0.826542891524841
[0.17479301 0.97632135]
[0.27357811 0.90151293]
[0.62913907 0.83735267]

biganqp

Start learning at 2020-11-09 23:59:34.920986
Stop learning 2020-11-09 23:59:42.421821
Elapsed learning 0:00:07.500835
Accuracy=0.8179473955647241,Precision=0.9360816788872487,F1=0.8636092021848807,Recall=0.8179473955647241
[0.17229437 0.97790648]
[0.27316404 0.89594183]
[0.6589404  0.82665458]

#256
wmgan
2468
Start learning at 2020-11-10 00:01:21.995119
Stop learning 2020-11-10 00:01:36.748247
Elapsed learning 0:00:14.753128
Accuracy=0.8839608045384219,Precision=0.9389350650779629,F1=0.9059763965047587,Recall=0.8839608045384219
[0.24831309 0.97675335]
[0.35282838 0.93626664]
[0.60927152 0.89900272]

aae
Start learning at 2020-11-09 23:45:50.163653
Stop learning 2020-11-09 23:45:57.733827
Elapsed learning 0:00:07.570174
Accuracy=0.8581743166580712,Precision=0.9376157111042037,F1=0.8895004273825307,Recall=0.8581743166580712
[0.21104972 0.97740228]
[0.31648716 0.92087849]
[0.63245033 0.8705349 ]


bigan
Start learning at 2020-11-10 00:05:11.831020
Stop learning 2020-11-10 00:05:26.509950
Elapsed learning 0:00:14.678930
Accuracy=0.8451091627986935,Precision=0.9364301524690655,F1=0.881015247603662,Recall=0.8451091627986935
[0.19469929 0.97704715]
[0.29773967 0.91295527]
[0.63245033 0.85675431]

biganqp
Start learning at 2020-11-10 00:13:32.314499
Stop learning 2020-11-10 00:13:46.982039
Elapsed learning 0:00:14.667540
Accuracy=0.8353102974041602,Precision=0.9364989481007195,F1=0.8748026788692008,Recall=0.8353102974041602
[0.18642447 0.97757284]
[0.28931751 0.9068637 ]
[0.64569536 0.84569356]
'''
'''
male
64:
aae
Start learning at 2020-11-10 00:47:22.198500
Stop learning 2020-11-10 00:48:48.370293
Elapsed learning 0:01:26.171793
Accuracy=0.7452294997421351,Precision=0.754235586760476,F1=0.7155141398197228,Recall=0.7452294997421351
[0.78608515 0.73712402]
[0.50534045 0.82843251]
[0.37235612 0.94556025]

biganqp
Start learning at 2020-11-10 01:55:16.532753
Stop learning 2020-11-10 01:56:43.047640
Elapsed learning 0:01:26.514887
Accuracy=0.7780642943097816,Precision=0.7814541129604837,F1=0.7618832831397431,Recall=0.7780642943097816
[0.79775281 0.77269744]
[0.6062824  0.84548175]
[0.48893261 0.93340381]
bigan
Start learning at 2020-11-10 01:58:47.872137
Stop learning 2020-11-10 02:00:14.201714
Elapsed learning 0:01:26.329577
Accuracy=0.7710159876224858,Precision=0.7776869259683614,F1=0.7510477999456975,Recall=0.7710159876224858
[0.80611354 0.76241438]
[0.58086847 0.84247871]
[0.45400885 0.94133192]
wmgan
Start learning at 2020-11-10 02:06:16.757242
Stop learning 2020-11-10 02:07:42.866903
Elapsed learning 0:01:26.109661
Accuracy=0.8722709300326629,Precision=0.8712255351448724,F1=0.8714016869366533,Recall=0.8722709300326629
[0.83558793 0.89037227]
[0.81213654 0.90324261]
[0.78996557 0.91649049]
#128
aae
Start learning at 2020-11-10 02:11:39.020113
Stop learning 2020-11-10 02:13:05.472486
Elapsed learning 0:01:26.452373
Accuracy=0.76912497851126,Precision=0.7750763465925498,F1=0.7490945109281035,Recall=0.76912497851126
[0.80052265 0.76140501]
[0.57780572 0.8411215 ]
[0.45204132 0.93948203]
wmgan
Start learning at 2020-11-10 02:14:53.488674
Stop learning 2020-11-10 02:16:19.776431
Elapsed learning 0:01:26.287757
Accuracy=0.7864878803506963,Precision=0.7914704768986569,F1=0.7711219351372446,Recall=0.7864878803506963
[0.81564246 0.77848379]
[0.62203287 0.85122185]
[0.50270536 0.93895349]
bigan
Start learning at 2020-11-10 02:17:48.099656
Stop learning 2020-11-10 02:19:14.389915
Elapsed learning 0:01:26.290259
Accuracy=0.7546845452982637,Precision=0.7632258835937212,F1=0.7287107725763387,Recall=0.7546845452982637
[0.79532164 0.74598205]
[0.53350768 0.83358601]
[0.40137727 0.94450317]
biganqp
Start learning at 2020-11-10 02:21:34.047252
Stop learning 2020-11-10 02:22:59.997648
Elapsed learning 0:01:25.950396
Accuracy=0.7414474815196836,Precision=0.7382979526749561,F1=0.7201845276445218,Recall=0.7414474815196836
[0.72434266 0.7457956 ]
[0.53175592 0.82142009]
[0.42006886 0.91411205]

256
aae
Start learning at 2020-11-10 02:25:44.403287
Stop learning 2020-11-10 02:28:38.599725
Elapsed learning 0:02:54.196438
Accuracy=0.7564036444902871,Precision=0.7645353999382579,F1=0.7312533111865384,Recall=0.7564036444902871
[0.79558541 0.7478534 ]
[0.53918699 0.83444328]
[0.40777177 0.94371036]
wmgan
Start learning at 2020-11-10 02:30:53.487731
Stop learning 2020-11-10 02:33:48.977689
Elapsed learning 0:02:55.489958
Accuracy=0.7686092487536531,Precision=0.7751368999499705,F1=0.7480862483352269,Recall=0.7686092487536531
[0.80264317 0.76035882]
[0.57512626 0.8410111 ]
[0.44810625 0.94080338]
bigan
Start learning at 2020-11-10 02:36:00.286461
Stop learning 2020-11-10 02:38:56.689855
Elapsed learning 0:02:56.403394
Accuracy=0.7459171394189444,Precision=0.7518667833328229,F1=0.7184641190030827,Recall=0.7459171394189444
[0.7739388  0.74000833]
[0.51477347 0.82789939]
[0.38563699 0.93948203]
biganqp
Start learning at 2020-11-10 02:47:59.866770
Stop learning 2020-11-10 02:50:54.765406
Elapsed learning 0:02:54.898636
Accuracy=0.7160048134777377,Precision=0.7060464769742898,F1=0.6924859391363175,Recall=0.7160048134777377
[0.66212766 0.7296424 ]
[0.48503741 0.80394019]
[0.38268569 0.89508457]

age
64
aae
Start learning at 2020-11-10 10:27:46.157658
Stop learning 2020-11-10 10:28:24.852695
Elapsed learning 0:00:38.695037
Accuracy=0.7490115179645865,Precision=0.7514448782723476,F1=0.750191851165396,Recall=0.7490115179645865
[0.8373425  0.48347107]
[0.83352338 0.49022346]
[0.82973893 0.49716714]
wmgan
Start learning at 2020-11-10 10:30:32.527571
Stop learning 2020-11-10 10:31:11.282755
Elapsed learning 0:00:38.755184
Accuracy=0.7555440948942754,Precision=0.7456531999552419,F1=0.7476454888528326,Recall=0.7498710675605983
[0.8295353  0.48396719]
[0.8361671  0.47148565]
[0.84290579 0.45963173]

bigan
Start learning at 2020-11-10 10:37:23.093760
Stop learning 2020-11-10 10:38:01.577972
Elapsed learning 0:00:38.484212
Accuracy=0.7325081657211621,Precision=0.7343153918212747,F1=0.7333948118126443,Recall=0.7325081657211621
[0.82545122 0.45      ]
[0.8228194  0.45441795]
[0.82020431 0.45892351]

biganqp
Start learning at 2020-11-10 10:37:23.093760
Stop learning 2020-11-10 10:38:01.577972
Elapsed learning 0:00:38.484212
Accuracy=0.7325081657211621,Precision=0.7343153918212747,F1=0.7333948118126443,Recall=0.7325081657211621
[0.82545122 0.45      ]
[0.8228194  0.45441795]
[0.82020431 0.45892351]
128
aae
Start learning at 2020-11-10 09:56:05.453966
Stop learning 2020-11-10 09:56:43.874155
Elapsed learning 0:00:38.420189
Accuracy=0.7490115179645865,Precision=0.7514448782723476,F1=0.750191851165396,Recall=0.7490115179645865
[0.8373425  0.48347107]
[0.83352338 0.49022346]
[0.82973893 0.49716714]
wmgan
Start learning at 2020-11-10 09:58:34.920939
Stop learning 2020-11-10 09:59:13.080621
Elapsed learning 0:00:38.159682
Accuracy=0.7555440948942754,Precision=0.7456531999552419,F1=0.7476454888528326,Recall=0.7498710675605983
[0.8295353  0.48396719]
[0.8361671  0.47148565]
[0.84290579 0.45963173]
bigan
Start learning at 2020-11-10 10:04:52.594829
Stop learning 2020-11-10 10:05:31.129511
Elapsed learning 0:00:38.534682
Accuracy=0.7325081657211621,Precision=0.7343153918212747,F1=0.7333948118126443,Recall=0.7325081657211621
[0.82545122 0.45      ]
[0.8228194  0.45441795]
[0.82020431 0.45892351]
biganqp
Start learning at 2020-11-10 10:09:35.950584
Stop learning 2020-11-10 10:10:13.779531
Elapsed learning 0:00:37.828947
Accuracy=0.7354306343476018,Precision=0.7587910639343328,F1=0.744524209632924,Recall=0.7354306343476018
[0.85330375 0.46394094]
[0.81810661 0.51497006]
[0.78569807 0.5786119 ]

256
aae
Start learning at 2020-11-10 09:23:19.892473
Stop learning 2020-11-10 09:24:37.376682
Elapsed learning 0:01:17.484209

Accuracy=0.7426508509541001,Precision=0.7370883656961059,F1=0.7396819202199955,Recall=0.7426508509541001
[0.82339858 0.4678274 ]
[0.83181665 0.45225027]
[0.84040863 0.43767705]

wmgan
Start learning at 2020-11-10 09:30:57.005374
Stop learning 2020-11-10 09:32:16.033353
Elapsed learning 0:01:19.027979
Accuracy=0.7564036444902871,Precision=0.7356756783451194,F1=0.7401104184001045,Recall=0.7459171394189444
[0.81975093 0.4733871 ]
[0.83544868 0.44268477]
[0.85175936 0.41572238]
bigan
Start learning at 2020-11-10 09:44:22.463351
Stop learning 2020-11-10 09:45:40.061300
Elapsed learning 0:01:17.597949
Accuracy=0.7433384906309094,Precision=0.7325537716372948,F1=0.737209204486737,Recall=0.7433384906309094
[0.81762653 0.46715328]
[0.83390811 0.43553875]
[0.85085131 0.40793201]
biganqp
Start learning at 2020-11-10 09:50:09.650016
Stop learning 2020-11-10 09:51:27.020079
Elapsed learning 0:01:17.370063
Accuracy=0.7000171909919203,Precision=0.7130667152972191,F1=0.7059054284185317,Recall=0.7000171909919203
[0.81486742 0.39548023]
[0.79777494 0.41930116]
[0.78138479 0.44617564]


samling
64
aae
Start learning at 2020-11-10 10:47:03.155441
Stop learning 2020-11-10 10:48:29.666634
Elapsed learning 0:01:26.511193
Accuracy=0.7158329035585353,Precision=0.7172096993521676,F1=0.7134839555281249,Recall=0.7158329035585353
[0.72854715 0.7071903 ]
[0.67479835 0.74767211]
[0.62843532 0.79306995]
wmgan
Start learning at 2020-11-10 10:53:32.301297
Stop learning 2020-11-10 10:54:59.158296
Elapsed learning 0:01:26.856999
Accuracy=0.7435104005501117,Precision=0.7441757765979874,F1=0.7421939320214146,Recall=0.7435104005501117
[0.75172975 0.7375    ]
[0.71230235 0.76861042]
[0.67680469 0.80246114]
bigan
Start learning at 2020-11-10 11:19:28.434710
Stop learning 2020-11-10 11:20:54.229013
Elapsed learning 0:01:25.794303
Accuracy=0.7271789582258896,Precision=0.7293600754380988,F1=0.7246209624780792,Recall=0.7271789582258896
[0.7459087  0.71473534]
[0.68580479 0.7589245 ]
biganqp
Start learning at 2020-11-10 11:23:09.295153
Stop learning 2020-11-10 11:24:34.888214
Elapsed learning 0:01:25.593061
Accuracy=0.7228812102458312,Precision=0.7225536133343442,F1=0.7223873077409075,Recall=0.7228812102458312
[0.71439539 0.72976339]
[0.69778778 0.74412698]
[0.68193477 0.75906736]
128:
aae
Start learning at 2020-11-10 11:29:14.209427
Stop learning 2020-11-10 11:30:40.221790
Elapsed learning 0:01:26.012363
Accuracy=0.7333677153171738,Precision=0.7341415634858,F1=0.7318127039910269,Recall=0.7333677153171738
[0.7421875  0.72703102]
[0.69947685 0.76038931]
[0.66141444 0.79695596]
wmgan
Start learning at 2020-11-10 11:32:01.816253
Stop learning 2020-11-10 11:33:57.188808
Elapsed learning 0:01:55.372555
Accuracy=0.7778923843905793,Precision=0.7785809009903785,F1=0.7769830010222177,Recall=0.7778923843905793
[0.78774529 0.77048193]
[0.75277459 0.79837703]
[0.72077684 0.82836788]
bigan
Start learning at 2020-11-10 11:35:34.766157
Stop learning 2020-11-10 11:37:01.501103
Elapsed learning 0:01:26.734946
Accuracy=0.7399002922468626,Precision=0.7421650390609,F1=0.7376080528925955,Recall=0.7399002922468626
[0.76005133 0.72635815]
[0.70140122 0.7696056 ]
[0.65115427 0.81832902]
biganqp
Start learning at 2020-11-10 11:38:35.928715
Stop learning 2020-11-10 11:40:02.259031
Elapsed learning 0:01:26.330316
Accuracy=0.6807632800412584,Precision=0.6820653579402197,F1=0.677297697499954,Recall=0.6807632800412584
[0.69106047 0.67411598]
[0.62941529 0.71961347]
[0.57786735 0.77169689]

256
aae
Start learning at 2020-11-10 11:48:51.873036
Stop learning 2020-11-10 11:51:46.873182
Elapsed learning 0:02:55.000146
Accuracy=0.7270070483066873,Precision=0.7280497992075095,F1=0.7251465789527505,Recall=0.7270070483066873
[0.73780742 0.71942657]
[0.69032761 0.75591761]
[0.64858923 0.79630829]
wmgan
Start learning at 2020-11-10 11:53:24.496651
Stop learning 2020-11-10 11:57:21.088357
Elapsed learning 0:03:56.591706
Accuracy=0.7417913013580884,Precision=0.7427460038948259,F1=0.7402630691080151,Recall=0.7417913013580884
[0.75257308 0.73406139]
[0.70880186 0.76806671]
[0.66984243 0.80537565]
bigan
Elapsed learning 0:02:55.179135
Accuracy=0.7276946879834967,Precision=0.7293287808463892,F1=0.7254712173201022,Recall=0.7276946879834967
[0.74289351 0.71734104]
[0.68855682 0.75809407]
[0.64162697 0.80375648]
biganqp
Start learning at 2020-11-10 12:11:17.144724
Stop learning 2020-11-10 12:14:11.887144
Elapsed learning 0:02:54.742420
Accuracy=0.6417397283823276,Precision=0.6410942665622352,F1=0.6391185762575192,Recall=0.6417397283823276
[0.63533361 0.64618521]
[0.59233177 0.68046611]
[0.55478197 0.71858808]


Joint
#64
aae
Accuracy=0.41206807632800413,Precision=0.41818578197066053,F1=0.31431621926769754,Recall=0.41206807632800413
[0.66666667 0.35784314 0.83333333 0.5        0.39364739 0.43261649
 0.         0.        ]
[0.02752294 0.14974359 0.02590674 0.01301518 0.50946882 0.54016559
 0.         0.        ]
[0.01405152 0.09468223 0.01315789 0.00659341 0.72185864 0.71888029
 0.         0.        ]
 Accuracy=0.5007735946364105,Precision=0.5122206868770943,F1=0.4460511970869748,Recall=0.5007735946364105
[0.45273632 0.45210728 0.56862745 0.36       0.47187758 0.56151711
 1.         0.        ]
[0.28980892 0.45559846 0.2406639  0.1785124  0.57830715 0.63212705
 0.00506329 0.        ]
[0.21311475 0.45914397 0.15263158 0.11868132 0.74672775 0.72304943
 0.00253807 0.        ]
 bigan
 Start learning at 2020-11-10 20:05:33.120231
Stop learning 2020-11-10 20:07:05.888403
Elapsed learning 0:01:32.768172
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Accuracy=0.42358604091456076,Precision=0.42910709887615156,F1=0.3338972111693177,Recall=0.42358604091456076
[0.44444444 0.36792453 0.5        1.         0.39329156 0.46260804
 0.         0.        ]
[0.01834862 0.26108787 0.02061856 0.00875274 0.50318321 0.56728111
 0.         0.        ]
[0.00936768 0.20233463 0.01052632 0.0043956  0.69829843 0.73317451
 0.         0.        ]
 biganqp
 Accuracy=0.4607185834622658,Precision=0.4035832994072968,F1=0.41475790693799675,Recall=0.4607185834622658
[0.35454545 0.37089618 0.45454545 0.3164557  0.49900299 0.49499762
 0.         0.        ]
[0.30911493 0.44046365 0.28776978 0.09363296 0.56649689 0.55002647
 0.         0.        ]
[0.27400468 0.54215305 0.21052632 0.05494505 0.65510471 0.61882073
 0.         0.        ]
128
aae
start learning at 2020-11-10 20:42:32.475551
Stop learning 2020-11-10 20:44:04.843996
Elapsed learning 0:01:32.368445
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Accuracy=0.4192882929345023,Precision=0.40231429683473496,F1=0.32733306934905937,Recall=0.4192882929345023
[0.3        0.41176471 0.6        0.66666667 0.37711006 0.4688113
 0.         0.        ]
[0.02684564 0.2147651  0.01558442 0.02586207 0.49755011 0.56527909
 0.         0.        ]
[0.01405152 0.14526589 0.00789474 0.01318681 0.73102094 0.71173317
 0.         0.        ]
wmgan
Start learning at 2020-11-10 21:09:54.944323
Stop learning 2020-11-10 21:12:24.040359
Elapsed learning 0:02:29.096036
Accuracy=0.5365308578304968,Precision=0.5347158092832534,F1=0.48343182526226347,Recall=0.5365308578304968
[0.5        0.49002494 0.61904762 0.37988827 0.52137133 0.57998213
 0.4        1.        ]
[0.34003091 0.49968214 0.3083004  0.21451104 0.62056174 0.66275211
 0.01002506 0.01086957]
[0.25761124 0.50972763 0.20526316 0.14945055 0.76636126 0.77307921
 0.00507614 0.00546448]
 bigan
 Start learning at 2020-11-10 22:11:05.875068
Stop learning 2020-11-10 22:12:39.383461
Elapsed learning 0:01:33.508393
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Accuracy=0.41636582430806257,Precision=0.33677495645586303,F1=0.3161434578989769,Recall=0.41636582430806257
[0.         0.48295455 0.66666667 0.         0.38505338 0.44369688
 0.         0.        ]
[0.         0.17951426 0.01044386 0.         0.4988474  0.55651788
 0.         0.        ]
[0.         0.11024643 0.00526316 0.         0.70811518 0.74627755
 0.         0.        ]
 biganqp
 Start learning at 2020-11-10 22:24:34.016514
Stop learning 2020-11-10 22:26:07.795827
Elapsed learning 0:01:33.779313
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Accuracy=0.37854564208354824,Precision=0.3056926234373102,F1=0.29662262998590455,Recall=0.37854564208354824
[0.5        0.31179775 0.         0.2        0.40484429 0.36612022
 0.         0.        ]
[0.06153846 0.19698314 0.         0.00860215 0.4875     0.47557411
 0.         0.        ]
[0.03278689 0.14396887 0.         0.0043956  0.61256545 0.67837999
 0.         0.        ]
#256
aae
Start learning at 2020-11-10 13:01:57.590459
Stop learning 2020-11-10 13:05:04.223072
Elapsed learning 0:03:06.632613

Accuracy=0.40312876052948254,Precision=0.35425695587432054,F1=0.3112110656935529,Recall=0.40312876052948254
[0.36363636 0.38095238 0.42857143 0.33333333 0.38609023 0.42123647
 0.         0.        ]
[0.01826484 0.19923372 0.01550388 0.00436681 0.4904489  0.53104359
 0.         0.        ]
[0.00936768 0.13488975 0.00789474 0.0021978  0.67212042 0.71828469
 0.         0.        ]

wmgan
Start learning at 2020-11-10 13:17:58.086952
Stop learning 2020-11-10 13:22:11.112345
Elapsed learning 0:04:13.025393
Accuracy=0.4992264053635895,Precision=0.5091181488993193,F1=0.44408841818589956,Recall=0.4992264053635895
[0.44278607 0.45244216 0.55102041 0.34868421 0.47186858 0.56020456
 1.         0.        ]
[0.28343949 0.45448677 0.22594142 0.17462932 0.57986374 0.62924282
 0.01010101 0.        ]
[0.20843091 0.45654994 0.14210526 0.11648352 0.75196335 0.7176891
 0.00507614 0.        ]

bigan
Start learning at 2020-11-10 18:43:20.170544
Stop learning 2020-11-10 18:46:26.499360
Elapsed learning 0:03:06.328816
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Accuracy=0.4029568506102802,Precision=0.3551279758736935,F1=0.3034534692157594,Recall=0.4029568506102802
[0.25       0.34463277 1.         0.         0.36584554 0.44925884
 0.         0.        ]
[0.00464037 0.12869198 0.00524934 0.         0.48499559 0.54849188
 0.         0.        ]
[0.00234192 0.07911803 0.00263158 0.         0.71924084 0.70399047
 0.         0.        ]
biganqp
Start learning at 2020-11-10 19:01:54.051758
Stop learning 2020-11-10 19:04:59.320160
Elapsed learning 0:03:05.268402
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Accuracy=0.35155578476878113,Precision=0.26113760523955376,F1=0.27384508498791293,Recall=0.35155578476878113
[0.33333333 0.30083565 0.         0.         0.37378641 0.34164038
 0.         0.        ]
[0.03125    0.19115044 0.         0.         0.44649446 0.44669004
 0.         0.        ]
[0.01639344 0.14007782 0.         0.         0.55431937 0.6450268
 0.         0.        ]
'''
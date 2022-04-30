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

# aae_64_gla_train = np.load('../Bigan_base/exp128/train_aae_age.npy')
# aae_64_gla_test = np.load('../Bigan_base/exp128/test_aae_age.npy')
# bigan_64_gla_train = np.load('../Bigan_base/exp128/train_bigan_age.npy')
# bigan_64_gla_test = np.load('../Bigan_base/exp128/test_bigan_age.npy')
# biganqp_64_gla_train = np.load('../Bigan_base/exp128/train_biganqp_age.npy')
# biganqp_64_gla_test = np.load('../Bigan_base/exp128/test_biganqp_age.npy')
# wmgan_64_gla_train = np.load('../Bigan_base/exp128/train_wmgan_age.npy')
# wmgan_64_gla_test = np.load('../Bigan_base/exp128/test_wmgan_age.npy')

aae_64_gla_train = np.load('../Bigan_base/exp256/train_aae_age.npy')
aae_64_gla_test = np.load('../Bigan_base/exp256/test_aae_age.npy')
bigan_64_gla_train = np.load('../Bigan_base/exp256/train_bigan_age.npy')
bigan_64_gla_test = np.load('../Bigan_base/exp256/test_bigan_age.npy')
biganqp_64_gla_train = np.load('../Bigan_base/exp256/train_biganqp_age.npy')
biganqp_64_gla_test = np.load('../Bigan_base/exp256/test_biganqp_age.npy')
wmgan_64_gla_train = np.load('../Bigan_base/exp256/train_wmgan_age.npy')
wmgan_64_gla_test = np.load('../Bigan_base/exp256/test_wmgan_age.npy')

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
#
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
Age                                      
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
#
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

# x_train = x_train1
# y_train = y_train1
# print(len(x_train))

# lenss =
# tmpl = min(y_aae_train.shape[0] - sum(y_aae_train),sum(y_aae_train))

'''
Joint
# '''
# x_train = x_aae_train
# y_train = y_aae_train

# x_train = x_wm_train
# y_train = y_wm_train

# x_train = x_bi_train
# y_train = y_bi_train
# #
x_train = x_qp_train
y_train = y_qp_train

'''
Glasses
'''
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
param_C = 5
param_gamma = 0.05
# classifier = svm.SVC(C=param_C,gamma=param_gamma,max_iter=500)
# random_forest = RandomForestClassifier()
# ada = AdaBoostClassifier(learning_rate=0.1,n_estimators=400)
# ada = AdaBoostClassifier(learning_rate=0.1,n_estimators=300)
# lda = LinearDiscriminantAnalysis()

# random_forest = RandomForestClassifier(n_estimators=270,min_samples_split=6,min_samples_leaf=2)
# random_forest = RandomForestClassifier(n_estimators=90,min_samples_split=20,min_samples_leaf=10)
random_forest = RandomForestClassifier(n_estimators=100)
# random_forest = RandomForestClassifier(n_estimators=120)
# random_forest = RandomForestClassifier(n_estimators=175,min_samples_split=10,min_samples_leaf=5)


#We learn the digits on train part
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
# classifier.fit(X_train, y_train)
#randomfit(X_train, y_train)
random_forest.fit(x_train, y_train)
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))

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
# predicted = random_forest.predict(X_aae_test)

# Y_test = []
# for i in range(Y_wm_test.shape[0]):
#     if int(Y_wm_test[i]) == 0 or int(Y_wm_test[i]) == 1 or int(Y_wm_test[i]) == 2 or int(Y_wm_test[i]) == 3:
#         Y_test.append(0)
#     else:
#         Y_test.append(1)
#
# #print(Y_test)
# #expected = Y_wm_test
# expected = Y_test
#
# predicted = random_forest.predict(X_wm_test)

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
# predicted = random_forest.predict(X_bi_test)

# Y_test = []
# for i in range(Y_qp_test.shape[0]):
#     if int(Y_qp_test[i]) == 0 or int(Y_qp_test[i]) == 1 or int(Y_qp_test[i]) == 2 or int(Y_qp_test[i]) == 3:
#         Y_test.append(0)
#     else:
#         Y_test.append(1)
# # expected = Y_qp_test
# expected = Y_test
# predicted = random_forest.predict(X_qp_test)

'''
Age
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
# predicted = random_forest.predict(X_aae_test)

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
# predicted = random_forest.predict(X_wm_test)

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
# predicted = random_forest.predict(X_bi_test)
# #
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
# predicted = random_forest.predict(X_qp_test)

'''
smaling
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
# predicted = random_forest.predict(X_aae_test)
#
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
# predicted = random_forest.predict(X_wm_test)


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
# predicted = random_forest.predict(X_bi_test)
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
# predicted = random_forest.predict(X_qp_test)
'''
Joint
'''
# expected = Y_aae_test
# predicted = random_forest.predict(X_aae_test)

# expected = Y_wm_test
# predicted = random_forest.predict(X_wm_test)

# expected = Y_bi_test
# predicted = random_forest.predict(X_bi_test)

expected = Y_qp_test
predicted = random_forest.predict(X_qp_test)

'''Glasses'''
#
# expected = Y_aae_test
# predicted = random_forest.predict(X_aae_test)

# expected = Y_wm_test
# predicted = random_forest.predict(X_wm_test)
# #
# expected = Y_bi_test
# predicted = random_forest.predict(X_bi_test)
#
# expected = Y_qp_test
# predicted = random_forest.predict(X_qp_test)

#
#
########################################################
# Now predict the value of the test
# expected = test_labels
# predicted = random_forest.predict(test_images)

# show_some_digits(X_test,predicted,title_text="Predicted {}")

print("Classification report for classifier %s:\n%s\n"
      % (random_forest, metrics.classification_report(expected, predicted)))

cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

# plot_confusion_matrix(cm)
# metrics.precision_score(expected, predicted)
# metrics.f1_score(expected, predicted)
# metrics.recall_score(expected, predicted)
# metrics.st

print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),metrics.precision_score(expected, predicted,average='weighted'),metrics.f1_score(expected, predicted,average='weighted'),metrics.recall_score(expected, predicted,average='weighted')))
print(metrics.precision_score(expected, predicted,average=None))
print(metrics.f1_score(expected, predicted,average=None))
print(metrics.recall_score(expected, predicted,average=None))

# random_forest = RandomForestClassifier(n_estimators=270,min_samples_split=2,min_samples_leaf=1)
# # random_forest = RandomForestClassifier(n_estimators=90,min_samples_split=20,min_samples_leaf=10)
# # random_forest = RandomForestClassifier(n_estimators=250)
# # random_forest = RandomForestClassifier(n_estimators=155,min_samples_split=10,min_samples_leaf=5)

'''
Glasses
64
aae

Start learning at 2020-11-11 11:18:55.022565
Stop learning 2020-11-11 11:18:56.190718
Elapsed learning 0:00:01.168153
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.25      0.56      0.34       302
         1.0       0.97      0.91      0.94      5515

    accuracy                           0.89      5817
   macro avg       0.61      0.73      0.64      5817
weighted avg       0.94      0.89      0.91      5817


Confusion matrix:
[[ 170  132]
 [ 514 5001]]
Accuracy=0.8889461921952897,Precision=0.9366056359807493,F1=0.9084666004867591,Recall=0.8889461921952897
[0.24853801 0.97428404]
[0.34482759 0.93933133]
[0.56291391 0.90679964]

wmgan
Start learning at 2020-11-11 11:19:45.293061
Stop learning 2020-11-11 11:19:46.417784
Elapsed learning 0:00:01.124723
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=125,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.27      0.53      0.36       302
         1.0       0.97      0.92      0.95      5515

    accuracy                           0.90      5817
   macro avg       0.62      0.73      0.65      5817
weighted avg       0.94      0.90      0.92      5817


Confusion matrix:
[[ 161  141]
 [ 437 5078]]
Accuracy=0.9006360667010487,Precision=0.9364467522828082,F1=0.9156058872266019,Recall=0.9006360667010487
[0.26923077 0.97298333]
[0.35777778 0.94615241]
[0.53311258 0.92076156]

bigan
Start learning at 2020-11-11 11:22:13.398625
Stop learning 2020-11-11 11:22:14.578514
Elapsed learning 0:00:01.179889
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.22      0.53      0.31       302
         1.0       0.97      0.90      0.93      5515

    accuracy                           0.88      5817
   macro avg       0.60      0.71      0.62      5817
weighted avg       0.93      0.88      0.90      5817


Confusion matrix:
[[ 160  142]
 [ 562 4953]]
Accuracy=0.8789754168815541,Precision=0.9331647952616297,F1=0.9013995144395419,Recall=0.8789754168815541
[0.22160665 0.97212954]
[0.3125    0.9336475]
[0.52980132 0.8980961 ]

biganqp
Start learning at 2020-11-11 11:22:57.372269
Stop learning 2020-11-11 11:22:58.507922
Elapsed learning 0:00:01.135653
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.19      0.74      0.30       302
         1.0       0.98      0.83      0.90      5515

    accuracy                           0.82      5817
   macro avg       0.59      0.78      0.60      5817
weighted avg       0.94      0.82      0.87      5817


Confusion matrix:
[[ 224   78]
 [ 965 4550]]
Accuracy=0.8206979542719615,Precision=0.9418850664321994,F1=0.8661916596366807,Recall=0.8206979542719615
[0.18839361 0.98314607]
[0.30046948 0.89717046]
[0.74172185 0.82502267]

128
aae
Start learning at 2020-11-11 11:24:18.275597
Stop learning 2020-11-11 11:24:19.454962
Elapsed learning 0:00:01.179365
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.24      0.56      0.34       302
         1.0       0.97      0.90      0.94      5515

    accuracy                           0.89      5817
   macro avg       0.61      0.73      0.64      5817
weighted avg       0.94      0.89      0.91      5817


Confusion matrix:
[[ 169  133]
 [ 524 4991]]
Accuracy=0.887055183084064,Precision=0.9361352931999454,F1=0.9071714032206467,Recall=0.887055183084064
[0.24386724 0.97404372]
[0.33969849 0.93824608]
[0.55960265 0.9049864 ]

wmgan
Start learning at 2020-11-11 11:25:32.546459
Stop learning 2020-11-11 11:25:33.978832
Elapsed learning 0:00:01.432373
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=125,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.31      0.60      0.41       302
         1.0       0.98      0.93      0.95      5515

    accuracy                           0.91      5817
   macro avg       0.64      0.76      0.68      5817
weighted avg       0.94      0.91      0.92      5817


Confusion matrix:
[[ 180  122]
 [ 398 5117]]
Accuracy=0.9106068420147843,Precision=0.942173158354287,F1=0.9234781826775135,Recall=0.9106068420147843
[0.31141869 0.97671311]
[0.40909091 0.9516459 ]
[0.59602649 0.92783318]

bigan
Start learning at 2020-11-11 11:26:45.401906
Stop learning 2020-11-11 11:26:46.572518
Elapsed learning 0:00:01.170612
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.21      0.55      0.31       302
         1.0       0.97      0.89      0.93      5515

    accuracy                           0.87      5817
   macro avg       0.59      0.72      0.62      5817
weighted avg       0.93      0.87      0.90      5817


Confusion matrix:
[[ 167  135]
 [ 624 4891]]
Accuracy=0.8695203713254255,Precision=0.9335783218432393,F1=0.8956816811956547,Recall=0.8695203713254255
[0.21112516 0.97313967]
[0.30558097 0.92799545]
[0.55298013 0.88685403]

biganqp
Start learning at 2020-11-11 11:28:04.610757
Stop learning 2020-11-11 11:28:05.775623
Elapsed learning 0:00:01.164866
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.15      0.69      0.25       302
         1.0       0.98      0.79      0.87      5515

    accuracy                           0.78      5817
   macro avg       0.57      0.74      0.56      5817
weighted avg       0.94      0.78      0.84      5817


Confusion matrix:
[[ 209   93]
 [1172 4343]]
Accuracy=0.7825339522090424,Precision=0.9360638657223913,F1=0.8404544834449967,Recall=0.7825339522090424
[0.15133961 0.97903517]
[0.24836601 0.8728771 ]
[0.69205298 0.78748867]

256
aae
Start learning at 2020-11-11 11:38:43.923993
Stop learning 2020-11-11 11:38:45.640762
Elapsed learning 0:00:01.716769
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.30      0.49      0.37       302
         1.0       0.97      0.94      0.95      5515

    accuracy                           0.91      5817
   macro avg       0.63      0.71      0.66      5817
weighted avg       0.94      0.91      0.92      5817


Confusion matrix:
[[ 149  153]
 [ 350 5165]]
Accuracy=0.913529310641224,Precision=0.9363088568807373,F1=0.9233764801976083,Recall=0.913529310641224
[0.29859719 0.97122979]
[0.37203496 0.9535678 ]
[0.49337748 0.93653672]

wmgan
Start learning at 2020-11-11 11:40:11.342369
Stop learning 2020-11-11 11:40:13.547415
Elapsed learning 0:00:02.205046
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=5, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, n_estimators=175,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.29      0.51      0.37       302
         1.0       0.97      0.93      0.95      5515

    accuracy                           0.91      5817
   macro avg       0.63      0.72      0.66      5817
weighted avg       0.94      0.91      0.92      5817


Confusion matrix:
[[ 153  149]
 [ 376 5139]]
Accuracy=0.9097472924187726,Precision=0.936384692401157,F1=0.9211260009002715,Recall=0.9097472924187726
[0.28922495 0.971823  ]
[0.36823105 0.95140239]
[0.50662252 0.9318223 ]


bigan
Start learning at 2020-11-11 11:43:21.133351
Stop learning 2020-11-11 11:43:22.804753
Elapsed learning 0:00:01.671402
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.23      0.55      0.33       302
         1.0       0.97      0.90      0.94      5515

    accuracy                           0.88      5817
   macro avg       0.60      0.72      0.63      5817
weighted avg       0.93      0.88      0.90      5817


Confusion matrix:
[[ 165  137]
 [ 539 4976]]
Accuracy=0.8837888946192195,Precision=0.9348478395902593,F1=0.9048101919809606,Recall=0.8837888946192195
[0.234375   0.97320555]
[0.32803181 0.93639443]
[0.54635762 0.90226655]

biganqp
Start learning at 2020-11-11 11:44:13.119428
Stop learning 2020-11-11 11:44:14.866952
Elapsed learning 0:00:01.747524
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.23      0.58      0.33       302
         1.0       0.98      0.90      0.93      5515

    accuracy                           0.88      5817
   macro avg       0.60      0.74      0.63      5817
weighted avg       0.94      0.88      0.90      5817


Confusion matrix:
[[ 176  126]
 [ 579 4936]]
Accuracy=0.8788035069623518,Precision=0.9365865944088047,F1=0.9021788351080264,Recall=0.8788035069623518
[0.23311258 0.97510865]
[0.33301798 0.93334594]
[0.58278146 0.8950136 ]
'''

'''
male
64
aae
Start learning at 2020-11-11 11:47:52.547384
Stop learning 2020-11-11 11:48:09.387699
Elapsed learning 0:00:16.840315
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.86      0.36      0.50      2033
           1       0.74      0.97      0.84      3784

    accuracy                           0.75      5817
   macro avg       0.80      0.66      0.67      5817
weighted avg       0.78      0.75      0.72      5817


Confusion matrix:
[[ 723 1310]
 [ 117 3667]]
Accuracy=0.83711905,Precision=0.78010015315768,F1=0.7204539972610629,Recall=0.7546845452982637
[0.86071429 0.73678923]
[0.50330665 0.83711905]
[0.35563207 0.96908034]

wmgan
Start learning at 2020-11-11 11:54:28.444597
Stop learning 2020-11-11 11:54:48.251900
Elapsed learning 0:00:19.807303
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=130,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.87      0.35      0.50      2033
           1       0.74      0.97      0.84      3784

    accuracy                           0.76      5817
   macro avg       0.80      0.66      0.67      5817
weighted avg       0.78      0.76      0.72      5817


Confusion matrix:
[[ 714 1319]
 [ 105 3679]]
Accuracy=0.7552002750558707,Precision=0.7835207713426149,F1=0.7200190177660153,Recall=0.7552002750558707
[0.87179487 0.73609444]
[0.50070126 0.83785015]
[0.35120512 0.97225159]

bigan
Start learning at 2020-11-11 11:58:21.667990
Stop learning 2020-11-11 11:58:37.733883
Elapsed learning 0:00:16.065893
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.86      0.34      0.49      2033
           1       0.73      0.97      0.83      3784

    accuracy                           0.75      5817
   macro avg       0.80      0.66      0.66      5817
weighted avg       0.78      0.75      0.71      5817


Confusion matrix:
[[ 692 1341]
 [ 111 3673]]
Accuracy=0.7503867973182052,Precision=0.7777101584467171,F1=0.7137055195279125,Recall=0.7503867973182052
[0.86176837 0.73254886]
[0.48801128 0.83496249]
[0.34038367 0.97066596]

biganqp
Start learning at 2020-11-11 12:00:14.650188
Stop learning 2020-11-11 12:00:31.028644
Elapsed learning 0:00:16.378456
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.85      0.81      0.83      2033
           1       0.90      0.92      0.91      3784

    accuracy                           0.88      5817
   macro avg       0.87      0.86      0.87      5817
weighted avg       0.88      0.88      0.88      5817


Confusion matrix:
[[1638  395]
 [ 293 3491]]
Accuracy=0.8817259755887915,Precision=0.8808476934334437,F1=0.8809907240268899,Recall=0.8817259755887915
[0.84826515 0.89835306]
[0.82643794 0.91029987]
[0.80570585 0.92256871]

128
aae
Start learning at 2020-11-11 12:04:41.038511
Stop learning 2020-11-11 12:04:57.377746
Elapsed learning 0:00:16.339235
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.89      0.37      0.52      2033
           1       0.74      0.98      0.84      3784

    accuracy                           0.76      5817
   macro avg       0.82      0.67      0.68      5817
weighted avg       0.79      0.76      0.73      5817


Confusion matrix:
[[ 746 1287]
 [  92 3692]]
Accuracy=0.762936221419976,Precision=0.7934841021150342,F1=0.7297635959066918,Recall=0.762936221419976
[0.8902148  0.74151436]
[0.51967955 0.8426338 ]
[0.3669454 0.9756871]

wmgan
Elapsed learning 0:00:17.401200
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=110,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.90      0.39      0.54      2033
           1       0.75      0.98      0.85      3784

    accuracy                           0.77      5817
   macro avg       0.82      0.68      0.69      5817
weighted avg       0.80      0.77      0.74      5817


Confusion matrix:
[[ 791 1242]
 [  91 3693]]
 0.8968254
 0.84711549
Accuracy=0.7708440777032834,Precision=0.8002269547670225,F1=0.7407279881015505,Recall=0.7708440777032834
[0.8968254  0.74832827]
[0.54271012 0.84711549]
[0.38908018 0.97595137]

bigan
Start learning at 2020-11-11 12:12:14.410675
Stop learning 2020-11-11 12:12:30.904503
Elapsed learning 0:00:16.493828
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.87      0.32      0.47      2033
           1       0.73      0.97      0.83      3784

    accuracy                           0.75      5817
   macro avg       0.80      0.65      0.65      5817
weighted avg       0.78      0.75      0.71      5817


Confusion matrix:
[[ 653 1380]
 [  97 3687]]
Accuracy=0.7460890493381468,Precision=0.7776329773098245,F1=0.7059637655257186,Recall=0.7460890493381468
[0.87066667 0.7276495 ]
[0.46927776 0.8331262 ]
[0.3212002  0.97436575]

biganqp
Start learning at 2020-11-11 12:14:20.049291
Stop learning 2020-11-11 12:14:36.163042
Elapsed learning 0:00:16.113751
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.74      0.42      0.53      2033
           1       0.75      0.92      0.82      3784

    accuracy                           0.74      5817
   macro avg       0.74      0.67      0.68      5817
weighted avg       0.74      0.74      0.72      5817


Confusion matrix:
[[ 850 1183]
 [ 303 3481]]
Accuracy=0.7445418600653257,Precision=0.7431579780421671,F1=0.722567127963679,Recall=0.7445418600653257
[0.73720729 0.74635506]
[0.53358443 0.82410038]
[0.41810133 0.919926  ]

256
aae
Start learning at 2020-11-11 16:39:59.834986
Stop learning 2020-11-11 16:40:23.945072
Elapsed learning 0:00:24.110086
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.88      0.28      0.42      2033
           1       0.72      0.98      0.83      3784

    accuracy                           0.73      5817
   macro avg       0.80      0.63      0.63      5817
weighted avg       0.77      0.73      0.69      5817


Confusion matrix:
[[ 570 1463]
 [  81 3703]]
Accuracy=0.7345710847515902,Precision=0.7722925024485725,F1=0.686728886839025,Recall=0.7345710847515902
[0.87557604 0.71680217]
[0.4247392  0.82748603]
[0.28037383 0.97859408]

wmgan
Start learning at 2020-11-11 16:50:50.812575
Stop learning 2020-11-11 16:51:21.242368
Elapsed learning 0:00:30.429793
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=120,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.89      0.31      0.46      2033
           1       0.73      0.98      0.83      3784

    accuracy                           0.75      5817
   macro avg       0.81      0.64      0.65      5817
weighted avg       0.78      0.75      0.70      5817


Confusion matrix:
[[ 628 1405]
 [  76 3708]]
Accuracy=0.7454014096613375,Precision=0.7835179626930198,F1=0.7026044118841517,Recall=0.7454014096613375
[0.89204545 0.72521025]
[0.4588966 0.8335394]
[0.3089031  0.97991543]

bigan
Start learning at 2020-11-11 16:53:28.616845
Stop learning 2020-11-11 16:53:52.833669
Elapsed learning 0:00:24.216824
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.86      0.27      0.41      2033
           1       0.71      0.98      0.82      3784

    accuracy                           0.73      5817
   macro avg       0.79      0.62      0.62      5817
weighted avg       0.77      0.73      0.68      5817


Confusion matrix:
[[ 556 1477]
 [  91 3693]]
Accuracy=0.7304452466907341,Precision=0.7650029190879781,F1=0.6816055611033789,Recall=0.7304452466907341
[0.85935085 0.71431335]
[0.41492537 0.82488273]
[0.27348746 0.97595137]

biganqp
Start learning at 2020-11-11 16:59:11.939358
Stop learning 2020-11-11 16:59:36.584630
Elapsed learning 0:00:24.645272
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.66      0.38      0.48      2033
           1       0.73      0.89      0.80      3784

    accuracy                           0.71      5817
   macro avg       0.69      0.64      0.64      5817
weighted avg       0.70      0.71      0.69      5817


Confusion matrix:
[[ 772 1261]
 [ 403 3381]]
Accuracy=0.7139418944473096,Precision=0.7034210219534185,F1=0.690252180824959,Recall=0.7139418944473096
[0.65702128 0.72834985]
[0.48129676 0.80251602]
[0.37973438 0.89349894]
'''

'''
Age
aae
64
Start learning at 2020-11-11 17:09:59.280056
Stop learning 2020-11-11 17:10:06.443232
Elapsed learning 0:00:07.163176
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.82      0.88      0.85      4405
           1       0.52      0.40      0.45      1412

    accuracy                           0.76      5817
   macro avg       0.67      0.64      0.65      5817
weighted avg       0.75      0.76      0.75      5817


Confusion matrix:
[[3883  522]
 [ 850  562]]
Accuracy=0.7641395908543923,Precision=0.7471131540735692,F1=0.752875347300539,Recall=0.7641395908543923
[0.82040989 0.51845018]
[0.84985774 0.45032051]
[0.8814983 0.398017 ]
wmgan
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=140,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.82      0.89      0.85      4405
           1       0.53      0.37      0.43      1412

    accuracy                           0.77      5817
   macro avg       0.67      0.63      0.64      5817
weighted avg       0.75      0.77      0.75      5817


Confusion matrix:
[[3941  464]
 [ 892  520]]
Accuracy=0.7668901495616297,Precision=0.7457748749446714,F1=0.7514698540226236,Recall=0.7668901495616297
[0.81543555 0.52845528]
[0.85321498 0.43405676]
[0.89466515 0.36827195]

bigan
Start learning at 2020-11-11 17:25:39.248629
Stop learning 2020-11-11 17:25:46.360766
Elapsed learning 0:00:07.112137
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.81      0.87      0.84      4405
           1       0.48      0.37      0.42      1412

    accuracy                           0.75      5817
   macro avg       0.64      0.62      0.63      5817
weighted avg       0.73      0.75      0.74      5817


Confusion matrix:
[[3831  574]
 [ 891  521]]
Accuracy=0.7481519683685749,Precision=0.7298682192219093,F1=0.7366029936035419,Recall=0.7481519683685749
[0.81130877 0.47579909]
[0.83948724 0.41563622]
[0.86969353 0.36898017]

biganqp
Start learning at 2020-11-11 17:28:42.453653
Stop learning 2020-11-11 17:28:49.561830
Elapsed learning 0:00:07.108177
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.85      0.78      0.81      4405
           1       0.45      0.57      0.50      1412

    accuracy                           0.73      5817
   macro avg       0.65      0.67      0.66      5817
weighted avg       0.75      0.73      0.74      5817


Confusion matrix:
[[3439  966]
 [ 610  802]]
Accuracy=0.7290699673371154,Precision=0.7532883473295104,F1=0.7385307684696596,Recall=0.7290699673371154
[0.84934552 0.45361991]
[0.81357937 0.50440252]
[0.78070375 0.56798867]

128
aae
Start learning at 2020-11-11 17:31:05.983839
Stop learning 2020-11-11 17:31:13.076689
Elapsed learning 0:00:07.092850
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.82      0.87      0.84      4405
           1       0.49      0.40      0.44      1412

    accuracy                           0.75      5817
   macro avg       0.65      0.63      0.64      5817
weighted avg       0.74      0.75      0.74      5817


Confusion matrix:
[[3813  592]
 [ 843  569]]
Accuracy=0.753309265944645,Precision=0.7391196515698549,F1=0.7446935477067225,Recall=0.753309265944645
[0.8189433  0.49009475]
[0.84162896 0.44228527]
[0.86560726 0.4029745 ]
wmgan
Start learning at 2020-11-11 17:32:12.909472
Stop learning 2020-11-11 17:32:19.965641
Elapsed learning 0:00:07.056169
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.82      0.88      0.85      4405
           1       0.51      0.39      0.44      1412

                   verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.82      0.90      0.86      4405
           1       0.55      0.39      0.46      1412

    accuracy                           0.77      5817
   macro avg       0.68      0.64      0.66      5817
weighted avg       0.75      0.77      0.76      5817


Confusion matrix:
[[3949  456]
 [ 860  552]]
Accuracy=0.7737665463297232,Precision=0.7547680872989241,F1=0.759842349185616,Recall=0.7737665463297232
[0.82116864 0.54761905]
[0.85717387 0.45619835]
[0.89648127 0.39093484]

bigan
Start learning at 2020-11-11 17:42:26.265883
Stop learning 2020-11-11 17:42:33.488602
Elapsed learning 0:00:07.222719
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.81      0.88      0.84      4405
           1       0.48      0.34      0.40      1412

    accuracy                           0.75      5817
   macro avg       0.64      0.61      0.62      5817
weighted avg       0.73      0.75      0.73      5817


Confusion matrix:
[[3883  522]
 [ 935  477]]
Accuracy=0.7495272477221936,Precision=0.7262070826929143,F1=0.7336824841137207,Recall=0.7495272477221936
[0.80593607 0.47747748]
[0.84202537 0.39568644]
[0.8814983 0.3378187]

biganqp
Start learning at 2020-11-11 17:44:48.018152
Stop learning 2020-11-11 17:44:55.472706
Elapsed learning 0:00:07.454554
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.81      0.78      0.80      4405
           1       0.39      0.43      0.41      1412

    accuracy                           0.70      5817
   macro avg       0.60      0.61      0.60      5817
weighted avg       0.71      0.70      0.70      5817


Confusion matrix:
[[3453  952]
 [ 810  602]]
Accuracy=0.6970947223654805,Precision=0.7074110351890529,F1=0.7018645463453633,Recall=0.6970947223654805
[0.80999296 0.38738739]
[0.79672358 0.40593392]
[0.78388195 0.42634561]

256
aae
tart learning at 2020-11-11 17:48:03.259398
Stop learning 2020-11-11 17:48:13.582582
Elapsed learning 0:00:10.323184
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.81      0.89      0.84      4405
           1       0.48      0.33      0.39      1412

    accuracy                           0.75      5817
   macro avg       0.64      0.61      0.62      5817
weighted avg       0.73      0.75      0.73      5817


Confusion matrix:
[[3904  501]
 [ 944  468]]
Accuracy=0.7515901667526216,Precision=0.7270444185008348,F1=0.7344275419990334,Recall=0.7515901667526216
[0.80528053 0.48297214]
[0.84383443 0.39311214]
[0.88626561 0.33144476]

wmgan
Start learning at 2020-11-11 17:50:33.349071
Stop learning 2020-11-11 17:50:43.927580
Elapsed learning 0:00:10.578509
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=160,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.81      0.92      0.86      4405
           1       0.55      0.31      0.39      1412

    accuracy                           0.77      5817
   macro avg       0.68      0.61      0.63      5817
weighted avg       0.74      0.77      0.75      5817


Confusion matrix:
[[4045  360]
 [ 978  434]]
Accuracy=0.7799845281072718,Precision=0.7425005643992673,F1=0.7453043703623748,Recall=0.7699845281072718
[0.80529564 0.5465995 ]
[0.85808231 0.39347235]
[0.91827469 0.30736544]

bigan
Start learning at 2020-11-11 17:59:37.780675
Stop learning 2020-11-11 17:59:48.353789
Elapsed learning 0:00:10.573114
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.80      0.89      0.84      4405
           1       0.48      0.33      0.39      1412

    accuracy                           0.75      5817
   macro avg       0.64      0.61      0.62      5817
weighted avg       0.73      0.75      0.73      5817


Confusion matrix:
[[3902  503]
 [ 950  462]]
Accuracy=0.7502148873990029,Precision=0.725206249808394,F1=0.7327593561543079,Recall=0.7502148873990029
[0.80420445 0.47875648]
[0.8430377  0.38872528]
[0.88581158 0.32719547]

biganqp
Start learning at 2020-11-11 18:03:33.580671
Stop learning 2020-11-11 18:03:44.126497
Elapsed learning 0:00:10.545826
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.81      0.79      0.80      4405
           1       0.39      0.43      0.41      1412

    accuracy                           0.70      5817
   macro avg       0.60      0.61      0.60      5817
weighted avg       0.71      0.70      0.70      5817


Confusion matrix:
[[3460  945]
 [ 808  604]]
Accuracy=0.6986419116383016,Precision=0.7085513940075083,F1=0.7032334040275418,Recall=0.6986419116383016
[0.81068416 0.38992899]
[0.79787847 0.40797028]
[0.78547106 0.42776204]
'''

'''
smaling
64
aae
Start learning at 2020-11-11 18:05:50.868752
Stop learning 2020-11-11 18:06:06.729771
Elapsed learning 0:00:15.861019
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.76      0.63      0.69      2729
           1       0.72      0.82      0.76      3088

    accuracy                           0.73      5817
   macro avg       0.74      0.73      0.73      5817
weighted avg       0.73      0.73      0.73      5817


Confusion matrix:
[[1730  999]
 [ 561 2527]]
Accuracy=0.7318205260443528,Precision=0.7347158773049869,F1=0.7290009094443698,Recall=0.7318205260443528
[0.75512876 0.71667612]
[0.68924303 0.76413668]
[0.63393184 0.81832902]

wmgan
Start learning at 2020-11-11 18:19:45.605526
Stop learning 2020-11-11 18:20:11.080279
Elapsed learning 0:00:25.474753
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=160,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.79      0.65      0.71      2729
           1       0.73      0.84      0.78      3088

    accuracy                           0.75      5817
   macro avg       0.76      0.75      0.75      5817
weighted avg       0.76      0.75      0.75      5817


Confusion matrix:
[[1779  950]
 [ 487 2601]]
Accuracy=0.7529654461062403,Precision=0.7571532298756234,F1=0.7501303201917069,Recall=0.7529654461062403
[0.78508385 0.73246973]
[0.71231231 0.78355174]
[0.65188714 0.84229275]

bigan
Start learning at 2020-11-11 18:33:12.570914
Stop learning 2020-11-11 18:33:28.593046
Elapsed learning 0:00:16.022132
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.75      0.63      0.69      2729
           1       0.71      0.82      0.76      3088

    accuracy                           0.73      5817
   macro avg       0.73      0.72      0.72      5817
weighted avg       0.73      0.73      0.73      5817


Confusion matrix:
[[1721 1008]
 [ 564 2524]]
Accuracy=0.7297576070139247,Precision=0.7327010881070865,F1=0.7268548567431989,Recall=0.7297576070139247
[0.75317287 0.71460929]
[0.68647786 0.76253776]
[0.63063393 0.81735751]

biganqp
Start learning at 2020-11-11 18:34:51.911294
Stop learning 2020-11-11 18:35:08.089349
Elapsed learning 0:00:16.178055
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.72      0.69      0.71      2729
           1       0.74      0.77      0.75      3088

    accuracy                           0.73      5817
   macro avg       0.73      0.73      0.73      5817
weighted avg       0.73      0.73      0.73      5817


Confusion matrix:
[[1880  849]
 [ 721 2367]]
Accuracy=0.7301014268523294,Precision=0.7298107799523745,F1=0.7296007055063501,Recall=0.7301014268523294
[0.72279892 0.73600746]
[0.7054409  0.75095178]
[0.68889703 0.76651554]

128
aae
Start learning at 2020-11-11 18:37:22.682709
Stop learning 2020-11-11 18:37:38.934886
Elapsed learning 0:00:16.252177
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.74      0.65      0.69      2729
           1       0.72      0.80      0.76      3088

    accuracy                           0.73      5817
   macro avg       0.73      0.72      0.73      5817
weighted avg       0.73      0.73      0.73      5817


Confusion matrix:
[[1761  968]
 [ 605 2483]]
Accuracy=0.7295856970947223,Precision=0.7311328355213585,F1=0.7274584499190242,Recall=0.7295856970947223
[0.74429417 0.71950159]
[0.69126595 0.75944334]
[0.64529132 0.80408031]

wmgan
Start learning at 2020-11-11 18:40:01.502376
Stop learning 2020-11-11 18:40:26.854118
Elapsed learning 0:00:25.351742
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=180,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.82      0.69      0.75      2729
           1       0.76      0.86      0.81      3088

    accuracy                           0.78      5817
   macro avg       0.79      0.78      0.78      5817
weighted avg       0.79      0.78      0.78      5817


Confusion matrix:
[[1880  849]
 [ 424 2664]]
Accuracy=0.7811586728554237,Precision=0.7853703889046378,F1=0.7789638591198048,Recall=0.7811586728554237
[0.81597222 0.75832622]
[0.74706934 0.80715043]
[0.68889703 0.8626943 ]

bigan
Start learning at 2020-11-11 18:43:14.245044
Stop learning 2020-11-11 18:43:30.122503
Elapsed learning 0:00:15.877459
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.76      0.64      0.70      2729
           1       0.72      0.82      0.77      3088

    accuracy                           0.74      5817
   macro avg       0.74      0.73      0.73      5817
weighted avg       0.74      0.74      0.73      5817


Confusion matrix:
[[1753  976]
 [ 551 2537]]
Accuracy=0.7374935533780299,Precision=0.7403192820866055,F1=0.7348608113715176,Recall=0.7374935533780299
[0.76085069 0.72217478]
[0.69660242 0.76867141]
[0.64235984 0.82156736]

biganqp
Start learning at 2020-11-11 18:44:20.893731
Stop learning 2020-11-11 18:44:38.451825
Elapsed learning 0:00:17.558094
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.68      0.59      0.63      2729
           1       0.68      0.76      0.71      3088

    accuracy                           0.68      5817
   macro avg       0.68      0.67      0.67      5817
weighted avg       0.68      0.68      0.68      5817


Confusion matrix:
[[1611 1118]
 [ 753 2335]]
Accuracy=0.6783565411724256,Precision=0.6786858699059138,F1=0.675805083728677,Recall=0.6783565411724256
[0.68147208 0.67622357]
[0.63263303 0.71395811]
[0.59032613 0.75615285]

256
aae
Start learning at 2020-11-11 18:51:35.984706
Stop learning 2020-11-11 18:51:59.737934
Elapsed learning 0:00:23.753228
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.75      0.61      0.67      2729
           1       0.71      0.82      0.76      3088

    accuracy                           0.72      5817
   macro avg       0.73      0.72      0.72      5817
weighted avg       0.73      0.72      0.72      5817


Confusion matrix:
[[1674 1055]
 [ 564 2524]]
Accuracy=0.7216778408114148,Precision=0.7252874844956969,F1=0.7181701315641809,Recall=0.7216778408114148
[0.74798928 0.70522492]
[0.67404872 0.75716214]
[0.61341151 0.81735751]

wmgan
Start learning at 2020-11-11 18:54:15.008622
Stop learning 2020-11-11 18:54:52.292907
Elapsed learning 0:00:37.284285
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=180,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.78      0.62      0.69      2729
           1       0.72      0.85      0.78      3088

    accuracy                           0.74      5817
   macro avg       0.75      0.74      0.74      5817
weighted avg       0.75      0.74      0.74      5817


Confusion matrix:
[[1700 1029]
 [ 466 2622]]
Accuracy=0.7629946707925047,Precision=0.7694500044663402,F1=0.7589504645775266,Recall=0.7429946707925047
[0.78485688 0.71815941]
[0.69458631 0.778157  ]
[0.62293881 0.84909326]

bigan
Start learning at 2020-11-11 18:59:14.027977
Stop learning 2020-11-11 18:59:37.543044
Elapsed learning 0:00:23.515067
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.74      0.62      0.68      2729
           1       0.71      0.81      0.75      3088

    accuracy                           0.72      5817
   macro avg       0.72      0.71      0.71      5817
weighted avg       0.72      0.72      0.72      5817


Confusion matrix:
[[1695 1034]
 [ 594 2494]]
Accuracy=0.7201306515385938,Precision=0.7226708012881589,F1=0.7171669897613457,Recall=0.7201306515385938
[0.74049803 0.7069161 ]
[0.67556796 0.75392987]
[0.62110663 0.80764249]

biganqp
Start learning at 2020-11-11 19:01:21.424414
Stop learning 2020-11-11 19:01:46.049070
Elapsed learning 0:00:24.624656
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

           0       0.62      0.54      0.58      2729
           1       0.63      0.70      0.67      3088

    accuracy                           0.63      5817
   macro avg       0.63      0.62      0.62      5817
weighted avg       0.63      0.63      0.63      5817


Confusion matrix:
[[1477 1252]
 [ 911 2177]]
Accuracy=0.628158844765343,Precision=0.6271992283979054,F1=0.625497220564657,Recall=0.628158844765343
[0.61850921 0.63487897]
[0.57729138 0.66809882]
[0.54122389 0.70498705]
'''

'''
Joint

64
aae
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.60      0.01      0.03       427
         1.0       0.41      0.10      0.16       771
         2.0       1.00      0.01      0.01       380
         3.0       0.33      0.01      0.01       455
         4.0       0.39      0.67      0.50      1528
         5.0       0.41      0.73      0.53      1679
         6.0       1.00      0.00      0.01       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.40      5817
   macro avg       0.52      0.19      0.16      5817
weighted avg       0.48      0.40      0.31      5817


Confusion matrix:
[[   6   31    0    1  237  152    0    0]
 [   2   78    0    1  197  493    0    0]
 [   2   19    2    4  209  144    0    0]
 [   0   43    0    3  183  226    0    0]
 [   0    5    0    0 1031  492    0    0]
 [   0   11    0    0  442 1226    0    0]
 [   0    3    0    0  263  127    1    0]
 [   0    2    0    0   59  122    0    0]]
Accuracy=0.40347258036788725,Precision=0.479015745919696,F1=0.3079153864279669,Recall=0.40347258036788725
[0.6        0.40625    1.         0.33333333 0.39336131 0.41113347
 1.         0.        ]
[0.02745995 0.16199377 0.0104712  0.01293103 0.49698723 0.52606737
 0.00506329 0.        ]
[0.01405152 0.10116732 0.00526316 0.00659341 0.67473822 0.73019655
 0.00253807 0.        ]
 
 wmgan
tart learning at 2020-11-11 19:09:50.882436
Stop learning 2020-11-11 19:10:12.422712
Elapsed learning 0:00:21.540276
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=120,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.67      0.01      0.02       427
         1.0       0.41      0.12      0.18       771
         2.0       0.00      0.00      0.00       380
         3.0       1.00      0.00      0.00       455
         4.0       0.40      0.70      0.51      1528
         5.0       0.43      0.75      0.55      1679
         6.0       1.00      0.00      0.01       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.42      5817
   macro avg       0.49      0.20      0.16      5817
weighted avg       0.48      0.42      0.32      5817


Confusion matrix:
[[   4   29    0    0  274  120    0    0]
 [   0   89    0    0  205  477    0    0]
 [   1   22    0    0  236  121    0    0]
 [   1   46    0    1  150  257    0    0]
 [   0    7    0    0 1068  453    0    0]
 [   0   14    0    0  402 1263    0    0]
 [   0    4    0    0  264  125    1    0]
 [   0    4    0    0   48  131    0    0]]
Accuracy=0.5270534639848719,Precision=0.57944077732145607,F1=0.5179688446764252,Recall=0.5270534639848719
[0.66666667 0.41395349 0.         1.         0.40347563 0.42857143
 1.         0.        ]
[0.01847575 0.18052738 0.         0.00438596 0.51161677 0.5460441
 0.00506329 0.        ]
[0.00936768 0.1154345  0.         0.0021978  0.69895288 0.75223347
 0.00253807 0.        ]
 
 Start learning at 2020-11-11 19:21:27.306615
Stop learning 2020-11-11 19:21:45.354594
Elapsed learning 0:00:18.047979
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.33      0.01      0.01       427
         1.0       0.36      0.09      0.15       771
         2.0       0.75      0.01      0.02       380
         3.0       0.50      0.01      0.01       455
         4.0       0.40      0.69      0.51      1528
         5.0       0.42      0.75      0.54      1679
         6.0       1.00      0.00      0.01       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.41      5817
   macro avg       0.47      0.19      0.16      5817
weighted avg       0.46      0.41      0.31      5817


Confusion matrix:
[[   3   33    0    1  258  132    0    0]
 [   1   71    1    0  196  502    0    0]
 [   4   24    3    0  241  108    0    0]
 [   0   45    0    3  149  258    0    0]
 [   1    4    0    0 1053  470    0    0]
 [   0   15    0    1  396 1267    0    0]
 [   0    4    0    1  255  133    1    0]
 [   0    3    0    0   54  126    0    0]]
Accuracy=0.4527557160048135,Precision=0.49596054031177125,F1=0.3131922662111657,Recall=0.4527557160048135
[0.33333333 0.35678392 0.75       0.5        0.4046887  0.4228972
 1.         0.        ]
[0.01376147 0.14639175 0.015625   0.01301518 0.50992736 0.54203209
 0.00506329 0.        ]
[0.00702576 0.0920882  0.00789474 0.00659341 0.68913613 0.75461584
 0.00253807 0.        ]
 
 
 Start learning at 2020-11-11 19:23:06.733523
Stop learning 2020-11-11 19:23:24.530865
Elapsed learning 0:00:17.797342
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.36      0.22      0.27       427
         1.0       0.42      0.50      0.46       771
         2.0       0.40      0.21      0.27       380
         3.0       0.33      0.13      0.18       455
         4.0       0.53      0.68      0.60      1528
         5.0       0.52      0.71      0.60      1679
         6.0       1.00      0.02      0.03       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.49      5817
   macro avg       0.45      0.31      0.30      5817
weighted avg       0.49      0.49      0.45      5817


Confusion matrix:
[[  95  145   39   12   85   51    0    0]
 [  48  388   18   34   39  244    0    0]
 [  80  107   78   43   37   35    0    0]
 [  28  189   43   57   21  117    0    0]
 [  10   29    8    7 1043  431    0    0]
 [   2   63    1   13  405 1195    0    0]
 [   0    6    4    2  267  109    6    0]
 [   1    7    2    3   61  109    0    0]]
Accuracy=0.4920061887570913,Precision=0.49216191028069967,F1=0.4455183041750296,Recall=0.4920061887570913
[0.35984848 0.41541756 0.40414508 0.33333333 0.53268641 0.52160629
 1.         0.        ]
[0.27496382 0.45513196 0.27225131 0.18210863 0.59839357 0.60201511
 0.03       0.        ]
[0.22248244 0.50324254 0.20526316 0.12527473 0.68259162 0.71173317
 0.01522843 0.        ]
 
 128
 aae
 Start learning at 2020-11-11 19:25:40.966723
Stop learning 2020-11-11 19:25:59.772919
Elapsed learning 0:00:18.806196
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.47      0.02      0.04       427
         1.0       0.37      0.10      0.16       771
         2.0       1.00      0.01      0.01       380
         3.0       0.33      0.00      0.00       455
         4.0       0.39      0.70      0.50      1528
         5.0       0.44      0.73      0.55      1679
         6.0       1.00      0.00      0.01       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.41      5817
   macro avg       0.50      0.20      0.16      5817
weighted avg       0.47      0.41      0.31      5817


Confusion matrix:
[[   8   33    0    0  280  106    0    0]
 [   2   80    0    2  238  449    0    0]
 [   7   29    2    0  236  106    0    0]
 [   0   45    0    1  164  245    0    0]
 [   0    7    0    0 1075  446    0    0]
 [   0   14    0    0  440 1225    0    0]
 [   0    5    0    0  289   99    1    0]
 [   0    3    0    0   65  115    0    0]]
Accuracy=0.41120852673199243,Precision=0.4707708212135639,F1=0.3145838925040618,Recall=0.41120852673199243
[0.47058824 0.37037037 1.         0.33333333 0.38571941 0.43891078
 1.         0.        ]
[0.03603604 0.1621074  0.0104712  0.00436681 0.49826188 0.54809843
 0.00506329 0.        ]
[0.01873536 0.10376135 0.00526316 0.0021978  0.70353403 0.72960095
 0.00253807 0.        ]
 
 wmgan
 Start learning at 2020-11-11 19:27:09.788490
Stop learning 2020-11-11 19:27:27.837457
Elapsed learning 0:00:18.048967
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
          precision    recall  f1-score   support

         0.0       1.00      0.00      0.00       427
         1.0       0.52      0.10      0.17       771
         2.0       0.00      0.00      0.00       380
         3.0       0.00      0.00      0.00       455
         4.0       0.44      0.74      0.55      1528
         5.0       0.45      0.84      0.59      1679
         6.0       1.00      0.00      0.01       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.45      5817
   macro avg       0.43      0.21      0.17      5817
weighted avg       0.46      0.45      0.34      5817


Confusion matrix:
[[   1   28    0    0  307   91    0    0]
 [   0   79    0    0  153  539    0    0]
 [   0    9    0    0  261  110    0    0]
 [   0   31    0    0  132  292    0    0]
 [   0    1    0    0 1138  389    0    0]
 [   0    4    0    0  271 1404    0    0]
 [   0    1    0    0  273  119    1    0]
 [   0    0    0    0   41  142    0    0]]
Accuracy=0.5509197180677325,Precision=0.55693607612807724,F1=0.33911932732538785,Recall=0.5509197180677325
[1.         0.51633987 0.         0.         0.44177019 0.45495787
 1.         0.        ]
[0.0046729  0.17099567 0.         0.         0.5545809  0.58929696
 0.00506329 0.        ]
[0.00234192 0.10246433 0.         0.         0.7447644  0.83621203
 0.00253807 0.        ]
 
 Start learning at 2020-11-11 19:32:06.942807
Stop learning 2020-11-11 19:32:25.164244
Elapsed learning 0:00:18.221437
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.33      0.00      0.01       427
         1.0       0.41      0.08      0.14       771
         2.0       0.00      0.00      0.00       380
         3.0       1.00      0.00      0.01       455
         4.0       0.39      0.66      0.49      1528
         5.0       0.42      0.76      0.54      1679
         6.0       1.00      0.00      0.01       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.40      5817
   macro avg       0.44      0.19      0.15      5817
weighted avg       0.45      0.40      0.30      5817

bigan
Confusion matrix:
[[   2   27    1    0  267  130    0    0]
 [   2   65    0    0  203  501    0    0]
 [   1   11    0    0  257  111    0    0]
 [   0   36    1    2  171  245    0    0]
 [   0    8    0    0 1011  509    0    0]
 [   0    9    0    0  399 1271    0    0]
 [   1    0    0    0  273  119    1    0]
 [   0    2    0    0   43  138    0    0]]
Accuracy=0.4043321299638989,Precision=0.4474695281916877,F1=0.3041858433579948,Recall=0.4043321299638989
[0.33333333 0.41139241 0.         1.         0.38528963 0.42030423
 1.         0.        ]
[0.00923788 0.13993541 0.         0.00875274 0.48699422 0.54050606
 0.00506329 0.        ]
[0.00468384 0.0843061  0.         0.0043956  0.66164921 0.75699821
 0.00253807 0.        ]
 
 biganqp
 Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.34      0.06      0.10       427
         1.0       0.32      0.18      0.23       771
         2.0       1.00      0.01      0.01       380
         3.0       0.28      0.02      0.03       455
         4.0       0.41      0.66      0.50      1528
         5.0       0.40      0.67      0.50      1679
         6.0       1.00      0.00      0.01       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.40      5817
   macro avg       0.47      0.20      0.17      5817
weighted avg       0.44      0.40      0.32      5817


Confusion matrix:
[[  24   77    0    2  212  112    0    0]
 [  17  135    0    7  207  405    0    0]
 [  11   50    2    5  172  140    0    0]
 [   7   77    0    8  121  242    0    0]
 [   6   27    0    3 1012  480    0    0]
 [   5   51    0    3  497 1123    0    0]
 [   0    6    0    0  213  174    1    0]
 [   0    1    0    1   49  132    0    0]]
Accuracy=0.39625236376138906,Precision=0.44449894719270966,F1=0.3176789907818993,Recall=0.39625236376138906
[0.34285714 0.31839623 1.         0.27586207 0.40757149 0.39992877
 1.         0.        ]
[0.09657948 0.22594142 0.0104712  0.03305785 0.50461232 0.50055717
 0.00506329 0.        ]
[0.05620609 0.17509728 0.00526316 0.01758242 0.66230366 0.66885051
 0.00253807 0.        ]
 
 256
 aae
 Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.40      0.00      0.01       427
         1.0       0.36      0.06      0.10       771
         2.0       0.00      0.00      0.00       380
         3.0       0.67      0.00      0.01       455
         4.0       0.38      0.67      0.49      1528
         5.0       0.40      0.70      0.51      1679
         6.0       1.00      0.00      0.01       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.39      5817
   macro avg       0.40      0.18      0.14      5817
weighted avg       0.41      0.39      0.29      5817


Confusion matrix:
[[   2   24    0    1  257  143    0    0]
 [   0   47    0    0  220  504    0    0]
 [   2   19    0    0  216  143    0    0]
 [   0   25    0    2  166  262    0    0]
 [   0    5    0    0 1031  492    0    0]
 [   0    6    0    0  494 1179    0    0]
 [   1    3    0    0  267  122    1    0]
 [   0    1    0    0   66  116    0    0]]
Accuracy=0.3888602372356885,Precision=0.41176489582682496,F1=0.28981141703349245,Recall=0.3888602372356885
[0.4        0.36153846 0.         0.66666667 0.37946264 0.39817629
 1.         0.        ]
[0.00925926 0.10432852 0.         0.00873362 0.48574794 0.50818966
 0.00506329 0.        ]
[0.00468384 0.06095979 0.         0.0043956  0.67473822 0.70220369
 0.00253807 0.        ]
 
 wmgan
 Start learning at 2020-11-11 19:38:57.712721
Stop learning 2020-11-11 19:40:05.623549
Elapsed learning 0:01:07.910828
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=6,
                       min_weight_fraction_leaf=0.0, n_estimators=270,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.00      0.00      0.00       427
         1.0       0.50      0.04      0.07       771
         2.0       0.00      0.00      0.00       380
         3.0       0.00      0.00      0.00       455
         4.0       0.41      0.69      0.52      1528
         5.0       0.41      0.78      0.54      1679
         6.0       0.00      0.00      0.00       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.41      5817
   macro avg       0.17      0.19      0.14      5817
weighted avg       0.29      0.41      0.30      5817


Confusion matrix:
[[   0    9    0    0  270  148    0    0]
 [   0   31    0    0  196  544    0    0]
 [   0    7    0    0  230  143    0    0]
 [   0   15    0    0  134  306    0    0]
 [   0    0    0    0 1057  471    0    0]
 [   0    0    0    0  366 1313    0    0]
 [   0    0    0    0  268  126    0    0]
 [   0    0    0    0   50  133    0    0]]
Accuracy=0.5127557160048135,Precision=0.4932910707991626,F1=0.30120033730250223,Recall=0.5127557160048135
[0.         0.5        0.         0.         0.41112408 0.41237437
 0.         0.        ]
[0.         0.07442977 0.         0.         0.51573555 0.53999589
 0.         0.        ]
[0.         0.04020752 0.         0.         0.69175393 0.7820131
 0.         0.        ]
 
 bigan
 Start learning at 2020-11-11 19:42:00.050736
Stop learning 2020-11-11 19:42:26.701576
Elapsed learning 0:00:26.650840
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.33      0.00      0.01       427
         1.0       0.37      0.06      0.11       771
         2.0       0.00      0.00      0.00       380
         3.0       0.00      0.00      0.00       455
         4.0       0.38      0.68      0.48      1528
         5.0       0.42      0.73      0.53      1679
         6.0       1.00      0.00      0.01       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.40      5817
   macro avg       0.31      0.18      0.14      5817
weighted avg       0.36      0.40      0.30      5817


Confusion matrix:
[[   2   20    0    0  276  129    0    0]
 [   0   48    0    0  268  455    0    0]
 [   1   17    0    0  245  117    0    0]
 [   2   23    0    0  178  252    0    0]
 [   1    7    0    0 1040  480    0    0]
 [   0   10    0    0  451 1218    0    0]
 [   0    3    0    0  254  136    1    0]
 [   0    2    0    0   60  121    0    0]]
Accuracy=0.3969400034381984,Precision=0.36058550751180785,F1=0.2954914428278776,Recall=0.3969400034381984
[0.33333333 0.36923077 0.         0.         0.37518038 0.41884457
 1.         0.        ]
[0.00923788 0.10654828 0.         0.         0.48372093 0.53106606
 0.00506329 0.        ]
[0.00468384 0.06225681 0.         0.         0.68062827 0.7254318
 0.00253807 0.        ]

biganqpStart learning at 2020-11-11 19:46:54.969452
Stop learning 2020-11-11 19:47:21.680223
Elapsed learning 0:00:26.710771
/home/huaqin/PycharmProjects/class_exp/venv/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for classifier RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.35      0.08      0.13       427
         1.0       0.32      0.17      0.22       771
         2.0       0.20      0.00      0.01       380
         3.0       0.11      0.00      0.01       455
         4.0       0.36      0.60      0.45      1528
         5.0       0.37      0.60      0.45      1679
         6.0       1.00      0.00      0.01       394
         7.0       0.00      0.00      0.00       183

    accuracy                           0.36      5817
   macro avg       0.34      0.18      0.16      5817
weighted avg       0.36      0.36      0.29      5817


Confusion matrix:
[[  34   57    0    1  202  133    0    0]
 [  18  128    2    5  231  387    0    0]
 [  10   37    1    4  172  156    0    0]
 [   9   56    1    2  119  268    0    0]
 [  12   37    0    1  919  559    0    0]
 [  12   73    0    2  586 1006    0    0]
 [   1    8    0    2  232  150    1    0]
 [   2    8    1    1   76   95    0    0]]
Accuracy=0.3594636410520887,Precision=0.3575370906296848,F1=0.2895027684667194,Recall=0.3594636410520887
[0.34693878 0.31683168 0.2        0.11111111 0.36223886 0.36528686
 1.         0.        ]
[0.12952381 0.21787234 0.00519481 0.00845666 0.45215252 0.45386871
 0.00506329 0.        ]
[0.07962529 0.16601816 0.00263158 0.0043956  0.60143979 0.59916617
 0.00253807 0.        ]
'''
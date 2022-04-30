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
# print(base_images[0])
# # np.save("targets.npy",targets)
# # np.save("images.npy",images)
#
# print(base_images.shape)

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


# Let's have a look at the random 16 images,
# We have to reshape each data row, from flat array of 784 int to 28x28 2D array

#pick  random indexes from 0 to size of our dataset
# show_some_digits(base_images*255.,base_targets)


#---------------- classification begins -----------------
#scale data for [0,255] -> [0,1]
#sample smaller size for testing
#rand_idx = np.random.choice(images.shape[0],10000)
#X_data =images[rand_idx]/255.0
#Y      = targets[rand_idx]

# #full dataset classification
# X_data = images/255.0
# Y = targets
#
# X_data = base_images
# Y = base_targets
# #split data to train and test
# # #from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.09, random_state=42)
# print(X_train.shape)

################ Classifier with good params ###########
# Create a classifier: a support vector classifier

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
'''
# x_train = x_aae_train
# y_train = y_aae_train

# x_train = x_wm_train
# y_train = y_wm_train

# x_train = x_bi_train
# y_train = y_bi_train
#
x_train = x_qp_train
y_train = y_qp_train

param_C = 5
param_gamma = 0.05

lda = LinearDiscriminantAnalysis()
#We learn the digits on train part
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
# classifier.fit(X_train, y_train)
#lda.fit(X_train, y_train)
lda.fit(x_train, y_train)
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))

expected = Y_qp_test
predicted = lda.predict(X_qp_test)

'''Glasses'''

# expected = Y_aae_test
# predicted = lda.predict(X_aae_test)

# expected = Y_wm_test
# predicted = lda.predict(X_wm_test)
# #
# expected = Y_bi_test
# predicted = lda.predict(X_bi_test)
#
# expected = Y_qp_test
# predicted = lda.predict(X_qp_test)

#
#
########################################################
# Now predict the value of the test
# expected = test_labels
# predicted = lda.predict(test_images)

# show_some_digits(X_test,predicted,title_text="Predicted {}")

print("Classification report for classifier %s:\n%s\n"
      % (lda, metrics.classification_report(expected, predicted)))

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

'''
Glasses
64
aae
Start learning at 2020-11-11 08:09:59.241736
Stop learning 2020-11-11 08:09:59.280506
Elapsed learning 0:00:00.038770
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.27      0.75      0.39       302
         1.0       0.98      0.89      0.93      5515

    accuracy                           0.88      5817
   macro avg       0.63      0.82      0.66      5817
weighted avg       0.95      0.88      0.91      5817


Confusion matrix:
[[ 225   77]
 [ 616 4899]]
Accuracy=0.8808664259927798,Precision=0.9473020518449321,F1=0.9058957157840923,Recall=0.8808664259927798
[0.26753864 0.98452572]
[0.39370079 0.93394338]
[0.74503311 0.88830462]
wmgan

Start learning at 2020-11-11 08:11:03.275658
Stop learning 2020-11-11 08:11:03.317767
Elapsed learning 0:00:00.042109
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.25      0.75      0.37       302
         1.0       0.98      0.88      0.93      5515

    accuracy                           0.87      5817
   macro avg       0.62      0.81      0.65      5817
weighted avg       0.95      0.87      0.90      5817


Confusion matrix:
[[ 226   76]
 [ 678 4837]]
Accuracy=0.8703799209214371,Precision=0.9463963492533536,F1=0.8989897698951479,Recall=0.8703799209214371
[0.25       0.98453084]
[0.3747927  0.92769467]
[0.74834437 0.87706256]
bigan
Start learning at 2020-11-11 08:13:06.463677
Stop learning 2020-11-11 08:13:06.507053
Elapsed learning 0:00:00.043376
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.22      0.70      0.33       302
         1.0       0.98      0.86      0.92      5515

    accuracy                           0.85      5817
   macro avg       0.60      0.78      0.62      5817
weighted avg       0.94      0.85      0.89      5817


Confusion matrix:
[[ 210   92]
 [ 761 4754]]
Accuracy=0.8533608389204057,Precision=0.9413122442863899,F1=0.8871583197581308,Recall=0.8533608389204057
[0.21627188 0.98101527]
[0.3299293  0.91767204]
[0.69536424 0.86201269]

biganqp
Start learning at 2020-11-11 08:13:58.434349
Stop learning 2020-11-11 08:13:58.483264
Elapsed learning 0:00:00.048915
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.28      0.77      0.41       302
         1.0       0.99      0.89      0.94      5515

    accuracy                           0.89      5817
   macro avg       0.63      0.83      0.67      5817
weighted avg       0.95      0.89      0.91      5817


Confusion matrix:
[[ 232   70]
 [ 593 4922]]
Accuracy=0.88602372356885,Precision=0.9493884006587074,F1=0.9096331995492162,Recall=0.88602372356885
[0.28121212 0.98597756]
[0.41171251 0.93689921]
[0.76821192 0.89247507]

128
aae
Start learning at 2020-11-11 08:16:16.202449
Stop learning 2020-11-11 08:16:16.245623
Elapsed learning 0:00:00.043174
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.24      0.73      0.36       302
         1.0       0.98      0.87      0.92      5515

    accuracy                           0.86      5817
   macro avg       0.61      0.80      0.64      5817
weighted avg       0.94      0.86      0.90      5817


Confusion matrix:
[[ 219   83]
 [ 703 4812]]
Accuracy=0.8648788035069623,Precision=0.9443390786641617,F1=0.89507689313965,Recall=0.8648788035069623
[0.23752711 0.98304392]
[0.35784314 0.92449568]
[0.72516556 0.87252947]
wmgan
Start learning at 2020-11-11 08:17:20.771660
Stop learning 2020-11-11 08:17:20.815160
Elapsed learning 0:00:00.043500
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.30      0.80      0.43       302
         1.0       0.99      0.90      0.94      5515

    accuracy                           0.89      5817
   macro avg       0.64      0.85      0.69      5817
weighted avg       0.95      0.89      0.91      5817


Confusion matrix:
[[ 242   60]
 [ 576 4939]]
Accuracy=0.8906652913873131,Precision=0.9520631776910012,F1=0.9131683897390478,Recall=0.8906652913873131
[0.29584352 0.9879976 ]
[0.43214286 0.93950923]
[0.8013245  0.89555757]
bigan
Start learning at 2020-11-11 08:18:39.091601
Stop learning 2020-11-11 08:18:39.132175
Elapsed learning 0:00:00.040574
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.23      0.77      0.35       302
         1.0       0.99      0.85      0.92      5515

    accuracy                           0.85      5817
   macro avg       0.61      0.81      0.63      5817
weighted avg       0.95      0.85      0.89      5817


Confusion matrix:
[[ 233   69]
 [ 802 4713]]
Accuracy=0.8502664603747636,Precision=0.9460907575066444,F1=0.8859821433897377,Recall=0.8502664603747636
[0.22512077 0.98557089]
[0.34854151 0.91541226]
[0.77152318 0.85457842]
biganqp
Start learning at 2020-11-11 08:15:13.300865
Stop learning 2020-11-11 08:15:13.340173
Elapsed learning 0:00:00.039308
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.24      0.75      0.37       302
         1.0       0.98      0.87      0.92      5515

    accuracy                           0.87      5817
   macro avg       0.61      0.81      0.65      5817
weighted avg       0.95      0.87      0.90      5817


Confusion matrix:
[[ 225   77]
 [ 704 4811]]
Accuracy=0.8657383531029741,Precision=0.9457222140049346,F1=0.8958848539322231,Recall=0.8657383531029741
[0.24219591 0.98424714]
[0.36555646 0.9249255 ]
[0.74503311 0.87234814]
256
aae
Start learning at 2020-11-11 08:20:18.412234
Stop learning 2020-11-11 08:20:18.511380
Elapsed learning 0:00:00.099146
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.27      0.69      0.39       302
         1.0       0.98      0.90      0.94      5515

    accuracy                           0.89      5817
   macro avg       0.62      0.79      0.66      5817
weighted avg       0.94      0.89      0.91      5817


Confusion matrix:
[[ 209   93]
 [ 570 4945]]
Accuracy=0.88602372356885,Precision=0.9445107633438089,F1=0.9085943132306601,Recall=0.88602372356885
[0.26829268 0.98154029]
[0.386679   0.93717426]
[0.69205298 0.89664551]

wmgan
Start learning at 2020-11-11 08:21:17.914501
Stop learning 2020-11-11 08:21:17.999030
Elapsed learning 0:00:00.084529
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.28      0.80      0.42       302
         1.0       0.99      0.89      0.94      5515

    accuracy                           0.89      5817
   macro avg       0.64      0.84      0.68      5817
weighted avg       0.95      0.89      0.91      5817


Confusion matrix:
[[ 241   61]
 [ 605 4910]]
Accuracy=0.89029918,Precision=0.9512386479037468,F1=0.9096651717102174,Recall=0.8855079938112429
[0.28486998 0.98772883]
[0.41986063 0.93648674]
[0.79801325 0.89029918]
bigan
Start learning at 2020-11-11 08:22:23.581144
Stop learning 2020-11-11 08:22:23.662245
Elapsed learning 0:00:00.081101
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.25      0.73      0.37       302
         1.0       0.98      0.88      0.93      5515

    accuracy                           0.87      5817
   macro avg       0.62      0.81      0.65      5817
weighted avg       0.95      0.87      0.90      5817


Confusion matrix:
[[ 221   81]
 [ 660 4855]]
Accuracy=0.8726147498710676,Precision=0.9455485088634645,F1=0.9002594029995411,Recall=0.8726147498710676
[0.25085131 0.98358995]
[0.37362637 0.92909769]
[0.73178808 0.88032638]
biganqp
Start learning at 2020-11-11 08:23:24.391665
Stop learning 2020-11-11 08:23:24.468241
Elapsed learning 0:00:00.076576
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.24      0.73      0.36       302
         1.0       0.98      0.87      0.93      5515

    accuracy                           0.87      5817
   macro avg       0.61      0.80      0.64      5817
weighted avg       0.94      0.87      0.90      5817


Confusion matrix:
[[ 220   82]
 [ 694 4821]]
Accuracy=0.8665979026989857,Precision=0.9447234141276886,F1=0.8962495199029333,Recall=0.8665979026989857
[0.24070022 0.98327555]
[0.36184211 0.92551353]
[0.72847682 0.87416138]
'''

'''
male
64
aae
Start learning at 2020-11-11 09:14:20.427816
Stop learning 2020-11-11 09:14:20.777415
Elapsed learning 0:00:00.349599
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.81      0.73      0.77      2033
           1       0.86      0.91      0.88      3784

    accuracy                           0.85      5817
   macro avg       0.84      0.82      0.83      5817
weighted avg       0.84      0.85      0.84      5817


Confusion matrix:
[[1494  539]
 [ 356 3428]]
Accuracy=0.8461406223139075,Precision=0.8443612250316842,F1=0.8443313397264458,Recall=0.8461406223139075
[0.80756757 0.86412906]
[0.76950811 0.88453103]
[0.73487457 0.90591966]

wmgan
Start learning at 2020-11-11 09:19:24.418505
Stop learning 2020-11-11 09:19:24.768288
Elapsed learning 0:00:00.349783
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.89      0.88      0.88      2033
           1       0.94      0.94      0.94      3784

    accuracy                           0.92      5817
   macro avg       0.91      0.91      0.91      5817
weighted avg       0.92      0.92      0.92      5817


Confusion matrix:
[[1787  246]
 [ 220 3564]]
Accuracy=0.9198899776517105,Precision=0.9196886221840598,F1=0.9197693364455037,Recall=0.9198899776517105
[0.89038366 0.93543307]
[0.88465347 0.93863577]
[0.87899656 0.94186047]

bigan
Start learning at 2020-11-11 09:17:05.284167
Stop learning 2020-11-11 09:17:05.665381
Elapsed learning 0:00:00.381214
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.82      0.80      0.81      2033
           1       0.89      0.91      0.90      3784

    accuracy                           0.87      5817
   macro avg       0.86      0.85      0.85      5817
weighted avg       0.87      0.87      0.87      5817


Confusion matrix:
[[1625  408]
 [ 359 3425]]
Accuracy=0.8681450919718068,Precision=0.867517504975138,F1=0.8677650084006324,Recall=0.8681450919718068
[0.81905242 0.89355596]
[0.80906149 0.89930419]
[0.79931136 0.90512685]

biganqp
Start learning at 2020-11-11 09:15:36.227614
Stop learning 2020-11-11 09:15:36.597370
Elapsed learning 0:00:00.369756
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.82      0.81      0.81      2033
           1       0.90      0.90      0.90      3784

    accuracy                           0.87      5817
   macro avg       0.86      0.86      0.86      5817
weighted avg       0.87      0.87      0.87      5817


Confusion matrix:
[[1644  389]
 [ 372 3412]]
Accuracy=0.8691765514870208,Precision=0.8689363762338811,F1=0.8690485209953981,Recall=0.8691765514870208
[0.81547619 0.89765851]
[0.81205236 0.8996704 ]
[0.80865716 0.90169133]

128
aae
Start learning at 2020-11-11 09:22:05.291887
Stop learning 2020-11-11 09:22:05.654717
Elapsed learning 0:00:00.362830
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.79      0.72      0.76      2033
           1       0.86      0.90      0.88      3784

    accuracy                           0.84      5817
   macro avg       0.83      0.81      0.82      5817
weighted avg       0.84      0.84      0.84      5817


Confusion matrix:
[[1473  560]
 [ 385 3399]]
Accuracy=0.8375451263537906,Precision=0.8355667209717769,F1=0.8357277009841702,Recall=0.8375451263537906
[0.79278794 0.85855014]
[0.75713184 0.87795428]
[0.72454501 0.89825581]
wmgan
Start learning at 2020-11-11 09:22:56.712127
Stop learning 2020-11-11 09:22:57.059222
Elapsed learning 0:00:00.347095
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.84      0.87      0.86      2033
           1       0.93      0.91      0.92      3784

    accuracy                           0.90      5817
   macro avg       0.88      0.89      0.89      5817
weighted avg       0.90      0.90      0.90      5817


Confusion matrix:
[[1769  264]
 [ 336 3448]]
Accuracy=0.8968540484785972,Precision=0.8979494493595325,F1=0.8972560385390772,Recall=0.8968540484785972
[0.84038005 0.92887931]
[0.85500242 0.91995731]
[0.87014265 0.91120507]
bigan
Start learning at 2020-11-11 09:24:35.765449
Stop learning 2020-11-11 09:24:36.125254
Elapsed learning 0:00:00.359805
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.80      0.80      0.80      2033
           1       0.89      0.89      0.89      3784

    accuracy                           0.86      5817
   macro avg       0.85      0.85      0.85      5817
weighted avg       0.86      0.86      0.86      5817


Confusion matrix:
[[1631  402]
 [ 408 3376]]
Accuracy=0.8607529654461062,Precision=0.8608496055510461,F1=0.860800311980368,Recall=0.8607529654461062
[0.79990191 0.89359449]
[0.80108055 0.89288548]
[0.80226267 0.89217759]
biganqp
Start learning at 2020-11-11 09:25:58.932185
Stop learning 2020-11-11 09:25:59.283142
Elapsed learning 0:00:00.350957
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.77      0.74      0.75      2033
           1       0.86      0.88      0.87      3784

    accuracy                           0.83      5817
   macro avg       0.82      0.81      0.81      5817
weighted avg       0.83      0.83      0.83      5817


Confusion matrix:
[[1495  538]
 [ 448 3336]]
Accuracy=0.8304968196664948,Precision=0.8290780976460786,F1=0.8295744342850807,Recall=0.8304968196664948
[0.76942872 0.86112545]
[0.75201207 0.87124576]
[0.73536645 0.88160677]

256
aae
Start learning at 2020-11-11 09:29:32.497093
Stop learning 2020-11-11 09:29:33.144620
Elapsed learning 0:00:00.647527
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.82      0.78      0.80      2033
           1       0.88      0.91      0.89      3784

    accuracy                           0.86      5817
   macro avg       0.85      0.84      0.84      5817
weighted avg       0.86      0.86      0.86      5817


Confusion matrix:
[[1578  455]
 [ 358 3426]]
Accuracy=0.8602372356884992,Precision=0.8591086617815528,F1=0.8594137199095943,Recall=0.8602372356884992
[0.81508264 0.88276217]
[0.79516251 0.89393346]
[0.77619282 0.90539112]

wmgan
Start learning at 2020-11-11 09:32:07.677477
Stop learning 2020-11-11 09:32:08.313802
Elapsed learning 0:00:00.636325
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.85      0.85      0.85      2033
           1       0.92      0.92      0.92      3784

    accuracy                           0.90      5817
   macro avg       0.89      0.89      0.89      5817
weighted avg       0.90      0.90      0.90      5817


Confusion matrix:
[[1738  295]
 [ 296 3488]]
Accuracy=0.8984012377514182,Precision=0.8984128495131898,F1=0.8984070150998338,Recall=0.8984012377514182
[0.85447394 0.92201956]
[0.85468404 0.92189771]
[0.85489424 0.9217759 ]

bigan
Start learning at 2020-11-11 09:35:06.244991
Stop learning 2020-11-11 09:35:06.904752
Elapsed learning 0:00:00.659761
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.85      0.85      0.85      2033
           1       0.92      0.92      0.92      3784

    accuracy                           0.90      5817
   macro avg       0.89      0.89      0.89      5817
weighted avg       0.90      0.90      0.90      5817


Confusion matrix:
[[1738  295]
 [ 296 3488]]
Accuracy=0.8984012377514182,Precision=0.8984128495131898,F1=0.8984070150998338,Recall=0.8984012377514182
[0.85447394 0.92201956]
[0.85468404 0.92189771]
[0.85489424 0.9217759 ]
biganqp
Start learning at 2020-11-11 09:33:13.288576
Stop learning 2020-11-11 09:33:13.912789
Elapsed learning 0:00:00.624213
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.77      0.73      0.75      2033
           1       0.86      0.88      0.87      3784

    accuracy                           0.83      5817
   macro avg       0.81      0.81      0.81      5817
weighted avg       0.83      0.83      0.83      5817


Confusion matrix:
[[1489  544]
 [ 449 3335]]
Accuracy=0.8292934502320783,Precision=0.8278001061501976,F1=0.8283096503321317,Recall=0.8292934502320783
[0.76831785 0.85975767]
[0.74993704 0.87041629]
[0.73241515 0.88134249]

'''

'''
Age
64
aae
Start learning at 2020-11-11 09:40:46.750522
Stop learning 2020-11-11 09:40:47.140135
Elapsed learning 0:00:00.389613
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.80      0.97      0.88      4405
           1       0.72      0.26      0.38      1412

    accuracy                           0.80      5817
   macro avg       0.76      0.61      0.63      5817
weighted avg       0.78      0.80      0.76      5817


Confusion matrix:
[[4267  138]
 [1050  362]]
Accuracy=0.7957710159876225,Precision=0.7834604712103587,F1=0.7566428268859129,Recall=0.7957710159876225
[0.80252022 0.724     ]
[0.87780292 0.37866109]
[0.96867196 0.25637394]
wmgan
Start learning at 2020-11-11 09:42:05.272501
Stop learning 2020-11-11 09:42:05.624749
Elapsed learning 0:00:00.352248
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.80      0.97      0.88      4405
           1       0.72      0.25      0.37      1412

    accuracy                           0.79      5817
   macro avg       0.76      0.61      0.62      5817
weighted avg       0.78      0.79      0.75      5817


Confusion matrix:
[[4272  133]
 [1064  348]]
Accuracy=0.8005997,Precision=0.87711734,F1=0.7534558076899331,Recall=0.8005997
[0.8005997  0.72349272]
[0.87711734 0.36767036]
[0.96980704 0.24645892]
bigan
Start learning at 2020-11-11 09:45:12.847560
Stop learning 2020-11-11 09:45:13.192485
Elapsed learning 0:00:00.344925
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.80      0.97      0.88      4405
           1       0.73      0.26      0.38      1412

    accuracy                           0.80      5817
   macro avg       0.77      0.61      0.63      5817
weighted avg       0.79      0.80      0.76      5817


Confusion matrix:
[[4271  134]
 [1045  367]]
Accuracy=0.7973182052604435,Precision=0.7862163309874979,F1=0.7585552380255466,Recall=0.7973182052604435
[0.80342363 0.73253493]
[0.87871618 0.38369054]
[0.96958002 0.25991501]

biganqp
Start learning at 2020-11-11 09:46:51.883062
Stop learning 2020-11-11 09:46:52.257498
Elapsed learning 0:00:00.374436
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.81      0.95      0.87      4405
           1       0.66      0.32      0.43      1412

    accuracy                           0.79      5817
   macro avg       0.74      0.63      0.65      5817
weighted avg       0.78      0.79      0.77      5817


Confusion matrix:
[[4171  234]
 [ 961  451]]
Accuracy=0.7945676465532061,Precision=0.775277287180709,F1=0.7667874074655334,Recall=0.7945676465532061
[0.81274357 0.65839416]
[0.87469854 0.43013829]
[0.94687855 0.3194051 ]

128
aae
Start learning at 2020-11-11 09:48:45.844989
Stop learning 2020-11-11 09:48:46.213876
Elapsed learning 0:00:00.368887
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.80      0.97      0.88      4405
           1       0.72      0.23      0.35      1412

    accuracy                           0.79      5817
   macro avg       0.76      0.60      0.61      5817
weighted avg       0.78      0.79      0.75      5817


Confusion matrix:
[[4277  128]
 [1084  328]]
Accuracy=0.7916451779267664,Precision=0.7787439156079881,F1=0.7485275371287172,Recall=0.7916451779267664
[0.79779892 0.71929825]
[0.87589597 0.35117773]
[0.97094211 0.23229462]
wmgan
Start learning at 2020-11-11 09:49:51.701397
Stop learning 2020-11-11 09:49:52.065374
Elapsed learning 0:00:00.363977
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.81      0.96      0.88      4405
           1       0.72      0.31      0.43      1412

    accuracy                           0.80      5817
   macro avg       0.77      0.64      0.66      5817
weighted avg       0.79      0.80      0.77      5817


Confusion matrix:
[[4236  169]
 [ 974  438]]
Accuracy=0.8035069623517277,Precision=0.88112324,F1=0.7725604009473959,Recall=0.8035069623517277
[0.81305182 0.72158155]
[0.88112324 0.43387816]
[0.96163451 0.3101983 ]
bigan
Start learning at 2020-11-11 10:03:32.745861
Stop learning 2020-11-11 10:03:33.142191
Elapsed learning 0:00:00.396330
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.81      0.96      0.88      4405
           1       0.71      0.28      0.40      1412

    accuracy                           0.80      5817
   macro avg       0.76      0.62      0.64      5817
weighted avg       0.78      0.80      0.76      5817


Confusion matrix:
[[4242  163]
 [1022  390]]
Accuracy=0.7962867457452295,Precision=0.781430056351676,F1=0.7608091146191037,Recall=0.7962867457452295
[0.80585106 0.70524412]
[0.87744338 0.39694656]
[0.96299659 0.27620397]

biganqp
Start learning at 2020-11-11 10:05:29.242321
Stop learning 2020-11-11 10:05:29.596040
Elapsed learning 0:00:00.353719
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.79      0.95      0.87      4405
           1       0.61      0.23      0.33      1412

    accuracy                           0.78      5817
   macro avg       0.70      0.59      0.60      5817
weighted avg       0.75      0.78      0.74      5817


Confusion matrix:
[[4199  206]
 [1094  318]]
Accuracy=0.7765171050369606,Precision=0.7480556605037957,F1=0.735495432423189,Recall=0.7765171050369606
[0.79331192 0.60687023]
[0.86595174 0.3285124 ]
[0.95323496 0.22521246]

256
aae
Start learning at 2020-11-11 10:11:20.253894
Stop learning 2020-11-11 10:11:20.880571
Elapsed learning 0:00:00.626677
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.81      0.96      0.88      4405
           1       0.72      0.29      0.41      1412

    accuracy                           0.80      5817
   macro avg       0.77      0.63      0.65      5817
weighted avg       0.79      0.80      0.77      5817


Confusion matrix:
[[4249  156]
 [1003  409]]
Accuracy=0.8007564036444903,Precision=0.7883606266472438,F1=0.7668134128658215,Recall=0.8007564036444903
[0.80902513 0.72389381]
[0.87998343 0.41375822]
[0.9645857  0.28966006]
wmgan
Start learning at 2020-11-11 10:15:53.050840
Stop learning 2020-11-11 10:15:53.676216
Elapsed learning 0:00:00.625376
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.82      0.96      0.89      4405
           1       0.75      0.36      0.49      1412

    accuracy                           0.82      5817
   macro avg       0.79      0.66      0.69      5817
weighted avg       0.81      0.82      0.79      5817


Confusion matrix:
[[4231  174]
 [ 899  513]]
Accuracy=0.8155406566958914,Precision=0.8058152316953079,F1=0.7906970346128414,Recall=0.8155406566958914
[0.82475634 0.74672489]
[0.88746723 0.48880419]
[0.96049943 0.36331445]
bigan
Start learning at 2020-11-11 10:14:52.770928
Stop learning 2020-11-11 10:14:53.414679
Elapsed learning 0:00:00.643751
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.82      0.95      0.88      4405
           1       0.67      0.34      0.45      1412

    accuracy                           0.80      5817
   macro avg       0.75      0.64      0.66      5817
weighted avg       0.78      0.80      0.77      5817


Confusion matrix:
[[4175  230]
 [ 937  475]]
Accuracy=0.7993811242908716,Precision=0.7820073102957327,F1=0.7733332670632287,Recall=0.7993811242908716
[0.81670579 0.67375887]
[0.87737732 0.44874823]
[0.94778661 0.33640227]

biganqp
Start learning at 2020-11-11 10:17:46.934563
Stop learning 2020-11-11 10:17:47.577237
Elapsed learning 0:00:00.642674
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.73      0.44      0.55      4405
           1       0.22      0.49      0.30      1412

    accuracy                           0.45      5817
   macro avg       0.47      0.47      0.43      5817
weighted avg       0.61      0.45      0.49      5817


Confusion matrix:
[[1940 2465]
 [ 719  693]]
 0.4526388172597559
 
Accuracy=0.72959759,Precision=0.605764220230728,F1=0.48955508588147856,Recall=0.4526388172597559
[0.72959759 0.21944269]
[0.54926387 0.30328228]
[0.44040863 0.4907932 ]

biganqp
Start learning at 2020-11-11 10:23:39.140672
Stop learning 2020-11-11 10:23:39.782430
Elapsed learning 0:00:00.641758
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.80      0.96      0.87      4405
           1       0.65      0.24      0.35      1412

    accuracy                           0.78      5817
   macro avg       0.72      0.60      0.61      5817
weighted avg       0.76      0.78      0.74      5817


Confusion matrix:
[[4223  182]
 [1079  333]]
Accuracy=0.7832215918858518,Precision=0.7601080826112316,F1=0.7427834377910645,Recall=0.7832215918858518
[0.79649189 0.64660194]
[0.87009375 0.34561495]
[0.95868331 0.23583569]
'''

'''
smaling
64
aae
Start learning at 2020-11-11 10:25:47.321562
Stop learning 2020-11-11 10:25:47.685437
Elapsed learning 0:00:00.363875
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.80      0.75      0.77      2729
           1       0.79      0.84      0.81      3088

    accuracy                           0.79      5817
   macro avg       0.79      0.79      0.79      5817
weighted avg       0.79      0.79      0.79      5817


Confusion matrix:
[[2040  689]
 [ 508 2580]]
Accuracy=0.7942238267148014,Precision=0.7945784137859002,F1=0.7936242729758352,Recall=0.7942238267148014
[0.80062794 0.78923218]
[0.77316657 0.81170363]
[0.74752657 0.83549223]
wmgan
Start learning at 2020-11-11 10:27:31.120377
Stop learning 2020-11-11 10:27:31.495213
Elapsed learning 0:00:00.374836
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.82      0.78      0.80      2729
           1       0.81      0.85      0.83      3088

    accuracy                           0.81      5817
   macro avg       0.81      0.81      0.81      5817
weighted avg       0.81      0.81      0.81      5817


Confusion matrix:
[[2121  608]
 [ 476 2612]]
Accuracy=0.8136496475846656,Precision=0.8137751682175731,F1=0.8132901526338737,Recall=0.8136496475846656
[0.81671159 0.81118012]
[0.79647015 0.82815472]
[0.77720777 0.84585492]
bigan
Start learning at 2020-11-11 10:28:48.480768
Stop learning 2020-11-11 10:28:48.829104
Elapsed learning 0:00:00.348336
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.81      0.75      0.78      2729
           1       0.79      0.84      0.82      3088

    accuracy                           0.80      5817
   macro avg       0.80      0.80      0.80      5817
weighted avg       0.80      0.80      0.80      5817


Confusion matrix:
[[2040  689]
 [ 485 2603]]
Accuracy=0.7981777548564553,Precision=0.798781515364791,F1=0.7974862474305793,Recall=0.7981777548564553
[0.80792079 0.79070474]
[0.7765512  0.81598746]
[0.74752657 0.84294041]
biganqp
Start learning at 2020-11-11 10:29:42.043776
Stop learning 2020-11-11 10:29:42.404993
Elapsed learning 0:00:00.361217
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.79      0.76      0.78      2729
           1       0.80      0.82      0.81      3088

    accuracy                           0.79      5817
   macro avg       0.79      0.79      0.79      5817
weighted avg       0.79      0.79      0.79      5817


Confusion matrix:
[[2084  645]
 [ 555 2533]]
Accuracy=0.7937080969571945,Precision=0.7935943924807358,F1=0.793460258820679,Recall=0.7937080969571945
[0.78969307 0.79704216]
[0.77645306 0.80849026]
[0.76364969 0.82027202]

128
aae
Start learning at 2020-11-11 10:31:14.610651
Stop learning 2020-11-11 10:31:14.961454
Elapsed learning 0:00:00.350803
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.80      0.76      0.78      2729
           1       0.80      0.84      0.82      3088

    accuracy                           0.80      5817
   macro avg       0.80      0.80      0.80      5817
weighted avg       0.80      0.80      0.80      5817


Confusion matrix:
[[2073  656]
 [ 504 2584]]
Accuracy=0.800584493725288,Precision=0.8007646069143525,F1=0.8001231870942858,Recall=0.800584493725288
[0.80442375 0.79753086]
[0.78137957 0.81668774]
[0.75961891 0.83678756]

wmgan
Start learning at 2020-11-11 10:32:18.114549
Stop learning 2020-11-11 10:32:18.529664
Elapsed learning 0:00:00.415115
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.85      0.82      0.83      2729
           1       0.84      0.87      0.86      3088

    accuracy                           0.85      5817
   macro avg       0.85      0.84      0.85      5817
weighted avg       0.85      0.85      0.85      5817


Confusion matrix:
[[2230  499]
 [ 393 2695]]
Accuracy=0.8466563520715146,Precision=0.8467730120478594,F1=0.8464315446875211,Recall=0.8466563520715146
[0.85017156 0.84376957]
[0.83333333 0.858007  ]
[0.81714914 0.87273316]

bigan
Start learning at 2020-11-11 10:33:26.813736
Stop learning 2020-11-11 10:33:27.165753
Elapsed learning 0:00:00.352017
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.81      0.76      0.79      2729
           1       0.80      0.85      0.82      3088

    accuracy                           0.81      5817
   macro avg       0.81      0.80      0.80      5817
weighted avg       0.81      0.81      0.81      5817


Confusion matrix:
[[2075  654]
 [ 476 2612]]
Accuracy=0.805741791301358,Precision=0.806159525361133,F1=0.805188323277855,Recall=0.805741791301358
[0.81340651 0.79975505]
[0.78598485 0.82215927]
[0.76035178 0.84585492]

biganqp
Start learning at 2020-11-11 10:34:21.419517
Stop learning 2020-11-11 10:34:21.765838
Elapsed learning 0:00:00.346321
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.74      0.70      0.72      2729
           1       0.75      0.78      0.76      3088

    accuracy                           0.74      5817
   macro avg       0.74      0.74      0.74      5817
weighted avg       0.74      0.74      0.74      5817


Confusion matrix:
[[1920  809]
 [ 678 2410]]
Accuracy=0.7443699501461234,Precision=0.7441526371782023,F1=0.7438815522857981,Recall=0.7443699501461234
[0.73903002 0.74867971]
[0.72085602 0.76423022]
[0.70355442 0.78044041]

256
aae
Start learning at 2020-11-11 10:36:15.884376
Stop learning 2020-11-11 10:36:16.519537
Elapsed learning 0:00:00.635161
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.81      0.78      0.79      2729
           1       0.81      0.83      0.82      3088

    accuracy                           0.81      5817
   macro avg       0.81      0.81      0.81      5817
weighted avg       0.81      0.81      0.81      5817


Confusion matrix:
[[2134  595]
 [ 516 2572]]
Accuracy=0.8090080797662025,Precision=0.8089151590807071,F1=0.8088116590281268,Recall=0.8090080797662025
[0.80528302 0.81212504]
[0.79345603 0.82238209]
[0.78197142 0.83290155]

wmgan
Start learning at 2020-11-11 10:37:30.785745
Stop learning 2020-11-11 10:37:31.406372
Elapsed learning 0:00:00.620627
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.86      0.81      0.84      2729
           1       0.84      0.88      0.86      3088

    accuracy                           0.85      5817
   macro avg       0.85      0.85      0.85      5817
weighted avg       0.85      0.85      0.85      5817


Confusion matrix:
[[2221  508]
 [ 356 2732]]
Accuracy=0.85146982980918,Precision=0.8519570292257215,F1=0.8511262359047094,Recall=0.85146982980918
[0.86185487 0.84320988]
[0.83716547 0.86346397]
[0.81385123 0.88471503]

bigan
Start learning at 2020-11-11 10:39:12.767860
Stop learning 2020-11-11 10:39:13.403623
Elapsed learning 0:00:00.635763
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.84      0.79      0.81      2729
           1       0.82      0.87      0.85      3088

    accuracy                           0.83      5817
   macro avg       0.83      0.83      0.83      5817
weighted avg       0.83      0.83      0.83      5817


Confusion matrix:
[[2151  578]
 [ 402 2686]]
Accuracy=0.8315282791817088,Precision=0.8321219508631218,F1=0.8310554714439063,Recall=0.8315282791817088
[0.84253819 0.82291667]
[0.81446422 0.84571788]
[0.78820081 0.86981865]

biganqp
Start learning at 2020-11-11 10:40:08.124972
Stop learning 2020-11-11 10:40:08.766794
Elapsed learning 0:00:00.641822
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

           0       0.73      0.71      0.72      2729
           1       0.75      0.77      0.76      3088

    accuracy                           0.74      5817
   macro avg       0.74      0.74      0.74      5817
weighted avg       0.74      0.74      0.74      5817


Confusion matrix:
[[1942  787]
 [ 717 2371]]
Accuracy=0.7414474815196836,Precision=0.7412014901270066,F1=0.7412167672068452,Recall=0.7414474815196836
[0.73034976 0.75079164]
[0.72086117 0.75920589]
[0.71161598 0.76781088]
'''

'''
Joint
64
aae
Start learning at 2020-11-11 10:42:50.339966
Stop learning 2020-11-11 10:42:50.675356
Elapsed learning 0:00:00.335390
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.46      0.40      0.43       427
         1.0       0.52      0.53      0.52       771
         2.0       0.51      0.38      0.43       380
         3.0       0.40      0.30      0.34       455
         4.0       0.59      0.71      0.64      1528
         5.0       0.63      0.79      0.70      1679
         6.0       0.49      0.09      0.15       394
         7.0       0.20      0.01      0.01       183

    accuracy                           0.57      5817
   macro avg       0.47      0.40      0.40      5817
weighted avg       0.54      0.57      0.54      5817


Confusion matrix:
[[ 170   74   34   19  103   27    0    0]
 [  49  411   13   70   53  172    2    1]
 [  81   20  144   47   67   14    6    1]
 [  16  122   57  135   43   78    4    0]
 [  39   33   16   14 1090  320   16    0]
 [   3   94    7   26  225 1321    2    1]
 [   8   12   11   13  243   71   35    1]
 [   0   32    3   12   31   98    6    1]]
Accuracy=0.5685061028021317,Precision=0.5423049460306901,F1=0.5374433894832644,Recall=0.5685061028021317
[0.46448087 0.51503759 0.50526316 0.40178571 0.58760108 0.62874822
 0.49295775 0.2       ]
[0.42875158 0.52390057 0.43308271 0.34134008 0.64439846 0.6989418
 0.15053763 0.0106383 ]
[0.39812646 0.53307393 0.37894737 0.2967033  0.71335079 0.78677784
 0.08883249 0.00546448]
 
 wmgan
 Start learning at 2020-11-11 10:44:05.403096
Stop learning 2020-11-11 10:44:05.753533
Elapsed learning 0:00:00.350437
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.46      0.42      0.44       427
         1.0       0.53      0.59      0.56       771
         2.0       0.50      0.43      0.46       380
         3.0       0.38      0.25      0.30       455
         4.0       0.60      0.73      0.65      1528
         5.0       0.65      0.79      0.72      1679
         6.0       0.48      0.07      0.12       394
         7.0       0.33      0.01      0.02       183

    accuracy                           0.58      5817
   macro avg       0.49      0.41      0.41      5817
weighted avg       0.56      0.58      0.55      5817


Confusion matrix:
[[ 178   79   45   20   92   11    2    0]
 [  41  458    9   56   45  160    2    0]
 [  87   28  163   38   51   11    2    0]
 [  27  160   57  112   30   67    2    0]
 [  37   32   28   17 1109  286   18    1]
 [   0   89    5   24  232 1325    2    2]
 [  14    7   16   12  269   48   27    1]
 [   1   17    4   12   31  115    1    2]]
Accuracy=0.59403372,Precision=0.5552750647394414,F1=0.5472011314864293,Recall=0.5800240673886883
[0.46233766 0.52643678 0.49847095 0.38487973 0.59655729 0.65496787
 0.48214286 0.33333333]
[0.43842365 0.55819622 0.46110325 0.3002681  0.65485681 0.71582928
 0.12       0.02116402]
[0.41686183 0.59403372 0.42894737 0.24615385 0.72578534 0.78916021
 0.06852792 0.01092896]
 
 bigan
 Start learning at 2020-11-11 10:44:52.837251
Stop learning 2020-11-11 10:44:53.188322
Elapsed learning 0:00:00.351071
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.43      0.42      0.43       427
         1.0       0.49      0.57      0.52       771
         2.0       0.53      0.43      0.48       380
         3.0       0.37      0.25      0.30       455
         4.0       0.61      0.72      0.66      1528
         5.0       0.64      0.77      0.70      1679
         6.0       0.53      0.08      0.14       394
         7.0       0.50      0.01      0.02       183

    accuracy                           0.57      5817
   macro avg       0.51      0.41      0.41      5817
weighted avg       0.56      0.57      0.54      5817


Confusion matrix:
[[ 179   81   42   30   75   19    1    0]
 [  65  438   20   65   40  141    2    0]
 [  88   29  163   41   42   14    3    0]
 [  34  161   53  115   27   65    0    0]
 [  29   38   16   15 1105  311   13    1]
 [   4  119    0   24  235 1292    4    1]
 [  14   20   11    7  244   67   31    0]
 [   0   17    1   12   33  114    4    2]]
Accuracy=0.5716004813477737,Precision=0.557449923464437,F1=0.5410504430197474,Recall=0.5716004813477737
[0.43341404 0.48504983 0.53267974 0.37216828 0.61354803 0.63865546
 0.53448276 0.5       ]
[0.42619048 0.52329749 0.47521866 0.30104712 0.66386302 0.69800108
 0.13716814 0.02139037]
[0.41920375 0.56809339 0.42894737 0.25274725 0.72316754 0.76950566
 0.0786802  0.01092896]
 
 biganqp
 Start learning at 2020-11-11 10:46:13.311662
Stop learning 2020-11-11 10:46:13.664805
Elapsed learning 0:00:00.353143
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.45      0.47      0.46       427
         1.0       0.57      0.63      0.60       771
         2.0       0.50      0.43      0.46       380
         3.0       0.41      0.33      0.36       455
         4.0       0.63      0.70      0.67      1528
         5.0       0.66      0.77      0.71      1679
         6.0       0.49      0.19      0.27       394
         7.0       0.36      0.05      0.09       183

    accuracy                           0.59      5817
   macro avg       0.51      0.45      0.45      5817
weighted avg       0.57      0.59      0.57      5817


Confusion matrix:
[[ 200   87   64   26   37   11    2    0]
 [  65  482   16   89   19   97    2    1]
 [ 111   32  162   47   14    9    4    1]
 [  42  147   61  148    9   45    2    1]
 [  20   19    8   19 1077  337   47    1]
 [   4   60    3   25  272 1294   12    9]
 [   5    3   10    4  232   61   74    5]
 [   0    9    3    6   39  109    7   10]]
Accuracy=0.592573491490459,Precision=0.5745856202856164,F1=0.5730128571295585,Recall=0.592573491490459
[0.44742729 0.57449344 0.49541284 0.40659341 0.6339023  0.65919511
 0.49333333 0.35714286]
[0.4576659  0.59875776 0.4582744  0.36141636 0.66749303 0.71059857
 0.27205882 0.09478673]
[0.46838407 0.62516213 0.42631579 0.32527473 0.70484293 0.77069684
 0.18781726 0.05464481]
 
 128
 Start learning at 2020-11-11 10:47:41.167608
Stop learning 2020-11-11 10:47:41.507859
Elapsed learning 0:00:00.340251
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.44      0.35      0.39       427
         1.0       0.50      0.51      0.50       771
         2.0       0.50      0.36      0.42       380
         3.0       0.41      0.29      0.34       455
         4.0       0.57      0.73      0.64      1528
         5.0       0.63      0.78      0.69      1679
         6.0       0.54      0.09      0.15       394
         7.0       0.40      0.01      0.02       183

    accuracy                           0.56      5817
   macro avg       0.50      0.39      0.40      5817
weighted avg       0.54      0.56      0.53      5817


Confusion matrix:
[[ 148   75   37   16  132   17    2    0]
 [  42  392   10   73   62  190    1    1]
 [  66   32  138   50   74   13    7    0]
 [  17  121   52  134   60   68    1    2]
 [  38   36   19   11 1112  297   15    0]
 [   5  102    3   22  241 1304    2    0]
 [  16   14   15    7  234   73   35    0]
 [   2   15    3   12   26  121    2    2]]
Accuracy=0.5612858861956335,Precision=0.5435770881845147,F1=0.5290674089700806,Recall=0.5612858861956335
[0.44311377 0.49809403 0.49819495 0.41230769 0.57290057 0.62602016
 0.53846154 0.4       ]
[0.38896189 0.50320924 0.42009132 0.34358974 0.64110695 0.69324827
 0.15250545 0.0212766 ]
[0.34660422 0.50843061 0.36315789 0.29450549 0.72774869 0.77665277
 0.08883249 0.01092896]
 
 wmgan
 Start learning at 2020-11-11 10:48:20.163750
Stop learning 2020-11-11 10:48:20.497850
Elapsed learning 0:00:00.334100
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.50      0.54      0.52       427
         1.0       0.58      0.68      0.62       771
         2.0       0.54      0.47      0.50       380
         3.0       0.44      0.35      0.39       455
         4.0       0.67      0.76      0.71      1528
         5.0       0.71      0.82      0.76      1679
         6.0       0.63      0.13      0.22       394
         7.0       0.55      0.03      0.06       183

    accuracy                           0.63      5817
   macro avg       0.58      0.47      0.47      5817
weighted avg       0.62      0.63      0.61      5817


Confusion matrix:
[[ 230   68   56   13   55    5    0    0]
 [  42  521    5   75   25  103    0    0]
 [  99   19  178   43   30    5    5    1]
 [  20  169   56  158   15   37    0    0]
 [  51   28   20   20 1156  236   17    0]
 [   4   80    1   31  180 1375    5    3]
 [  15    7   13    7  253   46   52    1]
 [   2   13    0   12   23  123    4    6]]
Accuracy=0.6319408629877944,Precision=0.6225829661379512,F1=0.6061673302305948,Recall=0.6319408629877944
[0.49676026 0.57569061 0.54103343 0.44011142 0.66551526 0.71243523
 0.62650602 0.54545455]
[0.51685393 0.62171838 0.50211566 0.38820639 0.70811639 0.76198393
 0.21802935 0.06185567]
[0.53864169 0.67574578 0.46842105 0.34725275 0.7565445  0.81893985
 0.1319797  0.03278689]
 
 bigan
 tart learning at 2020-11-11 10:49:05.815905
Stop learning 2020-11-11 10:49:06.151553
Elapsed learning 0:00:00.335648
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.46      0.46      0.46       427
         1.0       0.49      0.54      0.51       771
         2.0       0.51      0.39      0.44       380
         3.0       0.38      0.30      0.33       455
         4.0       0.61      0.69      0.65      1528
         5.0       0.65      0.79      0.71      1679
         6.0       0.48      0.11      0.18       394
         7.0       0.67      0.02      0.04       183

    accuracy                           0.57      5817
   macro avg       0.53      0.41      0.42      5817
weighted avg       0.56      0.57      0.55      5817


Confusion matrix:
[[ 195   79   45   16   73   15    4    0]
 [  50  419    9   93   50  149    1    0]
 [  88   34  150   51   43   10    4    0]
 [  27  160   46  135   28   56    3    0]
 [  45   39   27   14 1059  313   31    0]
 [   6   96    3   27  217 1324    4    2]
 [  12   10   16    6  246   59   45    0]
 [   1   21    0   16   26  113    2    4]]
Accuracy=0.5726319408629877,Precision=0.5615952490108731,F1=0.5463234281896534,Recall=0.5726319408629877
[0.45990566 0.48834499 0.50675676 0.37709497 0.60792193 0.64933791
 0.4787234  0.66666667]
[0.45828437 0.51442603 0.44378698 0.33210332 0.64770642 0.71221087
 0.18442623 0.04232804]
[0.45667447 0.54345006 0.39473684 0.2967033  0.69306283 0.78856462
 0.1142132  0.02185792]
 
 biganqp
Start learning at 2020-11-11 10:50:01.250704
Stop learning 2020-11-11 10:50:01.598025
Elapsed learning 0:00:00.347321
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.39      0.34      0.37       427
         1.0       0.42      0.47      0.45       771
         2.0       0.46      0.33      0.39       380
         3.0       0.35      0.24      0.28       455
         4.0       0.55      0.68      0.61      1528
         5.0       0.57      0.67      0.62      1679
         6.0       0.43      0.11      0.17       394
         7.0       0.26      0.06      0.10       183

    accuracy                           0.51      5817
   macro avg       0.43      0.36      0.37      5817
weighted avg       0.49      0.51      0.49      5817


Confusion matrix:
[[ 146   95   36   15   97   36    2    0]
 [  58  366   21   71   75  175    2    3]
 [  83   53  125   45   49   25    0    0]
 [  26  156   49  107   31   78    4    4]
 [  33   55   17   18 1038  341   24    2]
 [  16  114   12   35  341 1129   17   15]
 [   8   15    8    8  209   96   43    7]
 [   1   14    1    8   33  108    7   11]]
Accuracy=0.509712910434932,Precision=0.4895444977735696,F1=0.48615073695922356,Recall=0.509712910434932
[0.393531   0.42165899 0.46468401 0.3485342  0.55419114 0.56790744
 0.43434343 0.26190476]
[0.36591479 0.44661379 0.38520801 0.2808399  0.6104087  0.6157622
 0.17444219 0.09777778]
[0.34192037 0.47470817 0.32894737 0.23516484 0.67931937 0.67242406
 0.10913706 0.06010929]
 
 256
 aae
 Start learning at 2020-11-11 10:51:01.284702
Stop learning 2020-11-11 10:51:01.913152
Elapsed learning 0:00:00.628450
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.48      0.42      0.45       427
         1.0       0.54      0.60      0.57       771
         2.0       0.50      0.41      0.45       380
         3.0       0.42      0.30      0.35       455
         4.0       0.60      0.74      0.66      1528
         5.0       0.66      0.75      0.70      1679
         6.0       0.49      0.14      0.22       394
         7.0       0.18      0.01      0.02       183

    accuracy                           0.58      5817
   macro avg       0.48      0.42      0.43      5817
weighted avg       0.56      0.58      0.56      5817


Confusion matrix:
[[ 178   88   40   14   85   21    1    0]
 [  33  466   13   65   48  143    1    2]
 [  79   27  156   41   64    7    6    0]
 [  28  138   59  136   28   64    1    1]
 [  30   28   22   13 1138  266   31    0]
 [   9   97    6   38  252 1266    8    3]
 [  11    8   12    9  244   51   56    3]
 [   4   14    2    9   36  105   11    2]]
Accuracy=0.5841499054495445,Precision=0.5585226649828928,F1=0.5580883903064252,Recall=0.5841499054495445
[0.47849462 0.53810624 0.50322581 0.41846154 0.6005277  0.65834633
 0.48695652 0.18181818]
[0.44555695 0.56933415 0.45217391 0.34871795 0.66491382 0.70294281
 0.22003929 0.02061856]
[0.41686183 0.60440986 0.41052632 0.2989011  0.7447644  0.75402025
 0.14213198 0.01092896] 
 
 wmgan
 Start learning at 2020-11-11 10:51:43.788791
Stop learning 2020-11-11 10:51:44.402033
Elapsed learning 0:00:00.613242
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.58      0.58      0.58       427
         1.0       0.59      0.65      0.62       771
         2.0       0.57      0.50      0.53       380
         3.0       0.46      0.40      0.42       455
         4.0       0.69      0.77      0.73      1528
         5.0       0.71      0.83      0.77      1679
         6.0       0.48      0.16      0.25       394
         7.0       0.45      0.07      0.12       183

    accuracy                           0.65      5817
   macro avg       0.57      0.49      0.50      5817
weighted avg       0.63      0.65      0.63      5817


Confusion matrix:
[[ 246   62   47   18   47    6    1    0]
 [  44  500   11   92   17  104    2    1]
 [  87   28  189   46   20    7    2    1]
 [  17  149   58  181   10   36    4    0]
 [  16   23   12    8 1172  249   46    2]
 [   3   71    1   33  158 1397    8    8]
 [   7    4   14    7  240   53   65    4]
 [   1    8    2   12   24  115    8   13]]
Accuracy=0.6468970259583978,Precision=0.6277982673413575,F1=0.6255404965862807,Recall=0.6468970259583978
[0.58432304 0.59171598 0.56586826 0.4559194  0.6943128  0.71021861
 0.47794118 0.44827586]
[0.58018868 0.61881188 0.52941176 0.42488263 0.72885572 0.76631925
 0.24528302 0.12264151]
[0.57611241 0.64850843 0.49736842 0.3978022  0.76701571 0.83204288
 0.16497462 0.07103825]
 
 bigan
 Start learning at 2020-11-11 10:52:35.444085
Stop learning 2020-11-11 10:52:36.094892
Elapsed learning 0:00:00.650807
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.51      0.52      0.51       427
         1.0       0.56      0.62      0.59       771
         2.0       0.51      0.47      0.49       380
         3.0       0.41      0.31      0.35       455
         4.0       0.68      0.74      0.71      1528
         5.0       0.67      0.80      0.73      1679
         6.0       0.58      0.19      0.29       394
         7.0       0.24      0.02      0.04       183

    accuracy                           0.61      5817
   macro avg       0.52      0.46      0.46      5817
weighted avg       0.59      0.61      0.59      5817


Confusion matrix:
[[ 220   68   54   20   48   15    2    0]
 [  51  480   18   78   20  123    1    0]
 [  98   25  179   46   23    6    2    1]
 [  27  145   65  140   15   59    2    2]
 [  22   21   22   10 1132  289   31    1]
 [   6  100    1   26  189 1340   11    6]
 [   6   10   13   11  215   61   75    3]
 [   2   15    1   10   26  119    6    4]]
Accuracy=0.6137184115523465,Precision=0.5932365438546838,F1=0.591144163813998,Recall=0.6137184115523465
[0.50925926 0.55555556 0.50708215 0.41055718 0.67865707 0.66600398
 0.57692308 0.23529412]
[0.51222352 0.58715596 0.48840382 0.35175879 0.70838548 0.72609049
 0.28625954 0.04      ]
[0.51522248 0.62256809 0.47105263 0.30769231 0.7408377  0.7980941
 0.19035533 0.02185792]
 
 biganqp
 Start learning at 2020-11-11 10:53:17.770930
Stop learning 2020-11-11 10:53:18.391867
Elapsed learning 0:00:00.620937
Classification report for classifier LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001):
              precision    recall  f1-score   support

         0.0       0.41      0.38      0.40       427
         1.0       0.44      0.47      0.46       771
         2.0       0.43      0.31      0.36       380
         3.0       0.41      0.29      0.34       455
         4.0       0.53      0.67      0.59      1528
         5.0       0.58      0.66      0.61      1679
         6.0       0.40      0.09      0.15       394
         7.0       0.18      0.03      0.05       183

    accuracy                           0.51      5817
   macro avg       0.42      0.36      0.37      5817
weighted avg       0.49      0.51      0.48      5817


Confusion matrix:
[[ 164   89   34   13   92   28    5    2]
 [  74  366   16   66   67  177    2    3]
 [  73   44  118   37   75   27    5    1]
 [  27  126   61  134   34   68    2    3]
 [  31   52   23   20 1025  347   28    2]
 [  17  128    9   34  376 1100    6    9]
 [  10   11   12   12  236   74   36    3]
 [   2   20    1   12   47   91    5    5]]
Accuracy=0.5067904418084923,Precision=0.48536778244632933,F1=0.48306147054249027,Recall=0.5067904418084923
[0.4120603  0.43779904 0.43065693 0.40853659 0.52510246 0.57531381
 0.40449438 0.17857143]
[0.39757576 0.45550716 0.36085627 0.34227331 0.58908046 0.61264272
 0.14906832 0.04739336]
[0.38407494 0.47470817 0.31052632 0.29450549 0.67081152 0.65515188
 0.09137056 0.0273224 ]
'''

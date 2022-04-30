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
import argparse
import os
output_root = "/data1/JCST/results/encodes"

parser = argparse.ArgumentParser()
parser.add_argument('--iteration', type=int, default=10000, help='num of features')
parser.add_argument('--image_size', type=int, default=512, help='size of images')
parser.add_argument("--num_exp",type=int,default=21,help="id of experiments")
parser.add_argument("--nlat",type=int,default=512,help="id of experiments")
parser.add_argument("--name",type=str,default="mmds",help="algorithm")
parser.add_argument("--classifer",type=str,default="svm",help="classifier")

opt = parser.parse_args()
print(opt)
base_path = os.path.join(os.path.join(output_root,"exp%d" % opt.image_size),"%d" % opt.num_exp)
aae_512_gla_train = np.load(os.path.join(os.path.join(base_path, "aaes"), "train_%d.npy" % opt.iteration))
aae_512_gla_test = np.load(os.path.join(os.path.join(base_path, "aaes"), "test_%d.npy" % opt.iteration))
bigan_512_gla_train = np.load(os.path.join(os.path.join(base_path, "bigan"), "train_%d.npy" % opt.iteration))
bigan_512_gla_test = np.load(os.path.join(os.path.join(base_path, "bigan"), "test_%d.npy" % opt.iteration))
biganqp_512_gla_train = np.load(os.path.join(os.path.join(base_path, "biganqp"), "train_%d.npy" % opt.iteration))
biganqp_512_gla_test = np.load(os.path.join(os.path.join(base_path, "biganqp"), "test_%d.npy" % opt.iteration))
wmgan_512_gla_train = np.load(os.path.join(os.path.join(base_path, "mmds"), "train_%d.npy" % opt.iteration))
wmgan_512_gla_test = np.load(os.path.join(os.path.join(base_path, "mmds"), "test_%d.npy" % opt.iteration))

X_aae_train = aae_512_gla_train[:,0:-1]
Y_aae_train = aae_512_gla_train[:,-1]
X_aae_test = aae_512_gla_test[:,0:-1]
Y_aae_test = aae_512_gla_test[:,-1]

print(X_aae_train.shape)

X_qp_train = biganqp_512_gla_train[:,0:-1]
Y_qp_train = biganqp_512_gla_train[:,-1]
X_qp_test = biganqp_512_gla_test[:,0:-1]
Y_qp_test = biganqp_512_gla_test[:,-1]

X_bi_train = bigan_512_gla_train[:,0:-1]
Y_bi_train = bigan_512_gla_train[:,-1]
X_bi_test = bigan_512_gla_test[:,0:-1]
Y_bi_test = bigan_512_gla_test[:,-1]

X_wm_train = wmgan_512_gla_train[:,0:-1]
Y_wm_train = wmgan_512_gla_train[:,-1]
X_wm_test = wmgan_512_gla_test[:,0:-1]
Y_wm_test = wmgan_512_gla_test[:,-1]

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


def svm_classifer(name="mmds"):
    if 'mmds' in name:
        x_train = x_wm_train
        y_train = y_wm_train
        x_test = X_wm_test
        y_test = Y_wm_test
    elif 'biganqp' in name:
        x_train = x_qp_train
        y_train = y_qp_train
        x_test = X_qp_test
        y_test = Y_qp_test
    elif 'bigan' in name and 'qp' not in name:
        x_train = x_bi_train
        y_train = y_bi_train
        x_test = X_bi_test
        y_test = Y_bi_test
    elif 'aae' in name:
        x_train = x_aae_train
        y_train = y_aae_train
        x_test = X_aae_test
        y_test = Y_aae_test
    else:
        return

    '''
    Glasses
    '''

    param_C = 5
    param_gamma = 0.05
    # classifier = svm.SVC(C=param_C, gamma=param_gamma, max_iter=3000)
    # classifier = svm.SVC(C=param_C,gamma=param_gamma,max_iter=3000)
    classifier = svm.SVC(C=param_C,gamma=param_gamma,max_iter=1000)
    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    classifier.fit(x_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))
    '''
    Joint
    '''

    expected = y_test
    predicted = classifier.predict(x_test)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)

    print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),
                                                            metrics.precision_score(expected, predicted,
                                                                                    average='weighted'),
                                                            metrics.f1_score(expected, predicted, average='weighted'),
                                                            metrics.recall_score(expected, predicted,
                                                                                 average='weighted')))
    print(metrics.precision_score(expected, predicted, average=None))
    print(metrics.f1_score(expected, predicted, average=None))
    print(metrics.recall_score(expected, predicted, average=None))

def lda_classifier(name='mmds'):
    if 'mmds' in name:
        x_train = x_wm_train
        y_train = y_wm_train
        x_test = X_wm_test
        y_test = Y_wm_test
    elif 'biganqp' in name:
        x_train = x_qp_train
        y_train = y_qp_train
        x_test = X_qp_test
        y_test = Y_qp_test
    elif 'bigan' in name and 'qp' not in name:
        x_train = x_bi_train
        y_train = y_bi_train
        x_test = X_bi_test
        y_test = Y_bi_test
    elif 'aae' in name:
        x_train = x_aae_train
        y_train = y_aae_train
        x_test = X_aae_test
        y_test = Y_aae_test
    else:
        return

    param_C = 5
    param_gamma = 0.05

    lda = LinearDiscriminantAnalysis()
    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    # classifier.fit(X_train, y_train)
    # lda.fit(X_train, y_train)
    lda.fit(x_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

    expected = y_test
    predicted = lda.predict(x_test)

    print("Classification report for classifier %s:\n%s\n"
          % (lda, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)

    # plot_confusion_matrix(cm)
    # metrics.precision_score(expected, predicted)
    # metrics.f1_score(expected, predicted)
    # metrics.recall_score(expected, predicted)
    # metrics.st

    print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),
                                                            metrics.precision_score(expected, predicted,
                                                                                    average='weighted'),
                                                            metrics.f1_score(expected, predicted, average='weighted'),
                                                            metrics.recall_score(expected, predicted,
                                                                                 average='weighted')))
    print(metrics.precision_score(expected, predicted, average=None))
    print(metrics.f1_score(expected, predicted, average=None))
    print(metrics.recall_score(expected, predicted, average=None))


def rf_classifier(name='mmds'):
    if 'mmds' in name:
        x_train = x_wm_train
        y_train = y_wm_train
        x_test = X_wm_test
        y_test = Y_wm_test
    elif 'biganqp' in name:
        x_train = x_qp_train
        y_train = y_qp_train
        x_test = X_qp_test
        y_test = Y_qp_test
    elif 'bigan' in name and 'qp' not in name:
        x_train = x_bi_train
        y_train = y_bi_train
        x_test = X_bi_test
        y_test = Y_bi_test
    elif 'aae' in name:
        x_train = x_aae_train
        y_train = y_aae_train
        x_test = X_aae_test
        y_test = Y_aae_test
    else:
        return

    param_C = 5
    param_gamma = 0.05
    # classifier = svm.SVC(C=param_C,gamma=param_gamma,max_iter=500)
    # random_forest = RandomForestClassifier()
    # ada = AdaBoostClassifier(learning_rate=0.1,n_estimators=400)
    # ada = AdaBoostClassifier(learning_rate=0.1,n_estimators=300)
    # lda = LinearDiscriminantAnalysis()

    random_forest = RandomForestClassifier(n_estimators=270,min_samples_split=6,min_samples_leaf=2)
    # random_forest = RandomForestClassifier(n_estimators=90,min_samples_split=20,min_samples_leaf=10)
    # random_forest = RandomForestClassifier(n_estimators=100)
    # random_forest = RandomForestClassifier(n_estimators=120)
    # random_forest = RandomForestClassifier(n_estimators=175,min_samples_split=10,min_samples_leaf=5)

    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    # classifier.fit(X_train, y_train)
    # randomfit(X_train, y_train)
    random_forest.fit(x_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

    expected = y_test
    predicted = random_forest.predict(x_test)


    print("Classification report for classifier %s:\n%s\n"
          % (random_forest, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)

    # plot_confusion_matrix(cm)
    # metrics.precision_score(expected, predicted)
    # metrics.f1_score(expected, predicted)
    # metrics.recall_score(expected, predicted)
    # metrics.st

    print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),
                                                            metrics.precision_score(expected, predicted,
                                                                                    average='weighted'),
                                                            metrics.f1_score(expected, predicted, average='weighted'),
                                                            metrics.recall_score(expected, predicted,
                                                                                 average='weighted')))
    print(metrics.precision_score(expected, predicted, average=None))
    print(metrics.f1_score(expected, predicted, average=None))
    print(metrics.recall_score(expected, predicted, average=None))

    # random_forest = RandomForestClassifier(n_estimators=270,min_samples_split=2,min_samples_leaf=1)
    # # random_forest = RandomForestClassifier(n_estimators=90,min_samples_split=20,min_samples_leaf=10)
    # # random_forest = RandomForestClassifier(n_estimators=250)
    # # random_forest = RandomForestClassifier(n_estimators=155,min_samples_split=10,min_samples_leaf=5)

def gbdt_classifer(name='mmds'):
    if 'mmds' in name:
        x_train = x_wm_train
        y_train = y_wm_train
        x_test = X_wm_test
        y_test = Y_wm_test
    elif 'biganqp' in name:
        x_train = x_qp_train
        y_train = y_qp_train
        x_test = X_qp_test
        y_test = Y_qp_test
    elif 'bigan' in name and 'qp' not in name:
        x_train = x_bi_train
        y_train = y_bi_train
        x_test = X_bi_test
        y_test = Y_bi_test
    elif 'aae' in name:
        x_train = x_aae_train
        y_train = y_aae_train
        x_test = X_aae_test
        y_test = Y_aae_test
    else:
        return
    param_C = 5
    param_gamma = 0.05
    # classifier = svm.SVC(C=param_C,gamma=param_gamma,max_iter=500)
    # random_forest = RandomForestClassifier()
    # ada = AdaBoostClassifier(learning_rate=0.1,n_estimators=400)
    # ada = AdaBoostClassifier(learning_rate=0.1,n_estimators=300)
    # lda = LinearDiscriminantAnalysis()
    gbdt = GradientBoostingClassifier(n_estimators=115, learning_rate=0.1)
    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    # classifier.fit(X_train, y_train)
    # ada.fit(X_train, y_train)
    gbdt.fit(x_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))
    #
    #
    ########################################################
    # Now predict the value of the test
    expected = y_test
    predicted = gbdt.predict(x_test)

    # show_some_digits(X_test,predicted,title_text="Predicted {}")

    print("Classification report for classifier %s:\n%s\n"
          % (gbdt, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)

    # plot_confusion_matrix(cm)
    # metrics.precision_score(expected, predicted)
    # metrics.f1_score(expected, predicted)
    # metrics.recall_score(expected, predicted)
    # metrics.st

    print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),
                                                            metrics.precision_score(expected, predicted,
                                                                                    average='weighted'),
                                                            metrics.f1_score(expected, predicted, average='weighted'),
                                                            metrics.recall_score(expected, predicted,
                                                                                 average='weighted')))
    print(metrics.precision_score(expected, predicted, average=None))
    print(metrics.f1_score(expected, predicted, average=None))
    print(metrics.recall_score(expected, predicted, average=None))

def ada_classifier(name='mmds'):
    if 'mmds' in name:
        x_train = x_wm_train
        y_train = y_wm_train
        x_test = X_wm_test
        y_test = Y_wm_test
    elif 'biganqp' in name:
        x_train = x_qp_train
        y_train = y_qp_train
        x_test = X_qp_test
        y_test = Y_qp_test
    elif 'bigan' in name and 'qp' not in name:
        x_train = x_bi_train
        y_train = y_bi_train
        x_test = X_bi_test
        y_test = Y_bi_test
    elif 'aae' in name:
        x_train = x_aae_train
        y_train = y_aae_train
        x_test = X_aae_test
        y_test = Y_aae_test
    else:
        return
    ada = AdaBoostClassifier(learning_rate=0.25,n_estimators=480)
    # ada = AdaBoostClassifier(learning_rate=0.13
    #                          , n_estimators=400)
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))

    ada.fit(x_train, y_train)
    # We learn the digits on train part
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
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))


    expected = y_test
    predicted = ada.predict(x_test)


    print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),
                                                            metrics.precision_score(expected, predicted,
                                                                                    average='weighted'),
                                                            metrics.f1_score(expected, predicted, average='weighted'),
                                                            metrics.recall_score(expected, predicted,
                                                                                 average='weighted')))
    print(metrics.precision_score(expected, predicted, average=None))
    print(metrics.f1_score(expected, predicted, average=None))
    print(metrics.recall_score(expected, predicted, average=None))

if __name__ == '__main__':
    if 'svm' in opt.classifer:
        svm_classifer(opt.name)
    elif 'ada' in opt.classifer:
        ada_classifier(opt.name)
    elif 'lda' in opt.classifer:
        lda_classifier(opt.name)
    elif 'gbdt'in opt.classifer:
        gbdt_classifer(opt.name)
    elif 'rf' in opt.classifer:
        rf_classifier(opt.name)


# #          lda      random_forest   ada        SVM
# # BIGAE   0.6642    0.4372        0.4210      0.4875
# # BIGAN   0.6367    0.4160        0.3512      0.2826
# # BIGANQP 0.4847    0.3481        0.3171      0.3023
# # AAES    0.6412    0.4076        0.3114      0.1734


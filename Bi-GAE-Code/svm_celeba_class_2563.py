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
parser.add_argument('--image_size', type=int, default=256, help='size of images')
parser.add_argument("--num_exp",type=int,default=22,help="id of experiments")
parser.add_argument("--nlat",type=int,default=256,help="id of experiments")
parser.add_argument("--name",type=str,default="mmds",help="algorithm")
parser.add_argument("--classifer",type=str,default="svm",help="classifier")

opt = parser.parse_args()
print(opt)

classifer_output_root = "/data1/JCST/results/classifer"
os.makedirs(classifer_output_root,exist_ok=True)

classifer_output_size = os.path.join(classifer_output_root,"exp%d" % opt.image_size)
os.makedirs(classifer_output_size,exist_ok=True)

num_exp_dir = os.path.join(classifer_output_size,"%d" % opt.num_exp)
os.makedirs(num_exp_dir,exist_ok=True)

def svm_classifer(x_train,y_train,x_test,y_test,name='mmds',iteration=10000):
    '''
    Glasses
    '''
    # with open('./result_svm/data.txt', 'w') as f:
    basedirs = os.path.join(num_exp_dir,name)
    if not os.path.exists(basedirs):
        os.makedirs(basedirs,exist_ok=True)
    classifer_dir = os.path.join(basedirs,"svm")
    if not os.path.exists(classifer_dir):
        os.makedirs(classifer_dir,exist_ok=True)
    f = open(os.path.join(classifer_dir,"%d.txt" % iteration),"a+")
    param_C = 5
    param_gamma = 0.05
    # classifier = svm.SVC(C=param_C, gamma=param_gamma, max_iter=3000)
    # classifier = svm.SVC(C=param_C,gamma=param_gamma,max_iter=3000)
    classifier = svm.SVC(C=param_C,gamma=param_gamma,max_iter=2000)
    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    print('Start learning at {}'.format(str(start_time)),file = f)
    classifier.fit(x_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)),file = f)
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)),file = f)
    '''
    Joint
    '''
    expected = y_test
    predicted = classifier.predict(x_test)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)),file = f)

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm,file = f)

    print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),
                                                            metrics.precision_score(expected, predicted,
                                                                                    average='weighted'),
                                                            metrics.f1_score(expected, predicted, average='weighted'),
                                                            metrics.recall_score(expected, predicted,
                                                                                 average='weighted')),file = f)
    print(metrics.precision_score(expected, predicted, average=None),file = f)
    print(metrics.f1_score(expected, predicted, average=None),file = f)
    print(metrics.recall_score(expected, predicted, average=None),file = f)

    print("svm: %s iteration: %d finished at %f" % (name,iteration,time.time()))

def lda_classifier(x_train,y_train,x_test,y_test,name='mmds',iteration=10000):
    param_C = 5
    param_gamma = 0.05

    # with open('./result_svm/data.txt', 'w') as f:
    basedirs = os.path.join(num_exp_dir, name)
    if not os.path.exists(basedirs):
        os.makedirs(basedirs, exist_ok=True)
    classifer_dir = os.path.join(basedirs, "lda")
    if not os.path.exists(classifer_dir):
        os.makedirs(classifer_dir, exist_ok=True)
    f = open(os.path.join(classifer_dir, "%d.txt" % iteration), "a+")

    lda = LinearDiscriminantAnalysis()
    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    print('Start learning at {}'.format(str(start_time)), file=f)
    # classifier.fit(X_train, y_train)
    # lda.fit(X_train, y_train)
    lda.fit(x_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)), file=f)
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)), file=f)

    expected = y_test
    predicted = lda.predict(x_test)

    print("Classification report for classifier %s:\n%s\n"
          % (lda, metrics.classification_report(expected, predicted)),file=f)


    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm, file=f)

    print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),
                                                            metrics.precision_score(expected, predicted,
                                                                                    average='weighted'),
                                                            metrics.f1_score(expected, predicted, average='weighted'),
                                                            metrics.recall_score(expected, predicted,
                                                                                 average='weighted')), file=f)
    print(metrics.precision_score(expected, predicted, average=None), file=f)
    print(metrics.f1_score(expected, predicted, average=None), file=f)
    print(metrics.recall_score(expected, predicted, average=None), file=f)

    print("lda: %s iteration: %d finished at %f" % (name, iteration, time.time()))


def rf_classifier(x_train,y_train,x_test,y_test,name='mmds',iteration=10000):
    # with open('./result_svm/data.txt', 'w') as f:
    basedirs = os.path.join(num_exp_dir, name)
    if not os.path.exists(basedirs):
        os.makedirs(basedirs, exist_ok=True)
    classifer_dir = os.path.join(basedirs, "rf")
    if not os.path.exists(classifer_dir):
        os.makedirs(classifer_dir, exist_ok=True)
    f = open(os.path.join(classifer_dir, "%d.txt" % iteration), "a+")

    param_C = 5
    param_gamma = 0.05
    # classifier = svm.SVC(C=param_C,gamma=param_gamma,max_iter=500)
    # random_forest = RandomForestClassifier()
    # ada = AdaBoostClassifier(learning_rate=0.1,n_estimators=400)
    # ada = AdaBoostClassifier(learning_rate=0.1,n_estimators=300)
    # lda = LinearDiscriminantAnalysis()

    # random_forest = RandomForestClassifier(n_estimators=270,min_samples_split=6,min_samples_leaf=2)
    # random_forest = RandomForestClassifier(n_estimators=90,min_samples_split=20,min_samples_leaf=10)
    # random_forest = RandomForestClassifier(n_estimators=100)
    # random_forest = RandomForestClassifier(n_estimators=120)
    random_forest = RandomForestClassifier(n_estimators=175,min_samples_split=10,min_samples_leaf=5)

    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    print('Start learning at {}'.format(str(start_time)), file=f)
    # classifier.fit(X_train, y_train)
    # randomfit(X_train, y_train)
    random_forest.fit(x_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)), file=f)
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)), file=f)

    expected = y_test
    predicted = random_forest.predict(x_test)

    print("Classification report for classifier %s:\n%s\n"
          % (random_forest, metrics.classification_report(expected, predicted)), file=f)

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm, file=f)

    print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),
                                                            metrics.precision_score(expected, predicted,
                                                                                    average='weighted'),
                                                            metrics.f1_score(expected, predicted, average='weighted'),
                                                            metrics.recall_score(expected, predicted,
                                                                                 average='weighted')), file=f)
    print(metrics.precision_score(expected, predicted, average=None), file=f)
    print(metrics.f1_score(expected, predicted, average=None), file=f)
    print(metrics.recall_score(expected, predicted, average=None), file=f)

    print("rf: %s iteration: %d finished at %f" % (name, iteration, time.time()))

    # random_forest = RandomForestClassifier(n_estimators=270,min_samples_split=2,min_samples_leaf=1)
    # # random_forest = RandomForestClassifier(n_estimators=90,min_samples_split=20,min_samples_leaf=10)
    # # random_forest = RandomForestClassifier(n_estimators=250)
    # # random_forest = RandomForestClassifier(n_estimators=155,min_samples_split=10,min_samples_leaf=5)

def gbdt_classifer(x_train,y_train,x_test,y_test,name='mmds',iteration=10000):
    # with open('./result_svm/data.txt', 'w') as f:
    basedirs = os.path.join(num_exp_dir, name)
    if not os.path.exists(basedirs):
        os.makedirs(basedirs, exist_ok=True)
    classifer_dir = os.path.join(basedirs, "gbdt")
    if not os.path.exists(classifer_dir):
        os.makedirs(classifer_dir, exist_ok=True)
    f = open(os.path.join(classifer_dir, "%d.txt" % iteration), "a+")

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
    print('Start learning at {}'.format(str(start_time)), file=f)
    # classifier.fit(X_train, y_train)
    # ada.fit(X_train, y_train)
    gbdt.fit(x_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)), file=f)
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)), file=f)
    #
    #
    ########################################################
    # Now predict the value of the test
    expected = y_test
    predicted = gbdt.predict(x_test)

    # show_some_digits(X_test,predicted,title_text="Predicted {}")

    print("Classification report for classifier %s:\n%s\n"
          % (gbdt, metrics.classification_report(expected, predicted)), file=f)

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm, file=f)

    print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),
                                                            metrics.precision_score(expected, predicted,
                                                                                    average='weighted'),
                                                            metrics.f1_score(expected, predicted, average='weighted'),
                                                            metrics.recall_score(expected, predicted,
                                                                                 average='weighted')), file=f)
    print(metrics.precision_score(expected, predicted, average=None), file=f)
    print(metrics.f1_score(expected, predicted, average=None), file=f)
    print(metrics.recall_score(expected, predicted, average=None), file=f)

    print("gbdt: %s iteration: %d finished at %f" % (name, iteration, time.time()))

def ada_classifier(x_train,y_train,x_test,y_test,name='mmds',iteration=10000):
    # with open('./result_svm/data.txt', 'w') as f:
    basedirs = os.path.join(num_exp_dir, name)
    if not os.path.exists(basedirs):
        os.makedirs(basedirs, exist_ok=True)
    classifer_dir = os.path.join(basedirs, "ada")
    if not os.path.exists(classifer_dir):
        os.makedirs(classifer_dir, exist_ok=True)
    f = open(os.path.join(classifer_dir, "%d.txt" % iteration), "a+")

    ada = AdaBoostClassifier(learning_rate=0.25,n_estimators=480)
    # ada = AdaBoostClassifier(learning_rate=0.13
    #                          , n_estimators=400)
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    print('Start learning at {}'.format(str(start_time)), file=f)

    ada.fit(x_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)), file=f)
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)), file=f)


    expected = y_test
    predicted = ada.predict(x_test)

    print("Classification report for classifier %s:\n%s\n"
          % (ada, metrics.classification_report(expected, predicted)), file=f)

    cm = metrics.confusion_matrix(expected, predicted)

    print("Confusion matrix:\n%s" % cm, file=f)

    print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),
                                                            metrics.precision_score(expected, predicted,
                                                                                    average='weighted'),
                                                            metrics.f1_score(expected, predicted, average='weighted'),
                                                            metrics.recall_score(expected, predicted,
                                                                                 average='weighted')), file=f)
    print(metrics.precision_score(expected, predicted, average=None), file=f)
    print(metrics.f1_score(expected, predicted, average=None), file=f)
    print(metrics.recall_score(expected, predicted, average=None), file=f)

    print("ada %s iteration: %d finished at %f" % (name, iteration, time.time()))

def main(iteration,names = [],classifers=[]):
    base_path = os.path.join(os.path.join(output_root, "exp%d" % opt.image_size), "%d" % opt.num_exp)
    nommd_256_gla_train = np.load(os.path.join(os.path.join(base_path, "nommd"), "train_%d.npy" % iteration))
    nommd_256_gla_test = np.load(os.path.join(os.path.join(base_path, "nommd"), "valid_%d.npy" % iteration))
    nossim_256_gla_train = np.load(os.path.join(os.path.join(base_path, "nossim"), "train_%d.npy" % iteration))
    nossim_256_gla_test = np.load(os.path.join(os.path.join(base_path, "nossim"), "valid_%d.npy" % iteration))
    mmd2mse_256_gla_train = np.load(os.path.join(os.path.join(base_path, "mmd2mse"), "train_%d.npy" % iteration))
    mmd2mse_256_gla_test = np.load(os.path.join(os.path.join(base_path, "mmd2mse"), "valid_%d.npy" % iteration))
    percept_256_gla_train = np.load(os.path.join(os.path.join(base_path, "percept"), "train_%d.npy" % iteration))
    percept_256_gla_test = np.load(os.path.join(os.path.join(base_path, "percept"), "valid_%d.npy" % iteration))
    wmgan_256_gla_train = np.load(os.path.join(os.path.join(base_path, "mmds"), "train_%d.npy" % iteration))
    wmgan_256_gla_test = np.load(os.path.join(os.path.join(base_path, "mmds"), "valid_%d.npy" % iteration))

    X_nossim_train = nossim_256_gla_train[:, 0:-1]
    Y_nossim_train = nossim_256_gla_train[:, -1]
    X_nossim_test = nossim_256_gla_test[:, 0:-1]
    Y_nossim_test = nossim_256_gla_test[:, -1]

    print(X_nossim_train.shape)

    X_nommd_train = nommd_256_gla_train[:, 0:-1]
    Y_nommd_train = nommd_256_gla_train[:, -1]
    X_nommd_test = nommd_256_gla_test[:, 0:-1]
    Y_nommd_test = nommd_256_gla_test[:, -1]

    X_mmd2mse_train = mmd2mse_256_gla_train[:, 0:-1]
    Y_mmd2mse_train = mmd2mse_256_gla_train[:, -1]
    X_mmd2mse_test = mmd2mse_256_gla_test[:, 0:-1]
    Y_mmd2mse_test = mmd2mse_256_gla_test[:, -1]

    X_percept_train = percept_256_gla_train[:, 0:-1]
    Y_percept_train = percept_256_gla_train[:, -1]
    X_percept_test = percept_256_gla_test[:, 0:-1]
    Y_percept_test = percept_256_gla_test[:, -1]

    X_wm_train = wmgan_256_gla_train[:, 0:-1]
    Y_wm_train = wmgan_256_gla_train[:, -1]
    X_wm_test = wmgan_256_gla_test[:, 0:-1]
    Y_wm_test = wmgan_256_gla_test[:, -1]


    from sklearn.model_selection import train_test_split
    x_nossim_train, x_nossim_valid, y_nossim_train, y_nossim_valid = train_test_split(X_nossim_train, Y_nossim_train,
                                                                                      test_size=0.05, random_state=42)
    print(x_nossim_train.shape)
    x_wm_train, x_wm_valid, y_wm_train, y_wm_valid = train_test_split(X_wm_train, Y_wm_train, test_size=0.05,
                                                                      random_state=42)
    x_nommd_train, x_nommd_valid, y_nommd_train, y_nommd_valid = train_test_split(X_nommd_train, Y_nommd_train,
                                                                                  test_size=0.05, random_state=42)
    x_mmd2mse_train, x_mmd2mse_valid, y_mmd2mse_train, y_mmd2mse_valid = train_test_split(X_mmd2mse_train,
                                                                                          Y_mmd2mse_train,
                                                                                          test_size=0.05,
                                                                                          random_state=42)
    x_percept_train, x_percept_valid, y_percept_train, y_percept_valid = train_test_split(X_percept_train,
                                                                                          Y_percept_train,
                                                                                          test_size=0.05,
                                                                                          random_state=42)
    print(sum(Y_wm_test))
    print(sum(Y_wm_train))

    if not names or not classifers:
        return
    else:
        for name in names:
            if 'mmds' in name:
                x_train = x_wm_train
                y_train = y_wm_train
                x_test = X_wm_test
                y_test = Y_wm_test
            elif 'nossim' in name:
                x_train = x_nossim_train
                y_train = y_nossim_train
                x_test = X_nossim_test
                y_test = Y_nossim_test
            elif 'nommd' in name:
                x_train = x_nommd_train
                y_train = y_nommd_train
                x_test = X_nommd_test
                y_test = Y_nommd_test
            elif 'mmd2mse' in name:
                x_train = x_mmd2mse_train
                y_train = y_mmd2mse_train
                x_test = X_mmd2mse_test
                y_test = Y_mmd2mse_test
            elif 'percept' in name:
                x_train = x_percept_train
                y_train = y_percept_train
                x_test = X_percept_test
                y_test = Y_percept_test
            else:
                continue

            for classifer in classifers:
                if 'svm' in classifer:
                    svm_classifer(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,name=name,iteration=iteration)
                elif 'ada' in classifer:
                    ada_classifier(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,name=name,iteration=iteration)
                elif 'lda' in classifer:
                    lda_classifier(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,name=name,iteration=iteration)
                elif 'gbdt' in classifer:
                    gbdt_classifer(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,name=name,iteration=iteration)
                elif 'rf' in classifer:
                    rf_classifier(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,name=name,iteration=iteration)

if __name__ == '__main__':
    base_path = os.path.join(os.path.join(output_root, "exp%d" % opt.image_size), "%d" % opt.num_exp)
    files = os.listdir(base_path)
    names = []
    aim_iterations = []
    for file in files:
        if os.path.isdir(os.path.join(base_path,file)):
            name0 = file
            names.append(name0)
            # os.path.join( ,name0)
            aim_path = os.path.join(base_path,file)
            aim_files = os.listdir(aim_path)
            for item in aim_files:
                if "train_" in item and ".npy" in item:
                    item_split = item.split(".")[0].strip().split("_")
                    aim_iteration = int(item_split[-1].strip())
                    aim_iterations.append(aim_iteration)
        else:
            continue

    aim_iterations = list(set(aim_iterations))
    print(aim_iterations)
    print(names)
    aim_iterations = [17000, 19000, 5000, 13000]
    aim_iterations2 = [15000, 14000, 18000, 11000]
    aim_iterations3 = [6000, 10000, 7000]
    for aim_it in aim_iterations3:
        main(aim_it, names=names[:], classifers=['lda', 'svm', 'rf', 'ada'])





# #          lda      random_forest   ada        SVM
# # BIGAE   0.6642    0.4372        0.4210      0.4875
# # BIGAN   0.6367    0.4160        0.3512      0.2826
# # BIGANQP 0.4847    0.3481        0.3171      0.3023
# # AAES    0.6412    0.4076        0.3114      0.1734


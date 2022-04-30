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

base_images = np.load("data/aae_20000_train_data.npy")
base_targets = np.load("data/aae_20000_train_label.npy",allow_pickle=True)

test_images = np.load("data/aae_20000_test_data.npy")
test_labels = np.load("data/aae_20000_test_label.npy")
print(base_images[0])
# np.save("targets.npy",targets)
# np.save("images.npy",images)

print(base_images.shape)

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
X_data = base_images
Y = base_targets
# #split data to train and test
# #from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.09, random_state=42)
print(X_train.shape)

################ Classifier with good params ###########
# Create a classifier: a support vector classifier

param_C = 5
param_gamma = 0.05
# classifier = svm.SVC(C=param_C,gamma=param_gamma,max_iter=500)
# random_forest = RandomForestClassifier()
# ada = AdaBoostClassifier(learning_rate=0.1,n_estimators=400)
# ada = AdaBoostClassifier(learning_rate=0.1,n_estimators=300)
# lda = LinearDiscriminantAnalysis()
gbdt = GradientBoostingClassifier(n_estimators=115,learning_rate=0.1)
#We learn the digits on train part
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
# classifier.fit(X_train, y_train)
# ada.fit(X_train, y_train)
gbdt.fit(X_train, y_train)
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))
#
#
########################################################
# Now predict the value of the test
expected = test_labels
predicted = gbdt.predict(test_images)

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

print("Accuracy={},Precision={},F1={},Recall={}".format(metrics.accuracy_score(expected, predicted),metrics.precision_score(expected, predicted,average='weighted'),metrics.f1_score(expected, predicted,average='weighted'),metrics.recall_score(expected, predicted,average='weighted')))
print(metrics.precision_score(expected, predicted,average=None))
print(metrics.f1_score(expected, predicted,average=None))
print(metrics.recall_score(expected, predicted,average=None))

#base
'''
Elapsed learning 0:29:46.826336
Classification report for classifier GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False):
              precision    recall  f1-score   support

           0       0.97      0.99      0.98       980
           1       0.98      0.99      0.98      1135
           2       0.95      0.93      0.94      1032
           3       0.93      0.94      0.93      1010
           4       0.95      0.95      0.95       982
           5       0.94      0.92      0.93       892
           6       0.96      0.95      0.96       958
           7       0.95      0.92      0.94      1028
           8       0.92      0.93      0.92       974
           9       0.91      0.93      0.92      1009

    accuracy                           0.95     10000
   macro avg       0.95      0.94      0.94     10000
weighted avg       0.95      0.95      0.95     10000


Confusion matrix:
[[ 966    0    1    0    0    3    4    1    5    0]
 [   0 1119    2    1    1    1    4    0    7    0]
 [   7    0  958   16    7    1    7   12   19    5]
 [   1    0   12  948    0   13    2   11   16    7]
 [   0    0    5    0  933    1    4    1    5   33]
 [   4    1    2   25    4  823   10    4   11    8]
 [   8    4    1    0    9   14  914    1    7    0]
 [   2    7   22    7    4    2    0  950    5   29]
 [   6    4    5   13    7    8    5    6  907   13]
 [   7    8    2   14   20    5    0   10    8  935]]
Accuracy=0.9453,Precision=0.9454306956937331,F1=0.9452839657012164,Recall=0.9453
[0.96503497 0.97900262 0.94851485 0.92578125 0.94720812 0.94489093
 0.96210526 0.95381526 0.91616162 0.90776699]
[0.97526502 0.98244074 0.93829579 0.93215339 0.94865277 0.93363585
 0.95807128 0.93873518 0.92362525 0.91711623]
[0.98571429 0.98590308 0.92829457 0.93861386 0.95010183 0.92264574
 0.95407098 0.92412451 0.9312115  0.92666006]
'''

#wmgan
'''
Elapsed learning 0:19:04.062149
Classification report for classifier GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=115,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False):
              precision    recall  f1-score   support

           0       0.95      0.97      0.96       980
           1       0.96      0.98      0.97      1135
           2       0.92      0.92      0.92      1032
           3       0.90      0.91      0.90      1010
           4       0.95      0.93      0.94       982
           5       0.92      0.89      0.91       892
           6       0.95      0.95      0.95       958
           7       0.95      0.93      0.94      1028
           8       0.89      0.91      0.90       974
           9       0.92      0.92      0.92      1009

    accuracy                           0.93     10000
   macro avg       0.93      0.93      0.93     10000
weighted avg       0.93      0.93      0.93     10000


Confusion matrix:
[[ 946    1    7    2    1    5    9    2    6    1]
 [   1 1110    8    2    3    0    2    4    4    1]
 [  14    5  949   21    3    0    2   10   24    4]
 [   4    2   21  921    0   32    0    5   17    8]
 [   1    8    5    0  914    2   13    7    3   29]
 [   8    3    1   43    1  797   14    3   17    5]
 [   9    2    4    4   11    8  911    0    8    1]
 [   2    4   26    7    7    0    0  960    6   16]
 [   7    5    8   26    4   19    6    2  882   15]
 [   6   11    7    3   15    5    1   15   20  926]]
Accuracy=0.9316,Precision=0.9316987537691004,F1=0.9315865117607455,Recall=0.9316
[0.94789579 0.9643788  0.91602317 0.89504373 0.95307612 0.91820276
 0.95093946 0.95238095 0.89361702 0.92047714]
[0.95652174 0.97112861 0.91779497 0.90338401 0.94178259 0.90568182
 0.95093946 0.94302554 0.89954105 0.9191067 ]
[0.96530612 0.97797357 0.91957364 0.91188119 0.93075356 0.89349776
 0.95093946 0.93385214 0.90554415 0.91774034]
'''

#bigan
'''
Elapsed learning 0:21:30.613022
Classification report for classifier GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=115,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False):
              precision    recall  f1-score   support

           0       0.82      0.88      0.85       980
           1       0.94      0.96      0.95      1135
           2       0.79      0.82      0.81      1032
           3       0.78      0.73      0.75      1010
           4       0.83      0.83      0.83       982
           5       0.75      0.74      0.75       892
           6       0.82      0.84      0.83       958
           7       0.81      0.81      0.81      1028
           8       0.78      0.75      0.76       974
           9       0.80      0.80      0.80      1009

    accuracy                           0.82     10000
   macro avg       0.81      0.81      0.81     10000
weighted avg       0.82      0.82      0.82     10000


Confusion matrix:
[[ 858    1   14   16    4   20   42    4   16    5]
 [   0 1095   10    4    2    2    6    5    9    2]
 [  27    7  847   55   11    7    9   29   35    5]
 [  21   10   90  734    6   61   17   24   37   10]
 [   7    5   13    8  812    8   34   26    9   60]
 [  21   11   12   55    9  659   28   13   66   18]
 [  36    6    8    3   26   44  805    3   17   10]
 [  15   10   38   21   32    3    3  835   10   61]
 [  37    7   31   32   11   50   29   24  727   26]
 [  19    7    6   15   62   19    4   63   10  804]]
Accuracy=0.8176,Precision=0.8167968409447953,F1=0.8169482490311021,Recall=0.8176
[0.82420749 0.94477998 0.79232928 0.77836691 0.83282051 0.75486827
 0.82395087 0.81384016 0.7767094  0.8031968 ]
[0.84908461 0.95466434 0.80628272 0.75166411 0.82984159 0.74674221
 0.83204134 0.81304771 0.76125654 0.8       ]
[0.8755102  0.96475771 0.82073643 0.72673267 0.82688391 0.73878924
 0.84029228 0.81225681 0.74640657 0.79682854]
'''

#biganqp
'''
Elapsed learning 0:21:36.812626
Classification report for classifier GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=115,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False):
              precision    recall  f1-score   support

           0       0.94      0.96      0.95       980
           1       0.98      0.98      0.98      1135
           2       0.95      0.93      0.94      1032
           3       0.88      0.91      0.90      1010
           4       0.94      0.95      0.94       982
           5       0.92      0.93      0.93       892
           6       0.97      0.97      0.97       958
           7       0.95      0.91      0.93      1028
           8       0.90      0.91      0.90       974
           9       0.92      0.89      0.90      1009

    accuracy                           0.93     10000
   macro avg       0.93      0.93      0.93     10000
weighted avg       0.93      0.93      0.93     10000


Confusion matrix:
[[ 939    0    2    4    5    7    6    3   13    1]
 [   2 1110    2    5    2    0    4    2    7    1]
 [   6    3  955   28    4    1    3   13   15    4]
 [   7    2   12  924    3   27    1    7   22    5]
 [  11    3    5    0  928    0    3    1    6   25]
 [   5    0    4   20    1  834    6    2   14    6]
 [   5    3    1    1    0   11  930    0    6    1]
 [   1    4   15   24   11    4    0  938    5   26]
 [  12    5    5   32    4   15    4    1  886   10]
 [  10    6    8   12   29   10    0   22   15  897]]
Accuracy=0.9341,Precision=0.9344209361606419,F1=0.9341322117645771,Recall=0.9341
[0.94088176 0.97711268 0.94648167 0.88       0.9402229  0.91749175
 0.97178683 0.94843276 0.8958544  0.91905738]
[0.94944388 0.97754293 0.93581578 0.89708738 0.94261046 0.92615214
 0.97127937 0.9300942  0.90269995 0.90377834]
[0.95816327 0.97797357 0.9253876  0.91485149 0.94501018 0.93497758
 0.97077244 0.91245136 0.90965092 0.88899901]
'''

#aae
# Elapsed learning 0:21:46.121346
# Classification report for classifier GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
#                            learning_rate=0.1, loss='deviance', max_depth=3,
#                            max_features=None, max_leaf_nodes=None,
#                            min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=1, min_samples_split=2,
#                            min_weight_fraction_leaf=0.0, n_estimators=115,
#                            n_iter_no_change=None, presort='deprecated',
#                            random_state=None, subsample=1.0, tol=0.0001,
#                            validation_fraction=0.1, verbose=0,
#                            warm_start=False):
#               precision    recall  f1-score   support
#
#            0       0.80      0.87      0.84       980
#            1       0.92      0.96      0.94      1135
#            2       0.78      0.77      0.77      1032
#            3       0.76      0.77      0.76      1010
#            4       0.79      0.77      0.78       982
#            5       0.74      0.64      0.69       892
#            6       0.80      0.83      0.81       958
#            7       0.79      0.82      0.81      1028
#            8       0.73      0.67      0.70       974
#            9       0.77      0.79      0.78      1009
#
#     accuracy                           0.79     10000
#    macro avg       0.79      0.79      0.79     10000
# weighted avg       0.79      0.79      0.79     10000
#
#
# Confusion matrix:
# [[ 852    1   19   17    8   34   19   10   17    3]
#  [   2 1085    6    6    2    3    4    7   16    4]
#  [  23   29  790   42   16   13   45   17   43   14]
#  [  27   16   29  779   16   45   16   30   35   17]
#  [   5    9   14   11  752   15   38   29   25   84]
#  [  40   10   38   76   23  573   26   33   54   19]
#  [  36    5   43    4   25   16  791    9   20    9]
#  [  21   10   20   11   34    9   10  845   14   54]
#  [  36    9   52   67   25   40   36   22  649   38]
#  [  17   11    6   16   55   27    6   63   14  794]]
# Accuracy=0.791,Precision=0.7892152010230772,F1=0.7893961846452138,Recall=0.791
# [0.80453258 0.91561181 0.77679449 0.75704568 0.78661088 0.73935484
#  0.79818365 0.79342723 0.73167982 0.76640927]
# [0.83570378 0.93534483 0.77110786 0.76410005 0.77605779 0.68746251
#  0.81169831 0.80745342 0.69747448 0.77652812]
# [0.86938776 0.95594714 0.76550388 0.77128713 0.76578411 0.64237668
#  0.8256785  0.82198444 0.66632444 0.78691774]
#alae
'''
Elapsed learning 0:19:16.822682
Classification report for classifier GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False):
              precision    recall  f1-score   support

           0       0.97      0.98      0.97       980
           1       0.97      0.99      0.98      1135
           2       0.94      0.93      0.94      1032
           3       0.92      0.92      0.92      1010
           4       0.94      0.93      0.93       982
           5       0.94      0.91      0.92       892
           6       0.96      0.95      0.95       958
           7       0.95      0.93      0.94      1028
           8       0.92      0.93      0.92       974
           9       0.89      0.91      0.90      1009

    accuracy                           0.94     10000
   macro avg       0.94      0.94      0.94     10000
weighted avg       0.94      0.94      0.94     10000


Confusion matrix:
[[ 962    0    2    1    2    4    7    1    1    0]
 [   0 1120    2    3    1    1    0    0    7    1]
 [  10    2  964   11    4    5    6   18   11    1]
 [   1    0   18  933    0   23    0    9   20    6]
 [   0    0    3    3  912    0   12    2    2   48]
 [   8    0    6   32    1  814   10    1   18    2]
 [   7    3    5    1   14    7  911    2    5    3]
 [   0   12   17    4    3    0    0  952    2   38]
 [   3    3    9   11    7   12    6    4  901   18]
 [   4    9    2   11   31    4    1   17    8  922]]
Accuracy=0.9391,Precision=0.9391786285763838,F1=0.9390838326134459,Recall=0.9391
[0.96683417 0.97476066 0.93774319 0.92376238 0.93538462 0.93563218
 0.95592865 0.94632207 0.92410256 0.88739172]
[0.97417722 0.98073555 0.93592233 0.92376238 0.93203883 0.92395006
 0.95342752 0.93608653 0.92457671 0.90039062]
[0.98163265 0.98678414 0.93410853 0.92376238 0.9287169  0.91255605
 0.95093946 0.92607004 0.92505133 0.91377602]
'''

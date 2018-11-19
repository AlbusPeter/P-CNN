from sklearn.svm import SVC,LinearSVC
import numpy as np

import torch
from torchnet.meter import APMeter

X_train = np.load('SVM_PurePose/train_feature_list_hmdb.npy')
X_train = np.transpose(X_train)
y_train = np.load('SVM_PurePose/train_labels_list_hmdb.npy')
print('Training data loaded!')

X_test = np.load('SVM_PurePose/test_feature_list_hmdb.npy')
X_test = np.transpose(X_test)
y_test = np.load('SVM_PurePose/test_labels_list_hmdb.npy')
print('Test data loaded!')

meter = APMeter()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    
    print(output.shape)
    print(target.shape)
    
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
#    pred = output.unsqueeze(0)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k * 100.0 / batch_size)
    return res
    
print('Start training!')
#clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', probability=False, random_state=None, shrinking=True,
#    tol=0.01, verbose=True)
clf = LinearSVC(tol=0.0001, C=1.0, verbose=1, max_iter=1000)
clf.fit(X_train, y_train)

print('Start prediction!')

acc = clf.score(X_test,y_test)

y_pred = clf.decision_function(X_test)

#np.save('result_test.npy',y_pred)

# One hot encoding buffer that you create out of the loop and just keep reusing
labels_onehot = torch.FloatTensor(y_test.shape[0], X_test.shape[1])

# In your for loop
labels_onehot.zero_()
labels_onehot.scatter_(1, torch.tensor(y_test).view(-1, 1), 1)

meter.add(torch.tensor(y_pred),labels_onehot)

print('Done!')
#
##y_pred = np.load('result.npy')
#
#acc = accuracy(torch.tensor(y_pred),torch.tensor(y_test))

print(meter.value())

print(acc)
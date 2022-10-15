#!/usr/bin/env python
# coding: utf-8

import copy
import random
import argparse
import time

import pandas as pd
from load_dataset import load
from classifier import *
from utils import *
from metrics import *  # include fairness and corresponding derivatives
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
#
# # ignore all the warnings
# import warnings
#
# warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Random Seeds, Datasets, Scenarios, Sensitive Attributes')
parser.add_argument("-v", "--valseed", type=int, default=0)
parser.add_argument("-c", "--scenario", type=int, default=1)
parser.add_argument("-d", "--dataset", type=str, default='adult')
parser.add_argument("-s", "--sens", type=str, default='gender')
parser.add_argument("-m", "--metric", type=int, default=1)
parser.add_argument("-a", "--algo", type=str, default='012345678')
parser.add_argument("-p", "--predictor", type=int, default=0)
parser.add_argument("-t", "--th", type=float, default=0.03)
parser.add_argument("-k", "--ckpt", type=int, default=0)
args = parser.parse_args()

val_seeds = [42, 99, 67, 2, 23]
val_seed = val_seeds[args.valseed]

seed = 1
th = args.th

dataset = args.dataset
sens_attr = args.sens
scenario = args.scenario
print(dataset, '-', scenario)
print('seed:', val_seed, ', th:', args.th)

assert (args.metric == 1) or (args.metric == 0)
if args.metric == 1:
    A = 'y'
    A_val = 1
else:
    A = None
    A_val = None
fair_metric = 0 if (A is None) else 1
assert (A == 'y' and A_val == 1) or (A is None)

predictor_id = args.predictor
clf_name = ['LogisticRegression', 'SVM', 'NeuralNetwork'][predictor_id]

# ## Preparation

# **Load Dataset**
X_train, X_test, y_train, y_test = load(dataset)

# **Parametric Model**

X_train_orig = copy.deepcopy(X_train)
X_test_orig = copy.deepcopy(X_test)

# Scale data: regularization penalty default: ‘l2’, ‘lbfgs’ solvers support only l2 penalties. 
# Regularization makes the predictor dependent on the scale of the features.

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# clf = NeuralNetwork(input_size=X_train.shape[-1])
clf = eval(clf_name)(input_size=X_train.shape[-1])
# clf = SVM(input_size=X_train.shape[-1])
num_params = len(convert_grad_to_ndarray(list(clf.parameters())))
if isinstance(clf, LogisticRegression):
    loss_func = logistic_loss_torch
    c = 0.003
    epoch_num = 1000
elif isinstance(clf, SVM):
    loss_func = svm_loss_torch
    c = 0.003
    epoch_num = 1000
elif isinstance(clf, NeuralNetwork):
    loss_func = nn_loss_torch
    c = 0.0005
    epoch_num = 1000

# **Metrics: Initial state**

clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

clf.fit(X_train, y_train)

y_pred_test = clf.predict_proba(X_test)
y_pred_train = clf.predict_proba(X_train)

spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
print("Initial statistical parity: ", spd_0)

tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
print("Initial TPR parity: ", tpr_parity_0)

predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
print("Initial predictive parity: ", predictive_parity_0)

loss_0 = logistic_loss(y_test, y_pred_test)
print("Initial loss: ", loss_0)

accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
print("Initial accuracy: ", accuracy_0)

# Correlation Ranking

for col in X_test_orig.columns:
    print(f'corr. between {col} and label: {np.round(np.corrcoef(X_test_orig[col], y_test)[0][1], 2)}')

for col in X_test_orig.columns:
    print(
        f'corr. between {col} and {sens_attr}: {np.round(np.corrcoef(X_test_orig[col], X_test_orig[sens_attr])[0][1], 2)}')


# ### Definition

def train_clf(model, info_dict):
    val_iter = 10
    epoch = model.epoch_num
    model.epoch_num = val_iter
    val_num = epoch // val_iter
    min_loss = 2048
    for _ in range(val_num):
        model.fit_info(info_dict)
        loss = model.compute_loss(info_dict)
        if loss < min_loss:
            torch.save(model.state_dict(), f'best_params/best_{args.ckpt}.pth')
            min_loss = loss
    model.load_state_dict(torch.load(f'best_params/best_{args.ckpt}.pth'))
    model.adjust_threshold(info_dict)
    model.eval()
    return model


def record_statistics(clf, record=True):
    af_0 = computeFairness(clf.predict_proba(X_train_sampled), X_train_orig_sampled,
                           y_train_sampled, fair_metric, dataset)
    if record:
        BFs.append(-af_0)
    print("BF: ", -af_0)
    y_pred_test = clf.predict_proba(X_test)
    af_0 = computeFairness(y_pred_test, X_test_orig, y_test,
                           fair_metric, dataset)
    if record:
        AFs.append(-af_0)
    print("AF: ", -af_0)

    test_acc = computeAccuracy(y_test, y_pred_test, clf.threshold)
    test_f1 = computeF1(y_test, y_pred_test, clf.threshold)
    test_auc = computeAUC(y_test, y_pred_test)
    print("Test Acc: ", test_acc)
    print("Test F1: ", test_f1)
    print("Test AUC: ", test_auc)

    if record:
        accs.append(test_acc)
        f1s.append(test_f1)
        aucs.append(test_auc)

    # Group the training data based on parent (sensitive attr=0)
    protected_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & \
                                           (get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & \
                                           (get_attr(X_train_orig_sampled, y_train_sampled, parent_3) == w) & \
                                           (get_attr(X_train_orig_sampled, y_train_sampled, sens_attr) == 0)].index
                idx = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                if len(idx) > 0:
                    protected_groups.append(idx)

    score_protected_groups = []
    for idx in protected_groups:
        score_protected_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    # Group the training data based on parent (sensitive attr=1)
    privileged_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & \
                                           (get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & \
                                           (get_attr(X_train_orig_sampled, y_train_sampled, parent_3) == w) & \
                                           (get_attr(X_train_orig_sampled, y_train_sampled, sens_attr) == 1)].index
                idx = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                if len(idx) > 0:
                    privileged_groups.append(idx)

    score_privileged_groups = []
    for idx in privileged_groups:
        score_privileged_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    ub = np.max(score_privileged_groups) - np.min(score_protected_groups)
    lb = -np.max(score_protected_groups) + np.min(score_privileged_groups)

    print(f'Upper Bound: {ub}')
    print(f'Lower Bound: {lb}')

    if record:
        ubs.append(ub)
        lbs.append(lb)


## Inject Selection Bias

### SYN scenario 1

indicator = (dataset == 'syn')

if indicator:
    p = np.ones_like(X_train_orig.S).astype(float)
    p[(X_train_orig.X2 == 1) & (y_train == 0)] = 0.1
    p[(X_train_orig.X2 == 0) & (y_train == 1)] = 0.05

    np.random.seed(val_seed)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]]

    parent_1 = 'X2'
    parent_2 = 'y'
    parent_3 = 'y'


# ## Regularizer

BFs = []
AFs = []
ubs = []
lbs = []
accs = []
f1s = []
aucs = []

# ### Prepare Validation Set


sc = StandardScaler()
X_train_sampled = sc.fit_transform(X_train_orig_sampled)
X_val_sampled = X_train_sampled.copy()
X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
X_val_orig_sampled = X_train_orig_sampled.copy()

y_train_sampled = y_train_sampled.reset_index(drop=True)
y_val_sampled = y_train_sampled.copy()
X_test = sc.transform(X_test_orig)
X_train = sc.transform(X_train_orig)

# ### No Regularizer
if '0' in args.algo:
    info_dict = dict()
    info_dict['x_train'] = X_train_sampled
    info_dict['y_train'] = y_train_sampled.to_numpy()
    info_dict['x_val'] = X_val_sampled
    info_dict['y_val'] = y_val_sampled.to_numpy()

    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    t0 = time.time()
    clf = train_clf(clf, info_dict)
    print(f'TIME: {clf.__class__.__name__}', time.time() - t0)

    y_pred_test = clf.predict_proba(X_test)
    y_pred_train = clf.predict_proba(X_train)

    spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
    print("Initial TPR parity: ", tpr_parity_0)

    predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
    print("Initial predictive parity: ", predictive_parity_0)

    loss_0 = logistic_loss(y_test, y_pred_test)
    print("Initial loss: ", loss_0)

    accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

    record_statistics(clf)

# ### Bound Regularizer
if '1' in args.algo:
    info_dict = dict()
    info_dict['th'] = th
    info_dict['balance'] = 1
    if fair_metric==1:
        info_dict['balance'] = 1
    info_dict['x_train'] = X_train_sampled
    info_dict['y_train'] = y_train_sampled.to_numpy()
    info_dict['x_val'] = X_val_sampled
    info_dict['y_val'] = y_val_sampled.to_numpy()

    protected_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & (
                        get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled, parent_3) == w) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled,
                                                            sens_attr) == 0)].index
                idx = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                if len(idx) > 0:
                    protected_groups.append(idx)

    privileged_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & (
                        get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled, parent_3) == w) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled,
                                                            sens_attr) == 1)].index
                idx = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                if len(idx) > 0:
                    privileged_groups.append(idx)

    info_dict['train_regularizer'] = [protected_groups, privileged_groups]

    protected_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_val_orig_sampled[(get_attr(X_val_orig_sampled, y_val_sampled, parent_1) == v) & (
                        get_attr(X_val_orig_sampled, y_val_sampled, parent_2) == u) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, parent_3) == w) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, sens_attr) == 0)].index
                idx = idx.intersection(get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
                if len(idx) > 0:
                    protected_groups.append(idx)

    privileged_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_val_orig_sampled[(get_attr(X_val_orig_sampled, y_val_sampled, parent_1) == v) & (
                        get_attr(X_val_orig_sampled, y_val_sampled, parent_2) == u) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, parent_3) == w) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, sens_attr) == 1)].index
                idx = idx.intersection(get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
                if len(idx) > 0:
                    privileged_groups.append(idx)
    info_dict['val_regularizer'] = [protected_groups, privileged_groups]

    clf = eval(clf_name + '_Reg_Bound')(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)
    t0 = time.time()
    clf = train_clf(clf, info_dict)
    print('TIME:', time.time() - t0)

    y_pred_test = clf.predict_proba(X_test)

    spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
    print("Initial TPR parity: ", tpr_parity_0)

    predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
    print("Initial predictive parity: ", predictive_parity_0)

    loss_0 = logistic_loss(y_test, y_pred_test)
    print("Initial loss: ", loss_0)

    accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

    record_statistics(clf)

# ### Metric Regularizer
if '2' in args.algo:
    info_dict = dict()
    info_dict['th'] = th
    info_dict['balance'] = 0.5
    info_dict['x_train'] = X_test
    info_dict['y_train'] = y_test.to_numpy()
    info_dict['x_val'] = X_test
    info_dict['y_val'] = y_test.to_numpy()

    protected_idx = X_train_orig_sampled[X_train_orig_sampled[sens_attr] == 0].index.intersection(
        get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
    privileged_idx = X_train_orig_sampled[X_train_orig_sampled[sens_attr] == 1].index.intersection(
        get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
    info_dict['train_regularizer'] = [protected_idx, privileged_idx]

    protected_idx = X_val_orig_sampled[X_val_orig_sampled[sens_attr] == 0].index.intersection(
        get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
    privileged_idx = X_val_orig_sampled[X_val_orig_sampled[sens_attr] == 1].index.intersection(
        get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
    info_dict['val_regularizer'] = [protected_idx, privileged_idx]
    # protected_idx = X_test_orig[X_test_orig[sens_attr] == 0].index.intersection(
    #     get_A_idx(X_test_orig, y_test, A, A_val))
    # privileged_idx = X_test_orig[X_test_orig[sens_attr] == 1].index.intersection(
    #     get_A_idx(X_test_orig, y_test, A, A_val))
    # info_dict['train_regularizer'] = [protected_idx, privileged_idx]
    #
    # protected_idx = X_test_orig[X_test_orig[sens_attr] == 0].index.intersection(
    #     get_A_idx(X_test_orig, y_test, A, A_val))
    # privileged_idx = X_test_orig[X_test_orig[sens_attr] == 1].index.intersection(
    #     get_A_idx(X_test_orig, y_test, A, A_val))
    # info_dict['val_regularizer'] = [protected_idx, privileged_idx]

    clf = eval(clf_name + '_Reg_Metric')(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf = train_clf(clf, info_dict)

    y_pred_test = clf.predict_proba(X_test)
    y_pred_train = clf.predict_proba(X_train)

    spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
    print("Initial TPR parity: ", tpr_parity_0)

    predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
    print("Initial predictive parity: ", predictive_parity_0)

    loss_0 = logistic_loss(y_test, y_pred_test)
    print("Initial loss: ", loss_0)

    accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

    record_statistics(clf)

# ### Weighted Metric Regularizer
if '3' in args.algo:
    info_dict = dict()
    info_dict['th'] = th
    info_dict['balance'] = 3
    # if dataset == 'hmda':
    #     info_dict['balance'] = 1
    info_dict['x_train'] = X_train_sampled
    info_dict['y_train'] = y_train_sampled.to_numpy()
    info_dict['x_val'] = X_val_sampled
    info_dict['y_val'] = y_val_sampled.to_numpy()

    protected_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & (
                        get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled, parent_3) == w) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled,
                                                            sens_attr) == 0)].index
                idx = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                if len(idx) > 0:
                    protected_groups.append(idx)

    privileged_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & (
                        get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled, parent_3) == w) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled,
                                                            sens_attr) == 1)].index
                idx = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                if len(idx) > 0:
                    privileged_groups.append(idx)

    info_dict['train_regularizer'] = [protected_groups, privileged_groups]

    protected_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_val_orig_sampled[(get_attr(X_val_orig_sampled, y_val_sampled, parent_1) == v) & (
                        get_attr(X_val_orig_sampled, y_val_sampled, parent_2) == u) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, parent_3) == w) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, sens_attr) == 0)].index
                idx = idx.intersection(get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
                if len(idx) > 0:
                    protected_groups.append(idx)

    privileged_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_val_orig_sampled[(get_attr(X_val_orig_sampled, y_val_sampled, parent_1) == v) & (
                        get_attr(X_val_orig_sampled, y_val_sampled, parent_2) == u) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, parent_3) == w) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, sens_attr) == 1)].index
                idx = idx.intersection(get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
                if len(idx) > 0:
                    privileged_groups.append(idx)
    info_dict['val_regularizer'] = [protected_groups, privileged_groups]

    weights = [[], []]
    for s in range(2):
        s_idx = X_train_orig[X_train_orig[sens_attr] == s].index.intersection(
            get_A_idx(X_train_orig, y_train, A, A_val))
        for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
            for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
                for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                    idx = X_train_orig[(get_attr(X_train_orig, y_train, parent_1) == v) & (
                            get_attr(X_train_orig, y_train, parent_2) == u) & (
                                               get_attr(X_train_orig, y_train, parent_3) == w) & (
                                               get_attr(X_train_orig, y_train, sens_attr) == s)].index
                    idx = idx.intersection(get_A_idx(X_train_orig, y_train, A, A_val))
                    if len(idx) > 0:
                        weights[s].append(len(idx) / len(s_idx))

    info_dict['weights'] = weights

    if scenario == 7:
        info_dict['balance'] = 6
        info_dict['weights'] = [[0.636514, 0.363486], [0.589085, 0.410914]]
    elif scenario == 8:
        info_dict['balance'] = 6
        info_dict['weights'] = [[0.589058, 0.410942], [0.634008, 0.3659912]]

    clf = eval(clf_name + '_Reg_WeightedMetric')(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    t0 = time.time()
    clf = train_clf(clf, info_dict)
    print('TIME:', time.time() - t0)

    spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
    print("Initial TPR parity: ", tpr_parity_0)

    predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
    print("Initial predictive parity: ", predictive_parity_0)

    loss_0 = logistic_loss(y_test, y_pred_test)
    print("Initial loss: ", loss_0)

    accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

    record_statistics(clf)

# ### Est. Weighted Metric Regularizer (limited info)
if '4' in args.algo:
    info_dict = dict()
    extl_info = dict()
    info_dict['th'] = th
    info_dict['balance'] = 3
    # if dataset == 'hmda':
    #     info_dict['balance'] = 1
    info_dict['x_train'] = X_train_sampled
    info_dict['y_train'] = y_train_sampled.to_numpy()
    info_dict['x_val'] = X_val_sampled
    info_dict['y_val'] = y_val_sampled.to_numpy()

    protected_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & (
                        get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled, parent_3) == w) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled,
                                                            sens_attr) == 0)].index
                idx = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                if len(idx) > 0:
                    protected_groups.append(idx)

    privileged_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & (
                        get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled, parent_3) == w) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled,
                                                            sens_attr) == 1)].index
                idx = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                if len(idx) > 0:
                    privileged_groups.append(idx)

    info_dict['train_regularizer'] = [protected_groups, privileged_groups]

    protected_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_val_orig_sampled[(get_attr(X_val_orig_sampled, y_val_sampled, parent_1) == v) & (
                        get_attr(X_val_orig_sampled, y_val_sampled, parent_2) == u) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, parent_3) == w) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, sens_attr) == 0)].index
                idx = idx.intersection(get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
                if len(idx) > 0:
                    protected_groups.append(idx)

    privileged_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_val_orig_sampled[(get_attr(X_val_orig_sampled, y_val_sampled, parent_1) == v) & (
                        get_attr(X_val_orig_sampled, y_val_sampled, parent_2) == u) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, parent_3) == w) & (
                                                 get_attr(X_val_orig_sampled, y_val_sampled, sens_attr) == 1)].index
                idx = idx.intersection(get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
                if len(idx) > 0:
                    privileged_groups.append(idx)
    info_dict['val_regularizer'] = [protected_groups, privileged_groups]

    extl_info['data'] = X_train
    idxs = [[], []]
    for s in range(2):
        for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
            for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
                for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                    idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & (
                            get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & (
                                                       get_attr(X_train_orig_sampled, y_train_sampled,
                                                                parent_3) == w) & (
                                                       get_attr(X_train_orig_sampled, y_train_sampled,
                                                                sens_attr) == s)].index
                    idx_unbiased = X_train_orig[(get_attr(X_train_orig, y_train, parent_1) == v) & (
                            get_attr(X_train_orig, y_train, parent_2) == u) & (
                                                        get_attr(X_train_orig, y_train, parent_3) == w) & (
                                                        get_attr(X_train_orig, y_train, sens_attr) == s)].index
                    idx_a = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                    if len(idx_a) > 0:
                        idxs[s].append(len(idx_a) / len(idx) * len(idx_unbiased))

    weights = [[], []]
    for s in range(2):
        weights[s] = [w / sum(idxs[s]) for w in idxs[s]]
    info_dict['weights'] = weights

    clf = eval(clf_name + '_Reg_WeightedMetric')(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf = train_clf(clf, info_dict)

    y_pred_test = clf.predict_proba(X_test)
    y_pred_train = clf.predict_proba(X_train)

    spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
    print("Initial TPR parity: ", tpr_parity_0)

    predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
    print("Initial predictive parity: ", predictive_parity_0)

    loss_0 = logistic_loss(y_test, y_pred_test)
    print("Initial loss: ", loss_0)

    accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

    record_statistics(clf)

if '5' in args.algo:
    if clf_name == 'NeuralNetwork':
        print('Cov. Reg. not implemented for NN!')
        record_statistics(clf)
    else:
        info_dict = dict()
        info_dict['th'] = 0
        info_dict['fair_metric'] = fair_metric
        info_dict['balance'] = 8
        if dataset == 'law' and scenario == 4:
            info_dict['balance'] = 2
        info_dict['sens'] = X_train_orig_sampled[sens_attr].to_numpy()
        info_dict['x_train'] = X_train_sampled
        info_dict['y_train'] = y_train_sampled.to_numpy()
        info_dict['x_val'] = X_val_sampled
        info_dict['y_val'] = y_val_sampled.to_numpy()

        protected_idx = X_train_orig_sampled[X_train_orig_sampled[sens_attr] == 0].index.intersection(
            get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
        privileged_idx = X_train_orig_sampled[X_train_orig_sampled[sens_attr] == 1].index.intersection(
            get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
        info_dict['train_regularizer'] = [protected_idx, privileged_idx]

        protected_idx = X_val_orig_sampled[X_val_orig_sampled[sens_attr] == 0].index.intersection(
            get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
        privileged_idx = X_val_orig_sampled[X_val_orig_sampled[sens_attr] == 1].index.intersection(
            get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
        info_dict['val_regularizer'] = [protected_idx, privileged_idx]

        clf = eval(clf_name + '_Reg_Cov')(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

        clf = train_clf(clf, info_dict)

        y_pred_test = clf.predict_proba(X_test)
        y_pred_train = clf.predict_proba(X_train)

        spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
        print("Initial statistical parity: ", spd_0)

        tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
        print("Initial TPR parity: ", tpr_parity_0)

        predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
        print("Initial predictive parity: ", predictive_parity_0)

        loss_0 = logistic_loss(y_test, y_pred_test)
        print("Initial loss: ", loss_0)

        accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
        print("Initial accuracy: ", accuracy_0)

        record_statistics(clf)

# ### Adversarial Debiasing
if '6' in args.algo:
    clf = eval(clf_name + '_AD')(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c,
                                 learning_rate=0.01, adv_learning_rate=0.2)
    fair_eval = lambda x: computeFairness(x.predict_proba(X_val_sampled), X_val_orig_sampled,
                                          y_val_sampled, fair_metric, dataset)
    acc_eval = lambda x: computeAccuracy(y_val_sampled, x.predict_proba(X_val_sampled), clf.threshold)
    _ = clf.fit(X_train_sampled, y_train_sampled, X_train_orig_sampled[sens_attr],
                fair_eval, acc_eval, th=0, balance=0.8, loss_balance=8, load=True)

    y_pred_test = clf.predict_proba(X_test)
    y_pred_train = clf.predict_proba(X_train)

    spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
    print("Initial TPR parity: ", tpr_parity_0)

    predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
    print("Initial predictive parity: ", predictive_parity_0)

    loss_0 = logistic_loss(y_test, y_pred_test)
    print("Initial loss: ", loss_0)

    accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

    record_statistics(clf)

if '7' in args.algo:
    positive_data = X_train[y_train == 1]
    positive_source = sample_bool[y_train == 1]

    all_data = X_train
    all_source = sample_bool

    Pr_C_given_X = LogisticRegression(input_size=X_train.shape[-1])
    Pr_C_given_X.fit(all_data, all_source.ravel())

    Pr_C_given_YX = LogisticRegression(input_size=positive_data.shape[-1])
    Pr_C_given_YX.fit(positive_data, positive_source.ravel())

    training_weights = Pr_C_given_X.predict_proba(X_train_sampled) / Pr_C_given_YX.predict_proba(X_train_sampled)
    val_weights = Pr_C_given_X.predict_proba(X_val_sampled) / Pr_C_given_YX.predict_proba(X_val_sampled)

    info_dict = dict()
    info_dict['x_train'] = X_train_sampled
    info_dict['y_train'] = y_train_sampled.to_numpy()
    info_dict['x_val'] = X_val_sampled
    info_dict['y_val'] = y_val_sampled.to_numpy()
    info_dict['clf_numerator'] = Pr_C_given_X
    info_dict['clf_denominator'] = Pr_C_given_YX
    info_dict['th'] = th
    info_dict['balance'] = 3

    protected_idx = X_train_orig_sampled[X_train_orig_sampled[sens_attr] == 0].index.intersection(
        get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
    privileged_idx = X_train_orig_sampled[X_train_orig_sampled[sens_attr] == 1].index.intersection(
        get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
    info_dict['train_regularizer'] = [protected_idx, privileged_idx]

    protected_idx = X_val_orig_sampled[X_val_orig_sampled[sens_attr] == 0].index.intersection(
        get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
    privileged_idx = X_val_orig_sampled[X_val_orig_sampled[sens_attr] == 1].index.intersection(
        get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
    info_dict['val_regularizer'] = [protected_idx, privileged_idx]

    clf = eval(clf_name + '_IPW')(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)
    clf = train_clf(clf, info_dict=info_dict)

    y_pred_test = clf.predict_proba(X_test)
    y_pred_train = clf.predict_proba(X_train)

    spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
    print("Initial TPR parity: ", tpr_parity_0)

    predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
    print("Initial predictive parity: ", predictive_parity_0)

    loss_0 = logistic_loss(y_test, y_pred_test)
    print("Initial loss: ", loss_0)

    accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

    record_statistics(clf)

if '8' in args.algo:
    if scenario in [3, 6]:
        record_statistics(clf)
    else:
        info_dict = dict()
        info_dict['th'] = th
        info_dict['balance'] = 5
        info_dict['x_train'] = X_train_sampled
        info_dict['y_train'] = y_train_sampled.to_numpy()
        info_dict['x_val'] = X_val_sampled
        info_dict['y_val'] = y_val_sampled.to_numpy()

        protected_subgroup_idx = [[] for _ in
                                  range(len(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()))]
        privileged_subgroup_idx = [[] for _ in
                                   range(len(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()))]

        protected_groups = []
        for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
            for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
                for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                    idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & \
                                               (get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & \
                                               (get_attr(X_train_orig_sampled, y_train_sampled, parent_3) == w) & \
                                               (get_attr(X_train_orig_sampled, y_train_sampled, sens_attr) == 0)].index
                    idx = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                    if len(idx) > 0:
                        protected_subgroup_idx[int(v)].append(len(protected_groups))
                        protected_groups.append(idx)

        privileged_groups = []
        for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
            for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
                for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                    idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & \
                                               (get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & \
                                               (get_attr(X_train_orig_sampled, y_train_sampled, parent_3) == w) & \
                                               (get_attr(X_train_orig_sampled, y_train_sampled, sens_attr) == 1)].index
                    idx = idx.intersection(get_A_idx(X_train_orig_sampled, y_train_sampled, A, A_val))
                    if len(idx) > 0:
                        privileged_subgroup_idx[int(v)].append(len(privileged_groups))
                        privileged_groups.append(idx)

        info_dict['train_subgroup_idx'] = [protected_subgroup_idx, protected_subgroup_idx]
        info_dict['train_regularizer'] = [protected_groups, privileged_groups]
        print(len(privileged_groups), len(protected_groups))
        protected_subgroup_idx = [[] for _ in
                                  range(len(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()))]
        privileged_subgroup_idx = [[] for _ in
                                   range(len(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()))]

        protected_groups = []
        for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
            for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
                for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                    idx = X_val_orig_sampled[(get_attr(X_val_orig_sampled, y_val_sampled, parent_1) == v) & \
                                             (get_attr(X_val_orig_sampled, y_val_sampled, parent_2) == u) & \
                                             (get_attr(X_val_orig_sampled, y_val_sampled, parent_3) == w) & \
                                             (get_attr(X_val_orig_sampled, y_val_sampled, sens_attr) == 0)].index
                    idx = idx.intersection(get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
                    if len(idx) > 0:
                        protected_subgroup_idx[int(v)].append(len(protected_groups))
                        protected_groups.append(idx)

        privileged_groups = []
        for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
            for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
                for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                    idx = X_val_orig_sampled[(get_attr(X_val_orig_sampled, y_val_sampled, parent_1) == v) & \
                                             (get_attr(X_val_orig_sampled, y_val_sampled, parent_2) == u) & \
                                             (get_attr(X_val_orig_sampled, y_val_sampled, parent_3) == w) & \
                                             (get_attr(X_val_orig_sampled, y_val_sampled, sens_attr) == 1)].index
                    idx = idx.intersection(get_A_idx(X_val_orig_sampled, y_val_sampled, A, A_val))
                    if len(idx) > 0:
                        privileged_subgroup_idx[int(v)].append(len(privileged_groups))
                        privileged_groups.append(idx)

        info_dict['val_subgroup_idx'] = [protected_subgroup_idx, privileged_subgroup_idx]
        info_dict['val_regularizer'] = [protected_groups, privileged_groups]

        weights = [[], []]
        for s in range(2):
            s_idx = X_train_orig[X_train_orig[sens_attr] == s].index.intersection(
                get_A_idx(X_train_orig, y_train, A, A_val))
            for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
                idx = X_train_orig[(get_attr(X_train_orig, y_train, parent_1) == v) & \
                                   (get_attr(X_train_orig, y_train, sens_attr) == s)].index
                idx = idx.intersection(get_A_idx(X_train_orig, y_train, A, A_val))
                if len(idx) > 0:
                    weights[s].append(len(idx) / len(s_idx))

        info_dict['weights'] = weights
        if scenario == 7:
            info_dict['balance'] = 2
            info_dict['weights'] = [[0.636514, 0.363486], [0.589085, 0.410914]]
        elif scenario == 8:
            info_dict['balance'] = 2
            info_dict['weights'] = [[0.589058, 0.410942], [0.634008, 0.3659912]]

        clf = eval(clf_name + '_Reg_Tighter_Bound')(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

        clf = train_clf(clf, info_dict)

        y_pred_test = clf.predict_proba(X_test)

        spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
        print("Initial statistical parity: ", spd_0)

        tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
        print("Initial TPR parity: ", tpr_parity_0)

        predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
        print("Initial predictive parity: ", predictive_parity_0)

        loss_0 = logistic_loss(y_test, y_pred_test)
        print("Initial loss: ", loss_0)

        accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
        print("Initial accuracy: ", accuracy_0)

        record_statistics(clf)

# ### No Regularizer trained on the test data
if '9' in args.algo:
    info_dict = dict()
    info_dict['x_train'] = X_test
    info_dict['y_train'] = y_test.to_numpy()
    info_dict['x_val'] = X_test
    info_dict['y_val'] = y_test.to_numpy()

    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf = train_clf(clf, info_dict=info_dict)

    y_pred_test = clf.predict_proba(X_test)

    spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
    print("Initial TPR parity: ", tpr_parity_0)

    predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
    print("Initial predictive parity: ", predictive_parity_0)

    loss_0 = logistic_loss(y_test, y_pred_test)
    print("Initial loss: ", loss_0)

    accuracy_0 = computeAccuracy(y_test, y_pred_test, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

    record_statistics(clf)

dict_name = '_'
if fair_metric == 1:
    dict_name = 'EO' + dict_name
elif fair_metric == 0:
    dict_name = 'SPD' + dict_name
else:
    raise NotImplementedError

if clf_name == 'LogisticRegression':
    dict_name = dict_name + 'LR'
elif clf_name == 'SVM':
    dict_name = dict_name + 'SVM'
elif clf_name == 'NeuralNetwork':
    dict_name = dict_name + 'NN'
else:
    raise NotImplementedError

dict_name = dict_name + '.dict'

with open(dict_name, 'r') as f:
    txt = f.read()
d = json.loads(txt)
names = ['Orig', f'Bound Reg.-{round(th, 3)}', f'BF Reg.-{round(th, 3)}', f'AF Reg.-{round(th, 3)}',
         f'Est.AF Reg.-{round(th, 3)}', 'Cov. Reg.', 'Adv. Debias', 'IPW', f'TightBound Reg.-{round(th, 3)}',
         'Test']
print(args.algo)
name = []
for i in range(len(args.algo)):
    n_idx = int(args.algo[i])
    name.append(names[n_idx])
print(f'saving: {name}')
for ls in [AFs, BFs, ubs, lbs, accs, f1s]:
    assert len(ls) == len(name)
for n_idx, n in enumerate(name):
    if f'sc{scenario}' not in d[dataset]:
        d[dataset][f'sc{scenario}'] = dict()
        for vs in val_seeds:
            d[dataset][f'sc{scenario}'][str(vs)] = dict()
    d[dataset][f'sc{scenario}'][str(val_seed)][n] = [float(AFs[n_idx]), float(BFs[n_idx]), float(ubs[n_idx]),
                                                     float(lbs[n_idx]), float(accs[n_idx]), float(f1s[n_idx]),
                                                     float(aucs[n_idx])]
txt = json.dumps(d)
with open(dict_name, 'w+') as f:
    f.write(txt)
print(f'saved: {dict_name}')
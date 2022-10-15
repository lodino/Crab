#!/usr/bin/env python
# coding: utf-8

import copy
import random

import pandas as pd
import argparse
from load_dataset import load
from classifier import *
from utils import *
from metrics import *  # include fairness and corresponding derivatives
from sklearn.model_selection import train_test_split

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

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
parser.add_argument("-p", "--predictor", type=int, default=0)
args = parser.parse_args()

val_seeds = [42, 99, 67, 2, 23]
val_seed = val_seeds[args.valseed]

seed = 1
c = 0.003
epoch_num = 3000
th = 0.03

dataset = args.dataset
sens_attr = args.sens
scenario = args.scenario
print(dataset, '-', scenario)

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
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# **Loss function** (Log loss for logistic regression)

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

if dataset == 'hmda':
    epoch_num = 300


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
            torch.save(model.state_dict(), 'best_params/best1.pth')
            min_loss = loss
    model.load_state_dict(torch.load('best_params/best1.pth'))
    model.adjust_threshold(info_dict)
    model.eval()
    return model


def record_statistics(clf, record=True, fair_metric=fair_metric):
    af_0 = computeFairness(clf.predict_proba(X_train_sampled), X_train_orig_sampled,
                            y_train_sampled, fair_metric, dataset)
    y_pred_test = clf.predict_proba(X_test)
    if record:
        BFs.append(-af_0)
    print("BF: ", -af_0)
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

    if fair_metric == 1:
        A = 'y'
        A_val = 1
    else:
        A = None
        A_val = None

    # Group the training data based on parent (sensitive attr=0)
    protected_groups = []
    for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
        for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
            for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & (
                        get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled,
                                                            parent_3) == w) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled,
                                                            sens_attr) == 0)].index
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
                idx = X_train_orig_sampled[(get_attr(X_train_orig_sampled, y_train_sampled, parent_1) == v) & (
                        get_attr(X_train_orig_sampled, y_train_sampled, parent_2) == u) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled,
                                                            parent_3) == w) & (
                                                   get_attr(X_train_orig_sampled, y_train_sampled,
                                                            sens_attr) == 1)].index
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


# ## Inject Selection Bias

# ### HMDA scenario 1

indicator = ((dataset == 'hmda') and (scenario == 1))

# **Sample based on the conditional probability**

if indicator:
    p = np.ones_like(X_train_orig.race).astype(float)
    p[(X_train_orig.race == 0) & (X_train_orig.DI == 2)] = 0.1
    p[(X_train_orig.race == 0) & (X_train_orig.DI == 3)] = 0.2
    p[(X_train_orig.race == 1) & (X_train_orig.DI == 1)] = 0.1

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled.to_numpy(), use_sklearn=False)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'race'
    parent_1 = 'DI'
    parent_2 = 'race'
    parent_3 = 'race'

# **Bound**

if indicator:
    record_statistics(clf, record=False)

# In[ ]:


# ### HMDA scenario 4

indicator = ((dataset == 'hmda') and (scenario == 4))

# **Sample based on the conditional probability**

if indicator:
    p = np.ones_like(X_train_orig.race).astype(float)
    p[(X_train_orig.race == 0) & (X_train_orig.DI == 1)] = 0.2
    p[(X_train_orig.race == 0) & (X_train_orig.DI == 2)] = 0.3

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled.to_numpy(), use_sklearn=False)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'race'
    parent_1 = 'DI'
    parent_2 = 'race'
    parent_3 = 'race'

### HMDA scenario 7

indicator = ((dataset == 'hmda') and (scenario == 7))

if indicator:
    p = np.ones_like(X_train_orig.race).astype(float)
    p[(X_train_orig.race == 0) & (X_train_orig.loan_type == 1)] = 0.2
    #     p[(X_train_orig.race == 1) & (X_train_orig.loan_type == 1)] = 0.55

    p[(X_train_orig.race == 0) & (X_train_orig.loan_type == 2)] = 0.35
    #     p[(X_train_orig.race == 1) & (X_train_orig.loan_type == 2)] = 0.5

    #     p[(X_train_orig.race == 0) & (X_train_orig.loan_type == 3)] = 0.8
    p[(X_train_orig.race == 1) & (X_train_orig.loan_type == 3)] = 0.9

    p[(X_train_orig.race == 1) & (X_train_orig.loan_type == 4)] = 0.7

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled.to_numpy(), use_sklearn=False)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'race'
    parent_1 = 'race'
    parent_2 = 'race'
    parent_3 = 'loan_type'

if indicator:
    record_statistics(clf, record=False)

# **Bound**

if indicator:
    record_statistics(clf, record=False)

# ### Adult scenario 1: make collider to be the child of relationship and gender (try to use sampling to eliminate the correlation between them on the training set)

indicator = ((dataset == 'adult') and (scenario == 1))

if indicator:
    print(np.sum(X_train_orig.relationship[X_train_orig.gender == 0]) / np.sum(
        X_train_orig.relationship[X_train_orig.gender == 1]))

# **Pr(C=1|married,female)=1, Pr(C=1|not_married,high)=1, Pr(C=1|married,male)=0.11, Pr(C=1|not_married,low)=1**

# generate Pr(C=1|gender, income) for each training data point
if indicator:
    p = np.where(np.logical_and(X_train_orig.relationship, X_train_orig.gender), 0.11, 1.0)

# **Sample based on the conditional probability**

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled.to_numpy(), use_sklearn=False)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'gender'
    parent_1 = 'relationship'
    parent_2 = 'gender'
    parent_3 = 'gender'

# **Bound**

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)
    spd_0 = computeFairness(clf.predict_proba(X_train_sampled), X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("BF: ", -spd_0)
    y_pred_train = clf.predict_proba(X_train)
    spd_0 = computeFairness(clf.predict_proba(X_train), X_train_orig, y_train, 0, dataset)
    print("AF: ", -spd_0)

    # Group the training data based on parent (sensitive attr=0)
    protected_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & (X_train_orig_sampled[parent_2] == u) & (
                    X_train_orig_sampled[sens_attr] == 0)].index
            if len(idx) > 0:
                protected_groups.append(idx)

    score_protected_groups = []
    for idx in protected_groups:
        score_protected_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    # Group the training data based on parent (sensitive attr=1)
    privileged_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & (X_train_orig_sampled[parent_2] == u) & (
                    X_train_orig_sampled[sens_attr] == 1)].index
            if len(idx) > 0:
                privileged_groups.append(idx)

    score_privileged_groups = []
    for idx in privileged_groups:
        score_privileged_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    print(f'Upper Bound: {np.max(score_privileged_groups) - np.min(score_protected_groups)}')
    print(f'Lower Bound: {-np.max(score_protected_groups) + np.min(score_privileged_groups)}')

# In[ ]:


# ### Adult scenario 4: make collider to be the child of relationship and gender (try to use sampling to increase the correlation between them on the training set)

indicator = ((dataset == 'adult') and (scenario == 4))

if indicator:
    print(np.sum(X_train_orig.relationship[X_train_orig.gender == 0]) / np.sum(
        X_train_orig.relationship[X_train_orig.gender == 1]))

if indicator:
    p = np.ones_like(X_train_orig.gender).astype(float)
    #     p[(X_train_orig.relationship==1)&(X_train_orig.gender==0)] = 0.5
    p[(X_train_orig.relationship == 0) & (X_train_orig.gender == 1)] = 0.5

# **Sample based on the conditional probability**

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled.to_numpy(), use_sklearn=False)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'gender'
    parent_1 = 'relationship'
    parent_2 = 'gender'
    parent_3 = 'gender'

# **Bound**

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)
    spd_0 = computeFairness(clf.predict_proba(X_train_sampled), X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("BF: ", -spd_0)
    y_pred_train = clf.predict_proba(X_train)
    spd_0 = computeFairness(clf.predict_proba(X_train), X_train_orig, y_train, 0, dataset)
    print("AF: ", -spd_0)

    # Group the training data based on parent (sensitive attr=0)
    protected_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & (X_train_orig_sampled[parent_2] == u) & (
                    X_train_orig_sampled[sens_attr] == 0)].index
            if len(idx) > 0:
                protected_groups.append(idx)

    score_protected_groups = []
    for idx in protected_groups:
        score_protected_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    # Group the training data based on parent (sensitive attr=1)
    privileged_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & (X_train_orig_sampled[parent_2] == u) & (
                    X_train_orig_sampled[sens_attr] == 1)].index
            if len(idx) > 0:
                privileged_groups.append(idx)

    score_privileged_groups = []
    for idx in privileged_groups:
        score_privileged_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    print(f'Upper Bound: {np.max(score_privileged_groups) - np.min(score_protected_groups)}')
    print(f'Lower Bound: {-np.max(score_protected_groups) + np.min(score_privileged_groups)}')

# In[ ]:


# ### Adult scenario 2: make collider to be the child of relationship and education (try to use sampling to increase the correlation between gender and label on the training set)

indicator = ((dataset == 'adult') and (scenario == 2))

# **Pr(C=1|married,female)=1, Pr(C=1|not_married,high)=1, Pr(C=1|married,male)=0.11, Pr(C=1|not_married,low)=1**

# generate Pr(C=1|gender, income) for each training data point
if indicator:
    p = np.ones_like(X_train_orig.gender).astype(float)
    p[(X_train_orig.relationship == 1) & (X_train_orig.education == 0)] = 0.2
    p[(X_train_orig.relationship == 1) & (X_train_orig.education == 1)] = 0.4
    p[(X_train_orig.relationship == 1) & (X_train_orig.education == 2)] = 0.6
    p[(X_train_orig.relationship == 0) & (X_train_orig.education == 1)] = 0.6
    p[(X_train_orig.relationship == 0) & (X_train_orig.education == 2)] = 0.4
    p[(X_train_orig.relationship == 0) & (X_train_orig.education == 3)] = 0.2

# **Sample based on the conditional probability**

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled.to_numpy(), use_sklearn=False)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'gender'
    parent_1 = 'relationship'
    parent_2 = 'education'
    parent_3 = 'relationship'

# **Bound**

if indicator:
    record_statistics(clf, record=False)

# ### Adult scenario 3: make collider to be the child of relationship and label (try to use sampling to increase the correlation between gender and label on the training set)

indicator = ((dataset == 'adult') and (scenario == 3))

# **Pr(C=1|married,female)=1, Pr(C=1|not_married,high)=1, Pr(C=1|married,male)=0.11, Pr(C=1|not_married,low)=1**

# generate Pr(C=1|gender, income) for each training data point
if indicator:
    p = np.ones_like(X_train_orig.gender).astype(float)
    p[(X_train_orig.relationship == 1) & (y_train == 0)] = 0.2
    p[(X_train_orig.relationship == 0) & (y_train == 1)] = 0.2

# **Sample based on the conditional probability**

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled.to_numpy(), use_sklearn=False)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'gender'
    parent_1 = 'relationship'
    parent_2 = 'y'
    parent_3 = 'relationship'

# **Bound**

if indicator:
    record_statistics(clf, record=False)

# In[ ]:


# ### Law scenario 1: make collider to be the child of relationship and gender (try to use sampling to eliminate the correlation between them on the training set)

indicator = ((dataset == 'law') and (scenario == 1))

# generate Pr(C=1|race, decile3, lsat) for each training data point
if indicator:
    p = np.ones_like(X_train_orig.racetxt).astype(float)
    p[(X_train_orig.racetxt==0)&(X_train_orig.decile3==0)] = 0.2
    p[(X_train_orig.racetxt==1)&(X_train_orig.decile3==1)] = 0.5


# **Sample based on the conditional probability**

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'racetxt'
    parent_1 = 'lsat'
    parent_2 = 'lsat'
    parent_3 = 'racetxt'

# **Bound**

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)
    spd_0 = computeFairness(clf.predict_proba(X_train_sampled), X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("BF: ", -spd_0)
    y_pred_train = clf.predict_proba(X_train)
    spd_0 = computeFairness(clf.predict_proba(X_train), X_train_orig, y_train, 0, dataset)
    print("AF: ", -spd_0)

    # Group the training data based on parent (sensitive attr=0)
    protected_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & (X_train_orig_sampled[parent_2] == u) & (
                    X_train_orig_sampled[sens_attr] == 0)].index
            if len(idx) > 0:
                protected_groups.append(idx)

    score_protected_groups = []
    for idx in protected_groups:
        score_protected_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    # Group the training data based on parent (sensitive attr=1)
    privileged_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & (X_train_orig_sampled[parent_2] == u) & (
                    X_train_orig_sampled[sens_attr] == 1)].index
            if len(idx) > 0:
                privileged_groups.append(idx)

    score_privileged_groups = []
    for idx in privileged_groups:
        score_privileged_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    print(f'Upper Bound: {np.max(score_privileged_groups) - np.min(score_protected_groups)}')
    print(f'Lower Bound: {-np.max(score_protected_groups) + np.min(score_privileged_groups)}')

# ### Law scenario 2: make collider to be the child of lsat and decile3 (try to use sampling to increase the correlation between label and race on the training set)

indicator = ((dataset == 'law') and (scenario == 2))
if indicator:
    numerators = []
    denominators = []
    ratios = []
    for u in range(2):
        numerator = len(X_train_orig[(X_train_orig.decile3 == u) & (X_train_orig.lsat == 0)].index)
        denominator = len(X_train_orig[(X_train_orig.decile3 == u) & (X_train_orig.lsat == 1)].index)
        numerators.append(numerator)
        denominators.append(denominator)
        ratio = numerator / denominator
        ratios.append(ratio)

# generate Pr(C=1|race, decile3, lsat) for each training data point
if indicator:
    p = np.ones_like(X_train_orig.racetxt).astype(float)
    p[(X_train_orig.lsat == 1) & (X_train_orig.decile3 == 0)] = 0.1
    p[(X_train_orig.lsat == 0) & (X_train_orig.decile3 == 1)] = 0.3

# **Sample based on the conditional probability**

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

    numerators = []
    denominators = []
    ratios = []
    for u in range(2):
        numerator = len(
            X_train_orig_sampled[(X_train_orig_sampled.decile3 == u) & (X_train_orig_sampled.lsat == 0)].index)
        denominator = len(
            X_train_orig_sampled[(X_train_orig_sampled.decile3 == u) & (X_train_orig_sampled.lsat == 1)].index)
        numerators.append(numerator)
        denominators.append(denominator)
        ratio = numerator / denominator
        ratios.append(ratio)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'racetxt'
    parent_1 = 'decile3'
    parent_2 = 'lsat'
    parent_3 = 'lsat'

# **Bound**

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)
    spd_0 = computeFairness(clf.predict_proba(X_train_sampled), X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("BF: ", -spd_0)
    y_pred_train = clf.predict_proba(X_train)
    spd_0 = computeFairness(clf.predict_proba(X_train), X_train_orig, y_train, 0, dataset)
    print("AF: ", -spd_0)

    # Group the training data based on parent (sensitive attr=0)
    protected_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & (X_train_orig_sampled[parent_2] == u) & (
                    X_train_orig_sampled[sens_attr] == 0)].index
            if len(idx) > 0:
                protected_groups.append(idx)

    score_protected_groups = []
    for idx in protected_groups:
        score_protected_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    # Group the training data based on parent (sensitive attr=1)
    privileged_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & (X_train_orig_sampled[parent_2] == u) & (
                    X_train_orig_sampled[sens_attr] == 1)].index
            if len(idx) > 0:
                privileged_groups.append(idx)

    score_privileged_groups = []
    for idx in privileged_groups:
        score_privileged_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    print(f'Upper Bound: {np.max(score_privileged_groups) - np.min(score_protected_groups)}')
    print(f'Lower Bound: {-np.max(score_protected_groups) + np.min(score_privileged_groups)}')

# ### Law scenario 3: make collider to be the child of lsat and label (try to use sampling to increase the correlation between label and race on the training set)

indicator = ((dataset == 'law') and (scenario == 3))
if indicator:
    numerators = []
    denominators = []
    ratios = []
    for u in range(2):
        numerator = len(X_train_orig[(y_train == u) & (X_train_orig.lsat == 0)].index)
        denominator = len(X_train_orig[(y_train == u) & (X_train_orig.lsat == 1)].index)
        numerators.append(numerator)
        denominators.append(denominator)
        ratio = numerator / denominator
        ratios.append(ratio)

if indicator:
    p = np.ones_like(X_train_orig.racetxt).astype(float)
    p[(X_train_orig.lsat == 1) & (y_train == 0)] = 0.3
    p[(X_train_orig.lsat == 0) & (y_train == 1)] = 0.1

# **Sample based on the conditional probability**

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

    numerators = []
    denominators = []
    ratios = []
    for u in range(2):
        numerator = len(
            X_train_orig_sampled[(X_train_orig_sampled.decile3 == u) & (X_train_orig_sampled.lsat == 0)].index)
        denominator = len(
            X_train_orig_sampled[(X_train_orig_sampled.decile3 == u) & (X_train_orig_sampled.lsat == 1)].index)
        numerators.append(numerator)
        denominators.append(denominator)
        ratio = numerator / denominator
        ratios.append(ratio)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'racetxt'
    parent_1 = 'lsat'
    parent_2 = 'lsat'
    parent_3 = 'y'

# **Bound**

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)
    spd_0 = computeFairness(clf.predict_proba(X_train_sampled), X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("BF: ", -spd_0)
    y_pred_train = clf.predict_proba(X_train)
    spd_0 = computeFairness(clf.predict_proba(X_train), X_train_orig, y_train, 0, dataset)
    print("AF: ", -spd_0)

    # Group the training data based on parent (sensitive attr=0)
    protected_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & (X_train_orig_sampled[parent_2] == u) & (
                    X_train_orig_sampled[sens_attr] == 0)].index
            if len(idx) > 0:
                protected_groups.append(idx)

    score_protected_groups = []
    for idx in protected_groups:
        score_protected_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    # Group the training data based on parent (sensitive attr=1)
    privileged_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & (X_train_orig_sampled[parent_2] == u) & (
                    X_train_orig_sampled[sens_attr] == 1)].index
            if len(idx) > 0:
                privileged_groups.append(idx)

    score_privileged_groups = []
    for idx in privileged_groups:
        score_privileged_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    print(f'Upper Bound: {np.max(score_privileged_groups) - np.min(score_protected_groups)}')
    print(f'Lower Bound: {-np.max(score_protected_groups) + np.min(score_privileged_groups)}')

### Law scenario 4

# In[ ]:
indicator = ((dataset == 'law') and (scenario == 4))

if indicator:
    p = np.ones_like(X_train_orig.racetxt).astype(float)
    p[(X_train_orig.racetxt == 1) & (X_train_orig.lsat == 0)] = 0.7
    p[(X_train_orig.racetxt == 0) & (X_train_orig.lsat == 1)] = 0.8

if indicator:
    np.random.seed(0)
    sample_bool = np.zeros(len(X_train))
    for idx in range(len(X_train)):
        sample_bool[idx] = np.random.binomial(n=1, p=p[idx])  # True or False

if indicator:
    X_train_orig_sampled = X_train_orig.loc[np.where(sample_bool)[0]]
    X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
    X_train_sampled = X_train[np.where(sample_bool)[0]]
    y_train_sampled = y_train[np.where(sample_bool)[0]].reset_index(drop=True)

if indicator:
    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

    clf.fit(X_train_sampled, y_train_sampled)

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

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)

    spd_0 = computeFairness(y_pred_train, X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("Initial statistical parity: ", spd_0)

    accuracy_0 = computeAccuracy(y_train_sampled, y_pred_train, clf.threshold)
    print("Initial accuracy: ", accuracy_0)

if indicator:
    sens_attr = 'racetxt'
    parent_1 = 'racetxt'
    parent_2 = 'racetxt'
    parent_3 = 'lsat'

if indicator:
    y_pred_train = clf.predict_proba(X_train_sampled)
    spd_0 = computeFairness(clf.predict_proba(X_train_sampled), X_train_orig_sampled, y_train_sampled, 0, dataset)
    print("BF: ", -spd_0)
    y_pred_train = clf.predict_proba(X_train)
    spd_0 = computeFairness(clf.predict_proba(X_train), X_train_orig, y_train, 0, dataset)
    print("AF: ", -spd_0)

    # Group the training data based on parent (sensitive attr=0)
    protected_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & \
                                       (X_train_orig_sampled[parent_2] == u) & \
                                       (X_train_orig_sampled[sens_attr] == 0)].index
            if len(idx) > 0:
                protected_groups.append(idx)

    score_protected_groups = []
    for idx in protected_groups:
        score_protected_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    # Group the training data based on parent (sensitive attr=1)
    privileged_groups = []
    for v in np.sort(X_train_orig_sampled[parent_1].unique()):
        for u in np.sort(X_train_orig_sampled[parent_2].unique()):
            idx = X_train_orig_sampled[(X_train_orig_sampled[parent_1] == v) & \
                                       (X_train_orig_sampled[parent_2] == u) & \
                                       (X_train_orig_sampled[sens_attr] == 1)].index
            if len(idx) > 0:
                privileged_groups.append(idx)

    score_privileged_groups = []
    for idx in privileged_groups:
        score_privileged_groups.append(np.mean(clf.predict_proba(X_train_sampled)[idx]))

    print(f'Upper Bound: {np.max(score_privileged_groups) - np.min(score_protected_groups)}')
    print(f'Lower Bound: {-np.max(score_protected_groups) + np.min(score_privileged_groups)}')

# In[ ]:


# ## Regularizer

BFs = []
AFs = []
ubs = []
lbs = []
accs = []
f1s = []
aucs = []

# ### Prepare Validation Set

X_train_orig_sampled, X_val_orig_sampled, y_train_sampled, y_val_sampled = train_test_split(X_train_orig_sampled,
                                                                                            y_train_sampled,
                                                                                            test_size=0.25,
                                                                                            random_state=val_seed)

sc = StandardScaler()
X_train_sampled = sc.fit_transform(X_train_orig_sampled)
X_val_sampled = sc.transform(X_val_orig_sampled)
X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
X_val_orig_sampled = X_val_orig_sampled.reset_index(drop=True)
y_train_sampled = y_train_sampled.reset_index(drop=True)
y_val_sampled = y_val_sampled.reset_index(drop=True)
X_test = sc.transform(X_test_orig)
X_train = sc.transform(X_train_orig)

lfr_pth = f'/Volumes/GoogleDrive/My Drive/Fairness/processed/LFR_{val_seed}_{dataset}_{scenario}.npy'
reweighing_pth = f'/Volumes/GoogleDrive/My Drive/Fairness/processed/Reweighing_{val_seed}_{dataset}_{scenario}.npy'
lfr_data = np.load(lfr_pth, allow_pickle=True)[()]
reweighing_data = np.load(reweighing_pth, allow_pickle=True)[()]

### Reweighing

info_dict = dict()
info_dict['x_train'] = X_train_sampled
info_dict['y_train'] = y_train_sampled.to_numpy()
info_dict['x_val'] = X_val_sampled
info_dict['y_val'] = y_val_sampled.to_numpy()
info_dict['train_weights'] = torch.Tensor(reweighing_data['train_weights'])
info_dict['train_weights'].requires_grad = False
info_dict['val_weights'] = torch.Tensor(reweighing_data['val_weights'])
info_dict['val_weights'].requires_grad = False

print(f'---------- Reweighing --- SC{scenario} ----------')
clf = eval(clf_name+'_Reweighing')(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)
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

record_statistics(clf, True, 0)
record_statistics(clf, True, 1)

### LFR

X_train_sampled = sc.fit_transform(lfr_data['X_train'])
X_val_sampled = sc.transform(lfr_data['X_val'])
X_test = sc.transform(lfr_data['X_test'])

X_train_orig_sampled = X_train_orig_sampled.reset_index(drop=True)
X_val_orig_sampled = X_val_orig_sampled.reset_index(drop=True)

y_train_sampled = pd.Series(lfr_data['y_train'], name='outcome')
y_val_sampled = pd.Series(lfr_data['y_val'], name='outcome')

print(len(y_train_sampled), len(X_train_sampled))

info_dict = dict()
info_dict['x_train'] = X_train_sampled
info_dict['y_train'] = y_train_sampled.to_numpy()
info_dict['x_val'] = X_val_sampled
info_dict['y_val'] = y_val_sampled.to_numpy()

print(f'---------- LFR --- SC{scenario} ----------')
clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)
clf = train_clf(clf, info_dict=info_dict)

y_pred_test = clf.predict_proba(X_test)
# y_pred_train = clf.predict_proba(X_train)

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
# accs.append(accuracy_0)
# accs.append(accuracy_0)

record_statistics(clf, True, 0)
record_statistics(clf, True, 1)

dict_name = '_'

if clf_name == 'LogisticRegression':
    dict_name = dict_name + 'LR'
elif clf_name == 'SVM':
    dict_name = dict_name + 'SVM'
elif clf_name == 'NeuralNetwork':
    dict_name = dict_name + 'NN'
else:
    raise NotImplementedError

dict_name = dict_name + '.dict'
name = ['Reweighing', 'Reweighing', 'LFR', 'LFR']
for ls in [AFs, BFs, ubs, lbs, accs, f1s, aucs]:
    assert len(ls) == len(name)

for m in ['SPD', 'EO']:
    metric_dict_name = m + dict_name

    with open(metric_dict_name, 'r') as f:
        txt = f.read()
    d = json.loads(txt)
    for n_idx, n in enumerate(name):
        if ((n_idx % 2) == 1) and (m == 'EO'):
            d[dataset][f'sc{scenario}'][str(val_seed)][n] = [float(AFs[n_idx]), float(BFs[n_idx]), float(ubs[n_idx]),
                                                             float(lbs[n_idx]), float(accs[n_idx]), float(f1s[n_idx]),
                                                             float(aucs[n_idx])]
            print('saved: ', n, m)
        elif ((n_idx % 2) == 0) and (m == 'SPD'):
            d[dataset][f'sc{scenario}'][str(val_seed)][n] = [float(AFs[n_idx]), float(BFs[n_idx]), float(ubs[n_idx]),
                                                             float(lbs[n_idx]), float(accs[n_idx]), float(f1s[n_idx]),
                                                             float(aucs[n_idx])]
            print('saved: ', n, m)
    txt = json.dumps(d)
    with open(metric_dict_name, 'w+') as f:
        f.write(txt)
        f.close()
    print('saved: ', metric_dict_name)

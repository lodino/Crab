#!/usr/bin/env python
# coding: utf-8

import copy
import random
import argparse
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
parser.add_argument("--percent", type=int, default=100)
args = parser.parse_args()

val_seeds = [42, 99, 67, 2, 23]
val_seed = val_seeds[args.valseed]

seed = 1
th = args.th

dataset = args.dataset
sens_attr = args.sens
scenario = args.scenario
percent = args.percent

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
    spd_0 = computeFairness(clf.predict_proba(X_train_sampled), X_train_orig_sampled,
                            y_train_sampled, fair_metric, dataset)
    if record:
        BFs.append(-spd_0)
    print("BF: ", -spd_0)

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

### HMDA scenario 1

indicator = ((dataset == 'hmda') and (scenario == 1))

if indicator:
    p = np.ones_like(X_train_orig.race).astype(float)
    p[(X_train_orig.race == 1) & (X_train_orig.DI == 1)] = 0.1
    p[(X_train_orig.race == 0) & (X_train_orig.DI == 2)] = 0.1
    p[(X_train_orig.race == 0) & (X_train_orig.DI == 3)] = 0.2

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
    parent_3 = 'DI'

if indicator:
    record_statistics(clf, record=False)

### HMDA scenario 2

indicator = ((dataset == 'hmda') and (scenario == 2))

if indicator:
    p = np.ones_like(X_train_orig.race).astype(float)
    p[(X_train_orig.income_brackets == 1) & (X_train_orig.DI == 1)] = 0.7
    p[(X_train_orig.income_brackets == 1) & (X_train_orig.DI == 2)] = 0.8
    # p[(X_train_orig.income_brackets == 0) & (X_train_orig.DI == 3)] = 0.2

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
    parent_1 = 'income_brackets'
    parent_2 = 'income_brackets'
    parent_3 = 'DI'
if indicator:
    record_statistics(clf, record=False)

### HMDA scenario 3

indicator = ((dataset == 'hmda') and (scenario == 3))

if indicator:
    p = np.ones_like(X_train_orig.race).astype(float)
    p[(y_train == 1) & (X_train_orig.LV == 1)] = 0.05
    p[(y_train == 0) & (X_train_orig.LV == 1)] = 0.2

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
    parent_1 = 'LV'
    parent_2 = 'LV'
    parent_3 = 'y'
if indicator:
    record_statistics(clf, record=False)

### HMDA scenario 4

indicator = ((dataset == 'hmda') and (scenario == 4))

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
    parent_1 = 'race'
    parent_2 = 'race'
    parent_3 = 'DI'
if indicator:
    record_statistics(clf, record=False)

### HMDA scenario 5

indicator = ((dataset == 'hmda') and (scenario == 5))

if indicator:
    p = np.ones_like(X_train_orig.race).astype(float)
    p[(X_train_orig.income_brackets == 1) & (X_train_orig.DI == 1)] = 0.1
    p[(X_train_orig.income_brackets == 0) & (X_train_orig.DI == 2)] = 0.1
    p[(X_train_orig.income_brackets == 1) & (X_train_orig.DI == 3)] = 0.2

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
    parent_1 = 'income_brackets'
    parent_2 = 'income_brackets'
    parent_3 = 'DI'

if indicator:
    record_statistics(clf, record=False)

### HMDA scenario 6

indicator = ((dataset == 'hmda') and (scenario == 6))

if indicator:
    p = np.ones_like(X_train_orig.race).astype(float)
    p[(y_train == 1) & (X_train_orig.DI == 1)] = 0.5
    p[(y_train == 1) & (X_train_orig.DI == 2)] = 0.5
    p[(y_train == 0) & (X_train_orig.DI == 3)] = 0.2

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
    parent_2 = 'DI'
    parent_3 = 'y'
if indicator:
    record_statistics(clf, record=False)

### Adult scenario 1

indicator = ((dataset == 'adult') and (scenario == 1))

# generate Pr(C=1|gender, income) for each training data point
if indicator:
    p = np.where(np.logical_and(X_train_orig.relationship, X_train_orig.gender), 0.11, 1.0)

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
if indicator:
    record_statistics(clf, record=False)

### Adult scenario 2
indicator = ((dataset == 'adult') and (scenario == 2))

if indicator:
    p = np.ones_like(X_train_orig.gender).astype(float)
    p[(X_train_orig.relationship == 0) & (X_train_orig.hours == 0)] = 0.7
    p[(X_train_orig.relationship == 1) & (X_train_orig.hours == 1)] = 0.8

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
    parent_2 = 'relationship'
    parent_3 = 'hours'
if indicator:
    record_statistics(clf, record=False)

### Adult scenario 3

indicator = ((dataset == 'adult') and (scenario == 3))

if indicator:
    p = np.ones_like(X_train_orig.gender).astype(float)
    p[(X_train_orig.relationship == 1) & (y_train == 0)] = 0.2
    p[(X_train_orig.relationship == 0) & (y_train == 1)] = 0.2

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
    parent_2 = 'relationship'
    parent_3 = 'y'
if indicator:
    record_statistics(clf, record=False)

### Adult scenario 4

indicator = ((dataset == 'adult') and (scenario == 4))

if indicator:
    p = np.ones_like(X_train_orig.gender).astype(float)
    p[(X_train_orig.relationship == 0) & (X_train_orig.gender == 1)] = 0.5

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
    parent_1 = 'gender'
    parent_2 = 'gender'
    parent_3 = 'relationship'
if indicator:
    record_statistics(clf, record=False)

### Adult scenario 5

indicator = ((dataset == 'adult') and (scenario == 5))

if indicator:
    p = np.ones_like(X_train_orig.gender).astype(float)
    p[(X_train_orig.relationship == 0) & (X_train_orig.hours == 0)] = 0.154 / 0.658
    p[(X_train_orig.relationship == 1) & (X_train_orig.hours == 1)] = 0.240 / 1.52

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
    parent_2 = 'relationship'
    parent_3 = 'hours'
if indicator:
    record_statistics(clf, record=False)

### Adult scenario 6

indicator = ((dataset == 'adult') and (scenario == 6))

if indicator:
    p = np.ones_like(X_train_orig.gender).astype(float)
    p[(X_train_orig.relationship == 1) & (y_train == 1)] = 0.7
    p[(X_train_orig.relationship == 0) & (y_train == 0)] = 0.6

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
    parent_2 = 'relationship'
    parent_3 = 'y'
if indicator:
    record_statistics(clf, record=False)

### Law scenario 1

indicator = ((dataset == 'law') and (scenario == 1))

if indicator:
    p = np.ones_like(X_train_orig.racetxt).astype(float)
    p[(X_train_orig.racetxt == 0) & (X_train_orig.decile3 == 0)] = 0.2
    p[(X_train_orig.racetxt == 1) & (X_train_orig.decile3 == 1)] = 0.5

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
    parent_1 = 'decile3'
    parent_2 = 'decile3'
    parent_3 = 'racetxt'
if indicator:
    record_statistics(clf, record=False)

### Law scenario 2

indicator = ((dataset == 'law') and (scenario == 2))

if indicator:
    p = np.ones_like(X_train_orig.racetxt).astype(float)
    p[(X_train_orig.lsat == 1) & (X_train_orig.decile3 == 0)] = 0.1
    p[(X_train_orig.lsat == 0) & (X_train_orig.decile3 == 1)] = 0.3

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
    parent_1 = 'decile3'
    parent_2 = 'decile3'
    parent_3 = 'lsat'
if indicator:
    record_statistics(clf, record=False)

### Law scenario 3

indicator = ((dataset == 'law') and (scenario == 3))

if indicator:
    p = np.ones_like(X_train_orig.racetxt).astype(float)
    p[(X_train_orig.lsat == 1) & (y_train == 1)] = 0.1
    p[(X_train_orig.lsat == 0) & (y_train == 0)] = 0.7

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
    parent_3 = 'y'
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

### Law scenario 4

indicator = ((dataset == 'law') and (scenario == 4))

# generate Pr(C=1|race, decile3, lsat) for each training data point
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
    record_statistics(clf, record=False)

### Law scenario 5

indicator = ((dataset == 'law') and (scenario == 5))

if indicator:
    p = np.ones_like(X_train_orig.racetxt).astype(float)
    p[(X_train_orig.lsat == 0) & (X_train_orig.ugpa == 0)] = 0.7 / 5.71
    p[(X_train_orig.lsat == 1) & (X_train_orig.ugpa == 1)] = 5 / 11.48

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
    parent_1 = 'ugpa'
    parent_2 = 'ugpa'
    parent_3 = 'lsat'
if indicator:
    record_statistics(clf, record=False)

### Law scenario 6

indicator = ((dataset == 'law') and (scenario == 6))

if indicator:
    p = np.ones_like(X_train_orig.racetxt).astype(float)
    p[(X_train_orig.lsat == 0) & (y_train == 1)] = 0.5
    p[(X_train_orig.lsat == 1) & (y_train == 0)] = 0.7

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
    parent_3 = 'y'
if indicator:
    record_statistics(clf, record=False)

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

percent_seeds = [12, 34, 56, 78, 90]
np.random.seed(percent_seeds[args.valseed])
sc_sub = StandardScaler()
subsample_idx = (np.random.random(len(X_train)) < (float(percent) / 100))
X_train_orig_sub = X_train_orig.copy()[subsample_idx]
X_train_sub = sc_sub.fit_transform(X_train_orig_sub)
y_train_sub = y_train.copy()[subsample_idx]

# ### No Regularizer
if '0' in args.algo:
    info_dict = dict()
    info_dict['x_train'] = X_train_sampled
    info_dict['y_train'] = y_train_sampled.to_numpy()
    info_dict['x_val'] = X_val_sampled
    info_dict['y_val'] = y_val_sampled.to_numpy()

    clf = eval(clf_name)(input_size=X_train.shape[-1], epoch_num=epoch_num, c=c)

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

# ### Bound Regularizer
if '1' in args.algo:
    info_dict = dict()
    info_dict['th'] = th
    info_dict['balance'] = 0.5
    if dataset == 'adult':
        if scenario == 4:
            info_dict['balance'] = 5
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

# ### Metric Regularizer
if '2' in args.algo:
    info_dict = dict()
    info_dict['th'] = th
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
    info_dict['balance'] = 0.5
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
        s_idx = X_train_orig_sub[X_train_orig_sub[sens_attr] == s].index.intersection(
            get_A_idx(X_train_orig_sub, y_train_sub, A, A_val))
        for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
            for u in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_2).unique()):
                for w in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_3).unique()):
                    idx = X_train_orig_sub[(get_attr(X_train_orig_sub, y_train_sub, parent_1) == v) & (
                            get_attr(X_train_orig_sub, y_train_sub, parent_2) == u) & (
                                                   get_attr(X_train_orig_sub, y_train_sub, parent_3) == w) & (
                                                   get_attr(X_train_orig_sub, y_train_sub, sens_attr) == s)].index
                    idx = idx.intersection(get_A_idx(X_train_orig_sub, y_train_sub, A, A_val))
                    if len(idx) > 0:
                        weights[s].append(len(idx) / len(s_idx))

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

# ### Est. Weighted Metric Regularizer (limited info)
if '4' in args.algo:
    info_dict = dict()
    extl_info = dict()
    info_dict['th'] = th
    info_dict['balance'] = 0.5
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
                    idx_unbiased = X_train_orig_sub[(get_attr(X_train_orig_sub, y_train_sub, parent_1) == v) & (
                            get_attr(X_train_orig_sub, y_train_sub, parent_2) == u) & (
                                                            get_attr(X_train_orig_sub, y_train_sub, parent_3) == w) & (
                                                            get_attr(X_train_orig_sub, y_train_sub,
                                                                     sens_attr) == s)].index
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
    positive_data = X_train[(y_train == 1) & subsample_idx]
    positive_source = sample_bool[(y_train == 1) & subsample_idx]

    all_data = X_train[subsample_idx]
    all_source = sample_bool[subsample_idx]
    print('X_train_orig size:', len(X_train_orig))
    print('IPW sample size:', len(all_data))

    Pr_C_given_X = LogisticRegression(input_size=X_train.shape[-1], epoch_num=epoch_num)
    Pr_C_given_X.fit(all_data, all_source.ravel())

    Pr_C_given_YX = LogisticRegression(input_size=positive_data.shape[-1], epoch_num=epoch_num)
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
    info_dict['balance'] = 8

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
            s_idx = X_train_orig_sub[X_train_orig_sub[sens_attr] == s].index.intersection(
                get_A_idx(X_train_orig_sub, y_train, A, A_val))
            for v in np.sort(get_attr(X_train_orig_sampled, y_train_sampled, parent_1).unique()):
                idx = X_train_orig_sub[(get_attr(X_train_orig_sub, y_train, parent_1) == v) & \
                                       (get_attr(X_train_orig_sub, y_train, sens_attr) == s)].index
                idx = idx.intersection(get_A_idx(X_train_orig_sub, y_train, A, A_val))
                if len(idx) > 0:
                    weights[s].append(len(idx) / len(s_idx))

        info_dict['weights'] = weights

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
names = ['Orig', f'Bound Reg.-{round(th, 3)}', f'BF Reg.-{round(th, 3)}',
         f'AF Reg.-{round(th, 3)}-{percent}', f'Est.AF Reg.-{round(th, 3)}-{percent}',
         'Cov. Reg.', 'Adv. Debias', f'IPW-{percent}', f'TightBound Reg.-{round(th, 3)}-{percent}',
         'Test']
print(args.algo)
name = []
for i in range(len(args.algo)):
    n_idx = int(args.algo[i])
    name.append(names[n_idx])
print(f'saving: {name}')
for ls in [AFs, BFs, ubs, lbs, accs, f1s, aucs]:
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

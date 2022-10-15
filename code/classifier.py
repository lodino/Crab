from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from utils import *
import numpy as np
import torch.nn as nn
import torch
import json

seed = 42

with open('config.json', 'r') as f:
    txt = f.read()
    dtype = json.loads(txt)['dtype']
    f.close()
if dtype == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


class LogisticRegression(nn.Module):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        super(LogisticRegression, self).__init__()
        torch.manual_seed(seed)
        self.sklearn_lr = SGDClassifier(loss='log', warm_start=True, max_iter=epoch_num, random_state=0,
                                        average=True, shuffle=False, learning_rate='constant',
                                        eta0=learning_rate, alpha=c, verbose=0)
        self.lr = torch.nn.Linear(input_size, 1, bias=True)
        self.sm = torch.nn.Sigmoid()
        self.C = c
        self.epoch_num = epoch_num
        self.criterion = logistic_loss_torch
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        self.threshold = 0.5

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x.squeeze()

    def fit(self, x, y, verbose=False, use_sklearn=False):
        torch.manual_seed(0)
        if use_sklearn:
            self.sklearn_lr.fit(x, y)
            self.C = self.sklearn_lr.C
            self.lr.weight.data = torch.Tensor(self.sklearn_lr.coef_)
            self.lr.bias.data = torch.Tensor(self.sklearn_lr.intercept_)
        else:
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            x.requires_grad = False
            y.requires_grad = False
            self.train()
            for _ in range(self.epoch_num):
                loss = self.criterion(self, x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, x):
        self.eval()
        return self.forward(torch.Tensor(x)).detach().numpy()

    def load_weights_from_another_model(self, orig_model):
        self.C = orig_model.C
        self.lr.weight.data = orig_model.lr.weight.data.clone()
        self.lr.bias.data = orig_model.lr.bias.data.clone()

    def fit_info(self, info_dict, use_sklearn=False):
        x, y = info_dict['x_train'], info_dict['y_train']
        self.fit(x, y, use_sklearn)

    def compute_loss(self, info_dict):
        self.eval()
        x, y = info_dict['x_val'], info_dict['y_val']
        loss = self.criterion(self, x, y).detach().numpy()
        return loss

    def adjust_threshold(self, info_dict):
        y_true, y_pred = info_dict['y_val'], self.predict_proba(info_dict['x_val'])
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        # avoid divided by 0 by 0 adding 0.00001 to the denominator
        fscore = (2 * precision * recall) / (precision + recall + 0.00001)
        ix = np.argmax(fscore)
        self.threshold = thresholds[ix]
        print('Best Threshold=%f, G-mean=%.3f' % (self.threshold, fscore[ix]))


class LogisticRegression_Reg_Bound(LogisticRegression):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(LogisticRegression_Reg_Bound, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, regularizer, balance=1, th=0):
        torch.manual_seed(0)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x.requires_grad = False
        y.requires_grad = False
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                protected_idx, privileged_idx = self.get_regularizer_idx(regularizer[0], regularizer[1],
                                                                         self.predict_proba(x))
                add_loss = torch.mean(self(x[privileged_idx])) - torch.mean(self(x[protected_idx]))
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.fit(x, y, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        loss = self.criterion(self, x, y).detach().numpy()
        protected_idx, privileged_idx = self.get_regularizer_idx(regularizer[0], regularizer[1],
                                                                 self.predict_proba(x))

        add_loss = np.max(
            [np.abs(np.mean(self.predict_proba(x[privileged_idx])) - np.mean(self.predict_proba(x[protected_idx]))),
             th])
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        loss += balance * add_loss
        return loss

    def get_regularizer_idx(self, protected_idxs, privileged_idxs, pred):
        score_protected_groups = []
        for idx in protected_idxs:
            score_protected_groups.append(np.mean(pred[idx]))
        score_privileged_groups = []
        for idx in privileged_idxs:
            score_privileged_groups.append(np.mean(pred[idx]))
        if (np.max(score_privileged_groups) - np.min(score_protected_groups)) >= \
                (np.max(score_protected_groups) - np.min(score_privileged_groups)):
            return protected_idxs[np.argmin(score_protected_groups)], privileged_idxs[
                np.argmax(score_privileged_groups)]
        else:
            return privileged_idxs[np.argmin(score_privileged_groups)], protected_idxs[
                np.argmax(score_protected_groups)]


class LogisticRegression_Reg_Metric(LogisticRegression):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(LogisticRegression_Reg_Metric, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, regularizer=None, balance=1, th=0):
        torch.manual_seed(0)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x.requires_grad = False
        y.requires_grad = False
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                add_loss = torch.abs(torch.mean(self(x[regularizer[1]])) - torch.mean(self(x[regularizer[0]])))
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.fit(x, y, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        loss = self.criterion(self, x, y).detach().numpy()
        add_loss = np.max([np.abs(
            np.mean(self.predict_proba(x[regularizer[1]])) - np.mean(self.predict_proba(x[regularizer[0]]))), th])
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        loss += balance * add_loss
        return loss


class LogisticRegression_Reg_WeightedMetric(LogisticRegression):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(LogisticRegression_Reg_WeightedMetric, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, external_data_weights, regularizer=None, balance=1, th=0):
        '''external_data_weights corresponds to the conditional distribution: Pr(U|S,A)'''
        torch.manual_seed(0)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x.requires_grad = False
        y.requires_grad = False
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                weighted_privileged = 0
                for u_idx, u in enumerate(regularizer[1]):
                    weighted_privileged += external_data_weights[1][u_idx] * torch.mean(self(x[u]))
                weighted_protected = 0
                for u_idx, u in enumerate(regularizer[0]):
                    weighted_protected += external_data_weights[0][u_idx] * torch.mean(self(x[u]))
                add_loss = torch.abs(weighted_privileged - weighted_protected)
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        external_data_weights = info_dict['weights']
        balance = info_dict['balance']
        self.fit(x, y, external_data_weights, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        loss = self.criterion(self, x, y).detach().numpy()
        add_loss = np.max([np.abs(self.compute_AF_est(info_dict)), th])
        balance = info_dict['balance']
        loss += balance * add_loss
        return loss

    def compute_AF_est(self, info_dict):
        self.eval()
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        external_data_weights = info_dict['weights']
        weighted_privileged = 0
        for u_idx, u in enumerate(regularizer[1]):
            weighted_privileged += external_data_weights[1][u_idx] * np.mean(self.predict_proba(x[u]))
        weighted_protected = 0
        for u_idx, u in enumerate(regularizer[0]):
            weighted_protected += external_data_weights[0][u_idx] * np.mean(self.predict_proba(x[u]))
        est = weighted_privileged - weighted_protected
        return est


class LogisticRegression_Reg_Cov(LogisticRegression):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(LogisticRegression_Reg_Cov, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, z, fair_metric, regularizer=None, balance=1, th=0):
        torch.manual_seed(0)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        weights = torch.Tensor(z - np.mean(z)).ravel()
        x.requires_grad = False
        y.requires_grad = False
        weights.requires_grad = False
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                logits = self.lr(x).ravel()
                if fair_metric == 0:
                    add_loss = torch.abs(torch.mean(logits * weights))
                elif fair_metric == 1:
                    add_loss = torch.abs(
                        torch.mean(logits * weights * (~torch.logical_xor(torch.where(logits > 0, 1, 0), y))))
                else:
                    raise NotImplementedError
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        fair_metric = info_dict['fair_metric']
        sens = info_dict['sens']
        self.fit(x, y, sens, fair_metric, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        loss = self.criterion(self, x, y).detach().numpy()
        add_loss = np.abs(
            np.mean(self.predict_proba(x[regularizer[1]])) - np.mean(self.predict_proba(x[regularizer[0]])))
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        loss += balance * add_loss
        return loss


class LogisticRegression_AD(LogisticRegression):
    def __init__(self, input_size, learning_rate=0.005, adv_learning_rate=0.4, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(LogisticRegression_AD, self).__init__(input_size, learning_rate, c, epoch_num)
        self.c = 1
        self.lr2 = nn.Linear(3, 1)
        self.criterion_adv = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.lr.parameters(), lr=learning_rate, momentum=0.9)
        self.adv_optimizer = torch.optim.SGD(self.lr2.parameters(), lr=adv_learning_rate, momentum=0.9)
        self.clf_layers = [self.lr]
        self.adv_layers = [self.lr2]

    def forward_adv(self, x, y):
        s = torch.nn.Sigmoid()((1 + self.c) * self.lr(x)).ravel()
        z_tilda = torch.nn.Sigmoid()(self.lr2(torch.stack([s, s * y, s * (1 - y)], dim=1)))
        return z_tilda

    def loss_adv(self, x, y, z):
        z_tilda = self.forward_adv(x, y)
        loss = self.criterion_adv(z_tilda, z)
        return loss

    def fit(self, x, y, z, fair_metric, acc_metric, th=1, balance=0.5, loss_balance=0.5, load=True):
        torch.manual_seed(0)
        spds = []
        accs = []
        acc_advs = []
        min_loss = 10

        saved_flag = False
        x = torch.Tensor(x)
        x.requires_grad = False
        y = torch.Tensor(y)
        y.requires_grad = False
        z = torch.Tensor(z).reshape(-1, 1)
        z.requires_grad = False
        self.train()
        for e in range(self.epoch_num):
            for layer in self.clf_layers:
                layer.requires_grad = False
            for layer in self.adv_layers:
                layer.requires_grad = True
            adv_loss = self.loss_adv(x, y, z)
            self.adv_optimizer.zero_grad()
            adv_loss.backward()
            self.adv_optimizer.step()

            for layer in self.clf_layers:
                layer.requires_grad = True
            for layer in self.adv_layers:
                layer.requires_grad = False
            loss = self.criterion(self, x, y) - balance * self.loss_adv(x, y, z)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (e % 10) == 0:
                self.eval()
                spd_0 = fair_metric(self)
                # y_pred_val = self.predict_proba(x_val)
                # spd_0 = computeFairness(y_pred_val, x_val, y_val, fair_metric, dataset)
                spds.append(spd_0)
                accuracy_0 = acc_metric(self)
                # accuracy_0 = computeAccuracy(y_val, y_pred_val)
                accs.append(accuracy_0)
                weighted_loss = self.criterion(self, x, y) + loss_balance * np.max([np.abs(spd_0), th])
                if weighted_loss <= min_loss:
                    min_loss = weighted_loss
                    torch.save(self.state_dict(), 'best_params/best_LR_AD.pth')
                    saved_flag = True

                zt = self.forward_adv(x, y).detach().numpy()
                zt = np.where(zt > 0.5, 1, 0).ravel()
                acc_advs.append(np.sum(zt == z.detach().numpy().ravel()) / len(zt))
        if saved_flag and load:
            self.load_state_dict(torch.load('best_params/best_LR_AD.pth'))
        self.eval()
        return spds, accs, acc_advs


class LogisticRegression_IPW(LogisticRegression):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(LogisticRegression_IPW, self).__init__(input_size, learning_rate, c, epoch_num)

    def apply_weights(self, x, weights):
        return torch.clamp(self(x) * weights, 0, 1)

    def fit(self, x, y, regularizer=None, balance=1, th=1):
        x = torch.Tensor(x)
        x.requires_grad = False
        y = torch.Tensor(y)
        y.requires_grad = False
        weights = (self.clf_numerator(x) / self.clf_denominator(x)).detach().squeeze()
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                add_loss = torch.abs(torch.mean(self.apply_weights(x[regularizer[1]], weights[regularizer[1]])) \
                                     - torch.mean(self.apply_weights(x[regularizer[0]], weights[regularizer[0]])))
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        x, y = info_dict['x_train'], info_dict['y_train']
        th = info_dict['th']
        self.clf_numerator = info_dict['clf_numerator']
        self.clf_denominator = info_dict['clf_denominator']
        self.clf_numerator.requires_grad = False
        self.clf_denominator.requires_grad = False
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.fit(x, y, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        x, y = torch.Tensor(x), torch.Tensor(y)
        loss = self.criterion(self, x, y).detach().numpy()
        weights = (self.clf_numerator(x) / self.clf_denominator(x)).detach().squeeze()
        add_loss = np.abs(np.mean(self.apply_weights(x[regularizer[1]], weights[regularizer[1]]).detach().numpy()) \
                          - np.mean(self.apply_weights(x[regularizer[0]], weights[regularizer[0]]).detach().numpy()))
        return loss + balance * add_loss


class LogisticRegression_Reg_Tighter_Bound(LogisticRegression):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(LogisticRegression_Reg_Tighter_Bound, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, regularizer, subgroup_idx, balance=1, th=0):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x.requires_grad = False
        y.requires_grad = False
        protected_subgroup_idx, privileged_subgroup_idx = subgroup_idx
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                preds = self.predict_proba(x)
                first_terms = self.weights
                second_terms = []
                for u1 in range(len(protected_subgroup_idx)):
                    protected_idx, privileged_idx = [regularizer[0][i] for i in protected_subgroup_idx[u1]], \
                                                    [regularizer[1][i] for i in privileged_subgroup_idx[u1]]
                    protected_score, privileged_score = [np.mean(preds[i]) for i in protected_idx], [np.mean(preds[i])
                                                                                                     for i in
                                                                                                     privileged_idx]
                    second_terms.append([[protected_subgroup_idx[u1][np.argmax(protected_score)],
                                          protected_subgroup_idx[u1][np.argmin(protected_score)]],
                                         [privileged_subgroup_idx[u1][np.argmax(privileged_score)],
                                          privileged_subgroup_idx[u1][np.argmin(privileged_score)]]])

                loss_option1 = torch.stack(
                    [first_terms[1][u1] * torch.mean(self(x[regularizer[1][second_terms[u1][1][0]]])) for u1 in
                     range(len(privileged_subgroup_idx))]).sum() - \
                               torch.stack(
                                   [first_terms[0][u1] * torch.mean(self(x[regularizer[0][second_terms[u1][0][1]]])) for
                                    u1 in range(len(protected_subgroup_idx))]).sum()

                loss_option2 = torch.stack(
                    [first_terms[1][u1] * torch.mean(self(x[regularizer[1][second_terms[u1][1][1]]])) for u1 in
                     range(len(privileged_subgroup_idx))]).sum() - \
                               torch.stack(
                                   [first_terms[0][u1] * torch.mean(self(x[regularizer[0][second_terms[u1][0][0]]])) for
                                    u1 in range(len(protected_subgroup_idx))]).sum()

                add_loss = torch.max(torch.stack([torch.abs(loss_option1), torch.abs(loss_option2)]))
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.weights = info_dict['weights']
        subgroup_idx = info_dict['train_subgroup_idx']
        self.fit(x, y, regularizer, subgroup_idx, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        subgroup_idx = info_dict['val_subgroup_idx']
        protected_subgroup_idx, privileged_subgroup_idx = subgroup_idx

        loss = self.criterion(self, x, y).detach().numpy()
        preds = self.predict_proba(x)
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        first_terms = self.weights
        second_terms = []
        for u1 in range(len(protected_subgroup_idx)):
            protected_idx, privileged_idx = [regularizer[0][i] for i in protected_subgroup_idx[u1]], \
                                            [regularizer[1][i] for i in privileged_subgroup_idx[u1]]

            protected_score, privileged_score = [np.mean(preds[i]) for i in protected_idx], [np.mean(preds[i]) for i in
                                                                                             privileged_idx]
            second_terms.append([[np.max(protected_score), np.min(protected_score)],
                                 [np.max(privileged_score), np.min(privileged_score)]])
        loss_option1 = sum(
            [first_terms[1][u1] * second_terms[u1][1][0] for u1 in range(len(privileged_subgroup_idx))]) - \
                       sum([first_terms[0][u1] * second_terms[u1][0][1] for u1 in range(len(protected_subgroup_idx))])

        loss_option2 = sum(
            [first_terms[1][u1] * second_terms[u1][1][1] for u1 in range(len(privileged_subgroup_idx))]) - \
                       sum([first_terms[0][u1] * second_terms[u1][0][0] for u1 in range(len(protected_subgroup_idx))])

        add_loss = max([abs(loss_option1), abs(loss_option2), th])
        loss += balance * add_loss
        return loss


class LogisticRegression_Reweighing(LogisticRegression):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(LogisticRegression_Reweighing, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, use_sklearn=False, sample_weights=None):
        if use_sklearn:
            self.sklearn_lr.fit(x, y)
            self.C = self.sklearn_lr.C
            self.lr.weight.data = torch.Tensor(self.sklearn_lr.coef_)
            self.lr.bias.data = torch.Tensor(self.sklearn_lr.intercept_)
        else:
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            self.train()
            for _ in range(self.epoch_num):
                loss = self.criterion(self, x, y, sample_weights)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def fit_info(self, info_dict, use_sklearn=False):
        x, y = info_dict['x_train'], info_dict['y_train']
        if 'train_weights' in info_dict:
            sample_weights = info_dict['train_weights']
        else:
            sample_weights = None
        self.fit(x, y, use_sklearn, sample_weights)

    def compute_loss(self, info_dict):
        self.eval()
        if 'val_weights' in info_dict:
            sample_weights = info_dict['val_weights']
        else:
            sample_weights = None
        x, y = info_dict['x_val'], info_dict['y_val']
        loss = self.criterion(self, x, y, sample_weights).detach().numpy()
        return loss


class SVM(nn.Module):
    def __init__(self, input_size, learning_rate=0.05, c=0.1, epoch_num=100, kernel='linear'):
        super(SVM, self).__init__()
        torch.manual_seed(seed)
        self.sklearn_svc = SGDClassifier(random_state=0, warm_start=True, max_iter=epoch_num,
                                         average=True, shuffle=False, learning_rate='constant',
                                         eta0=learning_rate, alpha=c, loss='hinge')
        self.lr = torch.nn.Linear(input_size, 1, bias=True)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.8)
        self.smooth_hinge = torch.nn.Softplus(beta=1)
        self.C = c
        self.criterion = svm_loss_torch
        self.epoch_num = epoch_num
        self.threshold = 0.5
        if kernel != 'linear':
            raise NotImplementedError

    def decision_function(self, x):
        if ~isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = self.lr(x)
        return x.squeeze()

    def forward(self, x):
        if ~isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = self.lr(x)
        x = 1 / (1 + torch.exp(-x))
        return x.squeeze()

    def fit(self, x, y, use_sklearn=False):
        if use_sklearn:
            self.sklearn_svc.fit(x, y)
            self.C = self.sklearn_svc.C
            self.lr.weight.data = torch.Tensor(self.sklearn_svc.coef_)
            self.lr.bias.data = torch.Tensor(self.sklearn_svc.intercept_)
        else:
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            for _ in range(self.epoch_num):
                loss = self.criterion(self, x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, x):
        self.eval()
        return self.forward(torch.Tensor(x)).detach().numpy()

    def compute_loss(self, info_dict):
        self.eval()
        x, y = info_dict['x_val'], info_dict['y_val']
        loss = self.criterion(self, x, y).detach().numpy()
        return loss

    def fit_info(self, info_dict, use_sklearn=False):
        x, y = info_dict['x_train'], info_dict['y_train']
        self.fit(x, y, use_sklearn)

    def adjust_threshold(self, info_dict):
        y_true, y_pred = info_dict['y_val'], self.predict_proba(info_dict['x_val'])
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        # avoid divided by adding 0.00001 to the denominator
        fscore = (2 * precision * recall) / (precision + recall + 0.00001)
        ix = np.argmax(fscore)
        self.threshold = thresholds[ix]
        print('Best Threshold=%f, G-mean=%.3f' % (self.threshold, fscore[ix]))


class SVM_Reg_Bound(SVM):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(SVM_Reg_Bound, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, regularizer, balance=1, th=0):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x.requires_grad = False
        y.requires_grad = False
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                protected_idx, privileged_idx = self.get_regularizer_idx(regularizer[0], regularizer[1],
                                                                         self.predict_proba(x))
                add_loss = torch.mean(self(x[privileged_idx])) - torch.mean(self(x[protected_idx]))
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.fit(x, y, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        loss = self.criterion(self, x, y).detach().numpy()
        protected_idx, privileged_idx = self.get_regularizer_idx(regularizer[0], regularizer[1],
                                                                 self.predict_proba(x))

        add_loss = np.max(
            [np.abs(np.mean(self.predict_proba(x[privileged_idx])) - np.mean(self.predict_proba(x[protected_idx]))),
             th])
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        loss += balance * add_loss
        return loss

    def get_regularizer_idx(self, protected_idxs, privileged_idxs, pred):
        score_protected_groups = []
        for idx in protected_idxs:
            score_protected_groups.append(np.mean(pred[idx]))
        score_privileged_groups = []
        for idx in privileged_idxs:
            score_privileged_groups.append(np.mean(pred[idx]))
        if (np.max(score_privileged_groups) - np.min(score_protected_groups)) >= \
                (np.max(score_protected_groups) - np.min(score_privileged_groups)):
            return protected_idxs[np.argmin(score_protected_groups)], privileged_idxs[
                np.argmax(score_privileged_groups)]
        else:
            return privileged_idxs[np.argmin(score_privileged_groups)], protected_idxs[
                np.argmax(score_protected_groups)]


class SVM_Reg_Metric(SVM):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(SVM_Reg_Metric, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, regularizer=None, balance=1, th=0):
        torch.manual_seed(0)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x.requires_grad = False
        y.requires_grad = False
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                add_loss = torch.abs(torch.mean(self(x[regularizer[1]])) - torch.mean(self(x[regularizer[0]])))
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.fit(x, y, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        loss = self.criterion(self, x, y).detach().numpy()
        add_loss = np.max([np.abs(
            np.mean(self.predict_proba(x[regularizer[1]])) - np.mean(self.predict_proba(x[regularizer[0]]))), th])
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        loss += balance * add_loss
        return loss


class SVM_Reg_WeightedMetric(SVM):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(SVM_Reg_WeightedMetric, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, external_data_weights, regularizer=None, balance=1, th=0):
        '''external_data_weights corresponds to the conditional distribution: Pr(U|S,A)'''
        torch.manual_seed(0)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x.requires_grad = False
        y.requires_grad = False
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                weighted_privileged = 0
                for u_idx, u in enumerate(regularizer[1]):
                    weighted_privileged += external_data_weights[1][u_idx] * torch.mean(self(x[u]))
                weighted_protected = 0
                for u_idx, u in enumerate(regularizer[0]):
                    weighted_protected += external_data_weights[0][u_idx] * torch.mean(self(x[u]))
                add_loss = torch.abs(weighted_privileged - weighted_protected)
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        external_data_weights = info_dict['weights']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.fit(x, y, external_data_weights, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        external_data_weights = info_dict['weights']
        loss = self.criterion(self, x, y).detach().numpy()
        add_loss = np.max([np.abs(self.compute_AF_est(info_dict)), th])
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        loss += balance * add_loss
        return loss

    def compute_AF_est(self, info_dict):
        self.eval()
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        external_data_weights = info_dict['weights']
        weighted_privileged = 0
        for u_idx, u in enumerate(regularizer[1]):
            weighted_privileged += external_data_weights[1][u_idx] * np.mean(self.predict_proba(x[u]))
        weighted_protected = 0
        for u_idx, u in enumerate(regularizer[0]):
            weighted_protected += external_data_weights[0][u_idx] * np.mean(self.predict_proba(x[u]))
        est = weighted_privileged - weighted_protected
        return est


class SVM_Reg_Cov(SVM):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(SVM_Reg_Cov, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, z, fair_metric, regularizer=None, balance=1, th=0):
        torch.manual_seed(0)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        weights = torch.Tensor(z - np.mean(z)).ravel()
        x.requires_grad = False
        y.requires_grad = False
        weights.requires_grad = False
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                logits = self.lr(x).ravel()
                if fair_metric == 0:
                    add_loss = torch.abs(torch.mean(logits * weights))
                elif fair_metric == 1:
                    add_loss = torch.abs(
                        torch.mean(logits * weights * (~torch.logical_xor(torch.where(logits > 0, 1, 0), y))))
                else:
                    raise NotImplementedError
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        fair_metric = info_dict['fair_metric']
        sens = info_dict['sens']
        self.fit(x, y, sens, fair_metric, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        loss = self.criterion(self, x, y).detach().numpy()
        add_loss = np.abs(
            np.mean(self.predict_proba(x[regularizer[1]])) - np.mean(self.predict_proba(x[regularizer[0]])))
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        loss += balance * add_loss
        return loss


class SVM_AD(SVM):
    def __init__(self, input_size, learning_rate=0.005, adv_learning_rate=0.4, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(SVM_AD, self).__init__(input_size, learning_rate, c, epoch_num)
        self.c = 1
        self.lr2 = nn.Linear(3, 1)
        self.criterion_adv = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.lr.parameters(), lr=learning_rate, momentum=0.9)
        self.adv_optimizer = torch.optim.SGD(self.lr2.parameters(), lr=adv_learning_rate, momentum=0.9)
        self.clf_layers = [self.lr]
        self.adv_layers = [self.lr2]

    def forward_adv(self, x, y):
        s = torch.nn.Sigmoid()((1 + self.c) * self.lr(x)).ravel()
        z_tilda = torch.nn.Sigmoid()(self.lr2(torch.stack([s, s * y, s * (1 - y)], dim=1)))
        return z_tilda

    def loss_adv(self, x, y, z):
        z_tilda = self.forward_adv(x, y)
        loss = self.criterion_adv(z_tilda, z)
        return loss

    def fit(self, x, y, z, fair_metric, acc_metric, th=1, balance=0.5, loss_balance=0.5, load=True):
        torch.manual_seed(0)
        spds = []
        accs = []
        acc_advs = []
        min_loss = 10

        saved_flag = False
        x = torch.Tensor(x)
        x.requires_grad = False
        y = torch.Tensor(y)
        y.requires_grad = False
        z = torch.Tensor(z).reshape(-1, 1)
        z.requires_grad = False
        self.train()
        for e in range(self.epoch_num):
            for layer in self.clf_layers:
                layer.requires_grad = False
            for layer in self.adv_layers:
                layer.requires_grad = True
            adv_loss = self.loss_adv(x, y, z)
            self.adv_optimizer.zero_grad()
            adv_loss.backward()
            self.adv_optimizer.step()

            for layer in self.clf_layers:
                layer.requires_grad = True
            for layer in self.adv_layers:
                layer.requires_grad = False
            loss = self.criterion(self, x, y) - balance * self.loss_adv(x, y, z)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (e % 10) == 0:
                self.eval()
                spd_0 = fair_metric(self)
                spds.append(spd_0)
                accuracy_0 = acc_metric(self)
                accs.append(accuracy_0)
                weighted_loss = self.criterion(self, x, y) + loss_balance * np.max([np.abs(spd_0), th])
                if weighted_loss <= min_loss:
                    min_loss = weighted_loss
                    torch.save(self.state_dict(), 'best_params/best_SVM_AD.pth')
                    saved_flag = True

                zt = self.forward_adv(x, y).detach().numpy()
                zt = np.where(zt > 0.5, 1, 0).ravel()
                acc_advs.append(np.sum(zt == z.detach().numpy().ravel()) / len(zt))
        if saved_flag and load:
            self.load_state_dict(torch.load('best_params/best_SVM_AD.pth'))
        self.eval()
        return spds, accs, acc_advs


class SVM_IPW(SVM):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(SVM_IPW, self).__init__(input_size, learning_rate, c, epoch_num)

    def apply_weights(self, x, weights):
        return torch.clamp(self(x) * weights, 0, 1)

    def fit(self, x, y, regularizer=None, balance=1, th=1):
        x = torch.Tensor(x)
        x.requires_grad = False
        y = torch.Tensor(y)
        y.requires_grad = False
        weights = (self.clf_numerator(x) / self.clf_denominator(x)).detach().squeeze()
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                add_loss = torch.abs(torch.mean(self.apply_weights(x[regularizer[1]], weights[regularizer[1]])) \
                                     - torch.mean(self.apply_weights(x[regularizer[0]], weights[regularizer[0]])))
            loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict_proba(self, x):
        self.eval()
        x = torch.Tensor(x)
        return self(x).detach().numpy()

    def fit_info(self, info_dict):
        x, y = info_dict['x_train'], info_dict['y_train']
        th = info_dict['th']
        self.clf_numerator = info_dict['clf_numerator']
        self.clf_denominator = info_dict['clf_denominator']
        self.clf_numerator.requires_grad = False
        self.clf_denominator.requires_grad = False
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.fit(x, y, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        x, y = torch.Tensor(x), torch.Tensor(y)
        loss = self.criterion(self, x, y).detach().numpy()
        weights = (self.clf_numerator(x) / self.clf_denominator(x)).detach().squeeze()
        add_loss = np.abs(np.mean(self.apply_weights(x[regularizer[1]], weights[regularizer[1]]).detach().numpy()) \
                          - np.mean(self.apply_weights(x[regularizer[0]], weights[regularizer[0]]).detach().numpy()))
        return loss + balance * add_loss


class SVM_Reg_Tighter_Bound(SVM):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(SVM_Reg_Tighter_Bound, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, regularizer, subgroup_idx, balance=1, th=0):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x.requires_grad = False
        y.requires_grad = False
        protected_subgroup_idx, privileged_subgroup_idx = subgroup_idx
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y)
            if regularizer is not None:
                preds = self.predict_proba(x)
                first_terms = self.weights
                second_terms = []
                for u1 in range(len(protected_subgroup_idx)):
                    protected_idx, privileged_idx = [regularizer[0][i] for i in protected_subgroup_idx[u1]], \
                                                    [regularizer[1][i] for i in privileged_subgroup_idx[u1]]
                    protected_score, privileged_score = [np.mean(preds[i]) for i in protected_idx], [np.mean(preds[i])
                                                                                                     for i in
                                                                                                     privileged_idx]
                    second_terms.append([[protected_subgroup_idx[u1][np.argmax(protected_score)],
                                          protected_subgroup_idx[u1][np.argmin(protected_score)]],
                                         [privileged_subgroup_idx[u1][np.argmax(privileged_score)],
                                          privileged_subgroup_idx[u1][np.argmin(privileged_score)]]])

                loss_option1 = torch.stack(
                    [first_terms[1][u1] * torch.mean(self(x[regularizer[1][second_terms[u1][1][0]]])) for u1 in
                     range(len(privileged_subgroup_idx))]).sum() - \
                               torch.stack(
                                   [first_terms[0][u1] * torch.mean(self(x[regularizer[0][second_terms[u1][0][1]]])) for
                                    u1 in range(len(protected_subgroup_idx))]).sum()

                loss_option2 = torch.stack(
                    [first_terms[1][u1] * torch.mean(self(x[regularizer[1][second_terms[u1][1][1]]])) for u1 in
                     range(len(privileged_subgroup_idx))]).sum() - \
                               torch.stack(
                                   [first_terms[0][u1] * torch.mean(self(x[regularizer[0][second_terms[u1][0][0]]])) for
                                    u1 in range(len(protected_subgroup_idx))]).sum()

                add_loss = torch.max(torch.stack([torch.abs(loss_option1), torch.abs(loss_option2)]))
                loss += balance * torch.clamp(add_loss, min=th)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.weights = info_dict['weights']
        subgroup_idx = info_dict['train_subgroup_idx']
        self.fit(x, y, regularizer, subgroup_idx, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        subgroup_idx = info_dict['val_subgroup_idx']
        protected_subgroup_idx, privileged_subgroup_idx = subgroup_idx

        loss = self.criterion(self, x, y).detach().numpy()
        preds = self.predict_proba(x)
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        first_terms = self.weights
        second_terms = []
        for u1 in range(len(protected_subgroup_idx)):
            protected_idx, privileged_idx = [regularizer[0][i] for i in protected_subgroup_idx[u1]], \
                                            [regularizer[1][i] for i in privileged_subgroup_idx[u1]]

            protected_score, privileged_score = [np.mean(preds[i]) for i in protected_idx], [np.mean(preds[i]) for i in
                                                                                             privileged_idx]
            second_terms.append([[np.max(protected_score), np.min(protected_score)],
                                 [np.max(privileged_score), np.min(privileged_score)]])
        loss_option1 = sum(
            [first_terms[1][u1] * second_terms[u1][1][0] for u1 in range(len(privileged_subgroup_idx))]) - \
                       sum([first_terms[0][u1] * second_terms[u1][0][1] for u1 in range(len(protected_subgroup_idx))])

        loss_option2 = sum(
            [first_terms[1][u1] * second_terms[u1][1][1] for u1 in range(len(privileged_subgroup_idx))]) - \
                       sum([first_terms[0][u1] * second_terms[u1][0][0] for u1 in range(len(protected_subgroup_idx))])

        add_loss = max([abs(loss_option1), abs(loss_option2), th])
        loss += balance * add_loss
        return loss


class SVM_Reweighing(SVM):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300):
        torch.manual_seed(seed)
        super(SVM_Reweighing, self).__init__(input_size, learning_rate, c, epoch_num)

    def fit(self, x, y, sample_weights=None):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        self.train()
        for _ in range(self.epoch_num):
            loss = self.criterion(self, x, y, sample_weights)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict_proba(self, x):
        self.eval()
        return self.forward(torch.Tensor(x)).detach().numpy()

    def fit_info(self, info_dict):
        x, y = info_dict['x_train'], info_dict['y_train']
        if 'train_weights' in info_dict:
            sample_weights = info_dict['train_weights']
        else:
            sample_weights = None
        self.fit(x, y, sample_weights)

    def compute_loss(self, info_dict):
        self.eval()
        if 'val_weights' in info_dict:
            sample_weights = info_dict['val_weights']
        else:
            sample_weights = None
        x, y = info_dict['x_val'], info_dict['y_val']
        loss = self.criterion(self, x, y, sample_weights).detach().numpy()
        return loss


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300, batch_size=-1):
        torch.manual_seed(seed)
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 10)
        self.sm1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 1)
        self.sm2 = torch.nn.Sigmoid()
        self.input_size = input_size
        self.C = c
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.criterion = nn_loss_torch
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        self.threshold = 0.5
        # best result according to grid search
        self.sklearn_nn = MLPClassifier(random_state=0, alpha=c, learning_rate='adaptive', batch_size=batch_size,
                                        solver='adam', hidden_layer_sizes=(10,), activation='logistic')

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.sm1(x)
        x = self.fc2(x)
        x = self.sm2(x)
        return x.squeeze()

    def fit(self, x, y, use_sklearn=False):
        if self.batch_size < 0:
            self.batch_size = len(x)
        if use_sklearn:
            self.sklearn_nn.fit(x, y)
            self.fc1.weight.data = torch.Tensor(self.sklearn_nn.coefs_[0]).T
            self.fc1.bias.data = torch.Tensor(self.sklearn_nn.intercepts_[0]).T
            self.fc2.weight.data = torch.Tensor(self.sklearn_nn.coefs_[1]).T
            self.fc2.bias.data = torch.Tensor(self.sklearn_nn.intercepts_[1]).T
        else:
            torch.manual_seed(0)
            num_batches = len(x) // self.batch_size + 1 if len(x) % self.batch_size != 0 else len(x) // self.batch_size
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            self.train()
            for _ in range(self.epoch_num):
                for batch_id in torch.randperm(num_batches):
                    if batch_id < num_batches - 1:
                        x_ = x[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                        y_ = y[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                    else:
                        x_ = x[batch_id * self.batch_size:]
                        y_ = y[batch_id * self.batch_size:]
                    loss = self.criterion(self, x_, y_)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def predict_proba(self, x):
        self.eval()
        x = torch.Tensor(x).view(-1, self.input_size)
        return self.forward(x).detach().numpy()

    def compute_loss(self, info_dict):
        self.eval()
        x, y = info_dict['x_val'], info_dict['y_val']
        loss = self.criterion(self, x, y)
        return loss

    def fit_info(self, info_dict):
        x, y = info_dict['x_train'], info_dict['y_train']
        self.fit(x, y)

    def adjust_threshold(self, info_dict):
        y_true, y_pred = info_dict['y_val'], self.predict_proba(info_dict['x_val'])
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        # avoid divided by 0 adding 0.00001 to the denominator
        fscore = (2 * precision * recall) / (precision + recall + 0.00001)
        ix = np.argmax(fscore)
        self.threshold = thresholds[ix]
        print('Best Threshold=%f, G-mean=%.3f' % (self.threshold, fscore[ix]))


class NeuralNetwork_Reg_Bound(NeuralNetwork):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300, batch_size=-1):
        torch.manual_seed(seed)
        super(NeuralNetwork_Reg_Bound, self).__init__(input_size, learning_rate, c, epoch_num, batch_size)

    def fit(self, x, y, regularizer, balance=1, th=0):
        if self.batch_size < 0:
            self.batch_size = len(x)
        num_batches = len(x) // self.batch_size + 1 if len(x) % self.batch_size != 0 else len(x) // self.batch_size
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        self.train()
        for _ in range(self.epoch_num):
            for batch_id in torch.randperm(num_batches):
                if batch_id < num_batches - 1:
                    x_ = x[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                    y_ = y[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                else:
                    x_ = x[batch_id * self.batch_size:]
                    y_ = y[batch_id * self.batch_size:]
                loss = self.criterion(self, x_, y_)
                if regularizer is not None:
                    protected_idx, privileged_idx = self.get_regularizer_idx(regularizer[0], regularizer[1],
                                                                             self.predict_proba(x))
                    add_loss = torch.mean(self(x[privileged_idx])) - torch.mean(self(x[protected_idx]))
                    loss += balance * torch.clamp(add_loss, min=th)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.fit(x, y, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        loss = self.criterion(self, x, y).detach().numpy()
        protected_idx, privileged_idx = self.get_regularizer_idx(regularizer[0], regularizer[1],
                                                                 self.predict_proba(x))

        add_loss = np.max(
            [np.abs(np.mean(self.predict_proba(x[privileged_idx])) - np.mean(self.predict_proba(x[protected_idx]))),
             th])
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        loss += balance * add_loss
        return loss

    def get_regularizer_idx(self, protected_idxs, privileged_idxs, pred):
        score_protected_groups = []
        for idx in protected_idxs:
            score_protected_groups.append(np.mean(pred[idx]))
        score_privileged_groups = []
        for idx in privileged_idxs:
            score_privileged_groups.append(np.mean(pred[idx]))
        if (np.max(score_privileged_groups) - np.min(score_protected_groups)) >= \
                (np.max(score_protected_groups) - np.min(score_privileged_groups)):
            return protected_idxs[np.argmin(score_protected_groups)], privileged_idxs[
                np.argmax(score_privileged_groups)]
        else:
            return privileged_idxs[np.argmin(score_privileged_groups)], protected_idxs[
                np.argmax(score_protected_groups)]


class NeuralNetwork_Reg_Metric(NeuralNetwork):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300, batch_size=-1):
        torch.manual_seed(seed)
        super(NeuralNetwork_Reg_Metric, self).__init__(input_size, learning_rate, c, epoch_num, batch_size)

    def fit(self, x, y, regularizer=False, balance=1, th=0):
        if self.batch_size < 0:
            self.batch_size = len(x)
        torch.manual_seed(0)
        num_batches = len(x) // self.batch_size + 1 if len(x) % self.batch_size != 0 else len(x) // self.batch_size
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        self.train()
        for _ in range(self.epoch_num):
            for batch_id in torch.randperm(num_batches):
                if batch_id < num_batches - 1:
                    x_ = x[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                    y_ = y[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                else:
                    x_ = x[batch_id * self.batch_size:]
                    y_ = y[batch_id * self.batch_size:]
                loss = self.criterion(self, x_, y_)
                if regularizer is not None:
                    add_loss = torch.abs(torch.mean(self(x[regularizer[1]])) - torch.mean(self(x[regularizer[0]])))
                    loss += balance * torch.clamp(add_loss, min=th)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.fit(x, y, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        loss = self.criterion(self, x, y).detach().numpy()
        add_loss = np.max([np.abs(
            np.mean(self.predict_proba(x[regularizer[1]])) - np.mean(self.predict_proba(x[regularizer[0]]))), th])
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        loss += balance * add_loss
        return loss


class NeuralNetwork_Reg_WeightedMetric(NeuralNetwork):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300, batch_size=-1):
        torch.manual_seed(seed)
        super(NeuralNetwork_Reg_WeightedMetric, self).__init__(input_size, learning_rate, c, epoch_num, batch_size)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

    def fit(self, x, y, external_data_weights, regularizer=None, balance=1, th=0):
        '''external_data_weights corresponds to the conditional distribution: Pr(U|S,A)'''
        if self.batch_size < 0:
            self.batch_size = len(x)
        torch.manual_seed(0)
        num_batches = len(x) // self.batch_size + 1 if len(x) % self.batch_size != 0 else len(x) // self.batch_size
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        self.train()
        for _ in range(self.epoch_num):
            for batch_id in torch.randperm(num_batches):
                if batch_id < num_batches - 1:
                    x_ = x[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                    y_ = y[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                else:
                    x_ = x[batch_id * self.batch_size:]
                    y_ = y[batch_id * self.batch_size:]
                loss = self.criterion(self, x_, y_)
                if regularizer is not None:
                    weighted_privileged = 0
                    for u_idx, u in enumerate(regularizer[1]):
                        weighted_privileged += external_data_weights[1][u_idx] * torch.mean(self(x[u]))
                    weighted_protected = 0
                    for u_idx, u in enumerate(regularizer[0]):
                        weighted_protected += external_data_weights[0][u_idx] * torch.mean(self(x[u]))
                    add_loss = torch.abs(weighted_privileged - weighted_protected)
                    loss += balance * torch.clamp(add_loss, min=th)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        external_data_weights = info_dict['weights']
        balance = info_dict['balance']
        self.fit(x, y, external_data_weights, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        loss = self.criterion(self, x, y).detach().numpy()
        add_loss = np.max([np.abs(self.compute_AF_est(info_dict)), th])
        balance = info_dict['balance']
        loss += balance * add_loss
        return loss

    def compute_AF_est(self, info_dict):
        self.eval()
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        external_data_weights = info_dict['weights']
        weighted_privileged = 0
        for u_idx, u in enumerate(regularizer[1]):
            weighted_privileged += external_data_weights[1][u_idx] * np.mean(self.predict_proba(x[u]))
        weighted_protected = 0
        for u_idx, u in enumerate(regularizer[0]):
            weighted_protected += external_data_weights[0][u_idx] * np.mean(self.predict_proba(x[u]))
        est = weighted_privileged - weighted_protected
        return est


class NeuralNetwork_AD(NeuralNetwork):
    def __init__(self, input_size, learning_rate=0.005, adv_learning_rate=0.4,
                 c=0.01, epoch_num=300, batch_size=-1):
        torch.manual_seed(seed)
        super(NeuralNetwork_AD, self).__init__(input_size, learning_rate, c, epoch_num, batch_size)
        self.fc3 = nn.Linear(10, 1)
        self.criterion_adv = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(list(self.fc1.parameters()) + list(self.fc2.parameters()),
                                         lr=learning_rate, momentum=0.9)
        self.adv_optimizer = torch.optim.SGD(self.fc3.parameters(), lr=adv_learning_rate, momentum=0.9)
        self.clf_layers = [self.fc1, self.fc2]
        self.adv_layers = [self.fc3]

    def forward_adv(self, x, y):
        z_tilda = torch.nn.Sigmoid()(self.fc3(self.sm1(self.fc1(x))))
        return z_tilda

    def loss_adv(self, x, y, z):
        z_tilda = self.forward_adv(x, y)
        loss = self.criterion_adv(z_tilda, z)
        return loss

    def fit(self, x, y, z, fair_metric, acc_metric, th=1, balance=0.5, loss_balance=0.5, load=True):
        if self.batch_size < 0:
            self.batch_size = len(x)
        torch.manual_seed(0)
        biass = []
        accs = []
        acc_advs = []
        min_loss = 10

        saved_flag = False
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        z = torch.Tensor(z).reshape(-1, 1)
        x.requires_grad = False
        y.requires_grad = False
        z.requires_grad = False
        self.train()
        for e in range(self.epoch_num):
            for layer in self.clf_layers:
                layer.requires_grad = False
            for layer in self.adv_layers:
                layer.requires_grad = True
            adv_loss = self.loss_adv(x, y, z)
            self.adv_optimizer.zero_grad()
            adv_loss.backward()
            self.adv_optimizer.step()

            for layer in self.clf_layers:
                layer.requires_grad = True
            for layer in self.adv_layers:
                layer.requires_grad = False
            loss = self.criterion(self, x, y) - balance * self.loss_adv(x, y, z)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (e % 10) == 0:
                self.eval()
                bias = fair_metric(self)
                biass.append(bias)
                accuracy_0 = acc_metric(self)
                accs.append(accuracy_0)
                weighted_loss = self.criterion(self, x, y) + loss_balance * np.max([np.abs(bias), th])
                if weighted_loss <= min_loss:
                    min_loss = weighted_loss
                    torch.save(self.state_dict(), 'best_params/best_NN_AD.pth')
                    saved_flag = True

                zt = self.forward_adv(x, y).detach().numpy()
                zt = np.where(zt > 0.5, 1, 0).ravel()
                acc_advs.append(np.sum(zt == z.detach().numpy().ravel()) / len(zt))
        if saved_flag and load:
            self.load_state_dict(torch.load('best_params/best_NN_AD.pth'))
        self.eval()
        return biass, accs, acc_advs


class NeuralNetwork_IPW(NeuralNetwork):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300, batch_size=-1):
        torch.manual_seed(seed)
        super(NeuralNetwork_IPW, self).__init__(input_size, learning_rate, c, epoch_num, batch_size)

    def fit(self, x, y, regularizer=False, balance=1, th=0):
        if self.batch_size < 0:
            self.batch_size = len(x)
        torch.manual_seed(0)
        num_batches = len(x) // self.batch_size + 1 if len(x) % self.batch_size != 0 else len(x) // self.batch_size
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x.requires_grad = False
        y.requires_grad = False
        weights = (self.clf_numerator(x) / self.clf_denominator(x)).detach().squeeze()
        self.train()
        for _ in range(self.epoch_num):
            for batch_id in torch.randperm(num_batches):
                if batch_id < num_batches - 1:
                    x_ = x[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                    y_ = y[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                else:
                    x_ = x[batch_id * self.batch_size:]
                    y_ = y[batch_id * self.batch_size:]
                loss = self.criterion(self, x_, y_)
                if regularizer is not None:
                    add_loss = torch.abs(torch.mean(self.apply_weights(x[regularizer[1]], weights[regularizer[1]])) \
                                         - torch.mean(self.apply_weights(x[regularizer[0]], weights[regularizer[0]])))
                loss += balance * torch.clamp(add_loss, min=th)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def apply_weights(self, x, weights):
        return torch.clamp(self(x) * weights, 0, 1)

    def predict_proba(self, x):
        self.eval()
        x = torch.Tensor(x)
        return self(x).detach().numpy()

    def fit_info(self, info_dict):
        x, y = info_dict['x_train'], info_dict['y_train']
        th = info_dict['th']
        self.clf_numerator = info_dict['clf_numerator']
        self.clf_denominator = info_dict['clf_denominator']
        self.clf_numerator.requires_grad = False
        self.clf_denominator.requires_grad = False
        regularizer = info_dict['train_regularizer']
        balance = info_dict['balance']
        self.fit(x, y, regularizer, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        balance = info_dict['balance']
        x, y = torch.Tensor(x), torch.Tensor(y)
        loss = self.criterion(self, x, y).detach().numpy()
        weights = (self.clf_numerator(x) / self.clf_denominator(x)).detach().squeeze()
        add_loss = np.abs(np.mean(self.apply_weights(x[regularizer[1]], weights[regularizer[1]]).detach().numpy()) \
                          - np.mean(self.apply_weights(x[regularizer[0]], weights[regularizer[0]]).detach().numpy()))
        return loss + balance * add_loss


class NeuralNetwork_Reg_Tighter_Bound(NeuralNetwork):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300, batch_size=-1):
        torch.manual_seed(seed)
        super(NeuralNetwork_Reg_Tighter_Bound, self).__init__(input_size, learning_rate, c, epoch_num, batch_size)

    def fit(self, x, y, regularizer, subgroup_idx, balance=1, th=0):
        if self.batch_size < 0:
            self.batch_size = len(x)
        torch.manual_seed(0)
        num_batches = len(x) // self.batch_size + 1 if len(x) % self.batch_size != 0 else len(x) // self.batch_size
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x.requires_grad = False
        y.requires_grad = False
        protected_subgroup_idx, privileged_subgroup_idx = subgroup_idx
        self.train()
        for _ in range(self.epoch_num):
            for batch_id in torch.randperm(num_batches):
                if batch_id < num_batches - 1:
                    x_ = x[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                    y_ = y[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                else:
                    x_ = x[batch_id * self.batch_size:]
                    y_ = y[batch_id * self.batch_size:]
                loss = self.criterion(self, x_, y_)
                if regularizer is not None:
                    preds = self.predict_proba(x)
                    first_terms = self.weights
                    second_terms = []
                    for u1 in range(len(protected_subgroup_idx)):
                        protected_idx, privileged_idx = [regularizer[0][i] for i in protected_subgroup_idx[u1]], \
                                                        [regularizer[1][i] for i in privileged_subgroup_idx[u1]]
                        protected_score, privileged_score = [np.mean(preds[i]) for i in protected_idx], [np.mean(preds[i])
                                                                                                         for i in
                                                                                                         privileged_idx]
                        second_terms.append([[protected_subgroup_idx[u1][np.argmax(protected_score)],
                                              protected_subgroup_idx[u1][np.argmin(protected_score)]],
                                             [privileged_subgroup_idx[u1][np.argmax(privileged_score)],
                                              privileged_subgroup_idx[u1][np.argmin(privileged_score)]]])

                    loss_option1 = torch.stack(
                        [first_terms[1][u1] * torch.mean(self(x[regularizer[1][second_terms[u1][1][0]]])) for u1 in
                         range(len(privileged_subgroup_idx))]).sum() - \
                                   torch.stack(
                                       [first_terms[0][u1] * torch.mean(self(x[regularizer[0][second_terms[u1][0][1]]])) for
                                        u1 in range(len(protected_subgroup_idx))]).sum()

                    loss_option2 = torch.stack(
                        [first_terms[1][u1] * torch.mean(self(x[regularizer[1][second_terms[u1][1][1]]])) for u1 in
                         range(len(privileged_subgroup_idx))]).sum() - \
                                   torch.stack(
                                       [first_terms[0][u1] * torch.mean(self(x[regularizer[0][second_terms[u1][0][0]]])) for
                                        u1 in range(len(protected_subgroup_idx))]).sum()

                    add_loss = torch.max(torch.stack([torch.abs(loss_option1), torch.abs(loss_option2)]))
                    loss += balance * torch.clamp(add_loss, min=th)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def fit_info(self, info_dict):
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_train'], info_dict['y_train']
        regularizer = info_dict['train_regularizer']
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        self.weights = info_dict['weights']
        subgroup_idx = info_dict['train_subgroup_idx']
        self.fit(x, y, regularizer, subgroup_idx, balance, th)

    def compute_loss(self, info_dict):
        self.eval()
        th = info_dict['th']  # initialized with 0
        x, y = info_dict['x_val'], info_dict['y_val']
        regularizer = info_dict['val_regularizer']
        subgroup_idx = info_dict['val_subgroup_idx']
        protected_subgroup_idx, privileged_subgroup_idx = subgroup_idx

        loss = self.criterion(self, x, y).detach().numpy()
        preds = self.predict_proba(x)
        if 'balance' in info_dict.keys():
            balance = info_dict['balance']
        else:
            balance = 0.5
        first_terms = self.weights
        second_terms = []
        for u1 in range(len(protected_subgroup_idx)):
            protected_idx, privileged_idx = [regularizer[0][i] for i in protected_subgroup_idx[u1]], \
                                            [regularizer[1][i] for i in privileged_subgroup_idx[u1]]

            protected_score, privileged_score = [np.mean(preds[i]) for i in protected_idx], [np.mean(preds[i]) for i in
                                                                                             privileged_idx]
            second_terms.append([[np.max(protected_score), np.min(protected_score)],
                                 [np.max(privileged_score), np.min(privileged_score)]])
        loss_option1 = sum(
            [first_terms[1][u1] * second_terms[u1][1][0] for u1 in range(len(privileged_subgroup_idx))]) - \
                       sum([first_terms[0][u1] * second_terms[u1][0][1] for u1 in range(len(protected_subgroup_idx))])

        loss_option2 = sum(
            [first_terms[1][u1] * second_terms[u1][1][1] for u1 in range(len(privileged_subgroup_idx))]) - \
                       sum([first_terms[0][u1] * second_terms[u1][0][0] for u1 in range(len(protected_subgroup_idx))])

        add_loss = max([abs(loss_option1), abs(loss_option2), th])
        loss += balance * add_loss
        return loss


class NeuralNetwork_Reweighing(NeuralNetwork):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=300, batch_size=-1):
        torch.manual_seed(seed)
        super(NeuralNetwork_Reweighing, self).__init__(input_size, learning_rate, c, epoch_num, batch_size)

    def fit(self, x, y, sample_weights):
        if self.batch_size < 0:
            self.batch_size = len(x)
        num_batches = len(x) // self.batch_size + (1 if (len(x) % self.batch_size != 0) else 0)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        for _ in range(self.epoch_num):
            for batch_id in range(num_batches):
                if batch_id < num_batches - 1:
                    x_ = x[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                    y_ = y[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                    weights = sample_weights[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                else:
                    x_ = x[batch_id * self.batch_size:]
                    y_ = y[batch_id * self.batch_size:]
                    weights = sample_weights[batch_id * self.batch_size:]
                loss = self.criterion(self, x_, y_, weights)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, x):
        self.eval()
        return self.forward(torch.Tensor(x)).detach().numpy()

    def fit_info(self, info_dict):
        x, y = info_dict['x_train'], info_dict['y_train']
        if 'train_weights' in info_dict:
            sample_weights = info_dict['train_weights']
        else:
            sample_weights = None
        self.fit(x, y, sample_weights)

    def compute_loss(self, info_dict):
        self.eval()
        if 'val_weights' in info_dict:
            sample_weights = info_dict['val_weights']
        else:
            sample_weights = None
        x, y = info_dict['x_val'], info_dict['y_val']
        loss = self.criterion(self, x, y, sample_weights).detach().numpy()
        return loss

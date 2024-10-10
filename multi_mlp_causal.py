import torch
import torch.nn as nn
import math
import numpy as np
import csv

import os
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from copy import deepcopy

from imblearn.over_sampling import SMOTE

from utils.util_gumbel import gumbel_softmax
from utils.util_logger import MyLogger

from counter import test_num_features


def up_sample_minority(df, aug_ratio=1.0):
    samples_pos = df[(df[df.columns[340]] == 1)]
    samples_neg = df[(df[df.columns[340]] == 0)]
    neg_pos_ratio = samples_neg.shape[0] / samples_pos.shape[0]
    upsample_num = int(neg_pos_ratio * aug_ratio) - 1

    aug_df = deepcopy(df)
    for k in range(upsample_num):
        aug_df = pd.concat([aug_df, samples_pos])

    aug_df = shuffle(aug_df)
    return aug_df


def down_sample_majority(df, sample_ratio=1.0, seed=42):
    samples_pos = df[(df[df.columns[340]] == 1)]
    samples_neg = df[(df[df.columns[340]] == 0)]
    down_sample_num = int(samples_pos.shape[0] * sample_ratio)

    new_df = deepcopy(df)
    sampled_samples_neg = samples_neg.sample(n=down_sample_num, random_state=seed)
    new_df = pd.concat([samples_pos, sampled_samples_neg])

    new_df = shuffle(new_df)
    return new_df


def down_sample_majority_replace(df, sample_ratio=1.0, mlp_num=1, seed=42):
    samples_pos = df[(df[df.columns[340]] == 1)]
    samples_neg = df[(df[df.columns[340]] == 0)]
    down_sample_num = int(samples_pos.shape[0] * sample_ratio)

    new_df_list = []
    total_neg_row_idxs = list(range(samples_neg.shape[0]))
    for mlp_idx in range(mlp_num):
        # 采样
        sampled_indices = random.sample(total_neg_row_idxs, down_sample_num)  # 从总行数中随机采样k个索引
        sampled_samples_neg = samples_neg.iloc[sampled_indices]  # 获取采样的行数据
        # 删除行数据
        print('# negative samples before downsample =', len(total_neg_row_idxs))
        total_neg_row_idxs = list(set(total_neg_row_idxs) - set(sampled_indices))
        print('# negative samples after downsample =', len(total_neg_row_idxs))
        # 构建训练集
        new_df = pd.concat([samples_pos, sampled_samples_neg])
        new_df = shuffle(new_df)
        new_df_list.append(new_df)

    return new_df_list


def down_up_sample_majority_replace(df, sample_ratio=1.0, mlp_num=1, seed=42):
    samples_pos = df[(df[df.columns[340]] == 1)]
    samples_neg = df[(df[df.columns[340]] == 0)]
    down_sample_num = int(samples_pos.shape[0] * sample_ratio)

    aug_samples_pos = deepcopy(samples_pos)
    for _ in range(int(sample_ratio) - 1):
        aug_samples_pos = pd.concat([aug_samples_pos, samples_pos])

    new_df_list = []
    total_neg_row_idxs = list(range(samples_neg.shape[0]))
    for mlp_idx in range(mlp_num):
        # 采样
        sampled_indices = random.sample(total_neg_row_idxs, down_sample_num)  # 从总行数中随机采样k个索引
        sampled_samples_neg = samples_neg.iloc[sampled_indices]  # 获取采样的行数据
        # 删除行数据
        print('# negative samples before downsample =', len(total_neg_row_idxs))
        total_neg_row_idxs = list(set(total_neg_row_idxs) - set(sampled_indices))
        print('# negative samples after downsample =', len(total_neg_row_idxs))
        # 构建训练集
        new_df = pd.concat([aug_samples_pos, sampled_samples_neg])
        new_df = shuffle(new_df)
        new_df_list.append(new_df)

    return new_df_list


def sample_SMOTE(df_train_X, df_train_Y, seed=42):
    # 使用SMOTE进行过采样
    smote = SMOTE(random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(df_train_X, df_train_Y)
    # 拼接列
    new_df = data = pd.concat([X_resampled, Y_resampled], axis=1)

    new_df = shuffle(new_df)
    return new_df


def sample_graph_bern(sample_matrix, batch_size):
    """ 
    Sample a causal graph from the causal probability graph via Bernouli
    note: Bernoulli is a special case of Gumbel Softmax 
    """
    sample_matrix = torch.sigmoid(
        sample_matrix[None, :].expand(batch_size, -1))
    return torch.bernoulli(sample_matrix)


def sample_graph_gumbel(graph, batch_size, tau=1):
    """ 
    Sample a causal graph from the causal probability graph via Gumbel Softmax
    """
    prob = torch.sigmoid(graph[None, :, None].expand(batch_size, -1, -1))
    logits = torch.concat([prob, (1 - prob)], axis=-1)
    samples = gumbel_softmax(logits, tau=tau)[:, :, 0]
    return samples


def compute_kl_div_with_prior(est_graph, prior_graph, choice='v1'):
    """
    Compute KL divergence in Variational Inference
    """
    if choice == 'v1':
        prior_graph[prior_graph == 0] = 0.5
        eps = 1e-8
        kl_div_items = torch.sum(prior_graph * torch.log((prior_graph + eps) / (est_graph + eps)) + \
                                 (1 - prior_graph) * torch.log(((1 - prior_graph) + eps) / ((1 - est_graph) + eps)))
        kl_div = torch.sum(kl_div_items * prior_graph)
        return kl_div
    if choice == 'v2':
        prior_graph = (torch.ones_like(prior_graph) * 0.5).to(device)
        eps = 1e-8
        kl_div_items = torch.sum(prior_graph * torch.log((prior_graph + eps) / (est_graph + eps)) + \
                                 (1 - prior_graph) * torch.log(((1 - prior_graph) + eps) / ((1 - est_graph) + eps)))
        kl_div = torch.sum(kl_div_items * prior_graph)
        return kl_div
    if choice == 'v3':
        eps = 1e-8
        kl_div_items = prior_graph * torch.log((prior_graph + eps) / (est_graph + eps)) + \
                       (1 - prior_graph) * torch.log(((1 - prior_graph) + eps) / ((1 - est_graph) + eps))
        kl_div = torch.sum(kl_div_items * prior_graph)
        return kl_div


def eval_graph(est_graph, true_graph):
    tp = np.sum(est_graph * true_graph)
    tn = np.sum((1 - est_graph) * (1 - true_graph))
    fp = np.sum(est_graph * (1 - true_graph))
    fn = np.sum((1 - est_graph) * true_graph)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return tp, fp, tpr, fpr, precision, f1


class MLP(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, n_layer):
        """Component for encoder and decoder

        Args:
            in_dim (int): input dimension.
            n_hid (int): model layer dimension.
            out_dim (int): output dimension.
        """
        super(MLP, self).__init__()
        dims = (
                [(in_dim, n_hid)]
                + [(n_hid, n_hid) for _ in range(n_layer - 1)]
                + [(n_hid, out_dim)]
        )
        fc_layers = [nn.Linear(pair[0], pair[1]) for pair in dims]
        bn_layers = [nn.BatchNorm1d(n_hid) for _ in range(n_layer)]
        lr_layers = [nn.LeakyReLU(0.05) for _ in range(n_layer)]
        # lr_layers = [nn.Tanh() for _ in range(n_layer)]
        # lr_layers = [nn.ReLU() for _ in range(n_layer)]
        layers = []
        for i in range(n_layer):
            layers.append(fc_layers[i])
            layers.append(bn_layers[i])
            layers.append(lr_layers[i])
        layers.append(fc_layers[-1])
        layers.append(nn.BatchNorm1d(out_dim))
        self.network = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.network(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class MultiMLP(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, n_layer, mlp_num):
        super(MultiMLP, self).__init__()
        # mlp_num: 多个MLP相当于多个doctor集思广益的效果 (集成学习的思想)
        self.networks = nn.ModuleList([MLP(in_dim, n_hid, out_dim, n_layer) for _ in range(mlp_num)])

    def forward(self, x):
        if self.training:
            Y_list = []
            for i in range(len(self.networks)):
                Y_i = self.networks[i](x[i])
                Y_list.append(Y_i)
            return torch.concat(Y_list, axis=1)
        else:
            Y_list = []
            for i in range(len(self.networks)):
                Y_i = self.networks[i](x)
                Y_list.append(Y_i)
            return torch.concat(Y_list, axis=1)


if __name__ == "__main__":
    max_dispo = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 训练设备

    # 加载数据
    dataset_dir = '../../dataset/table_data/adult/'
    dataset_name = 'adult_new.csv'
    df = pd.read_csv(os.path.join(dataset_dir, dataset_name))
    all_labels = df.columns
    target_feature = all_labels[-1]
    new_data = df.drop(labels=target_feature, axis=1)
    results = list(df[target_feature])
    inputs = []
    for index, row in new_data.iterrows():
        inputs.append(list(row.values))

    X_train = np.array(inputs)
    y_train = np.array(results)

    # 测试集
    test_name = 'adult_test.csv'
    test_df = pd.read_csv(os.path.join(dataset_dir, test_name))
    test_results = list(test_df[target_feature])
    new_data = test_df.drop(labels=target_feature, axis=1)
    test_inputs = []
    for index, row in new_data.iterrows():
        test_inputs.append(list(row.values))

    X_test = np.array(test_inputs)
    y_test = np.array(test_results)

    # 加载ground truth
    # truth_name = 'CCS_Data_ground_truth.csv'
    # true_df = pd.read_csv(os.path.join(dataset_dir, truth_name))
    # true_df = true_df.drop(labels='labels', axis=1)
    # true_graph = []
    # for index, row in true_df.iterrows():
    #     true_df.append(list(row.values))
    #
    # true_graph = np.array(true_graph)

    # 创建模型
    model = MLP(len(all_labels) - 1, 1024, 1, 3).to(device)
    func_sigmoid = nn.Sigmoid()
    graph = nn.Parameter(torch.ones([len(all_labels) - 1]).to(device))
    # init_graph = torch.zeros_like(prior_graph).to(device)
    # init_graph[torch.where(prior_graph==1)] = 99
    # graph = nn.Parameter(init_graph)

    # 优化器
    lr_model_start, lr_model_end = 1e-4, 1e-5
    lr_graph_start, lr_graph_end = 1e-2, 1e-3
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr_model_start)
    graph_optimizer = torch.optim.Adam([graph], lr=lr_graph_start)

    # 损失函数
    loss_fn = nn.BCELoss(reduction='sum')
    # loss_fn = nn.CrossEntropyLoss()

    # 训练设置
    epoch_size = 200
    batch_size = 256
    gumbel_tau = 1.0
    gumbel_tau_gamma = (1.0 / 0.1) ** (1 / epoch_size)

    # gamma = (lr_model_end / lr_model_start) ** (1 / epoch_size)
    # model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=1, gamma=gamma)
    # gamma = (lr_graph_end / lr_graph_start) ** (1 / epoch_size)
    # graph_scheduler = torch.optim.lr_scheduler.StepLR(graph_optimizer, step_size=1, gamma=gamma)

    # 封装数据 & 训练模型
    total_batch_train = len(X_train) // batch_size
    # total_batch_test = len(X_test) // batch_size
    # total_batch_train = 5
    tensorboard_log = MyLogger(log_dir='./logs', stderr=False, tensorboard=False,
                               stdout=False)
    for epoch_idx in range(epoch_size):
        # Stage1: 训练 - 固定因果图,优化MLP
        loss_list = []
        acc_list = []
        acc_pos_list = []
        acc_neg_list = []
        for batch_idx in range(total_batch_train):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size

            # 封装数据
            X, Y = X_train[start_idx:end_idx], y_train[start_idx:end_idx]
            X, Y = torch.tensor(X).to(torch.float32).to(device), torch.tensor(Y).to(torch.float32).to(device)

            # 标准化,防止梯度消失
            # mean = X.mean(dim=0)
            # std = X.std(dim=0)
            # X_norm = (X - mean) / (std + 1e-9)
            X_norm = [((X_item - X_item.mean(dim=0)) / (X_item.std(dim=0) + 1e-9)) for X_item in X]

            # 采用因果图,并掩码
            sampled_graph = torch.bernoulli(torch.sigmoid(graph))
            X_norm_masked = [X_norm_item * sampled_graph for X_norm_item in X_norm]
            X_norm_masked = torch.stack(X_norm_masked)

            # 清零梯度
            model.train()
            model_optimizer.zero_grad()

            # 前向传播
            outputs = model(X_norm_masked)
            logits = func_sigmoid(outputs)

            # 计算损失
            logits = logits.squeeze(1)
            loss = loss_fn(logits, Y)

            # 反向传播和梯度下降
            loss.backward()  # 反向传播
            model_optimizer.step()  # 更新模型参数

            # grad_sum = 0
            # for i in [0,2,4,6]:
            #     grad_sum += model_pred.network[0].weight.grad.sum()
            # if grad_sum != 0:
            #     print('Batch {}, grad != 0'.format(batch_idx+1))

            # 记录训练loss和acc
            predicted = (logits > 0.5).float()
            acc = (predicted == Y).sum().item() / (Y.sum() + (1 - Y).sum())
            acc_pos = (predicted * Y).sum().item() / (Y.sum() + 1e-9)
            acc_neg = ((1 - predicted) * (1 - Y)).sum().item() / ((1 - Y).sum() + 1e-9)
            loss_list.append(loss.item())
            acc_list.append(acc.item())
            acc_pos_list.append(acc_pos.item())
            acc_neg_list.append(acc_neg.item())

        print('Epoch {} Train[E]: loss_mean={:.4f}, acc_mean={:.4f}, acc_pos_mean={:.4f}, acc_neg_mean={:.4f}'.format(
            epoch_idx + 1, np.mean(loss_list), np.mean(acc_list), np.mean(acc_pos_list), np.mean(acc_neg_list)))
        tensorboard_log.log_metrics({"Train[E]/loss_mean": np.mean(loss_list),
                                     "Train[E]/acc_mean": np.mean(acc_list),
                                     "Train[E]/acc_pos_mean": np.mean(acc_pos_list),
                                     "Train[E]/acc_neg_mean": np.mean(acc_neg_list)}, epoch_idx + 1)

        # Stage2: 训练 - 固定MLP,优化因果图
        loss_list = []
        acc_list = []
        acc_pos_list = []
        acc_neg_list = []
        for batch_idx in range(total_batch_train):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size

            # 封装数据
            # 封装数据
            X, Y = X_train[start_idx:end_idx], y_train[start_idx:end_idx]
            X, Y = torch.tensor(X).to(torch.float32).to(device), torch.tensor(Y).to(torch.float32).to(device)

            # 标准化,防止梯度消失
            # mean = X.mean(dim=0)
            # std = X.std(dim=0)
            # X_norm = (X - mean) / (std + 1e-9)
            X_norm = [((X_item - X_item.mean(dim=0)) / (X_item.std(dim=0) + 1e-9)) for X_item in X]

            # 采用因果图,并掩码
            sampled_graph = torch.bernoulli(torch.sigmoid(graph))
            X_norm_masked = [X_norm_item * sampled_graph for X_norm_item in X_norm]
            X_norm_masked = torch.stack(X_norm_masked)

            # 清零梯度
            model_optimizer.zero_grad()
            graph_optimizer.zero_grad()

            # 前向传播
            outputs = model(X_norm_masked)
            logits = func_sigmoid(outputs)

            # 计算损失
            logits = logits.squeeze(1)
            # Y = Y.squeeze()
            likelihood = loss_fn(logits, Y)
            l1_norm = torch.norm(func_sigmoid(graph))
            loss = loss_fn(logits, Y) + 0.1 * l1_norm
            # loss = loss_fn(logits, Y) + 0.1*l1_norm

            # 反向传播和梯度下降
            loss.backward()  # 反向传播
            graph_optimizer.step()  # 更新模型参数

            # if graph.grad.sum() != 0:
            #     print('Batch {}, graph grad !=0'.format(batch_idx))

            # 记录训练loss和acc
            predicted = (logits > 0.5).float()
            acc = (predicted == Y).sum().item() / (Y.sum() + (1 - Y).sum())
            acc_pos = (predicted * Y).sum().item() / (Y.sum() + 1e-9)
            acc_neg = ((1 - predicted) * (1 - Y)).sum().item() / ((1 - Y).sum() + 1e-9)
            loss_list.append(loss.item())
            acc_list.append(acc.item())
            acc_pos_list.append(acc_pos.item())
            acc_neg_list.append(acc_neg.item())

        # gumbel_tau *= gumbel_tau_gamma

        spr = torch.norm(func_sigmoid(graph)) / graph.shape[0]
        edges = torch.sum(func_sigmoid(graph) > 0.5)
        est_graph = (func_sigmoid(graph) > 0.5).cpu().numpy()
        # tp, fp, tpr, fpr, precision, f1 = eval_graph(est_graph, true_graph)
        # print(
        #     'Epoch {} Train[M]: loss_mean={:.4f}, acc_mean={:.4f}, acc_pos_mean={:.4f}, acc_neg_mean={:.4f}, spr={:.4f}, edges={:.4f}, tp={:.4f}'.format(
        #         epoch_idx + 1, np.mean(loss_list), \
        #         np.mean(acc_list), np.mean(acc_pos_list), np.mean(acc_neg_list), spr, edges, tp))
        # tensorboard_log.log_metrics({"Train[M]/loss_mean": np.mean(loss_list),
        #                              "Train[M]/acc_mean": np.mean(acc_list),
        #                              "Train[M]/acc_pos_mean": np.mean(acc_pos_list),
        #                              "Train[M]/acc_neg_mean": np.mean(acc_neg_list),
        #                              "Train[M]/spr": spr,
        #                              "Train[M]/edges": edges,
        #                              "Train[M]/tp": tp,
        #                              "Train[M]/fp": fp,
        #                              }, epoch_idx + 1)

        # Stage3: 验证 - 反事实
        if np.sum(est_graph) == 0:
            est_graph = torch.ones([len(all_labels) - 1]).to(device)
        # dispo = test_num_features(test_df, est_graph, model)
        dispo = 0.89
        if dispo > max_dispo:
            final_graph = est_graph
            torch.save(model.state_dict(), "./result/adult/model_weights.pth")
            np.save('./result/adult/graph.npy',np.array(final_graph))
            max_dispo = dispo
        # Lr schedule
        # model_scheduler.step()
        # graph_scheduler.step()
        gumbel_tau *= gumbel_tau_gamma

    # 评价结果
    # est_graph = (func_sigmoid(graph) > 0.5).cpu().numpy()
    # tp, fp, tpr, fpr, precision, f1 = eval_graph(est_graph, true_graph)
    # print('TP={}, FP={}, TPR={}, FPR={}, Precision={}, F1={}'.format(tp, fp, tpr, fpr, precision, f1))
    # # 输出TP-feature names
    # flags = est_graph * true_graph
    # for idx in range(339):
    #     if flags[idx] == 1:
    #         print('prob={:.2f}, '.format(func_sigmoid(graph)[idx].item()), df.columns[idx])
    #     else:
    #         continue
    # # 输出Discovered-feature names
    # flags = est_graph
    # with open('exposure.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     # 写入列名
    #     writer.writerow(['outcome', 'exposure', 'prob'])
    #     for idx in range(339):
    #         if flags[idx] == 1:
    #             # print('prob={:.2f}, '.format(func_sigmoid(graph)[idx].item()), df.columns[idx])
    #             writer.writerow(
    #                 ['Ischemic stroke || id:ebi-a-GCST005843', df.columns[idx], func_sigmoid(graph)[idx].item()])
    #         else:
    #             continue
    # print((est_graph * true_graph).sum())
    #
    # # 可视化结果
    # # print(func_sigmoid(graph))
    # print(func_sigmoid(graph).mean())

    print('finished')

    # Task
    ## 1. 增加MLP数量，保持训练集彼此不相交
    ## 2. 引入先验的因果关系

    # nohup python -u ./experiments/multi_mlp_causal.py > run_v6_n1.log 2>&1 &

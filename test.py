import numpy as np
import pandas as pd
import time
import torch
from torch import optim, nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import load_dataset
from castle.algorithms import PC
from algorithm.utils.evaluation import MetricGenenral


class medical_Dataset(Dataset):
    def __init__(self, data, result):
        self.data = data
        self.result = result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.result[index]


class MLP(torch.nn.Module):
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


# adult
# data = pd.read_csv('../../dataset/table_data/adult/adult_new.csv')
# cf_results = pd.read_csv('./result/adult/income_cf_results.csv')

# ccs
# data = pd.read_csv('../../dataset/table_data/CCS/CCS_Data_new.csv')
# cf_results = pd.read_csv('./result/ccs/strength_cf_results.csv')

# abalone
# data = pd.read_csv('../../dataset/table_data/Abalone/Abalone_Data_new.csv')
# cf_results = pd.read_csv('./result/abalone/Age_cf_results.csv')

# auto-mpg
# data = pd.read_csv('../../dataset/table_data/auto-mpg/Auto-mpg_Data_new.csv')
# cf_results = pd.read_csv('./result/autompg/Fuel consumption_cf_results.csv')


# travel
# data = pd.read_csv('../../dataset/travel/数据处理2113.csv')
# cf_results = pd.read_csv('./result/travel/大学生_cf_results.csv', encoding='gbk')

data = pd.read_csv('../../dataset/travel/tob.csv',encoding='gbk')
cf_results = pd.read_csv('./result/travel/tob/increase_cf_results.csv', encoding='gbk')

graph = torch.tensor(cf_results.prob > 0.05, dtype=torch.float64)


# graph = torch.bernoulli(torch.tensor(cf_results.prob, dtype=torch.float64))


# true_dag=pd.read_csv('../../dataset/table_data/')

def get_numpy_data(data):
    data_lists = []
    for index, row in data.iterrows():
        data_lists.append(list(row.values))
    data_numpy = np.array(data_lists)
    return data_numpy


# def test_pc(data):
#     data = get_numpy_data(data)
#     print('data ok')
#     # X=X.astype(float)
#     pc = PC(variant='original')
#     pc.learn(X)
#     est_graph = pc.causal_matrix
#     print(est_graph)
#     np.save(f'./result/exp_{exp}_winsize100.npy', est_graph)
#     est_graph = np.load(f'./result/exp_{exp}_winsize100.npy')
#     MetricGenenral(est_graph, true_dag)

def test_cf_result(data, graph):
    # train parameters
    epochs = 200
    learning_rate = 1e-3
    batch_size = 16  # 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 训练设备

    labels = data.columns
    data = get_numpy_data(data)

    # travel
    # data = data[:, 3:]
    data = data.astype(int)
    X = np.array(data[:, :-1])
    y = np.array(data[:, -1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # under sampling
    # rus = RandomUnderSampler(sampling_strategy=0.75, random_state=2023, replacement=False)
    # X_train, y_train = rus.fit_resample(X_train, y_train)
    # print('Resampled under dataset shape %s' % Counter(y_train))

    X_train = torch.tensor(X_train, dtype=torch.float64).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float64).view((len(X_train), 1)).to(device)
    train_data = medical_Dataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, sampler=None,
                                               batch_sampler=None,
                                               num_workers=0, collate_fn=None, pin_memory=False, drop_last=True,
                                               timeout=0,
                                               worker_init_fn=None, multiprocessing_context=None)

    X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float64).view((len(X_test), 1)).to(device)
    test_data = medical_Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, sampler=None,
                                              batch_sampler=None,
                                              num_workers=0, collate_fn=None, pin_memory=False, drop_last=True,
                                              timeout=0,
                                              worker_init_fn=None, multiprocessing_context=None)
    num_i = len(labels) - 1
    num_h = 256
    num_o = 1
    model = MLP(num_i, num_h, num_o, 3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_pre = torch.nn.BCELoss(reduction='sum')
    func_sigmoid = nn.Sigmoid()
    e = tqdm(total=epochs)
    tp = []
    fp = []
    tn = []
    fn = []
    loss_train = []
    loss_eval = []
    epoch_list = []
    acc_0 = []
    acc_1 = []

    try:
        for epoch in range(epochs):
            sum_loss = 0
            # sum_loss1 = 0
            # sum_loss2 = 0
            # sum_loss3 = 0
            sum_length = 0
            true_positive = 0
            false_positive = 0
            true_negative = 0
            false_negative = 0
            for i, data in enumerate(train_loader):
                # inputs = torch.tensor(data, dtype=torch.float64)
                inputs, results = data
                length = batch_size
                if len(inputs) < batch_size:
                    length = len(inputs)
                inputs = torch.tensor(inputs * graph, dtype=torch.float64).to(device)
                # inputs = torch.tensor(inputs, dtype=torch.float64).to(device)
                model.train()
                output = model(inputs)
                final_output = func_sigmoid(output)
                # final_output = final_output.type(torch.float64).requires_grad_(True)
                # for d in range(length):
                #     final_output[d] = output[d].argmax()
                loss = loss_pre(final_output, results)
                sum_loss += loss.item()
                sum_length += length
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # postfix = f"i={i}, len={sum_length},loss={sum_loss / sum_length}"
                # t.set_postfix_str(postfix)
            loss_train.append(sum_loss / sum_length)
            epoch_list.append(epoch)

            # model test
            model.eval()
            sum_loss = 0
            sum_length = 0
            for j, data in enumerate(test_loader):
                inputs, results = data
                length = batch_size
                if len(inputs) < batch_size:
                    length = len(inputs)
                # inputs = torch.tensor(inputs * graph, dtype=torch.float64).to(device)
                inputs = torch.tensor(inputs, dtype=torch.float64).to(device)
                output = model(inputs)
                # final_output = torch.argmax(output, dim=1)
                final_output = func_sigmoid(output)
                for d in range(length):
                    if results[d] == 0:
                        if final_output[d] > 0.5:
                            false_positive += 1
                        else:
                            true_negative += 1
                    else:
                        if final_output[d] > 0.5:
                            true_positive += 1
                        else:
                            false_negative += 1
                loss = loss_pre(final_output, results)
                sum_loss += loss.item()
                sum_length += length
                postfix = f"i={i}, len={sum_length},loss={loss.item()}"
                e.set_postfix_str(postfix)
            e.update(1)
            loss_eval.append(sum_loss / sum_length)
            tp.append(true_positive)
            fp.append(false_positive)
            tn.append(true_negative)
            fn.append(false_negative)
            acc_0.append(false_positive / (true_negative + false_positive))
            acc_1.append(true_positive / (true_positive + false_negative))
            if true_positive + false_positive == 0:
                pr = 0
            else:
                pr = true_positive / (true_positive + false_positive)
            print(f'TP={true_positive}, FP={false_positive},TN={true_negative},FN={false_negative},'
                  f'pr={pr},'
                  f'acc={(true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)}')
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        plt.figure()
        # draw evaluation in one figure
        plt.subplot(2, 1, 1)
        plt.xlabel('epochs')
        plt.ylabel('num')
        plt.title('evaluation with epochs increasing')
        # plt.plot(epoch_list, tp, label='tp')
        # plt.plot(epoch_list, fp, label='fp')
        # plt.plot(epoch_list, tn, label='tn')
        # plt.plot(epoch_list, fn, label='fn')
        plt.plot(epoch_list, acc_0, label='FPR')
        plt.plot(epoch_list, acc_1, label='TPR')
        plt.legend()

        # draw loss in another figure
        plt.subplot(2, 1, 2)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('loss with epochs increasing')
        plt.plot(epoch_list, loss_train, label='loss_train')
        plt.plot(epoch_list, loss_eval, label='loss_eval')
        plt.legend()
        plt.show()
        # plt.savefig('./result/kl_new.png')


test_cf_result(data, graph)

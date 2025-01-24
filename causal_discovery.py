import numpy as np
import pandas as pd
import torch
from flatbuffers.packer import float64
from sdcd.models import SDCD
from sdcd.utils import create_intervention_dataset
from castle.algorithms import PC, Notears, ICALiNGAM, DirectLiNGAM, RL
# from castle.metrics import MetricsDAG

from algorithm.utils.evaluation import MetricGenenral

# data_path = '../../dataset/table_data/Abalone/Abalone_Data.csv'
# ground_truth_path = '../../dataset/table_data/Abalone/Abalone_data_ground_truth.csv'
# dataname = 'abalone'

# data_path = '../../dataset/table_data/CCS/CCS_Data.csv'
# ground_truth_path = '../../dataset/table_data/CCS/CCS_Data_ground_truth.csv'
# dataname = 'ccs'


# data_path = '../../dataset/table_data/auto-mpg/Auto-mpg_Data.csv'
# ground_truth_path = '../../dataset/table_data/auto-mpg/Auto-mpg_Data_groung_truth.csv'
# dataname='autompg'

data_path = '../../dataset/table_data/liver_disorders/liver_disorder_Data.csv'
ground_truth_path = '../../dataset/table_data/liver_disorders/liver_disorders_Data_ground_truth.csv'
dataname='live_disorder'

# data_path = '../../dataset/table_data/adult/adult_new.csv'
# ground_truth_path = '../../dataset/table_data/liver_disorders/liver_disorders_Data_ground_truth.csv'
# dataname='adult'

# data_path = '../../dataset/table_data/arrhythmia/arrhythmia.csv'
# ground_truth_path = '../../dataset/table_data/liver_disorders/liver_disorders_Data_ground_truth.csv'
# dataname='arrhythmia'

def test_PC(data_path, ground_truth_path, dataname):
    data = pd.read_csv(data_path, encoding='utf-8')
    data_list = []
    for index, row in data.iterrows():
        data_list.append(list(row.values))
    data_list = np.array(data_list)
    true_data = pd.read_csv(ground_truth_path, encoding='utf-8')
    true_data = true_data.iloc[:, 1:]
    true_dag = []
    for index, row in true_data.iterrows():
        true_dag.append(list(row.values))
    true_dag = np.array(true_dag)
    print('data ok')
    # X=X.astype(float)
    pc = PC(variant='original')
    pc.learn(data_list)
    est_graph = pc.causal_matrix
    print(est_graph)
    np.save(f'./result/{dataname}/pc.npy', est_graph)
    est_graph = np.load(f'./result/{dataname}/pc.npy')
    # metrics = MetricsDAG(est_graph, true_dag)
    # print("pc",metrics.metrics)
    print("pc")
    # MetricGenenral(est_graph,true_dag)


def test_notears(data_path, ground_truth_path, dataname):
    data = pd.read_csv(data_path, encoding='utf-8')
    data_list = []
    for index, row in data.iterrows():
        data_list.append(list(row.values))
    data_list = np.array(data_list)
    true_data = pd.read_csv(ground_truth_path, encoding='utf-8')
    true_data = true_data.iloc[:, 1:]
    true_dag = []
    for index, row in true_data.iterrows():
        true_dag.append(list(row.values))
    true_dag = np.array(true_dag)
    print('data ok')
    n = Notears(max_iter=1000)
    n.learn(data_list)
    est_graph = n.causal_matrix
    print(est_graph)
    np.save(f'./result/{dataname}/notears.npy', est_graph)
    est_graph = np.load(f'./result/{dataname}/notears.npy')
    # metrics = MetricsDAG(est_graph, true_dag)
    # print("notears",metrics.metrics)
    print("notears")
    # MetricGenenral(est_graph, true_dag)


def test_ica(data_path, ground_truth_path, dataname):
    data = pd.read_csv(data_path, encoding='utf-8')
    data_list = []
    for index, row in data.iterrows():
        data_list.append(list(row.values))
    data_list = np.array(data_list)
    true_data = pd.read_csv(ground_truth_path, encoding='utf-8')
    true_data = true_data.iloc[:, 1:]
    true_dag = []
    for index, row in true_data.iterrows():
        true_dag.append(list(row.values))
    true_dag = np.array(true_dag)
    print('data ok')
    n = ICALiNGAM()
    n.learn(data_list)
    est_graph = n.causal_matrix
    print(est_graph)
    np.save(f'./result/{dataname}/ica.npy', est_graph)
    est_graph = np.load(f'./result/{dataname}/ica.npy')
    # metrics = MetricsDAG(est_graph, true_dag)
    # print("ica",metrics.metrics)
    print("ica")
    # MetricGenenral(est_graph, true_dag)


def test_direct(data_path, ground_truth_path, dataname):
    data = pd.read_csv(data_path, encoding='utf-8')
    data_list = []
    for index, row in data.iterrows():
        data_list.append(list(row.values))
    data_list = np.array(data_list)
    true_data = pd.read_csv(ground_truth_path, encoding='utf-8')
    true_data = true_data.iloc[:, 1:]
    true_dag = []
    for index, row in true_data.iterrows():
        true_dag.append(list(row.values))
    true_dag = np.array(true_dag)
    print('data ok')
    n = DirectLiNGAM()
    n.learn(data_list)
    est_graph = n.causal_matrix
    print(est_graph)
    np.save(f'./result/{dataname}/direct.npy', est_graph)
    est_graph = np.load(f'./result/{dataname}/direct.npy')
    # metrics = MetricsDAG(est_graph, true_dag)
    # print("direct",metrics.metrics)
    print("direct")
    # MetricGenenral(est_graph, true_dag)


def test_rl(data_path, ground_truth_path, dataname):
    data = pd.read_csv(data_path, encoding='utf-8')
    data_list = []
    for index, row in data.iterrows():
        data_list.append(list(row.values))
    data_list = np.array(data_list, dtype=np.float64)
    # data_list = np.array(data_list)
    # true_data = pd.read_csv(ground_truth_path, encoding='utf-8')
    # true_data = true_data.iloc[:, 1:]
    # true_dag = []
    # for index, row in true_data.iterrows():
    #     true_dag.append(list(row.values))
    # true_dag = np.array(true_dag)
    print('data ok')
    n = RL(device_type='cpu', nb_epoch=1000)
    n.learn(data_list)
    est_graph = n.causal_matrix
    print(est_graph)
    np.save(f'./result/{dataname}/rl.npy', est_graph)
    est_graph = np.load(f'./result/{dataname}/rl.npy')
    # metrics = MetricsDAG(est_graph, true_dag)
    # print("rl", metrics.metrics)
    print("rl")
    # MetricGenenral(est_graph, true_dag)


def test_sdcd(data_path, ground_truth_path, dataname):
    data = pd.read_csv(data_path, encoding='utf-8')
    # data = data.astype('float64')
    print(data.head())
    true_data = pd.read_csv(ground_truth_path, encoding='utf-8')
    true_data = true_data.iloc[:, 1:]
    true_dag = []
    for index, row in true_data.iterrows():
        true_dag.append(list(row.values))
    true_dag = np.array(true_dag)
    print('data ok')
    X_dataset = create_intervention_dataset(data, perturbation_colname="perturbation_label")
    double_dataset=torch.utils.data.TensorDataset(*(t.double() for t in X_dataset.tensors))
    model = SDCD()
    model.train(double_dataset, finetune=True)

    est_graph = model.get_adjacency_matrix(threshold=True)

    print(est_graph)
    np.save(f'./result/{dataname}/sdcd.npy', est_graph)
    est_graph = np.load(f'./result/{dataname}/sdcd.npy')
    # metrics = MetricsDAG(est_graph, true_dag)
    # print("rl", metrics.metrics)
    print("sdcd")
    MetricGenenral(est_graph, true_dag)


# test_PC(data_path, ground_truth_path, dataname)
# test_notears(data_path, ground_truth_path, dataname)
# test_ica(data_path, ground_truth_path, dataname)
# test_direct(data_path, ground_truth_path, dataname)
# test_rl(data_path, ground_truth_path, dataname)
test_sdcd(data_path, ground_truth_path, dataname)

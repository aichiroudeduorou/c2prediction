import sys
import os
# 将项目根目录添加到 sys.path
# __file__ 是 counter.py 的路径
# os.path.dirname(__file__) 是 .../algorithm/counter
# os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 是 .../Causal_Discovery
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
nice_package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'NICE'))
sys.path.insert(0, project_root)
sys.path.insert(0, nice_package_path)

from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from tabpfn import TabPFNClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from algorithm.NICE.nice import NICE
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# load data
# data_path = '../../dataset/table_data/auto-mpg/Auto-mpg_Data_new.csv'
data_path = '../../dataset/table_data/adult/adult_new.csv'
# data_path = '../../dataset/table_data/Abalone/Abalone_Data_new.csv'
# data_path = '../../dataset/table_data/CCS/CCS_Data_new.csv'
# data_path = '../../dataset/table_data/liver_disorders/liver_disorder_Data_new.csv'
# data_path = '../../dataset/table_data/arrhythmia/arrhythmia_new.csv'
data = pd.read_csv(data_path, encoding="utf")


# data_path = '../../dataset/travel/tob.csv'
# data = pd.read_csv(data_path, encoding="gbk")


def test_feature(data, results, target_feature, graph, function):
    data_l = 'adult'
    labels = data.columns
    inputs = []
    for index, row in data.iterrows():
        inputs.append(list(row.values))

    X = np.array(inputs)
    y = np.array(results)

    # X = np.array([X_norm_item * graph for X_norm_item in X])
    X = np.array(X * graph)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(labels)
    cat_labels = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                  'race', 'sex', 'native-country']  # adult

    # cat_labels = []  # ccs
    # cat_labels.remove(target_feature)  # test_cat_feature need do this
    # if target_feature in cat_labels:
    #     cat_labels.remove(target_feature)
    cat_feat = [np.where(labels == label)[0][0] for label in cat_labels]  # first
    # cat_feat = []  # second

    # num_labels = []  # ccs
    # # cat_labels.remove(target_feature)  # test_cat_feature need do this
    # if target_feature in num_labels:
    #     num_labels.remove(target_feature)
    # num_feat = [np.where(labels == label)[0][0] for label in num_labels]

    # cat_feat = []
    # for i in range(len(labels)):
    #     if i not in num_feat:
    #         cat_feat.append(i)

    # for label in cat_labels:
    #     # 查找label在labels中的索引
    #     index_arr = np.where(labels == label)[0]
    #     if index_arr.size > 0:  # 或者使用len(index_arr) > 0
    #         cat_feat.append(index_arr[0])
    #     else:
    #         print(f"标签'{label}'未在labels中找到。")

    num_feat = []
    for i in range(len(labels)):
        if i not in cat_feat:
            num_feat.append(i)

    clf = Pipeline([
        ('PP', ColumnTransformer([
            ('num', StandardScaler(), num_feat),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feat)])),
        ('RF', MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42))])

    clf.fit(X_train, y_train)

    predict_fn = lambda x: clf.predict_proba(x)

    NICE_mal = NICE(
        X_train=X_train,
        predict_fn=predict_fn,
        y_train=y_train,
        cat_feat=cat_feat,
        num_feat=num_feat,
        distance_metric='HEOM',
        num_normalization='minmax',
        optimization='proximity',
        justified_cf=True
    )

    NICE_eval = NICE(
        X_train=X_train,
        predict_fn=predict_fn,
        y_train=y_train,
        cat_feat=cat_feat,
        num_feat=num_feat,
        distance_metric='HEOM',
        num_normalization='minmax',
        optimization='none',
        justified_cf=True
    )

    cf_instance = []
    eval_x_train = []
    eval_y_train = []
    eval_x_test = []
    eval_y_test = []
    print(len(X_train), len(X_test))
    tq = tqdm(total=len(X_test))
    for i in range(len(X_test)):
        # for i in range(2):
        # print(i)
        to_explain = X_test[i:i + 1, :]

        CF = NICE_mal.explain(to_explain)
        cf_instance.append(CF[0])
        eval_x_train.append(X_test[i])  # x
        eval_x_train.append(CF[0])  # C
        eval_y_train.append(y_test[i])
        eval_y_train.append(abs(y_test[i] - 1))  # (0-1)=1 (1-1)=0

        neighbor = NICE_eval.explain(to_explain)

        eval_x_test.append(neighbor[0])
        eval_y_test.append(abs(y_test[i] - 1))
        postfix = f"i={i}, len={len(X_test)}"
        tq.set_postfix_str(postfix)
        tq.update(1)

    # train_df = pd.DataFrame(data=X_train, columns=labels)
    # train_df.to_csv(f'./result/{data_l}/{target_feature}_train_x.csv', index=False)
    # train_result = pd.DataFrame(data=y_train)
    # train_result.to_csv(f'./result/{data_l}/{target_feature}_train_y.csv', index=False)
    #
    # df = pd.DataFrame(data=cf_instance, columns=labels)
    # df.to_csv(f'./result/{data_l}/{target_feature}_cf.csv', index=False)
    #
    # ori_df = pd.DataFrame(data=X_test, columns=labels)
    # ori_df.to_csv(f'./result/{data_l}/{target_feature}_test_x.csv', index=False)
    # results_df = pd.DataFrame(data=y_test)
    # results_df.to_csv(f'./result/{data_l}/{target_feature}_test_y.csv', index=False)

    nn_clf = Pipeline([
        ('PP', ColumnTransformer([
            ('num', StandardScaler(), num_feat),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feat)])),
        ('RF', MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42))])

    nn_clf.fit(np.array(eval_x_train), np.array(eval_y_train))

    nn_predict_fn = lambda x: nn_clf.predict_proba(x)

    r = nn_predict_fn(np.array(eval_x_test))
    re = np.argmax(r, axis=1)
    pre = 1 - sum(abs(re - eval_y_test)) / len(eval_y_test)
    print(target_feature, pre)
    return pre


# cat_labels = ['尿素', '尿胆原', '总胆汁酸', '直接胆红素', '胆红素', '葡萄糖',
#               '蛋白质', '酮体', '隐血', 'label']
# mark_path = '../../dataset/medical/medical_data_II/processed_data/first/mark_0.8_出生胎儿畸形.csv'
# mark_data = pd.read_csv(mark_path, encoding="gbk")  # first


# mark_data = pd.read_csv(mark_path, encoding="utf")  # second

def test_cat_feature(data, cat_labels):
    data = data.iloc[:, 1:]
    data = data.drop(labels='身高', axis=1)
    data = data.drop(labels='孕前体重', axis=1)
    for i in range(len(cat_labels)):
        target_feature = cat_labels[i]
        results = list(data[target_feature])
        new_data = data.drop(labels=target_feature, axis=1)
        # new_data = new_data.iloc[:, 1:]
        test_feature(new_data, results, target_feature)


# test_cat_feature(data, cat_labels)


def test_num_features(data, graph, function):
    # data = data.iloc[:, 3:]
    all_labels = data.columns

    # test all features
    # for i in range(len(all_labels)):
    #     # if all_labels[i] not in cat_labels:
    #     target_feature = all_labels[i]
    #     new_data = data.drop(labels=target_feature, axis=1)
    #     results = list(data[target_feature])
    #     test_feature(new_data, results, target_feature)
    # graph = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target_feature = all_labels[-1]
    new_data = data.drop(labels=target_feature, axis=1)
    results = list(data[target_feature])
    dispo = test_feature(new_data, results, target_feature, graph, function)
    return dispo
    '''
    for i in range(-5, 0):
        # if all_labels[i] not in cat_labels:
        target_feature = all_labels[i]
        new_data = data.drop(labels=target_feature, axis=1)
        for j in range(-5, 0):
            if j != i:
                new_data = new_data.drop(labels=all_labels[j], axis=1)
        results = list(data[target_feature])
        test_feature(new_data, results, target_feature)
    '''


'''
dataname = 'adult'
graph = np.load(f'./result/{dataname}/rl.npy')
# graph = np.load(f'./result/{dataname}/direct.npy')
# graph = np.load(f'./result/{dataname}/ica.npy')
# graph = np.load(f'./result/{dataname}/notears.npy')
# graph = np.load(f'./result/{dataname}/pc.npy')
graph_new = graph[-1, :-1]
print(graph, graph_new)
function = []
test_num_features(data, graph_new, function)
'''

import sys
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, f1_score, recall_score
import seaborn as sns

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
sys.path.append(os.path.join(os.path.dirname(__file__), 'NCART', 'NCART_EXP'))
from NCART.ncart import NCARTClassifier

'''
real-world dataset experiment for event prediction task
'''
# data_dir = 'causal_discovery/dataset/real_world/'
# train_data_path = data_dir + 'adult/adult_new.csv'
# test_data_path = data_dir + 'adult/adult_test.csv'

# train_data_path = data_dir + 'Abalone/Abalone_data_train.csv'
# test_data_path = data_dir + 'Abalone/Abalone_data_test.csv'

# train_data_path = data_dir + 'arrhythmia/arrhythmia_train.csv'
# test_data_path = data_dir + 'arrhythmia/arrhythmia_test.csv'

# train_data_path = data_dir + 'CCS/CCS_Data_train.csv'
# test_data_path = data_dir + 'CCS/CCS_Data_test.csv'

# train_data_path = data_dir + 'auto-mpg/Auto-mpg_Data_train.csv'
# test_data_path = data_dir + 'auto-mpg/Auto-mpg_Data_test.csv'

# train_data_path = data_dir + 'liver_disorders/liver_disorder_Data_train.csv'
# test_data_path = data_dir + 'liver_disorders/liver_disorder_Data_test.csv'

'''
synthetic dataset experiment for event prediction task
'''
data_dir = 'causal_discovery/dataset/syn_dataset/'
n_samples=10000
n_features=400
train_data_path = data_dir + f"{n_samples}samples_{n_features}features/scm_train.csv"
test_data_path = data_dir + f"{n_samples}samples_{n_features}features/scm_test.csv"

data_train = pd.read_csv(train_data_path, encoding='utf-8')
data_test = pd.read_csv(test_data_path, encoding='utf-8')
# mean_values = data_train.mean()
# data_train = data_train.fillna(mean_values)
# mean_values = data_test.mean()
# data_test = data_test.fillna(mean_values)
col = data_train.columns
train = data_train.iloc[:, :-1]
test = data_test.iloc[:, :-1]
y_train = list(data_train.iloc[:, -1])
y_test = list(data_test.iloc[:, -1])
x_train = []
for index, row in train.iterrows():
    x_train.append(list(row.values))

# Split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
# for i in range(len(x_train)):
#     x_train[i].append(y_train[i])
# x_df = pd.DataFrame(x_train, columns=col)
# x_df.to_csv('../../dataset/table_data/arrhythmia/arrhythmia_train.csv', index=False)
# for i in range(len(x_test)):
#     x_test[i].append(y_test[i])
# x_df = pd.DataFrame(x_test, columns=col)
# x_df.to_csv('../../dataset/table_data/arrhythmia/arrhythmia_test.csv', index=False)


x_test = []
for index, row in test.iterrows():
    x_test.append(list(row.values))


def test_xgboost(x_train, y_train, x_test, y_test):
    # Initialize the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Train the model
    print('Xgboost begin training')
    # calculate train time
    
    start_time = time.time()
    xgb_clf.fit(x_train, y_train)
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')

    # Make predictions
    y_pred = xgb_clf.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('Xgboost')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision:{precision:.2f}')
    print(f'Recall:{recall:.2f}')
    print(f'F1:{f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)


def test_SVC(x_train, y_train, x_test, y_test):
    # Initialize the SVC classifier
    svc_clf = SVC(kernel='linear', random_state=42)

    # Train the model
    print('SVC begin training')
    start_time = time.time()
    svc_clf.fit(x_train, y_train)
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')
    # Make predictions
    y_pred = svc_clf.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('SVC')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision:{precision:.2f}')
    print(f'Recall:{recall:.2f}')
    print(f'F1:{f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)


def test_randomforest(x_train, y_train, x_test, y_test):
    # Initialize the Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    print('Random Forest begin training')
    start_time = time.time()
    rf_clf.fit(x_train, y_train)
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')

    # Make predictions
    y_pred = rf_clf.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('random_forest')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision:{precision:.2f}')
    print(f'Recall:{recall:.2f}')
    print(f'F1:{f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)


def test_neural_network(x_train, y_train, x_test, y_test):
    # Initialize the Neural Network classifier
    mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

    # Train the model
    print('Neural Network begin training')
    start_time = time.time()
    mlp_clf.fit(x_train, y_train)
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')

    # Make predictions
    y_pred = mlp_clf.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('neural_network')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision:{precision:.2f}')
    print(f'Recall:{recall:.2f}')
    print(f'F1:{f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)


def test_logistic(x_train, y_train, x_test, y_test):
    # Initialize the Logistic Regression classifier
    logreg = LogisticRegression(max_iter=200, random_state=42)

    # Train the model
    print('Logistic Regression begin training')
    start_time = time.time()
    logreg.fit(x_train, y_train)
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')

    # Make predictions
    y_pred = logreg.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('logistic_regression')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision:{precision:.2f}')
    print(f'Recall:{recall:.2f}')
    print(f'F1:{f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)


def test_tabnet(x_train,y_train,x_test,y_test):
    X_train, X_valid, Y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    clf = TabNetClassifier()  # TabNetRegressor()
    
    print('TabNet begin training')
    start_time = time.time()
    clf.fit(
        np.array(X_train), np.array(Y_train),
        eval_set=[(np.array(X_valid), np.array(y_valid))]
    )
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')
    y_pred = clf.predict(np.array(x_test))

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('tabnet')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision:{precision:.2f}')
    print(f'Recall:{recall:.2f}')
    print(f'F1:{f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

def test_tabpfn(x_train,y_train,x_test,y_test):
    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

    print('TabPFN begin training')
    start_time = time.time()
    classifier.fit(x_train, y_train)
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')
    y_pred, p_eval = classifier.predict(x_test, return_winning_probability=True)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('tabpfn')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision:{precision:.2f}')
    print(f'Recall:{recall:.2f}')
    print(f'F1:{f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)


def test_tabpfn_v2(x_train,y_train,x_test,y_test):
    # classifier = TabPFNClassifier(device='cpu')
    classifier=TabPFNClassifier.create_default_for_version(ModelVersion.V2,ignore_pretraining_limits=True,model_path='/workspace/causal_discovery/code/TabPFN-new/model/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')

    print('TabPFN V2 begin training')
    start_time = time.time()
    classifier.fit(x_train, y_train)
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')
    y_pred= classifier.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('tabpfn')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision:{precision:.2f}')
    print(f'Recall:{recall:.2f}')
    print(f'F1:{f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

def test_ncart(x_train,y_train,x_test,y_test):
    

    # Convert to numpy arrays for NCARTClassifier compatibility
    x_train_np = np.array(x_train)
    x_test_np = np.array(x_test)
    classifier = NCARTClassifier(epochs=100, n_trees=8, n_layers=2, n_selected=6)
    
    print('NCART begin training')
    start_time = time.time()
    classifier.fit(x_train_np, y_train)
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')
    y_pred = classifier.predict(x_test_np)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('ncart')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision:{precision:.2f}')
    print(f'Recall:{recall:.2f}')
    print(f'F1:{f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

test_xgboost(x_train, y_train, x_test, y_test)
test_SVC(x_train, y_train, x_test, y_test)
test_randomforest(x_train, y_train, x_test, y_test)
test_neural_network(x_train, y_train, x_test, y_test)
test_logistic(x_train, y_train, x_test, y_test)


test_tabnet(x_train,y_train, x_test, y_test)
# test_tabpfn(x_train, y_train, x_test, y_test)

test_ncart(x_train, y_train, x_test, y_test)
test_tabpfn_v2(x_train, y_train, x_test, y_test)
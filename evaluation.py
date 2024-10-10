import pandas as pd
import numpy as np

# load data
# adult
# data_path = '../../dataset/table_data/adult/adult_new.csv'

# ccs
# data_path = '../../dataset/table_data/CCS/CCS_Data_new.csv'

# abalone
# data_path = '../../dataset/table_data/Abalone/Abalone_Data_new.csv'

# auto-mpg
# data_path = '../../dataset/table_data/auto-mpg/Auto-mpg_Data_new.csv'
# data = pd.read_csv(data_path, encoding="utf")

# travel
data_path = '../../dataset/travel/tob.csv'
data = pd.read_csv(data_path, encoding="gbk")


def get_one_result(data):
    data_l = 'travel/tob'
    all_labels = data.columns
    target_feature = all_labels[-1]

    test_x_path = f'./result/{data_l}/{target_feature}_test_x.csv'
    cf_path = f'./result/{data_l}/{target_feature}_cf.csv'
    x_data = pd.read_csv(test_x_path, encoding="utf")
    cf_data = pd.read_csv(cf_path, encoding='utf')
    labels = x_data.columns

    results = np.zeros(len(labels))
    for i in range(len(x_data)):
        for j in range(len(labels)):
            if x_data.iloc[i][j] != cf_data.iloc[i][j]:
                results[j] += 1

    results = results / len(x_data)
    df = pd.DataFrame({'feature': labels, 'prob': results})
    df.to_csv(f'./result/{data_l}/{target_feature}_cf_results.csv', index=False,encoding='gbk')


get_one_result(data)

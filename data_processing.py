import numpy as np
import pandas as pd

# 替换 'delimiter' 为你的文件中实际使用的分隔符
data_path = '../../dataset/table_data/GSE10245/'
# col = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
#        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
# #
# data = pd.read_csv(data_path + 'arrhythmia.data', delimiter=',')  # 假设数据是用制表符分隔的
# data.to_csv(data_path + 'arrhythmia.csv', index=False, encoding='utf-8')  # 将数据保存为 CSV 文件

# data_path = '../../dataset/table_data/sido0_text/'
# col = ['Age', 'Length', 'Shell weight', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight']

# for i in range(5, 10):
#     data = pd.read_csv(data_path + f'000{i}.txt', delimiter=' ', header=None)  # 假设数据是用制表符分隔的
#     data.to_csv(data_path + f'Abalone_Data_{i}.csv', index=False, encoding='utf', header=False)  # 将数据保存为 CSV 文件
# for i in range(33, 38):
#     data = pd.read_csv(data_path + f'00{i}.txt', delimiter=' ', header=None)  # 假设数据是用制表符分隔的
#     data.to_csv(data_path + f'liver_disorder_Data.csv', index=False, encoding='utf', header=False)
#     data.to_csv(data_path + f'liver_disorder_{i}.csv', index=False, encoding='utf', header=False)  # 将数据保存为 CSV 文件
# data = pd.read_csv(data_path+'sido0_train.targets', delimiter=' ', header=None)  # 假设数据是用制表符分隔的
# data.to_csv(data_path + 'sido0_train_result_Data.csv', index=False, encoding='utf', header=False)
# data.to_csv(data_path + f'liver_disorder_{i}.csv', index=False, encoding='utf', header=False)

data = pd.read_csv(data_path + 'GSE10245.csv', encoding='utf-8')  # 假设数据是用制表符分隔的
cols = list(data.iloc[:, 0])
data = data.iloc[:, 1:]
# label = data.columns
data_list = []

for index, row in data.iterrows():
    data_list.append(list(row.values))
# data_list.append(label)
# data_list = np.array(data_list)
# data_list = data_list.transpose()
for i in range(58):
    print(i)
    data_list.append(list(data.iloc[:, i]))
#
data = pd.DataFrame(data, columns=cols)
data.to_csv(data_path + 'GSE10245_new.csv', index=False, encoding='utf-8')  # 将数据保存为 CSV 文件

# travel
# data_path = '../../dataset/travel/数据处理_2376.csv'
# data = pd.read_csv(data_path, encoding='utf')
# labels = data.columns
# data_list = []
# for index, row in data.iterrows():
#     data_list.append(list(row.values))
# new_data_list = []
# new_data_list.append(data_list[0])
# for i in range(1, len(data_list)):
#     if data_list[i][0] == new_data_list[len(new_data_list) - 1][0]:
#         for j in range(1, len(data_list[i])):
#             if data_list[i][j] == 1:
#                 new_data_list[len(new_data_list) - 1][j] == 1
#     else:
#         new_data_list.append(data_list[i])
#
# df = pd.DataFrame(new_data_list, columns=labels)
# df.to_csv('../../dataset/travel/数据处理_new.csv', index=False, encoding='gbk')

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

'''
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
'''
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

'''
import numpy as np
import pandas as pd
from fancyimpute import IterativeImputer

data_path = '../../dataset/table_data/arrhythmia/arrhythmia.csv'
data = pd.read_csv(data_path, encoding="utf")
# 创建一个包含缺失值的DataFrame

imputer=IterativeImputer()
df_imputed=pd.DataFrame(imputer.fit_transform(data),columns=data.columns)
df_imputed.to_csv('../../dataset/table_data/arrhythmia/arrhythmia_new.csv')
'''


import matplotlib.pyplot as plt
import numpy as np

def draw_sensitivity():
    size = 50  # 增大字体大小

    # 准备数据
    x = np.arange(len([0.1, 0.2, 0.3, 0.4, 0.5]))  # 转换为数组索引
    y_1 = [0.79, 0.8, 0.83, 0.79, 0.78]
    y_2 = [0.79, 0.79, 0.81, 0.79, 0.8]

    # 设置柱宽和间隔
    bar_width = 0.6  # 调整宽度以增加间隔

    # 第一张柱状图：y_1
    plt.figure(figsize=(14, 10))  # 增大绘图区域尺寸
    plt.bar(x, y_1, width=bar_width, color='skyblue', label='λ_s=0.1', edgecolor='black')

    # 设置x轴刻度标签对应的原始x值
    plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5'], fontweight='bold', fontsize=size)

    # 设置x和y轴标签
    plt.xlabel('λ_s', fontweight='bold', fontsize=size + 10)
    plt.ylabel('F1', fontweight='bold', fontsize=size + 10)

    # 设置y轴刻度字体
    plt.yticks(np.arange(0, 1.2, 0.2), fontweight='bold', fontsize=size)

    # 设置y轴范围
    plt.ylim(0, 1)

    # 调整图像的边距，将绘图整体向右上移动
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)

    # 保存并显示
    plt.tight_layout()  # 确保标签完整显示
    plt.savefig('./result/adult/sensitivity_s.pdf')
    plt.show()

    # 第二张柱状图：y_2
    plt.figure(figsize=(14, 10))  # 增大绘图区域尺寸
    plt.bar(x, y_2, width=bar_width, color='salmon', label='λ_c=0.1', edgecolor='black')

    # 设置x轴刻度标签对应的原始x值
    plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5'], fontweight='bold', fontsize=size)

    # 设置x和y轴标签
    plt.xlabel('λ_c', fontweight='bold', fontsize=size + 10)
    plt.ylabel('F1', fontweight='bold', fontsize=size + 10)

    # 设置y轴刻度字体
    plt.yticks(np.arange(0, 1.2, 0.2), fontweight='bold', fontsize=size)

    # 设置y轴范围
    plt.ylim(0, 1)

    # 调整图像的边距，将绘图整体向右上移动
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)

    # 保存并显示
    plt.tight_layout()  # 确保标签完整显示

    # 保存并显示
    plt.savefig('./result/adult/sensitivity_c.pdf')
    plt.show()

def split_dataset(dir_path):
    from sklearn.model_selection import train_test_split
    n_samples=10000
    n_features=400
    data_path = dir_path + f"{n_samples}samples_{n_features}features/scm_classification.csv"
    data = pd.read_csv(data_path, encoding='utf-8')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data.to_csv(dir_path + f"{n_samples}samples_{n_features}features/scm_train.csv", index=False, encoding='utf-8')
    test_data.to_csv(dir_path + f"{n_samples}samples_{n_features}features/scm_test.csv", index=False, encoding='utf-8')    

dir_path = '/workspace/causal_discovery/dataset/syn_dataset/'
split_dataset(dir_path)
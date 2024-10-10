import pandas as pd
import numpy as np

people_path = '../../dataset/travel/中国旅游数据库-年度（省）.csv'
cluster_path = '../../dataset/travel/聚类.csv'

people_data = pd.read_csv(people_path, encoding='gbk')
cluster_data = pd.read_csv(cluster_path, encoding='utf-8')

labels = set(list(cluster_data.iloc[:, -1]))
labels = list(labels)
print(len(labels), labels)
citys = list(people_data.iloc[:, 1])
print(len(citys), citys)

data_list = np.zeros([270, 51])
for city in range(len(citys)):
# for city in range(1):
    for year in range(9):
        for index, row in cluster_data.iterrows():
            if row['地区'] == citys[city]:
                if str(year + 2010) in row['时间']:
                    for label in range(len(labels)):
                        if labels[label] == row['类型']:
                            label_index = label
                            break
                    data_list[city * 9 + year][label_index] = data_list[city * 9 + year][label_index] + 1

people_data = people_data.iloc[:, 2:]
people_list = []
for index, row in people_data.iterrows():
    people_list.append(list(row.values))
for i in range(len(people_list)):
    for j in range(len(people_list[i]) - 1):
        if people_list[i][j] < people_list[i][j + 1]:
            data_list[i * 9 + j][-1] = 1

# labels = list(labels)
labels.append('increase')
data_df = pd.DataFrame(data_list, columns=labels)
data_df.to_csv('../../dataset/travel/tob.csv', index=False, encoding='gbk')

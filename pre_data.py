import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('./concrete.csv', header=0)
df = [raw_data.iloc[i, 1]/raw_data.iloc[i, 0] for i in range(raw_data.shape[0])]
strength = raw_data['strength']
raw_data = raw_data.iloc[:, :6]
raw_data['water_cement'] = df
# 归一化
min_max_scaler = preprocessing.MinMaxScaler()  # 默认为范围0~1，拷贝操作
raw_data_minmax0 = min_max_scaler.fit_transform(raw_data)
columns = [i for i in raw_data.columns]
raw_data_minmax = pd.DataFrame(raw_data_minmax0, columns=columns)
# # print('raw_data_minmax\n', raw_data_minmax.iloc[:5,:])
# # print(raw_data_minmax.shape)
#
# # PCA降维
# pca_data = PCA(n_components=0.9)
# pca_data0 = pca_data.fit_transform(raw_data_minmax.iloc[:, [2, 3, 4]])
# pca_data0 = pd.DataFrame(pca_data0)
# # print('pca data\n', pca_data0.iloc[:5, :])
# # print(pca_data0.shape)
#
# # 整理最后得到数据
# final_data0 = pd.concat([raw_data_minmax['water_cement'], pca_data0], axis=1)
# final_data0 = pd.concat([final_data0, raw_data_minmax['age']], axis=1)
# final_data = pd.concat([final_data0, strength], axis=1)
# # print('final data\n', final_data.iloc[:5, :]) # shape=(1030, 5)

final_data0 = pd.concat([raw_data_minmax['water_cement'], raw_data_minmax['age']], axis=1)
final_data = pd.concat([final_data0, strength], axis=1)

# 划分训练集和测试集
import numpy as np
train_X, test_X, train_y, test_y = train_test_split(final_data.iloc[:, :5], final_data['strength'], test_size=0.2, random_state=0)
list0 = []
for data in [train_X, test_X, train_y, test_y]:
    data_array = np.array(data)
    # 然后转化为list形式
    data_list = data_array.tolist()
    list0.append(data_list)
train_X, test_X, train_y, test_y = list0[0], list0[1], list0[2], list0[3]




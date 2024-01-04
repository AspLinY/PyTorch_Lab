import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import datetime
import warnings
from sklearn import preprocessing

warnings.filterwarnings("ignore")  # 忽略所有的警告信息
features = pd.read_csv('temps.csv')  # 使用Pandas库中的read_csv函数读取名为temps.csv的文件，并将数据存储在变量features中
print(features.head())  # 预览DataFrame（即features变量）中的前几行（默认为前5行）

print('数据维度', features.shape)  # 返回 数据维度 (348, 9) 即348行，9列

# 处理时间
# 分别得到年月日
years = features['year']
months = features['month']
days = features['day']

# datetime格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
         zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]  # 使用strptime函数将日期字符串转换为datetime对象
print(dates)

# 独热编码
features = pd.get_dummies(features)  # 如果features中有任何非数值型（通常是字符串）的列，get_dummies会为这些列中的每个唯一值创建一个新列。每个新列将用0和1表示原始数据中的值是否存在
print(features.head(5))

# 标签
labels = np.array(features['actual'])  # 提取名为'actual'的列，并将其转换为NumPy数组。这个数组labels被用作标签或目标变量，即您的模型需要预测的值。
print(labels)

# 在特征中去标签
features = features.drop('actual',
                         axis=1)  # 从features DataFrame中删除了'actual'列 确保features只包含输入特征（x），不包含需要预测的标签 参数axis=1指定了要删除的是列而不是行
print(features)

# 名字单独保存一下，以备后患
features_list = list(features.columns)
print(features_list)

# 转换成合适的形式
features = np.array(features)  # 将features DataFrame转换为NumPy数组
print(features)

input_features = preprocessing.StandardScaler().fit_transform(
    features)  # 特征标准化 确保所有特征都在相同的尺度上，这样一个特征就不会由于其数值范围大([1,2,3],[1000,1001,1002])而比其他特征对模型产生更大的影响
print(input_features)

# ====================构建网络模型====================

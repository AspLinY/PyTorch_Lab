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
labels = np.array(features['actual'])  # [y]:提取名为'actual'的列，并将其转换为NumPy数组。这个数组labels被用作标签或目标变量，即您的模型需要预测的值。
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
    features)  # [x]特征标准化 确保所有特征都在相同的尺度上，这样一个特征就不会由于其数值范围大([1,2,3],[1000,1001,1002])而比其他特征对模型产生更大的影响
print(input_features)

# ====================构建网络模型====================
x = torch.tensor(input_features, dtype=torch.float)  # 标准化后的特征，作为网络的输入
y = torch.tensor(labels, dtype=torch.float)  # 目标标签，是网络应该学习预测的值

# 权重参数初始化
# -----第一层（隐藏层）的权重和偏置
weights = torch.randn((14, 128), dtype=torch.float, requires_grad=True)  # 输入层14个神经元 隐藏层128个神经元
biases = torch.randn(128, dtype=torch.float, requires_grad=True)  # 因为隐藏层有128个神经元 为隐藏层中的每个神经元提供一个偏置值
# -----第二层（输出层）的权重和偏置
weights2 = torch.randn((128, 1), dtype=torch.float, requires_grad=True)  # 128对应于隐藏层的128个神经元，1对应于输出层的单个神经元
biases2 = torch.randn(1, dtype=torch.float, requires_grad=True)  # 为输出层的单个神经元提供一个偏置值
'''torch.float 是在PyTorch框架中定义的一个数据类型，它通常用于指定张量的数据类型。'''

learning_rate = 0.001
losses = []

for i in range(1000):
    # ==================前向传播==================
    # 计算隐藏层
    hidden = x.mm(weights) + biases
    # 加入激活函数
    hidden = torch.relu(hidden)  # 增加网络的非线性能力
    # 预测结果
    predictions = hidden.mm(weights2) + biases2  # y^
    '''在每一层，输出 y 通常是通过对上一层的输出 x 进行加权求和再加上偏置，然后通过激活函数进行转换得到的。即 y = f(wx + b) f是激活函数'''
    '''隐藏层：几乎总是需要激活函数。'''
    '''输出层是否需要激活函数取决于特定的任务:
        1、回归任务（如预测价格或温度），通常不使用激活函数'''
    '''输入层：输入层通常不使用激活函数。'''
    # ===========================================

    # 通过计算损失
    loss = torch.mean((predictions - y) ** 2)  # Mean Squared Error, MSE 损失函数
    losses.append(loss.data.numpy())

    # 打印
    if i % 100 == 0:
        print('loss:', loss)

    # 后向传播 -- 自动计算损失函数（loss）关于模型参数（通常是权重和偏置）的梯度（grad）
    loss.backward()
    '''梯度的作用
        指示了损失函数相对于每个参数的变化率，告诉我们如何调整参数以减少损失
        例如，如果某个权重的梯度是正数，意味着增加这个权重会增加损失，因此我们在更新时应该减小这个权重。'''

    # 更新参数 新参数=旧参数−(学习率×该层梯度)
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)

    # 每次迭代记得要清空
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

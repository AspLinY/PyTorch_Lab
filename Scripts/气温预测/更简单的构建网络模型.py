import torch.nn
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import 气温数据集与任务介绍

# =====================构建神经网络=====================
input_features = 气温数据集与任务介绍.input_features
features_list = 气温数据集与任务介绍.features_list
labels = 气温数据集与任务介绍.labels
input_size = input_features.shape[1]  # 样本数量 shape[1]将返回第二个元素，即 （列数或特征数）
hidden_size = 128  # 隐藏层的大小为128
output_size = 1  # 输出层的大小为1
batch_size = 16  # 设置批处理大小为16。这表示在神经网络训练过程中，每次前向和后向传播将处理16个数据样本
my_nn = torch.nn.Sequential(  # torch.nn.Sequential 是一个容器，按顺序包含了定义的各个层。这里包括两个 线性层(Linear) 层和一个 Sigmoid 激活函数。
    torch.nn.Linear(input_size, hidden_size),  # 这是网络的输入层（第一层），负责将输入数据映射到隐藏层
    torch.nn.Sigmoid(),  # Sigmoid激活函数。Sigmoid函数能够将输入值压缩到0和1之间，常用于二分类问题。
    torch.nn.Linear(hidden_size, output_size),  # 这是网络的输出层（第二层），负责将隐藏层的输出映射到最终的输出
)
cost = torch.nn.MSELoss(reduction='mean')  # 损失函数
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)  # 优化器为Adam-可以动态调整学习率
# my_nn.parameters() 提供了网络中需要训练的所有参数
# lr=0.001 设置了学习率为0.001。

#  ===========训练网络===========
losses = []  # 用于存储训练过程中每100次迭代的平均损失值
for i in range(1000):
    batch_loss = []  # 用于存储每个批次的损失
    # MINI-Batch 方法来进行训练
    for start in range(0, len(input_features), batch_size):  # 在每次外部循环中，数据将被按照batch_size（步长）分割成多个批次进行处理
        end = start + batch_size if start + batch_size < len(input_features) else len(
            input_features)  # 确保了最后一个批次能正确处理剩余的数据
        '''
            if start + batch_size < len(input_features):
                end = start + batch_size
            else:
                end = len(input_features)        
        '''
        x2 = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        y2 = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        prediction = my_nn(x2)  # 通过神经网络 my_nn 传递当前批次的数据 x2，得到预测结果
        loss = cost(prediction, y2)  # 损失函数
        optimizer.zero_grad()  # 清除累积的梯度
        loss.backward(retain_graph=True)  # 计算损失的梯度 retain_graph是否可重复执行
        optimizer.step()  # 更新神经网络的权重
        batch_loss.append(loss.data.numpy())  # 将当前批次的损失转换为NumPy格式，并添加到 batch_loss 列表中

    # 打印损失
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

# ==============预测训练结果==============
x = torch.tensor(input_features, dtype=torch.float)
predict = my_nn(x).data.numpy()

# 转换日期格式
# 分别得到年月日
features = pd.read_csv('temps.csv')  # 使用Pandas库中的read_csv函数读取名为temps.csv的文件，并将数据存储在变量features中
years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
         zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]  # 使用strptime函数将日期字符串转换为datetime对象

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

# 同理，再创建一个来存日期和其对应的模型预测值
months = features.iloc[:, features_list.index('month')]
days = features.iloc[:, features_list.index('day')]
years = features.iloc[:, features_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(
    data={'date': test_dates, 'prediction': predict.reshape(-1)})  # .reshape(-1) 能够将这个多维数组“展平”为一维数组

# =============画图===============
# 真实值
plt.plot(true_data['date'], true_data['actual'], '-b', label='actul')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation=60)
plt.legend()

#  图名
plt.xlabel('date')
plt.ylabel('Max tempereture(F)')
plt.title('Actual and Prediction')
plt.show()

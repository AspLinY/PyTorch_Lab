# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<使用GPU训练>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""只需要把数据和模型传入cuda里面就可以了"""
import torch
import torch.nn as nn
import numpy as np

x_values = [i for i in range(12)]  # 生成一个从 0 到 10 的整数列表
x_train = np.array(x_values, dtype=np.float32)  # 将使用 NumPy 库将 x_values 列表转换为一个 NumPy 数组，并指定数组的数据类型为 float32
x_train = x_train.reshape(-1, 1)  # 行数自己计算，列数固定为2

y_values = [2 * i + 1 for i in x_values]  # 生成一个从 0 到 10 的整数列表
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


class LinearRegressionModel(nn.Module):  # 定义用到了那些层

    def __init__(self, input_dim, output_dim):  # 它接收两个参数：input_dim（输入维度）和 output_dim（输出维度）
        super(LinearRegressionModel, self).__init__()  # 调用父类 nn.Module 的构造函数
        self.linear = nn.Linear(input_dim, output_dim)  # 创建了一个线性层（nn.Linear），并将其作为类的一个属性。

    def forward(self, x):  # 前向传输 定义怎么用的
        out = self.linear(x)  # 将输入数据 x 传递给之前定义的线性层,线性层计算y=wx+b并将结果赋给 out
        return out


# 模型既只有一个输入特征，也只有一个输出值。这是标准的单变量线性回归设置。
input_dim = 1  # 模型只有一个输入特征
output_dim = 1  # 模型只有一个输出值
model = LinearRegressionModel(input_dim, output_dim)  # 利用这些维度参数创建了线性回归模型的一个实例

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型传给cuda

# =============指定好参数和损失函数=============
epochs = 1000  # 模型将遍历训练数据 1000 次
learning_rate = 0.01  # 学习率，这是优化器在更新模型权重时使用的步长
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化器（SGD），用于更新模型的参数。这里使用了刚才设置的学习率
critersion = nn.MSELoss()  # 函数调用均方误差（MSE）

# =============训练模型=============
for epoch in range(epochs):
    # epoch += 1
    # 注意要转格式为tensor,将 NumPy 数组转换为 PyTorch 张量，这是 PyTorch 模型处理的数据格式
    inputs = torch.from_numpy(x_train).to(device)  # 数据传给cuda
    labels = torch.from_numpy(y_train).to(device)  # 数据传给cuda

    # 梯度每次清零
    optimizer.zero_grad()  # 这是必要的，因为默认情况下，梯度是累积的

    # 前向传播
    outputs = model(inputs)  # 将输入数据传递给模型

    # 计算损失
    loss = critersion(outputs, labels)  # 计算模型输出和真实标签之间的损失。

    # 反向传播
    loss.backward()  # 计算损失函数关于模型参数的梯度

    # 更新权重参数
    optimizer.step()  # 根据计算出的梯度更新模型的权重
    if epoch % 50 == 0:
        print('epoch:{},loss:{}'.format(epoch, loss.item()))  # 每 50 个周期打印一次当前的周期数和损失值

# =============测试模型预测结果=============
predicted = model(torch.from_numpy(x_train)).detach().numpy()
print(predicted)

# =============模型的保存与读取=============
torch.save(model.state_dict(), '../../Data/model.pkl')
model.load_state_dict(torch.load('../../Data/model.pkl'))

import torch
from torch import tensor

'''值（Scalar）组成特征（Vector），特征（Vector）组成矩阵（Matrix）'''
# ===========Scalar===========
# 通常就是一个数值
x = tensor(42)
print(x)  # x就是scalar

print(x * 2, x + x, x % 2, x / 2, x - 1)  # 可执行相关数学运算 （x/2结果为21. 说明转成了float）
print(x.dim())  # dim获取维度 因为只是一个值，所以为0，列表为1，矩阵为2，立方体空间为3
print(x.item())  # 从一个只包含单个元素的张量中提取这个元素的值,也就是说只有当维度为0的时候，才能用item

# ===========Vector===========
# 向量，通常指特征，eg：词向量特征，某一维度特征等等
v = tensor([-1.5, 0.5, 1.0])
print(v)
print(v.dim())
print(v * 2, v + v, v / 2, v - 1)
print(v.size())  # 打印出向量v的尺寸 返回torch.Size([3])它有一个维度，该维度上有三个元素
# eg:torch.Size([3,1])则表示一个二维张量，其中一个维度上有3个元素，另一个维度上只有1个元素。3行1列

# ===========Matrix===========
# 矩阵
M = tensor([[1, 2], [3, 4]])
print(M)
print(M.matmul(M))  # 矩阵乘法（自身相乘）
print(tensor([1, 0]).matmul(M))  # 向量与矩阵乘法
print(M * M)  # 元素级乘法 M * M，不同于matmul，这里的乘法是逐元素的
'''矩阵乘法与元素级乘法不同'''
'''在[元素级乘法]中，两个矩阵或数组的同位置元素相乘。这意味着进行这种乘法的两个矩阵或数组必须具有相同的尺寸'''
'''[矩阵乘法]是一种更复杂的运算，其中一个矩阵的行与另一个矩阵的列相乘，然后求和。'''

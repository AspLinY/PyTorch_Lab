import torch
from torch import tensor

#  Scalar
# 通常就是一个数值
x = tensor(42)
print(x)  # x就是scalar

x.dim()  # 获取维度 因为只是一个值，所以为0，列表为1，矩阵为2，立方体空间为3
print(x.dim())

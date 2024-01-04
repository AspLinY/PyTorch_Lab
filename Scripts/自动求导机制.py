import torch

x = torch.randn(3, 4, requires_grad=True)  # 张量x
print(x)

b = torch.randn(3, 4)
b.requires_grad = True  # 标记为需要梯度
print(b)  # 张量b

t = x + b
print(t)  # 张量t

y = t.sum()
print(y)

y.backward()  # 反向传播(是在 PyTorch 中实现自动梯度计算的关键步骤),自动计算了因变量 y 相对于自变量（在此例中是 b(因为被标记)）的梯度
var = b.grad  # 将 b 的梯度赋值给变量 var;grad：这是存储计算后的梯度的属性
print(var)

print(x.requires_grad, b.requires_grad, t.requires_grad, y.requires_grad)
# requires_grad 属性默认为 False，除非在创建张量时明确设置为 True 或者它是由已经设置为 True 的张量派生出来的。

import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]

loss_list = []

w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])
# Tensor 中保存两个值，一个data，一个grad,其中grad类型为Tensor
w1.requires_grad = True  # 需要计算梯度
w2.requires_grad = True
b.requires_grad = True

def forward(x):
    return x * x * w1 + x * w2 + b  # w为Tensor型，运算后返回Tensor型


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(300):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()  # backward将计算图所有需要计算梯度的地方求出梯度，并保存在变量中,计算图被释放
        print(x, y, "\tgrad=", w1.grad.item(), w2.grad.item(), b.grad.item())   # item是把里面的值直接拿出来变成一个标量
        w1.data = w1.data - 0.001 * w1.grad.data  # 取data进行更新不会建立计算图，此处仅需修改值
        w2.data = w2.data - 0.001 * w2.grad.data
        b.data = b.data - 0.001 * b.grad.data
        w1.grad.data.zero_()  # 将Tensor中的grad清零，准备进行下一次计算
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print("epoch=:", epoch+1, "loss=", l.item())
    loss_list.append(l.item())
print(w1.data.item(), w2.data.item(), b.data.item())
print("predict:x=5,y_pred=", forward(5).data.item())

plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(range(300), loss_list)
plt.show()


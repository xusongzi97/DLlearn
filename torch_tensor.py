import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

loss_list = []

w = torch.Tensor([1.0])
# Tensor 中保存两个值，一个data，一个grad,其中grad类型为Tensor
w.requires_grad = True  # 需要计算梯度


def forward(x):
    return x * w  # w为Tensor型，运算后返回Tensor型


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(300):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()  # backward将计算图所有需要计算梯度的地方求出梯度，并保存在变量中,计算图被释放
        print(x, y, "\tgrad=", w.grad.item())   # item是把里面的值直接拿出来变成一个标量
        w.data = w.data - 0.001 * w.grad.data  # 取data进行更新不会建立计算图，此处仅需修改值
        w.grad.data.zero_()  # 将Tensor中的grad清零，准备进行下一次计算

    print("epoch=:", epoch+1, "loss=", l.item())
    loss_list.append(l.item())

print("predict:x=4,y_pred=", forward(4).data.item())

plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(range(300), loss_list)
plt.show()


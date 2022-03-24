import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.Tensor([[0], [0], [1], [1]])

loss_list = []

w = torch.Tensor([1.0])
# Tensor 中保存两个值，一个data，一个grad,其中grad类型为Tensor
w.requires_grad = True  # 需要计算梯度


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # in_feature, out_feature, bias=True

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=True)  # 继承自nn.Module， 会构建计算图
# 使用方法： criterion(y_pred, y)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)  # 前传
    loss = criterion(y_pred, y_data)
    print("epoch= %d  loss=%.4f" % (epoch+1, loss.item()))
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()  # 反馈
    optimizer.step()  # 更新权重

print("w=", model.linear.weight.item())
print("b=", model.linear.bias.item())

x_test = torch.Tensor([[6.0], [7.0]])
y_test = model(x_test)
print("y_pred=",y_test.data)

plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(range(1000), loss_list)
plt.show()
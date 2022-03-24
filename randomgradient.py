import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [2, 4, 6]

loss_list = []

w = 1
lr = 0.001

def forward(x):
    return w * x


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2



def gradient(xs, ys):
    grad = 2 * x * (x * w - y)
    return grad


for epoch in range(1000):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        grad_val = gradient(x_data, y_data)
        w -= lr * grad_val
    loss_list.append(loss_val)
    print("Epoch:", epoch+1, "w=", w, "loss=", loss_val)


plt.plot(range(1000),loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

print("prediction:when x = 4, y = ", forward(4))


import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [2, 4, 6]

loss_list = []

w = 1
lr = 0.01

def forward(x):
    return w * x


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


for epoch in range(1000):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= lr * grad_val
    loss_list.append(cost_val)
    print("Epoch:", epoch+1, "w=", w, "loss=", cost_val)


plt.plot(range(1000),loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

print("prediction:when x = 4, y = ", forward(4))


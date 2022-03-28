import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

accuracy_list = []
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../dataset/mnist', train=True, download=True, transform=transform)
# train_set:60000, test_set:10000, classes:10   imgsize: 1*28*28
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist', train=False, download=True, transform=transform)
# 下载训练集和测试集
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class ResBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv1(y)
        return F.relu(x + y)


class CNNModule(torch.nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.resblock1 = ResBlock(16)
        self.resblock2 = ResBlock(32)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)

        x = self.pooling(self.relu(self.conv1(x)))
        x = self.resblock1(x)
        x = self.pooling(self.relu(self.conv2(x)))
        x = self.resblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = CNNModule()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        y_pred = model(inputs)

        model.zero_grad()
        loss = criterion(y_pred, targets)
        loss.backward()

        running_loss += loss.item()

        optimizer.step()

        if i % 300 == 299:
            print(f"epoch:{epoch + 1}, idx:{i+1} loss={running_loss}")
            running_loss = 0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("total: %d , correct: %d" % (total, correct))
        print("Accuracy = %.2f %%" % (100.0*correct/total))
        accuracy_list.append(correct/total)


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

    plt.plot(range(10), accuracy_list)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
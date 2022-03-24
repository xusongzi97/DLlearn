import torch
from torch.utils.data import Dataset  # 抽象类
from torch.utils.data import DataLoader
import numpy as np


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, dtype=np.float32, delimiter=',')
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])  # 要最后一列，且最后得到的是个矩阵，所以要加[]

    def __getitem__(self, item):  # dataset[index]  魔法方法，使实例化后的对象支持下标操作
        return self.x_data[item], self.y_data[item]

    def __len__(self):  # 魔法方法，使用 len() 时可返回mini_batch的条数
        return self.len


dataset = DiabetesDataset('diabetes.csv.gz')
train_loder = DataLoader(dataset=dataset,
                         batch_size=32,
                         shuffle=True,
                         num_workers=0)


class DiabetesModule(torch.nn.Module):
    def __init__(self):
        super(DiabetesModule, self).__init__()
        self.liner1 = torch.nn.Linear(8, 6)
        self.liner2 = torch.nn.Linear(6, 4)
        self.liner3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.liner1(x))
        x = self.sigmoid(self.liner2(x))
        x = self.sigmoid(self.liner3(x))
        return x


model = DiabetesModule()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loder, 0):  # 每次进行一个Mini——Batch
            inputs, lables = data
            y_pred = model(inputs)
            loss = criterion(y_pred, lables)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

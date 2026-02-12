from typing import Any, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn, optim
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, Dataset

# torch: dataset


def mock_samples() -> Tuple[Tensor, Tensor]:
    # 设置两个高斯分布的均值向量和协方差矩阵
    mu1 = -3 * torch.ones(2)
    mu2 = 3 * torch.ones(2)
    sigma1 = torch.eye(2) * 0.5
    sigma2 = torch.eye(2) * 2

    # 实例化两个二元高斯分布 m1 和 m2
    m1 = MultivariateNormal(mu1, sigma1)
    m2 = MultivariateNormal(mu2, sigma2)
    # 生成 100 个样本
    x1 = m1.sample(torch.Size((100,)))
    x2 = m2.sample(torch.Size((100,)))

    # 设置样本对应的标签 y
    y = torch.zeros((200, 1))
    y[100:] = 1

    # 使用 cat 函数将 x1 和 x2 组合
    x = torch.cat([x1, x2], dim=0)
    # 打乱样本和标签的顺序
    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]

    # 绘制样本
    plt.scatter(x1.numpy()[:, 0], x1.numpy()[:, 1])
    plt.scatter(x2.numpy()[:, 0], x2.numpy()[:, 1])

    return x, y


class CustomDataset(Dataset):
    def __init__(self, data: Tensor, labels: Tensor):
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Any:
        return self.data[idx], self.labels[idx]


def test_torch_dataset():
    data, labels = mock_samples()
    dataset = CustomDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 训练循环中使用
    for batch_data, batch_labels in dataloader:
        print(f"train: train={batch_data.shape}, label={batch_labels.shape}")
        # to impl training ...


# model: logistic


def my_linear(x, w, b):
    """实现 y = x*A.t + b 线性模型."""
    return torch.mm(x, w.t()) + b


def my_sigmoid(x):
    """实现 sigmoid 函数."""
    x = 1 / (1 + torch.exp(-x))
    return x


def my_loss(x, y):
    """实现损失函数."""
    loss = -torch.mean(torch.log(x) * y + torch.log(1 - x) * (1 - y))
    return loss


def test_logistic_process():
    # step1: 数据准备
    x, y = mock_samples()

    # step2: 线性方程
    # 定义了线性模型的输入维度 D_in 和输出维度 D_out
    # Logistic 回归是二分类模型, 预测的是变量为正类的概率, 所以输出的维度应该为 D_out=1
    D_in, D_out = 2, 1

    # 实例化 nn.Linear, 将线性模型应用到数据 x 上, 得到计算结果 output
    linear = nn.Linear(D_in, D_out, bias=True)
    output = linear(x)
    print(x.shape, linear.weight.shape, linear.bias.shape, output.shape)

    # step3: 激活函数
    # 将线性模型的计算结果映射到 0-1
    sigmoid = nn.Sigmoid()
    scores = sigmoid(output)
    print("scores:", scores)

    # step4: 损失函数
    loss = nn.BCELoss()
    result = loss(sigmoid(output), y)
    print("loss:", result)


class LogisticRegression(nn.Module):
    def __init__(self, D_in: int):
        super().__init__()
        self.linear = nn.Linear(D_in, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Any) -> Any:
        x = self.linear(x)
        output = self.sigmoid(x)
        return output


def test_logistic_model():
    x, y = mock_samples()
    lr_model = LogisticRegression(2)
    loss = nn.BCELoss()

    result = loss(lr_model(x), y)
    print("loss:", result)

    # 模型训练
    optimizer = optim.SGD(lr_model.parameters(), lr=0.03)

    iters = 10
    batch_size = 10
    for _ in range(iters):
        for i in range(int(len(x) / batch_size)):
            inputs = x[i * batch_size : (i + 1) * batch_size]
            target = y[i * batch_size : (i + 1) * batch_size]
            # 前向传播
            outputs = lr_model(inputs)
            l = loss(outputs, target)
            # 反向传播和优化
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

    # 模型评估
    lr_model.eval()
    with torch.no_grad():  # 不需要计算梯度
        predictions = lr_model(x)
        l = loss(predictions, y)
        print("final loss:", l.item())


# model: neural net


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.predict = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.hidden(x))
        return self.predict(x)


def test_neuralnet_model():
    train_data, targets = mock_samples()

    # 准备模型, 损失函数和优化器
    model = NeuralNet(input_size=10, hidden_size=20, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模式 (启用 dropout, 批归一化使用批统计量)
    model.train()
    for epoch in range(100):
        # 前向传播
        outputs = model(train_data)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 评估/推理模式 (禁用 dropout, 批归一化使用运行统计量)
    test_data = torch.rand(3, 3)
    model.eval()
    with torch.no_grad():  # 禁用梯度计算, 节省内存
        predictions = model(test_data)
        loss = criterion(predictions, targets)
        print("final loss:", loss.item())


if __name__ == "__main__":
    pass

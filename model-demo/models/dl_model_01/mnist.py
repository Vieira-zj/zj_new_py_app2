import os

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# 预处理: 将两个步骤整合在一起
compose_transform = transforms.Compose(
    {
        transforms.ToTensor(),  # 转为 Tensor, 范围改为 0-1
        transforms.Normalize((0.1307,), (0.3081)),  # 数据归一化, 即均值为 0, 标准差为 1
    }
)

# 训练数据集
train_data = MNIST(
    root="./data", train=True, download=False, transform=compose_transform
)
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

# 测试数据集
test_data = MNIST(
    root="./data", train=False, download=False, transform=compose_transform
)
test_loader = DataLoader(test_data, shuffle=False, batch_size=64)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 10)  # 10 个手写数字对应的 10 个输出

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 784)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return x


def model_train():
    model = Model()
    criterion = nn.CrossEntropyLoss()
    # 第一个参数是初始化参数值, 第二个参数是学习率
    optimizer = torch.optim.SGD(model.parameters(), 0.8)

    for index, data in enumerate(train_loader):
        input_data, target = data
        optimizer.zero_grad()
        y_predict = model(input_data)
        loss = criterion(y_predict, target)
        loss.backward()
        optimizer.step()
        if index % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")
            print("loss: %.2f" % loss.item())

    print("model training finished")


def load_model() -> Model:
    path = "./model/model.pkl"
    if not os.path.exists(path):
        raise ValueError("no model found:", path)

    model = Model()
    model.load_state_dict(torch.load(path))  # 加载模型参数
    return model


def model_test():
    model = load_model()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            input_data, target = data
            # output 输出 10 个预测取值, 其中最大的即为预测的数
            output: Tensor = model(input_data)
            # 返回一个元组, 第一个为最大概率值, 第二个为最大值的下标
            _, predict = torch.max(output.data, dim=1)
            # target是形状为 (batch_size,1) 的矩阵, 使用 size(0) 取出该批的大小
            total += target.size(0)
            # predict 和 target 均为 (batch_size,1) 的矩阵, sum() 求出相等的个数
            correct += (predict == target).sum().item()

        print("accuracy: %.2f" % (correct / total))


def main_test():
    model = load_model()

    image = Image.open("./test/test_one.png")
    image = image.resize((28, 28))  # 裁剪尺寸为 28*28
    image = image.convert("L")  # 转换为灰度图像

    transform = transforms.ToTensor()
    image = transform(image)
    image = image.resize(1, 1, 28, 28)

    output = model(image)
    probability, predict = torch.max(output.data, dim=1)
    print("predict:%d, probability:%.2f" % (predict[0], probability))


if __name__ == "__main__":
    pass

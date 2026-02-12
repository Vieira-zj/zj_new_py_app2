import torch
import torch.optim as optim
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义数据预处理操作
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # 数据增强: 随机翻转图片
        transforms.RandomCrop(32, padding=4),  # 数据增强: 随机裁剪图片
        transforms.ToTensor(),  # 将 PIL.Image 或者 numpy.ndarray 数据类型转化为 torch.FloadTensor, 并归一化到 [0.0, 1.0]
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),  # 标准化 (这里的均值和标准差是 CIFAR10 数据集的)
    ]
)

# 下载并加载训练数据集
trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# 下载并加载测试数据集
testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


class CnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 输入通道数 3, 输出通道数 6, 卷积核大小 5
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化, 核大小 2, 步长 2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道数 6, 输出通道数 16, 卷积核大小 5
        self.linear1 = nn.Linear(
            16 * 5 * 5, 120
        )  # 全连接层, 输入维度 16*5*5, 输出维度 120
        self.linear2 = nn.Linear(120, 84)  # 全连接层, 输入维度 120, 输出维度 84
        self.linear3 = nn.Linear(
            84, 10
        )  # 全连接层, 输入维度 84, 输出维度 10 (CIFAR10 有 10 类)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 第一层 卷积+激活函数+池化
        x = self.pool(torch.relu(self.conv2(x)))  # 第二层 卷积+激活函数+池化
        x = x.view(-1, 16 * 5 * 5)  # 将特征图展平
        x = torch.relu(self.linear1(x))  # 第一层 全连接+激活函数
        x = torch.relu(self.linear2(x))  # 第二层 全连接+激活函数
        x = self.linear3(x)  # 第三层 全连接
        return x


def model_train():
    model = CnnNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1, 3):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            inputs, labels = data  # 获取输入数据
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            if i % 2000 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch, i, running_loss / 2000))
                running_loss = 0.0

    torch.save(model.state_dict(), "./cifar_net.pth")  # 保存模型
    print("finished training")


def model_test():
    model = CnnNet()
    model.load_state_dict(torch.load("./cifar_net.pth"))  # 加载模型参数

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs: Tensor = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("accuracy of the network on test images: %d %%" % (100 * correct / total))


if __name__ == "__main__":
    pass

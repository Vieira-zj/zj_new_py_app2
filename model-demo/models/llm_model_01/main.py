import torch
from model import QdogBabyLearnConfig, QdogBabyLearnLLM
from torch import Tensor, nn
from transformers import AutoTokenizer

# uv pip install "transformers[torch]"


def main_train():
    #  原始语料数据 (假设这是分词后的 ID 序列)
    # 对应文本: QQ 浏览器广告后台开发
    data = [51, 51, 586, 240, 6262, 1179, 5046, 799, 2507, 3158, 1335]
    data_tensor = torch.tensor(data)

    # 1. 构造输入 (Input): 取序列的前 N-1 个字
    # 也就是: "QQ浏览器广告后台开发"
    x = data_tensor[:-1]
    print(f"x = {x}")  # x = [51, 51, 586, ..., 3158]

    # 2. 构造目标 (Target): 取序列的后 N-1 个字 (即每个位置对应的 "下一个字")
    # 也就是: "Q浏览器广告后台开发组"
    y = data_tensor[1:]
    print(f"y = {y}")  # y = [51, 586, 240, ..., 1335]

    print(f"输入长度: {len(x)}, 目标长度: {len(y)}")
    # 输入长度: 10, 目标长度: 10

    # 3. 扔进模型计算
    config = QdogBabyLearnConfig()
    model = QdogBabyLearnLLM(config=config)
    logits: Tensor = model(x.unsqueeze(0))  # 增加 batch 维度

    # 4. 计算 Loss (交叉熵损失)
    # 这一步是在问模型:
    # 当输入第 1 个 'Q' 时, 你预测是第 2 个 'Q' 的概率大吗?
    # 当输入 '浏' 时, 你预测是 '览' 的概率大吗?
    # ...
    # 当输入 '发' 时, 你预测是 '组' 的概率大吗?
    loss = nn.CrossEntropyLoss(logits.view(-1, config.vocab_size), y.view(-1))
    loss.backward()  # 反向传播, 修正参数


def main_predict():
    # 首先将输入文本进行编码
    # tokenizer.encode() 方法将文本字符串转换为模型可理解的数字序列 (token IDs)
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    inputs = tokenizer.encode("QQ浏览器广告后台开发")
    print("encode token ids:", inputs)

    # 将编码结果转换为 PyTorch 张量, 并添加批次维度
    # torch.tensor([inputs]) 创建了一个形状为 [1, sequence_length] 的张量
    batch = torch.tensor([inputs])

    # 将批次数据输入模型进行推理
    # model(batch) 返回模型的输出, 通常是 logits 或概率分布
    config = QdogBabyLearnConfig()
    model = QdogBabyLearnLLM(config=config)

    output: Tensor = model(batch)
    print("model output:", output)
    # 输出形状: torch.Size([1, 10, 6400])
    print("output shape:", output.shape)

    # 使用 argmax 获取每个位置概率最大的 token 索引
    # dim=-1 表示在最后一个维度 (通常是词汇表维度) 上取最大值
    predictions = torch.argmax(output, dim=-1)
    print("predict result (token index):", predictions)

    # 将预测的 token 索引解码回文本
    # tokenizer.decode() 将数字序列转换回可读的文本字符串
    print("decode char:", tokenizer.decode(predictions[0]))


if __name__ == "__main__":
    pass

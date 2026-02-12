import torch
from torch import Tensor, nn

# example: torch


def test_torch_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device.type)


def test_torch_tensor_01():
    print(f"tensor default type: {torch.tensor(1).dtype}")

    data = [51, 51, 586, 240, 6262, 1179, 5046, 799, 2507, 3158, 1335]
    data_tensor = torch.tensor(data)
    x = data_tensor[:-1]
    y = data_tensor[1:]
    print(f"input len: {len(x)}, target len: {len(y)}")

    t = torch.tensor([1.0, 2.0])
    print(f"\n{t.shape}:\n{t}")

    t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
    print(f"\n{t.shape}:\n{t}")


def test_torch_tensor_02():
    t = torch.arange(5)
    print(f"arange:\n{t}")
    print()

    t = torch.empty(2, 2)
    print(f"empty:\n{t}")
    t = torch.zeros((2, 2))
    print(f"zeros:\n{t}")
    t = torch.ones((2, 2))
    print(f"ones:\n{t}")
    print()

    t = torch.rand(3, 3)
    print(f"rand:\n{t}")
    t = torch.randn(3, 3)
    print(f"randn:\n{t}")


def test_torch_calculate_01():
    x = torch.tensor([-1.1, 0.5, 0.501, 0.99])
    y = torch.round(x)
    print(f"round:\n{y}")

    x = torch.tensor([[1, 2], [6, 4]])
    idx = torch.argmax(x)
    y = x.view(4)
    print(f"\nflat tensor:\n{y}")
    print(f"argmax={idx}, max={y[idx].data}")


def test_torch_calculate_02():
    x = torch.tensor([1, 2])
    y = torch.tensor([3, 4])

    # 逐元素相乘
    z = x * y
    print(f"x * y: {z}")

    # 点积运算
    z = x @ y
    print(f"x @ y: {z}")  # 1*3 + 2*4 = 11


def test_torch_dot_mm_01():
    # a,b 相同维度
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[1, 2], [3, 4]])

    # 点积运算
    # 1*1 + 2*3 = 7
    # 1*2 + 2*4 = 10
    # 3*1 + 4*3 = 15
    # 3*2 + 4*4 = 22
    c = a @ b
    print(f"dot:\n{c}")

    # 矩阵乘法
    c = torch.mm(a, b)
    print(f"mm:\n{c}")


def test_torch_dot_mm_02():
    # a,b 不同维度
    a = torch.tensor([[1, 2, 3], [3, 4, 5]])
    b = torch.tensor([[1, 2], [3, 4], [5, 6]])

    # 点积运算
    # 1*1 + 2*3 + 3*5 = 22
    # 1*2 + 2*4 + 3*6 = 28
    # 3*1 + 4*3 + 5*5 = 40
    # 3*2 + 4*4 + 5*6 = 52
    c = a @ b
    print("dot")
    print("shape:", c.shape)
    print(f"value:\n{c}")

    # 矩阵乘法
    print("\nmm")
    c = torch.mm(a, b)
    print("shape:", c.shape)
    print(f"value:\n{c}")


def test_torch_transform():
    # 增加维度
    x = torch.randn(3, 4)
    x_unsq = x.unsqueeze(0)
    print(f"unsqueeze:\n{x_unsq.shape}")

    # 减少维度
    x_sq = x_unsq.squeeze(0)
    print(f"squeeze:\n{x_sq.shape}")

    # 转置
    y = x.transpose(0, 1)
    print(f"transpose:\n{y.shape}")


def test_torch_reshape():
    x = torch.randn(2, 3, 4)
    print("is_contiguous:", x.is_contiguous())
    print(x)

    # 连续使用 view
    y = x.view(6, 4)
    print(y)

    # 转置后变为非连续
    print("\ntranspose:")
    x_t = x.transpose(1, 2)
    print(x_t.is_contiguous())  # false

    # 非连续使用 reshape
    y_t = x_t.reshape(6, 4)
    print(y_t)


def test_torch_grad_01():
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 2 * x + 2
    # 计算梯度
    y.backward()
    print(f"在 x=2 处的导数: {x.grad}")


def test_torch_grad_02():
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    print(f"x + 2:\n{y}")
    # y 是计算结果, 所以它有 grad_fn 属性
    print(f"y grad fn:\n{y.grad_fn}")

    z = y * y * 3
    print(f"y * y * 3:\n{z}")
    out = z.mean()
    print(f"mean: {out}")

    # out.backward() 等同于 out.backward(torch.tensor(1.))
    # 如果 out 不是一个标量, 因为 tensor 是矩阵, 那么在调用 .backward() 时需要传入一个与 out 相同 shape 的权重向量进行相乘
    out.backward()
    print(f"\ngrad:\n{x.grad}")


def test_torch_regexp():
    x = torch.randn(32, 100)
    dropout = nn.Dropout(p=0.2)  # 丢弃 20% 神经元
    x_dropped = dropout(x)  # 训练时随机置零部分元素
    print(f"dropout: {x_dropped}")


# example: attention


def test_tf_attention():
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1) - 第 1 个 token 嵌入向量
            [0.55, 0.87, 0.66],  # journey  (x^2) - 第 2 个 token 嵌入向量
            [0.57, 0.85, 0.64],  # starts   (x^3) - 第 3 个 token 嵌入向量
            [0.22, 0.58, 0.33],  # with     (x^4) - 第 4 个 token 嵌入向量
            [0.77, 0.25, 0.10],  # one      (x^5) - 第 5 个 token 嵌入向量
            [0.05, 0.80, 0.55],  # step     (x^6) - 第 6 个 token 嵌入向量
        ]
    )

    dim = inputs.shape[1]
    print("dim:", dim)

    # inputs 形状: [seq_len, dim] (6*3)
    # inputs.T 形状: [dim, seq_len] (3*6)
    # 结果 attn_scores 形状: [seq_len, seq_len] (6*6 关系矩阵)
    attn_scores = inputs @ inputs.T  # query @ key
    print(f"attation scores:\n{attn_scores}")

    my_attn_scores = get_my_attn_scores(inputs)
    print(f"my attation scores:\n{my_attn_scores}")

    # Softmax: 将分数转化为概率 (权重), 相当于分配注意力比例
    # 比如 [0.1, 0.8, 0.1] 表示主要关注中间那个词
    attn_weight = torch.softmax(attn_scores, dim=-1)  # 6*6 关系矩阵
    print(f"attn weight:\n{attn_weight}")

    # MatMul: 根据权重, 提取并融合信息
    # 权重大的向量会被更多地 "吸取" 进来, 权重小的则被忽略
    context_vec = attn_weight @ inputs
    print(f"results:\n{context_vec}")  # 6*3 关系矩阵


def get_my_attn_scores(inputs: Tensor) -> Tensor:
    attn_scores = torch.empty(6, 6)

    # 计算注意力分数矩阵: 遍历所有 token 对, 计算它们之间的点积相似度
    # 外层循环: 遍历每个 token 作为查询 (query)
    for i, x_i in enumerate(inputs):
        # 内层循环: 遍历每个 token 作为键 (key)
        for j, x_j in enumerate(inputs):
            # 注意这里的 token 是一个多维度的向量, 所以要展开计算
            dot_product = 0.0
            # 遍历向量的每个维度进行计算
            for k, _ in enumerate(x_i):
                dot_product += x_i[k] * x_j[k]

            attn_scores[i, j] = dot_product

    return attn_scores


if __name__ == "__main__":
    # test_torch_device()

    # test_torch_tensor_01()
    # test_torch_tensor_02()

    # test_torch_dot_mm_01()
    test_torch_dot_mm_02()
    # test_torch_calculate_01()
    # test_torch_calculate_02()

    # test_torch_transform()
    # test_torch_view()

    # test_torch_grad_01()
    # test_torch_grad_02()

    # test_torch_regexp()

    # test_tf_attention()

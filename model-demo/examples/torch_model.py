import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from torch import Tensor


def test_torch_dropout():
    x = torch.randn(32, 100)
    dropout = torch.nn.Dropout(p=0.2)  # 丢弃 20% 神经元
    x_dropped = dropout(x)  # 训练时随机置零部分元素
    print(f"dropout: {x_dropped}")


# example: embedding


def test_embedding_st_model_01():
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "今天天气很好",
        "我喜欢机器学习",
    ]
    embeddings = st_model.encode(sentences, normalize_embeddings=True)
    print("embedding shape:", embeddings.shape)

    sums = np.sum(embeddings, axis=1)
    print("sum embeddings:", sums)  # 按行求和


def test_embedding_st_model_02():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "宠物"
    docs = ["猫是一种动物", "狗是人类的朋友", "今天天气不错"]

    query_embedding = model.encode(query)
    doc_embeddings = model.encode(docs)
    print("query embedding shape:", query_embedding.shape)
    print("doc embeddings shape:", doc_embeddings.shape)

    cos_scores = util.cos_sim(query_embedding, doc_embeddings)
    dot_scores = util.dot_score(query_embedding, doc_embeddings)
    # 当向量已归一化时, 内积 = 余弦相似度
    print(f"cos_sim_scores={cos_scores}, dot_scores={dot_scores}")


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
    # test_torch_dropout()

    # test_embedding_st_model_01()
    test_embedding_st_model_02()

    # test_tf_attention()

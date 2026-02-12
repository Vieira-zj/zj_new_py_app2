import torch
from torch import Tensor, nn

# ==========================================
# 1. Config
# ==========================================


class QdogBabyLearnConfig:
    def __init__(self):
        self.model_name: str = "qdogbabylearn"
        self.version: str = "1.0.0"

        self.num_hidden_layers: int = 16  # 盖多少层楼 (Transformer Block的数量)
        self.num_heads: int = 8  # 雇多少个专家同时看 (多头注意力的头数)
        self.emb_dim: int = 512  # 词向量的宽度 (每个词包含多少信息量)
        self.dropout: float = 0.0  # 随机扔掉多少神经元 (防止死记硬背)
        self.context_size: int = 512  # 一次能看多长的文章 (上下文窗口大小)
        self.vocab_size: int = 6400  # 字典里有多少个字 (词表大小)


# ==========================================
# 2. LLM Model
# ==========================================


class QdogBabyLearnLLM(nn.Module):
    def __init__(self, config: QdogBabyLearnConfig):
        super().__init__()
        self.config = config

        # 查表升维: 把离散的字变成向量
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        # 加入位置信息: 让模型知道 "我爱你" 和 "你爱我" 顺序不同
        self.pos_emb = nn.Embedding(config.context_size, config.emb_dim)

        self.dropout = nn.Dropout(config.dropout)

        # 叠罗汉: 堆叠 16 层 Transformer Block
        tf_blocks = [
            TransformerBlock(
                config.emb_dim,
                config.num_heads,
                config.dropout,
                config.context_size,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.transformer_blocks = nn.Sequential(*tf_blocks)

        # 最后输出前的标准化
        self.norm = LayerNorm(config.emb_dim)
        # ★ 映射回字典: 把向量变回预测下一个字的概率分数
        self.out = nn.Linear(config.emb_dim, config.vocab_size)  # emb_dim -> vocab_size

    def forward(self, x: Tensor) -> Tensor:
        # x 的形状: [batch_size, seq_len] (比如: [[1, 2, 3]])

        # 1. 查词表
        tok_embeds = self.tok_emb(x)
        # 2. 查位置表
        pos_embeds = self.pos_emb(torch.arange(x.shape[1], device=x.device))

        # 3. 信息融合 (词义 + 位置)
        x = tok_embeds + pos_embeds
        x = self.dropout(x)

        # 4. 经过 16 层 "思考"
        x = self.transformer_blocks(x)

        # 5. 最终整理
        x = self.norm(x)
        x = self.out(x)  # 输出形状: [batch_size, seq_len, vocab_size]
        return x


# ==========================================
# 3. Transformer Block
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float, context_size: int):
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(
            emb_dim, num_heads, dropout, context_size
        )
        self.layer_norm1 = LayerNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim, dropout)
        self.layer_norm2 = LayerNorm(emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        # --- 第一步: 注意力机制 (词与词找关系) ---
        residual = x  # 做模型不能忘本: 先记住原始输入
        x = self.layer_norm1(x)  # 拉齐起跑线
        x = residual + self.multi_head_attn(x)  # 原始信息 + 新学到的关系

        # --- 第二步: 前馈网络 (思考/记忆) ---
        residual = x  # 再次记住当前状态
        x = self.layer_norm2(x)  # 拉齐起跑线
        x = residual + self.feed_forward(x)  # 原始信息 + 思考后的结果

        return x


# ==========================================
# 4. 多头注意力 (核心: 我要啥, 你有啥, 我是啥)
# ==========================================


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float, context_size: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # 每个 head 负责的维度 (512/8 = 64)

        # 定义 Q,K,V 映射层
        self.W_querys = nn.Linear(emb_dim, emb_dim)  # 我要啥
        self.W_keys = nn.Linear(emb_dim, emb_dim)  # 你有啥
        self.W_values = nn.Linear(emb_dim, emb_dim)  # 我是啥

        self.out = nn.Linear(emb_dim, emb_dim)  # 信息汇总
        self.dropout = nn.Dropout(dropout)

        # Mask (不让看答案): 创建一个下三角矩阵
        self.register_buffer("mask", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, emb_dim = x.shape

        # 1. 线性变换生成 Q,K,V
        querys: Tensor = self.W_querys(x)
        keys: Tensor = self.W_keys(x)
        values: Tensor = self.W_values(x)

        # 2. 切分成多头 (8个人分工合作)
        # 形状变换: [B, T, 512] -> [B, T, 8, 64]
        querys = querys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # 3. 转置, 把头 (Head) 放到前面, 方便并行计算
        # 形状变换: [B, 8, T, 64]
        querys = querys.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 4. 计算相似度
        # Q @ K^T
        attn_score = querys @ keys.transpose(-1, -2)

        # 5. Mask 操作 (不让偷看后面的词)
        # 把 mask 为 0 的位置填成负无穷大 (-inf)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_score = attn_score.masked_fill(mask_bool == 0, float("-inf"))

        # 6. 归一化 (概率化)
        # 除以根号 dk 是为了防止梯度消失
        attn_weight: Tensor = torch.softmax(attn_score / (self.head_dim**0.5), dim=-1)
        attn_weight = self.dropout(attn_weight)

        # 7. 加权求和
        # Weight @ V
        context_vec: Tensor = attn_weight @ values

        # 8. 还原形状 (把8个头的结果拼回去)
        # [B, 8, T, 64] -> [B, T, 8, 64] -> [B, T, 512]
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, emb_dim)
        )

        # 9. 最终线性输出
        context_vec = self.out(context_vec)
        return context_vec


# ==========================================
# 5. 前馈网络 (发散思维 -> 收敛总结)
# ==========================================


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim * 4)
        self.linear2 = nn.Linear(emb_dim * 4, emb_dim)
        print("to impl dropout:", dropout)

    def forward(self, x: Tensor) -> Tensor:
        # 发散思维: 维度放大 4 倍 (512 -> 2048)
        x = self.linear1(x)
        # 非线性激活 (增加脑回路的复杂度)
        x = nn.GELU()(x)
        # 收敛总结: 维度变回原样 (2048 -> 512)
        x = self.linear2(x)
        return x


# ==========================================
# 6. 层归一化 (拉齐起跑线)
# ==========================================


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可学习的参数: 缩放 (scale) 和 平移 (shift)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        # 计算平均值
        mean = x.mean(dim=-1, keepdim=True)
        # 计算方差
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化公式: (x - 均值) / 根号(方差 + 极小值)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return norm * self.scale + self.shift

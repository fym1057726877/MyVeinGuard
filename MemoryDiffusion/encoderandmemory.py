import copy
import math
import torch as th
from torch import nn
from torch.nn import functional as F
from DirectActor.ActorModel import Resnet34Actor


class EncoderAndMemory(nn.Module):
    def __init__(self, latent_dims=256, MEM_DIM=200, addressing="sparse", image_size=128):
        super(EncoderAndMemory, self).__init__()

        self.MEM_DIM = MEM_DIM
        self.latent_dims = latent_dims
        assert isinstance(image_size, (int, list, tuple)), "image_size must be int, list or tuple"
        if isinstance(image_size, int):
            self.img_width = self.img_height = image_size
        if isinstance(image_size, (list, tuple)):
            self.img_width = image_size[0]
            self.img_height = image_size[1]

        self.encoder = Resnet34Actor(latent_dim=self.latent_dims)

        self.memory = th.zeros((self.MEM_DIM, self.latent_dims))
        nn.init.kaiming_uniform_(self.memory)
        self.memory = nn.Parameter(self.memory)

        self.Cosine_Similiarity = nn.CosineSimilarity(dim=2)
        self.attn = MultiHeadedAttention(head_num=5, MEM_DIM=self.MEM_DIM, dropout=0.1)
        self.addressing = addressing
        if self.addressing == 'sparse':
            self.threshold = 1 / self.MEM_DIM
            self.epsilon = 1e-12

        self.linear = nn.Linear(self.latent_dims, self.img_width * self.img_height)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        z = self.encoder(x)  # (B, C, H, W) -> (B, latent_dims)

        ex_mem = self.memory.unsqueeze(0).repeat(B, 1, 1)  # [b, mem_dim, fea]
        ex_z = z.unsqueeze(1).repeat(1, self.MEM_DIM, 1)  # [b, mem_dim, fea]

        mem_logit = self.Cosine_Similiarity(ex_z, ex_mem)  # [b, mem_dim]
        mem_weight = self.attn(mem_logit, mem_logit, mem_logit)  # [b, num_mem]

        # soft寻址和稀疏寻址
        z_hat = None
        if self.addressing == "soft":
            z_hat = th.matmul(mem_weight, self.memory)
        elif self.addressing == "sparse":
            mem_weight = (self.relu(mem_weight - self.threshold) * mem_weight) \
                         / (th.abs(mem_weight - self.threshold) + self.epsilon)
            mem_weight = F.normalize(mem_weight, p=1, dim=1)
            z_hat = th.matmul(mem_weight, self.memory)

        assert z_hat is not None, "model parameter：addressing is wrong"

        x_hat = self.linear(z_hat).view(x.shape)
        return dict(x_hat=x_hat, z_hat=z_hat, mem_weight=mem_weight)


# 注意力机制
def attention(query, key, value, mask=None, dropout=None):
    # query, key, value：3个输入, q:原始文本；k:给定文本的关键词；v：大脑对文本的延申
    # mask：掩码张量
    # dropout：nn.Dropout的实例

    # 取query最后一维，代表嵌入维度
    d_k = query.size(-1)
    # 按照注意力计算公式
    scores = th.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 完成与value的乘法, 返回注意力表示
    return th.matmul(p_attn, value), p_attn


# 多头注意力机制
# 首先定义克隆函数，克隆某个模型，要用到copy包实现深度拷贝
def clone(model, N):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, MEM_DIM, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert MEM_DIM % head_num == 0  # head_num必须得被embedding_dim整除
        self.d_k = MEM_DIM // head_num
        self.head_num = head_num
        # q,k,v个需要一个，最后拼接还需要一个，总共四个线性层
        self.linears = clone(nn.Linear(MEM_DIM, MEM_DIM), 4)
        self.attn = None  # 最后得到的注意力张量
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        # 进入多头处理环节
        # 得到每个头的输入
        query, key, value = \
            [model(x).view(batch_size, self.head_num, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]

        # 得到每个头的注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 把每个头的矩阵聚合起来,相当于得到输入步骤的逆操作
        # 当我们转置矩阵后矩阵的存储不是连续的，此后无法使用view进行重构，contiguous()函数能将矩阵的存储恢复为连续状态，
        x = x.transpose(1, 2).contiguous().view(batch_size, self.d_k * self.head_num)
        x = self.linears[-1](x)

        return x


# if __name__ == '__main__':
#     model = EncoderAndMemory()
#     x = th.randn((16, 1, 128, 128))
#     y = th.randn((16, 200))
#     att = MultiHeadedAttention(head_num=5, latent_dims=200)
#     out = att(y, y, y)
#     print(out.shape)

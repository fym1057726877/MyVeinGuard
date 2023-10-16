import math
import torch as th
from torch import nn
from torch.nn import functional as F
from DirectActor.ActorModel import Resnet34Encoder
from pywt import dwt2, idwt2


class WaveletsMemory(nn.Module):
    def __init__(self, feature_dims=2048, MEM_DIM=200, image_size=64):
        super(WaveletsMemory, self).__init__()

        self.MEM_DIM = MEM_DIM
        self.feature_dims = feature_dims
        assert isinstance(image_size, (int, list, tuple)), "image_size must be int, list or tuple"
        if isinstance(image_size, int):
            self.img_width = self.img_height = image_size
        if isinstance(image_size, (list, tuple)):
            self.img_width = image_size[0]
            self.img_height = image_size[1]

        # self.encoder = Resnet34Encoder(self.feature_dims)
        self.encoder = ConvEncoder(image_channels=1, conv_channels=32)
        self.decoder = ConvDecoder(image_channels=1, conv_channels=32)

        self.memory = th.zeros((self.MEM_DIM, self.feature_dims))
        nn.init.kaiming_uniform_(self.memory)
        self.memory = nn.Parameter(self.memory)

        self.Cosine_Similiarity = nn.CosineSimilarity(dim=2)
        # self.attn = MultiHeadedAttention(in_dim=MEM_DIM, num_heads=8)

        self.threshold = 1 / self.MEM_DIM
        self.epsilon = 1e-12

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        z = self.encoder(x)  # (B, C, H, W) -> (B, fea)
        ex_z = z.unsqueeze(1).repeat(1, self.MEM_DIM, 1)  # [b, mem_dim, fea]
        ex_mem = self.memory.unsqueeze(0).repeat(B, 1, 1)  # [b, mem_dim, fea]

        mem_logit = self.Cosine_Similiarity(ex_z, ex_mem)  # [b, mem_dim]
        # mem_weight = self.attn(mem_logit)  # [b, num_mem]
        mem_weight = F.softmax(mem_logit, dim=-1)

        # 稀疏寻址
        mem_weight = ((self.relu(mem_weight - self.threshold) * mem_weight)
                      / (th.abs(mem_weight - self.threshold) + self.epsilon))
        mem_weight = F.normalize(mem_weight, p=1, dim=1)
        z_hat = th.matmul(mem_weight, self.memory)

        x_recon = self.decoder(z_hat)

        return dict(x_recon=x_recon, z_hat=z_hat, mem_weight=mem_weight)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, in_dim, embed_dim=None, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        embed_dim = embed_dim or in_dim
        assert embed_dim % num_heads == 0  # head_num必须得被embedding_dim整除
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # q,k,v个需要一个，最后拼接还需要一个，总共四个线性层
        self.linear_q = nn.Linear(in_dim, self.embed_dim)
        self.linear_k = nn.Linear(in_dim, self.embed_dim)
        self.linear_v = nn.Linear(in_dim, self.embed_dim)

        self.scale = 1 / math.sqrt(self.embed_dim // self.num_heads)

        self.final_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B, L = x.shape
        assert L == self.in_dim
        # 进入多头处理环节
        # 得到每个头的输入
        dk = self.embed_dim // self.num_heads
        q = self.linear_q(x).view(B, self.num_heads, dk)  # B,H,dk
        k = self.linear_k(x).view(B, self.num_heads, dk)
        v = self.linear_v(x).view(B, self.num_heads, dk)

        dist = th.matmul(q, k.transpose(1, 2)) * self.scale  # B,H
        dist = th.softmax(dist, dim=-1)
        dist = self.dropout(dist)

        attn = th.matmul(dist, v)  # B,H,dk
        attn = attn.contiguous().view(B, self.embed_dim)
        out = self.final_linear(attn)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, image_channels=1, conv_channels=32):
        super(ConvEncoder, self).__init__()
        self.image_channel = image_channels
        self.conv_channel = conv_channels
        self.block1 = self._conv_block(
            in_channels=self.image_channel,
            out_channels=self.conv_channel // 2,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.block2 = self._conv_block(
            in_channels=self.conv_channel // 2,
            out_channels=self.conv_channel,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.block3 = self._conv_block(
            in_channels=self.conv_channel,
            out_channels=self.conv_channel * 2,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.final_conv = nn.Conv2d(
            in_channels=self.conv_channel * 2,
            out_channels=self.conv_channel * 4,
            kernel_size=1,
            stride=2,
            padding=0,
        )

    @staticmethod
    def _conv_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 2,
            padding: int = 1,
    ) -> nn.Sequential():
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final_conv(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, image_channels=1, conv_channels=32):
        super(ConvDecoder, self).__init__()
        self.image_channels = image_channels
        self.conv_channels = conv_channels

        self.block1 = self._transconv_block(
            in_channels=self.conv_channels * 4,
            out_channels=self.conv_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.block2 = self._transconv_block(
            in_channels=self.conv_channels * 2,
            out_channels=self.conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.block3 = self._transconv_block(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.final_transconv = nn.ConvTranspose2d(
            in_channels=self.conv_channels // 2,
            out_channels=image_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

    @staticmethod
    def _transconv_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 2,
            padding: int = 1,
            output_padding: int = 0,
    ) -> nn.Sequential():
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        # x:[b, self.conv_channel*4*4*4]
        x = x.view(-1, self.conv_channels * 4, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final_transconv(x)
        return x


def testencoder():
    e = ConvEncoder(image_channels=1, conv_channels=32)
    x = th.randn((16, 1, 64, 64))
    o = e(x)
    print(o.shape)


def testdecoder():
    d = ConvDecoder(image_channels=1, conv_channels=32)
    x = th.randn((16, 2048))
    o = d(x)
    print(o.shape)


def testmemory():
    model = WaveletsMemory()
    x = th.randn((16, 1, 64, 64))
    out = model(x)["z_hat"]
    print(out.shape)


if __name__ == '__main__':
    testencoder()

import torch as th
from torch import nn
from DirectActor.ActorModel import Resnet34Actor
from torch.nn import functional as F

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
        mem_weight = mem_logit.softmax(dim=1)  # [b, num_mem]

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


if __name__ == '__main__':
    model = EncoderAndMemory()
    x = th.randn((16, 1, 128, 128))
    out = model(x)["x_hat"]
    print(out.shape)
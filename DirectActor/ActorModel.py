import torch.nn as nn
import torchvision
import torch
from data.mydataset import Vein600_128x128
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


class Resnet34Actor(nn.Module):
    def __init__(self, latent_dim=256):
        super(Resnet34Actor, self).__init__()
        self.action_bound = 10
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.resnet = torchvision.models.resnet34()
        self.linear_out = nn.Linear(1000, latent_dim)


    def forward(self, img):
        z = self.conv_in(img)
        z = self.resnet(z)
        z = self.linear_out(z)
        z = torch.tanh(z)
        z = z * self.action_bound
        return z


def test_actormodel():
    transform = transforms.ToTensor()
    dataset = Vein600_128x128(transform=transform)
    dataloder = DataLoader(dataset, batch_size=100, shuffle=True)
    print(len(dataloder))
    x, y = next(iter(dataloder))

    print(x.shape, y.shape)

    model = Resnet34Actor()
    out = model(x)
    print(out.shape)

if __name__ == '__main__':
    test_actormodel()
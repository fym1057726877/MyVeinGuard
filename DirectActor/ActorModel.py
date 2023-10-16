import torch.nn as nn
import torchvision
import torch as th


class Resnet34Encoder(nn.Module):
    def __init__(self, feature_dims=2048):
        super(Resnet34Encoder, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.resnet = torchvision.models.resnet34()
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-3]
        )

    def forward(self, img):
        B = img.shape[0]
        z = self.conv_in(img)
        z = self.resnet(z)
        z = z.view(B, -1)
        return z


def test_actormodel():
    x = th.randn((16, 1, 64, 64))
    model = Resnet34Encoder()
    out = model(x)
    print(out.shape)

# test_actormodel()






import os
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from TransGAN.TransGanModel import SwinTransGenerator, SwinTransDiscriminator


# dataset_name = "Handvein"
# dataset_name = "Handvein3"
dataset_name = "Fingervein2"
batchsize = 16
g_lr = 1e-5
d_lr = 1e-5
delay = 3
device = "cuda"
start_epoch = 0
total_epochs = 8000

g_embed_dim = 512

d_embed_dim = 768

# g_depths = [2, 4, 2, 2]
g_depths = [4, 2, 2, 2]
# d_depths = [2, 2, 6, 2]
d_depths = [4, 2, 2, 2]

# is_local = True
is_local = False
is_peg = True
# is_peg = False

if dataset_name == "Handvein" or dataset_name == 'Handvein3':
    width = 64
    height = 64
    bottom_width = 8
    bottom_height = 8
    window_size = 4

if dataset_name == 'Fingervein2':
    width = 128
    height = 64
    bottom_width = 16
    bottom_height = 8
    window_size = 4



class TrainTransGAN:
    def __init__(self):
        super(TrainTransGAN, self).__init__()
        self.train_data = DataLoader()
        self.latent_dim = 256
        self.generator = SwinTransGenerator(
            embed_dim=g_embed_dim,
            bottom_width=bottom_width,
            bottom_height=bottom_height,
            window_size=window_size,
            depth=g_depths,
            is_local=is_local,
            is_peg=is_peg
        )
        self.discriminator = SwinTransDiscriminator(
            img_height=height,
            img_width=width,
            patch_size=window_size,
            embed_dim=d_embed_dim,
            depth=d_depths,
            is_local=is_local,
            is_peg=is_peg
        )
        self.generator.to(device)
        self.discriminator.to(device)
        ckp_parent_path = os.path.join(
            "TransGAN", "train_ckp"
        )
        self.g_ckp = os.path.join(
            ckp_parent_path,
            "TransGenerator.ckp"
        )
        self.d_ckp = os.path.join(
            ckp_parent_path,
            "TransDiscriminator.ckp"
        )

    def train(self):
        optim_g = optim.Adam(
            self.generator.parameters(),
            lr=self.g_lr,
            weight_decay=1e-3,
            betas=(0, 0.99)
        )
        optim_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.d_lr,
            weight_decay=1e-3,
            betas=(0, 0.99)
        )
        for epoch in range(self.total_epochs):
            iter_object = self.train_data
            for i, (img, label) in enumerate(iter_object):
                img = img.to(self.device)
                label = label.to(self.device)

                B = img.size(0)
                z = torch.randn((B, self.latent_dim)).to(self.device)
                self.generator.eval()
                self.discriminator.train()
                optim_d.zero_grad()
                fake_img = self.generator(z)

                gp = self.gradient_penalty(img, fake_img, label)
                loss_d = -torch.mean(self.discriminator(img)) + torch.mean(self.discriminator(fake_img)) + gp
                gp.backward(retain_graph=True)
                loss_d.backward()
                optim_d.step()

                if i % self.delay == 0:
                    self.generator.train()
                    self.discriminator.eval()
                    optim_g.zero_grad()
                    fake_img = self.generator(z)
                    loss_g = -torch.mean(self.discriminator(fake_img))
                    loss_g.backward()
                    optim_g.step()
            self.save_model()

    def save_model(self):
        torch.save(self.generator.state_dict(), self.g_ckp)
        torch.save(self.discriminator.state_dict(), self.d_ckp)

    def gradient_penalty(self, img, fake_img, label):
        B = img.size(0)
        eps = torch.randn((B, 1, 1, 1)).to(self.device)  
        x_inter = (eps * img + (1 - eps) * fake_img).requires_grad_(True).to(self.device)  
        d_x_inter = self.discriminator(x_inter).to(self.device)
        grad_tensor = Variable(torch.Tensor(B, 1).fill_(1.0), requires_grad=False).to(self.device)
        grad = torch.autograd.grad(
            outputs=d_x_inter,
            inputs=x_inter,
            grad_outputs=grad_tensor,
            retain_graph=True,  
            create_graph=True,  
            only_inputs=True  
            
        )[0]  
        grad = grad.reshape(B, -1)
        gradient_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return 10 * gradient_penalty


def trainTransGAN():
    # seed = 1  # seed for random function
    worker = TrainTransGAN()
    worker.train()

if __name__ == '__main__':
    trainTransGAN()

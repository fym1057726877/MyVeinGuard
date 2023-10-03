import os

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ConvGAN.ConvGanModel import ConvGenerator, ConvDiscriminator, ConvGeneratorFingervein1, \
    ConvDiscriminatorFingervein1

# customize these function according to the path of your own data
import getOriginalDatasetData, getOriginalTensorData


class TrainConvGAN:
    def __init__(self):
        super(TrainConvGAN, self).__init__()
        # init model
        if dataset_name == 'Fingervein1' or 'Fingervein2':
            self.generator = ConvGeneratorFingervein1(latent_dim)
            self.discriminator = ConvDiscriminatorFingervein1()
        else:
            self.generator = ConvGenerator(latent_dim)
            self.discriminator = ConvDiscriminator()


        self.generator.to(device)
        self.discriminator.to(device)

        self.train_loader = DataLoader(
            getOriginalDatasetData(dataset_name=dataset_name, phase="Train"),
            batch_size=batchsize,
            shuffle=True
        )

        ckp_root_path = os.path.join("ConvGAN", "checkpoints")

        self.g_ckp_path = os.path.join(
            ckp_root_path,
            "ConvGenerator.ckp"
        )
        self.d_ckp_path = os.path.join(
            ckp_root_path,
            "ConvDiscriminator.ckp"
        )

    def train(self):
        print("start train")
        optim_g = optim.Adam(
            self.generator.parameters(),
            lr=g_lr, weight_decay=1e-3, betas=(0, 0.99)
        )
        optim_d = optim.Adam(
            self.discriminator.parameters(),
            lr=d_lr, weight_decay=1e-3, betas=(0, 0.99)
        )
        lr_scheduler_g = optim.lr_scheduler.StepLR(optim_g, step_size=10, gamma=0.9)
        lr_scheduler_d = optim.lr_scheduler.StepLR(optim_d, step_size=10, gamma=0.9)
        for epoch in range(total_epochs):
            train_data_set = enumerate(self.train_loader)
            for i, (img, label) in train_data_set:
                img = img.to(device)
                
                z = torch.randn((img.size(0), latent_dim)).to(device)
                
                self.generator.eval()
                
                fakeImg = self.generator(z)
                
                self.discriminator.train()
                optim_d.zero_grad()
                
                gp = self.gradient_penalty(img, fakeImg)
                loss_d = -torch.mean(self.discriminator(img)) + torch.mean(self.discriminator(fakeImg)) + gp
                gp.backward(retain_graph=True)
                loss_d.backward()
                optim_d.step()

                
                if i % delay == 0:
                    self.generator.train()
                    self.discriminator.eval()
                    optim_g.zero_grad()
                    fake_img = self.generator(z)
                    loss_g = -torch.mean(self.discriminator(fake_img))
                    loss_g.backward()
                    optim_g.step()
                print("[G loss: %f] [D loss: %f]" % (loss_g.item(), loss_d.item()))
            
            
            if optim_g.state_dict()['param_groups'][0]['lr'] > g_lr / 1e2:
                lr_scheduler_g.step()
                lr_scheduler_d.step()
            self.save_model()
        
    def gradient_penalty(self, img, fake_img):
        B = img.size(0)
        eps = torch.randn((B, 1, 1, 1)).to(device)  
        x_inter = (eps * img + (1 - eps) * fake_img).requires_grad_(True).to(device)  
        d_x_inter = self.discriminator(x_inter).to(device)
        grad_tensor = Variable(torch.Tensor(B, 1).fill_(1.0), requires_grad=False).to(device)
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

    def save_model(self):
        print("Save the model...")
        torch.save(self.generator.state_dict(), self.g_ckp_path)
        torch.save(self.discriminator.state_dict(), self.d_ckp_path)


# the name of dataset
# dataset_name = "Handvein"
# dataset_name = "Fingervein1"
dataset_name = "Fingervein2"

latent_dim = 256
g_lr = 1e-5
d_lr = 1e-5
total_epochs = 4000
delay = 15

batchsize = 64

device = "cuda"



def trainConvGANmain():

    train_Conv_GAN = TrainConvGAN()
    train_Conv_GAN.train()


if __name__ == "__main__":
    trainConvGANmain()

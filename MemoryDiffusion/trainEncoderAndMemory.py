import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from gaussiandiffusion import UNetModel
from respace import SpacedDiffusion
from encoderandmemory import EncoderAndMemory
from utils import get_project_path
from data.mydataset import trainloader, testloader
from utils import draw_ori_and_recon_images16


class EntropyLoss(nn.Module):
    def __init__(self, entropy_loss_coef=0.002):
        super(EntropyLoss, self).__init__()
        self.entropy_loss_coef = entropy_loss_coef

    def forward(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum()
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


class Trainer:
    def __init__(
            self,
            lr=5e-4,
            device="cuda",
    ):
        super(Trainer, self).__init__()
        self.model_name = "encoder_memory"
        self.device = device
        self.train_loader, self.test_loader = trainloader, testloader
        self.diffsuion = SpacedDiffusion(num_ddim_timesteps=100)
        self.unet = UNetModel(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 3, 4),
            num_res_blocks=2,
        ).to(device)
        self.unet_path = os.path.join(get_project_path(), "pretrained", "ddim_eps_64.pth")
        self.unet.load_state_dict(torch.load(self.unet_path))

        self.encoder_memory = EncoderAndMemory(
            feature_dims=4096,
            MEM_DIM=600,
        ).to(self.device)
        self.save_path = os.path.join(get_project_path(), "pretrained", f"{self.model_name}.pth")
        self.encoder_memory.load_state_dict((torch.load(self.save_path)))

        # loss function
        self.loss_fun1 = nn.MSELoss()
        self.loss_fun2 = EntropyLoss()
        self.lr = lr

        self.optimer = optim.AdamW(self.encoder_memory.parameters(), lr=self.lr, weight_decay=0.005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimer, step_size=10, gamma=0.99)

    def train(self, epochs):
        for e in range(epochs):
            train_num = 0
            epoch_loss = 0
            self.unet.eval()
            self.encoder_memory.train()
            batch_count = len(self.train_loader)
            for index, (img, label) in tqdm(enumerate(self.train_loader), desc=f"train {e+1}/{epochs}",
                                            total=batch_count):
                self.optimer.zero_grad()
                img, label = img.to(self.device), label.to(self.device)
                out = self.encoder_memory(img)
                z_hat, mem_weight = out["z_hat"], out["mem_weight"]
                noise_imgs, x_recon = self.diffsuion.restore_img(self.unet, z_hat, t=20)
                loss = self.loss_fun1(x_recon, img) + self.loss_fun2(mem_weight)

                loss.backward()
                self.optimer.step()

                train_num += label.size(0)
                epoch_loss += loss

            if self.optimer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                self.scheduler.step()

            epoch_loss /= batch_count
            print(f"[Epoch {e+1}/{epochs}   Loss:{epoch_loss:.6f}]")
            torch.save(self.encoder_memory.state_dict(), self.save_path)

    def eval(self):
        self.encoder_memory.eval()
        self.unet.eval()
        imgs, labels = next(iter(self.test_loader))
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        z_hat = self.encoder_memory(imgs)["z_hat"]
        recon_imgs = self.diffsuion.restore_img(self.unet, z_hat, t=20)[1]
        draw_ori_and_recon_images16(imgs, recon_imgs)


def main(device="cuda"):
    train_model = Trainer(lr=5e-4, device=device)
    train_model.train(100)
    # train_model.eval()


if __name__ == "__main__":
    main()

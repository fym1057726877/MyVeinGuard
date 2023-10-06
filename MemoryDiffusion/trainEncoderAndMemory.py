import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from gaussiandiffusion import GaussianDiffusion, UNetModel, ModelMeanType
from encoderandmemory import EncoderAndMemory
from utils import get_project_path
from data.mydataset import get_Vein600_128x128_Dataloader
from utils import draw_ori_and_recon_images32


class EntropyLoss(nn.Module):
    def __init__(self, entropy_loss_coef=0.002):
        super(EntropyLoss, self).__init__()
        self.entropy_loss_coef = entropy_loss_coef

    def forward(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum()
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


class TrainEM:
    def __init__(
            self,
            total_epochs=10,
            lr=5e-5,
            device="cuda",
            batchsize=30
    ):
        super(TrainEM, self).__init__()
        self.model_name = "encoder_memory"
        # customize these object according to the path of your own data
        self.device = device
        self.train_loader, self.test_loader = get_Vein600_128x128_Dataloader(batch_size=batchsize, shuffle=True)
        self.diffsuion = GaussianDiffusion(
            betas_schedule="linear",
            time_steps=1000,
            ddim_step=100,
            mean_type=ModelMeanType.START_X,
        )
        self.unet = UNetModel(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 2),
            attention_resolutions=[],
            num_res_blocks=2,
        ).to(self.device)
        self.unet.load_state_dict(torch.load(os.path.join(get_project_path(), "pretrained", "ddim.pth")))

        self.encoder_memory = EncoderAndMemory(
            latent_dims=256,
            MEM_DIM=200,
            addressing="sparse",
            image_size=128
        ).to(self.device)
        self.save_path = os.path.join(get_project_path(), "pretrained", f"{self.model_name}.pth")

        # loss function
        self.loss_fun1 = nn.MSELoss()
        self.loss_fun2 = EntropyLoss()
        self.total_epochs = total_epochs
        self.lr = lr
        self.batch_size = batchsize

        self.optimer = optim.AdamW(self.encoder_memory.parameters(), lr=self.lr, weight_decay=0.05)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimer, step_size=10, gamma=0.99)

    def train(self):
        for e in range(self.total_epochs):
            train_num = 0
            epoch_loss = 0
            self.unet.eval()
            self.encoder_memory.train()
            # start train
            batch_count = len(self.train_loader)
            for index, (img, label) in tqdm(enumerate(self.train_loader), desc=f"train {e+1}/{self.total_epochs}",
                                            total=batch_count):
                self.optimer.zero_grad()
                img, label = img.to(self.device), label.to(self.device)
                out = self.encoder_memory(img)
                x_hat = out["x_hat"]
                x_recon = self.diffsuion.ddim_sample_loop(self.unet, shape=x_hat.shape, noise=x_hat, progress=False)[0]
                mem_weight = out["mem_weight"]
                loss = self.loss_fun1(x_recon, img) + self.loss_fun2(mem_weight)

                loss.backward()
                self.optimer.step()

                train_num += label.size(0)
                epoch_loss += loss

            if self.optimer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                self.scheduler.step()

            epoch_loss /= batch_count
            print(
                f"[Epoch {e}/{self.total_epochs}   Loss:{epoch_loss:.6f}]"
            )
            torch.save(self.encoder_memory.state_dict(), self.save_path)

        self.eval()

    def eval(self):
        self.encoder_memory.eval()
        imgs, labels = next(iter(self.test_loader))
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        imgs = self.encoder_memory(imgs)["x_hat"]
        recon_imgs = self.diffsuion.ddim_sample_loop(self.unet, shape=imgs.shape, noise=imgs, progress=True)[0]
        draw_ori_and_recon_images32(imgs, recon_imgs)


def train(epochs=10, device="cuda"):
    train_model = TrainEM(
        total_epochs=epochs,
        lr=5e-5,
        device=device,
        batchsize=30
    )
    train_model.train()


if __name__ == "__main__":
    train(epochs=10, device="cuda")

import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from WaveletMemory.waveletmemory import WaveletsMemory
from utils import get_project_path
from data.mydataset import trainloader, testloader
from utils import draw_ori_and_recon_images16


class EntropyLoss(nn.Module):
    def __init__(self, entropy_loss_coef=0.0002):
        super(EntropyLoss, self).__init__()
        self.entropy_loss_coef = entropy_loss_coef

    def forward(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum()
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


class TrainMemory:
    def __init__(
            self,
            lr=5e-4,
            device="cuda",
            batchsize=30
    ):
        super(TrainMemory, self).__init__()
        self.model_name = "waveletsmemory"
        self.device = device
        self.train_loader, self.test_loader = trainloader, testloader
        self.memory = WaveletsMemory(
            feature_dims=2048,
            MEM_DIM=1000,
            image_size=64,
        ).to(self.device)
        self.save_path = os.path.join(get_project_path(), "pretrained", f"{self.model_name}.pth")
        self.memory.load_state_dict(torch.load(self.save_path))

        # loss function
        self.loss_fun1 = nn.MSELoss()
        self.loss_fun2 = EntropyLoss()
        self.lr = lr
        self.batch_size = batchsize

        self.optimer = optim.AdamW(self.memory.parameters(), lr=self.lr, weight_decay=0.005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimer, step_size=10, gamma=0.99)

    def train(self, epochs):
        for e in range(epochs):
            train_num = 0
            epoch_loss = 0
            self.memory.train()
            # start train
            batch_count = len(self.train_loader)
            for index, (img, label) in tqdm(enumerate(self.train_loader), desc=f"train {e+1}/{epochs}",
                                            total=batch_count):
                self.optimer.zero_grad()
                img, label = img.to(self.device), label.to(self.device)
                out = self.memory(img)
                x_recon = out["x_recon"]
                mem_weight = out["mem_weight"]
                loss = self.loss_fun1(x_recon, img) + self.loss_fun2(mem_weight)

                loss.backward()
                self.optimer.step()

                train_num += label.size(0)
                epoch_loss += loss

            if self.optimer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                self.scheduler.step()

            epoch_loss /= batch_count
            print(f"[Epoch {e+1}/{epochs}   Loss:{epoch_loss:.6f}]")
            torch.save(self.memory.state_dict(), self.save_path)

    def eval(self):
        self.memory.eval()
        imgs, labels = next(iter(self.test_loader))
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        recon_imgs = self.memory(imgs)["x_recon"]
        # torch.save(recon_imgs, os.path.join(get_project_path(), "recon_imgs", f"{i}.pth"))
        draw_ori_and_recon_images16(imgs, recon_imgs)


def main(device="cuda"):
    train_model = TrainMemory(
        lr=5e-4,
        device=device,
        batchsize=30
    )
    # train_model.train(100)
    train_model.eval()


if __name__ == "__main__":
    main()

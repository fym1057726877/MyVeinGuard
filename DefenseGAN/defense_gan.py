

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader


def accuracy(y, y1):
    return y.max(dim=1, keep_dim=True)[1].eq(y1.view_as(y)).sum()


class GdRecParams:
    device = "cuda"
    gd_rec_batchsize = 32
    L = 200
    R = 10
    latent_dim = 256
    gd_rec_lr = 0.015


class GradientDescentReconstruct:

    def __init__(self, model_name, dataset_name):
        super(GradientDescentReconstruct, self).__init__()
        print("---defenseGAN method for searching RecImg---")

        # The pre-trianed generator of generative adversarial network will be loaded in this funcion
        self.generator = getTrainedConvGenerator()

    def reconstruct(self):
        # The adversarial samples will be loaded in this funcion
        # So we should generate them in advance
        adv_dataset = getAdvTestDataset()
        total_num = len(adv_dataset)
        adv_dataloader = DataLoader(adv_dataset, batch_size=GdRecParams.gd_rec_batchsize, shuffle=False)

        # The pre-trained classifier will be loaded in this funcion
        classifier = getPreTrainedClsModel()

        normal_acc, adv_acc, rec_acc, num = 0, 0, 0, 0
        iter_object = adv_dataloader
        for batchIndex, (img, adv_img, label) in enumerate(iter_object):
            img = img.to(GdRecParams.device)
            adv_img = adv_img.to(GdRecParams.device)
            label = label.to(GdRecParams.device)

            with torch.no_grad():
                y = classifier(img)
                normal_acc += accuracy(y, label)

                adv_y = classifier(adv_img)
                adv_acc += accuracy(adv_y, label)

            rec_img = self.reconstructBatchImg(adv_img, GdRecParams.L, GdRecParams.R)

            with torch.no_grad():
                rec_y = classifier(rec_img)
                rec_acc += accuracy(rec_y, label)

            num += label.size(0)


        print("[Num: %d/%d] [NormalAcc:   %f] [AdvAcc:   %f] [RecAcc:   %f]" % (
            num, total_num,
            torch.true_divide(normal_acc, num).item(),
            torch.true_divide(adv_acc, num).item(),
            torch.true_divide(rec_acc, num).item()))

    def reconstructBatchImg(self, img, L, R):
        B = img.size(0)
        img = img.repeat(R, 1, 1, 1).to(GdRecParams.device)
        z = torch.randn((B * R, GdRecParams.latent_dim)).to(GdRecParams.device)
        z.requires_grad = True
        optim_g = optim.RMSprop([z], lr=GdRecParams.gd_rec_lr)
        lr_scheduler_g = optim.lr_scheduler.StepLR(optim_g, step_size=5, gamma=0.99)
        loss_fun1 = nn.MSELoss()

        for iterIndex in range(L):
            fake_img = self.generator(z)
            optim_g.zero_grad()
            loss_g = loss_fun1(fake_img, img)
            reward = -F.mse_loss(fake_img, img, reduction="none").view(img.size(0), -1).sum(1).mean()
            print(
                "[L:  {}/{}] [loss_g:  {:f}] Reward:  {:f}".format(
                    iterIndex, L, loss_g.item(), reward.item()
                )
            )
            loss_g.backward()
            optim_g.step()
            if optim_g.state_dict()['param_groups'][0]['lr'] * 1000 > 1:
                lr_scheduler_g.step()

        fake_img = self.generator(z)
        distance = torch.mean(((fake_img - img) ** 2).view(B * R, -1), dim=1)

        rec_img = None

        distance = distance.view(R, B).transpose(1, 0)
        for imgIndex in range(B):
            j = torch.argmin(distance[imgIndex], dim=0)
            index = j * B + imgIndex
            current_rec_img = fake_img[index].unsqueeze(0)
            if rec_img is None:
                rec_img = current_rec_img
            else:
                rec_img = torch.cat((rec_img, current_rec_img), dim=0)
        return rec_img
    

attack_type = "FGSM"
eps = 0.01

gd = GradientDescentReconstruct()
gd.reconstruct(attack_type, eps)

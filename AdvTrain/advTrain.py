import os

from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from Classifier.trainClassifier import getDefinedClsModel

# customize these Object according to your own data
import HandVeinDataset, AdvTestDataset, HandVeinDataset2, HandVeinDataset3, getOriginalDatasetData, \
    HsjaHandVeinDataset

from torch.utils.data import DataLoader
import torch
from torch import optim
import torch.nn as nn
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
import numpy as np



# attack_type = "RandFGSM"
attack_type = "FGSM"
# attack_type = "PGD"
# attack_type = "HSJA"

# eps = 0.01
eps = 0.015
# eps = 0.02
# eps = 0.03

# dataset_name = "Handvein"
# dataset_name = "Handvein3"
dataset_name = "Fingervein2"

model_name = "Resnet18"
# model_name = "GoogleNet"
# model_name = "ModelB"
# model_name = "MSMDGANetCnn_wo_MaxPool"
# model_name = "Tifs2019Cnn_wo_MaxPool"
# model_name = "FVRASNet_wo_Maxpooling"
# model_name = "LightweightDeepConvNN"
genAdvExampleModel_name = model_name

device = "cuda"

batchsize = 48
lr = 5e-5
total_epochs = 1



def accuracy(y, y1):
    return y.max(dim=1, keep_dim=True)[1].eq(y1.view_as(y)).sum()


class TrainClsByAdvTraining:
    def __init__(self):
        super(TrainClsByAdvTraining, self).__init__()
        if attack_type == 'HSJA':
            self.train_loader = DataLoader(
                HsjaHandVeinDataset(
                    dataset_name=dataset_name,
                    model_name=model_name
                ),
                batch_size=batchsize,
                shuffle=True
            )
        else:
            self.train_loader = DataLoader(
                getOriginalDatasetData(
                    dataset_name=dataset_name,
                    phase='Train'
                ),
                batch_size=batchsize,
                shuffle=True
            )
        self.val_loader = DataLoader(
            getOriginalDatasetData(
                dataset_name=dataset_name,
                phase='Val'
            ),
            batch_size=batchsize,
            shuffle=True
        )
        self.test_loader = DataLoader(
            getOriginalDatasetData(
                dataset_name=dataset_name,
                phase='Test'
            ),
            batch_size=batchsize,
            shuffle=True
        )
        self.ckp_path = os.path.join(
            "AdvTrain", "advTrain.ckp"
        )
        self.loss_fun = nn.CrossEntropyLoss()
        self.classifier = getDefinedClsModel(
            dataset_name=dataset_name,
            model_name=model_name,
            device=device
        )
        self.classifier.load_state_dict(torch.load("分类器模型路径"))
        self.optimer = optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=0.05)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimer, step_size=10, gamma=0.99)

    def mainProcess(self):
        for e in range(self.total_epochs):
            currentEpoch_train_loss, currentEpoch_train_acc = self.train()

            if self.attack_type == 'HSJA':
                val_acc = self.evalModel(data_loader=self.val_loader, adv=False, attack=False)
            else:
                val_acc = self.evalModel(data_loader=self.val_loader, adv=False, attack=True)

            print(
                "[Epoch %d/%d] [Train_loss:   %f]\n [Train_acc:   %f] [Val_acc:   %f]" % (
                    e, self.total_epochs,
                    currentEpoch_train_loss,
                    currentEpoch_train_acc,
                    val_acc
                )
            )
            torch.save(self.classifier.state_dict(), self.ckp_path)
        self.test()

    def train(self):
        correct_num = 0  
        epoch_num = 0  
        epoch_loss = 0  
        self.classifier.train()
        for index, (img, label) in enumerate(self.train_loader):
            img, label = img.to(device), label.to(device)
            ori_pred = self.classifier(img)
            ori_cls_loss = self.loss_fun(ori_pred, label.long())
            if self.attack_type != 'HSJA':                
                adv_img = self.attack(img, label)
                adv_pred = self.classifier(adv_img)
                adv_cls_loss = self.loss_fun(adv_pred, label.long())
                cls_loss = 0.5*ori_cls_loss + 0.5*adv_cls_loss
                correct_num = correct_num + accuracy(ori_pred, label) + accuracy(adv_pred, label)
                epoch_num += 2*label.size(0)
            else:
                cls_loss = ori_cls_loss
                correct_num += accuracy(ori_pred, label)
                epoch_num += label.size(0)
            epoch_loss += cls_loss
            self.optimer.zero_grad()
            cls_loss.backward()
            self.optimer.step()
        train_acc = torch.true_divide(correct_num, epoch_num)
        if self.optimer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
            self.scheduler.step()
        return epoch_loss.item(), train_acc.item()

    
    def evalModel(self, data_loader, adv=False, attack=False):
        correct_num = 0  
        epoch_num = 0  
        self.classifier.eval()
        for index, batch_data in enumerate(data_loader):
            if adv:
                img = batch_data[1].to(device)
                label = batch_data[2].to(device)
            else:
                img = batch_data[0].to(device)
                label = batch_data[1].to(device)
            if attack:
                img = self.attack(img, label)
            pred = self.classifier(img)
            correct_num += accuracy(pred, label)
            epoch_num += label.size(0)
        train_acc = torch.true_divide(correct_num, epoch_num)
        return train_acc.item()

    def test(self):
        test_acc = self.evalModel(self.test_loader, adv=False, attack=False)
        print("before...")
        print("[Test_acc:   %.5f]\n" % test_acc)
        print("after...")
        test_acc = self.evalModel(self.test_loader, adv=False, attack=True)
        print("[Test_acc:   %.5f]\n" % test_acc)

    def attack(self, x, label):
        if self.attack_type == "RandFGSM":
            alpha = 0.005
            x = torch.clip(x + alpha * torch.sign(torch.randn(x.shape).to(device)), 0, 1)
            eps2 = eps - alpha
            x_adv = fast_gradient_method(self.classifier, x, eps2, np.inf)
        elif self.attack_type == "FGSM":
            x_adv = fast_gradient_method(self.classifier, x, eps, np.inf)
        elif self.attack_type == "PGD":
            x_adv = projected_gradient_descent(
                self.classifier, x,
                eps=eps, eps_iter=1 / 255, nb_iter=min(255 * eps + 4, 1.25 * (eps * 255)), norm=np.inf)
        elif self.attack_type == "HSJA":
            x_adv = hop_skip_jump_attack(
                self.classifier, x, norm=2,
                initial_num_evals=1, max_num_evals=50,
                num_iterations=30, batch_size=batchsize, verbose=False
            )
        else:
            raise RuntimeError("The attack type {} is invalid!".format(self.attack_type))
        return x_adv


def advTraining():
    clsWithAdv = TrainClsByAdvTraining()
    clsWithAdv.mainProcess()


if __name__ == "__main__":
    advTraining()

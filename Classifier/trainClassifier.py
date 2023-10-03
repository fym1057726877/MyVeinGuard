import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from Classifier.model import FVRASNet_wo_Maxpooling, FineTuneClassifier, LightweightDeepConvNN, MSMDGANetCnn_wo_MaxPool, TargetedModelB, Tifs2019CnnWithoutMaxPool


def getDefinedClsModel(dataset_name, model_name, device) -> nn.Module:
    if dataset_name == "Handvein":
        out_channel = 500
    elif dataset_name == "Handvein3":
        out_channel = 600
    elif dataset_name == "Fingervein2":
        out_channel = 600
    else:
        raise RuntimeError(f"dataset_name:{dataset_name} is not valid!")

    # define the model
    if model_name == "FV-CNN":   # FV-CNN
        if dataset_name == "Fingervein2":
            model = Tifs2019CnnWithoutMaxPool(out_channel, fingervein1=True)
        else:
            model = Tifs2019CnnWithoutMaxPool(out_channel)
    elif model_name == "ModelB":
        if dataset_name == "Fingervein1" or "Fingervein2":
            model = TargetedModelB(out_channel, fingervein1=True)
        else:
            model = TargetedModelB(out_channel)
    elif model_name == "PV-CNN":
        if dataset_name == "Fingervein1" or "Fingervein2":
            model = MSMDGANetCnn_wo_MaxPool(out_channel, fingervein1=True)
        else:
            model = MSMDGANetCnn_wo_MaxPool(out_channel)
    elif model_name == "FVRASNet_wo_Maxpooling":   # FVRAS-Net
        if dataset_name == "Fingervein1" or "Fingervein2":
            model = FVRASNet_wo_Maxpooling(out_channel, fingervein1=True)
        else:
            model = FVRASNet_wo_Maxpooling(out_channel)
    elif model_name == "LightweightDeepConvNN":   # Lightweight_CNN
        if dataset_name == "Fingervein1" or "Fingervein2":
            model = LightweightDeepConvNN(out_channel, fingervein1=True)
        else:
            model = LightweightDeepConvNN(out_channel)
    else:
        model = FineTuneClassifier(model_name, out_channel)
    model.to(device)
    return model



# lr = 1e-5
lr = 5e-5
# lr = 1e-4
batchsize = 48
total_epochs = 1000
device = "cuda"


# dataset_name = "Handvein"
# dataset_name = "Handvein3"
dataset_name = "Fingervein2"

model_name = "Resnet18"
# model_name = "GoogleNet"
# model_name = "ModelB"
# model_name = "MSMDGANetCnn_wo_MaxPool"   # PV-CNN
# model_name = "Tifs2019Cnn_wo_MaxPool"   # FV-CNN
# model_name = "FVRASNet_wo_Maxpooling"
# model_name = "LightweightDeepConvNN"



class TrainClassifier:
    def __init__(
            self,
            model,
    ):
        super(TrainClassifier, self).__init__()
        # customize these object according to the path of your own data
        self.train_loader = DataLoader()
        self.val_loader = DataLoader()
        self.test_loader = DataLoader()
        self.classifier = model
        self.ckp_path = os.path.join(
            "Classifier", "ckp"
        )
        # loss function
        self.loss_fun = nn.CrossEntropyLoss()
        self.classifier.to(device)

    def train(self):
        optimer = optim.AdamW(self.classifier.parameters(), lr=self.lr, weight_decay=0.05)
        scheduler = optim.lr_scheduler.StepLR(optimer, step_size=10, gamma=0.99)
        for e in range(self.total_epochs):
            correct_num = 0
            train_num = 0
            self.classifier.train()
            iter_object = self.train_loader
            # start train
            for index, (img, label) in enumerate(iter_object):
                img, label = img.to(self.device), label.to(self.device)
                pred = self.classifier(img)
                cls_loss = self.loss_fun(pred, label.long())
                optimer.zero_grad()
                cls_loss.backward()
                optimer.step()
                correct_num += pred.max(dim=1, keep_dim=True)[1].eq(label.view_as(pred)).sum()
                train_num += label.size(0)
            if optimer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()
            train_acc = torch.true_divide(correct_num, train_num)
            val_acc = self.eval(eval_type="val")
            print(
                f"[Epoch {e}/{self.total_epochs}] \n"
                f"[Train_acc:   {train_acc.item()}]"
                f"[Val_acc:   {val_acc.item()}]"
            )
            torch.save(self.classifier.state_dict(), self.ckp_path)
        test_acc = self.eval(eval_type="test")
        print(f"[Test_acc:   {test_acc.item()}\n")

    def eval(self, eval_type="val"):
        correct_num, eval_num = 0, 0
        self.classifier.eval()
        if eval_type == "test":
            dataloader = self.test_loader
        elif eval_type == "val":
            dataloader = self.val_loader
        else:
            raise RuntimeError("eval_type: {} is not valid".format(eval_type))
        iter_object = dataloader
        for index, (x, label) in enumerate(iter_object):
            x, label = x.to(self.device), label.to(self.device)
            pred = self.classifier(x)
            correct_num += pred.max(dim=1, keep_dim=True)[1].eq(label.view_as(pred)).sum()
            eval_num += label.size(0)
        return torch.true_divide(correct_num, eval_num)


def trainCls():
    # seed = 1  # the seed for random function
    model = getDefinedClsModel(
        dataset_name=dataset_name,
        model_name=model_name,
        device=device
    )
    train_Classifier = TrainClassifier(model)
    train_Classifier.train()


if __name__ == "__main__":
    trainCls()

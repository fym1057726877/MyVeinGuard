import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils import get_project_path
from data.mydataset import trainloader, testloader
from Classifier.model import (FVRASNet_wo_Maxpooling, FineTuneClassifier, LightweightDeepConvNN,
                              MSMDGANetCnn_wo_MaxPool, TargetedModelB, Tifs2019CnnWithoutMaxPool)


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
    if model_name == "FV-CNN":  # FV-CNN
        if dataset_name == "Fingervein2":
            model = Tifs2019CnnWithoutMaxPool(out_channel, fingervein1=True)
        else:
            model = Tifs2019CnnWithoutMaxPool(out_channel)
    elif model_name == "ModelB":
        model = TargetedModelB(out_channel)
    elif model_name == "PV-CNN":
        if dataset_name == "Fingervein1" or "Fingervein2":
            model = MSMDGANetCnn_wo_MaxPool(out_channel, fingervein1=True)
        else:
            model = MSMDGANetCnn_wo_MaxPool(out_channel)
    elif model_name == "FVRASNet_wo_Maxpooling":  # FVRAS-Net
        if dataset_name == "Fingervein1" or "Fingervein2":
            model = FVRASNet_wo_Maxpooling(out_channel, fingervein1=True)
        else:
            model = FVRASNet_wo_Maxpooling(out_channel)
    elif model_name == "LightweightDeepConvNN":  # Lightweight_CNN
        if dataset_name == "Fingervein1" or "Fingervein2":
            model = LightweightDeepConvNN(out_channel, fingervein1=True)
        else:
            model = LightweightDeepConvNN(out_channel)
    else:
        model = FineTuneClassifier(model_name, out_channel)
    model.to(device)
    return model


# lr = 1e-5
# lr = 5e-5
# lr = 1e-4
# batchsize = 48
# total_epochs = 1000
# device = "cuda"


# dataset_name = "Handvein"
# dataset_name = "Handvein3"
# dataset_name = "Fingervein2"

# model_name = "Resnet18"
# model_name = "GoogleNet"
# model_name = "ModelB"
# model_name = "MSMDGANetCnn_wo_MaxPool"   # PV-CNN
# model_name = "Tifs2019Cnn_wo_MaxPool"   # FV-CNN
# model_name = "FVRASNet_wo_Maxpooling"
# model_name = "LightweightDeepConvNN"


class TrainClassifier:
    def __init__(
            self,
            total_epochs=100,
            lr=5e-5,
            device="cuda",
            dataset_name="Fingervein2",
            model_name="ModelB",
            batchsize=30
    ):
        super(TrainClassifier, self).__init__()
        # customize these object according to the path of your own data
        self.device = device
        self.train_loader, self.test_loader = trainloader, testloader
        self.classifier = getDefinedClsModel(dataset_name, model_name, device)
        self.save_path = os.path.join(get_project_path(), "pretrained", f"{model_name}.pth")
        self.classifier.load_state_dict(torch.load(self.save_path))

        # loss function
        self.loss_fun = nn.CrossEntropyLoss()
        self.total_epochs = total_epochs
        self.lr = lr
        self.batch_size = batchsize

        self.optimer = optim.AdamW(self.classifier.parameters(), lr=self.lr, weight_decay=0.05)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimer, step_size=10, gamma=0.99)

    def train(self):
        for e in range(self.total_epochs):
            correct_num = 0
            train_num = 0
            epoch_loss = 0
            self.classifier.train()
            # start train
            batch_count = len(self.train_loader)
            for index, (img, label) in tqdm(enumerate(self.train_loader), desc=f"train {e}/{self.total_epochs}",
                                            total=batch_count):
                self.optimer.zero_grad()
                img, label = img.to(self.device), label.to(self.device)
                pred = self.classifier(img)
                cls_loss = self.loss_fun(pred, label.long())

                cls_loss.backward()
                self.optimer.step()

                correct_num += (pred.max(dim=1)[1] == label).sum()
                train_num += label.size(0)
                epoch_loss += cls_loss

            if self.optimer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                self.scheduler.step()

            train_acc = torch.true_divide(correct_num, train_num)
            test_acc = self.eval()
            epoch_loss /= batch_count
            print(
                f"[Epoch {e}/{self.total_epochs}   Loss:{epoch_loss:.6f}   "
                f"Train_acc: {train_acc.item():.6f}   Test_acc: {test_acc.item():.6f}]\n"
            )
            torch.save(self.classifier.state_dict(), self.save_path)

    def eval(self):
        correct_num, eval_num = 0, 0
        self.classifier.eval()
        for index, (x, label) in tqdm(enumerate(self.test_loader), desc="test step", total=len(self.test_loader)):
            x, label = x.to(self.device), label.to(self.device)
            pred = self.classifier(x)
            correct_num += (pred.max(dim=1)[1] == label).sum()
            eval_num += label.size(0)
        return torch.true_divide(correct_num, eval_num)


def trainCls(dataset_name, model_name, device, epochs):
    # seed = 1  # the seed for random function
    train_Classifier = TrainClassifier(
        dataset_name=dataset_name,
        model_name=model_name,
        device=device,
        total_epochs=epochs
    )
    # train_Classifier.train()
    print(train_Classifier.eval())


if __name__ == "__main__":
    trainCls(dataset_name="Handvein3", model_name="ModelB", device="cuda", epochs=20)

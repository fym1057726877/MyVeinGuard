import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from Classifier.trainClassifier import getDefinedClsModel
from MemoryDiffusion.encoderandmemory import EncoderAndMemory
from MemoryDiffusion.gaussiandiffusion import UNetModel, GaussianDiffusion, ModelMeanType
from utils import get_project_path
from Classifier.attackClassifier import generateAdvImage
from data.mydataset import testloader, normalize
from utils import draw_img_groups


# dataset_name = "Handvein"
# dataset_name = "Handvein3"
dataset_name = "Fingervein2"


class DirectTrainActorReconstruct:
    def __init__(
            self,
            classifier_name="ModelB",
            attack_type="FGSM",   # "RandFGSM", "FGSM", "PGD", "HSJA"
            eps=0.03,
    ):
        super(DirectTrainActorReconstruct, self).__init__()
        self.device = "cuda"
        self.batch_size = 20
        self.attack_dataloder = testloader
        self.attack_type = attack_type
        self.eps = eps

        # encoder
        self.actor = EncoderAndMemory(feature_dims=4096, MEM_DIM=600)
        self.actor.load_state_dict(torch.load(os.path.join(get_project_path(), "pretrained", "encoder_memory.pth")))
        self.actor.to(self.device)

        # diffusion
        self.diffsuion = GaussianDiffusion(mean_type=ModelMeanType.START_X)
        self.unet = UNetModel(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 3, 4),
            num_res_blocks=2,
        ).to(self.device)
        self.unet_path = os.path.join(get_project_path(), "pretrained", "ddim_x0_64.pth")
        self.unet.load_state_dict(torch.load(self.unet_path))

        # classifier
        self.classifier_name = classifier_name
        self.classifier = getDefinedClsModel(
            dataset_name=dataset_name,
            model_name=self.classifier_name,
            device=self.device
        )
        self.classifier_path = os.path.join(get_project_path(), "pretrained", f"{self.classifier_name}.pth")
        self.classifier.load_state_dict(torch.load(self.classifier_path))

        # adversial
        self.adv_path = os.path.join(
            get_project_path(),
            "data",
            "adv_imgs",
            f"600_{self.classifier_name}_{self.attack_type}_{self.eps}.pth"
        )

    def defense(self, img):
        z_tmp = self.actor(img)["z_hat"]
        rec_img = self.diffsuion.restore_img_0(model=self.unet, x_start=z_tmp, t=50)[1]
        return rec_img

    def test(self, progress=False):
        advDataloader = self.getAdvDataLoader(
            progress=True,
            shuffle=True
        )

        def accuracy(y, y1):
            return (y.max(dim=1)[1] == y1).sum()

        self.classifier.eval()
        self.actor.eval()
        self.unet.eval()

        normal_acc, rec_acc, adv_acc, rec_adv_acc, diff_adv_acc, num = 0, 0, 0, 0, 0, 0
        total_num = len(advDataloader)

        iterObject = enumerate(advDataloader)
        if progress:
            iterObject = tqdm(iterObject, total=total_num)

        for i, (img, adv_img, label) in iterObject:
            img, adv_img, label = img.to(self.device), adv_img.to(self.device), label.to(self.device)
            # rec_img = self.defense(img)
            # rec_adv_img = self.defense(adv_img)
            diff_adv = self.diffsuion.restore_img_0(model=self.unet, x_start=adv_img, t=50)[1]
            # draw_img_groups(img_groups=[img, adv_img, diff_adv], imgs_every_row=8)
            # return

            y = self.classifier(img)
            normal_acc += accuracy(y, label)

            # rec_y = self.classifier(rec_img)
            # rec_acc += accuracy(rec_y, label)

            adv_y = self.classifier(adv_img)
            adv_acc += accuracy(adv_y, label)

            # rec_adv_y = self.classifier(rec_adv_img)
            # rec_adv_acc += accuracy(rec_adv_y, label)

            diff_adv_y = self.classifier(diff_adv)
            diff_adv_acc += accuracy(diff_adv_y, label)

            num += label.size(0)

        print(f"-------------------------------------------------\n"
              f"test result:\n"
              f"NorAcc:{torch.true_divide(normal_acc, num).item():.6f}\n"
              # f"RecAcc:{torch.true_divide(rec_acc, num).item():.6f}\n"
              f"AdvAcc:{torch.true_divide(adv_acc, num).item():.6f}\n"
              # f"RAvAcc:{torch.true_divide(rec_adv_acc, num).item():.6f}\n"
              f"DAvAcc:{torch.true_divide(diff_adv_acc, num).item():.6f}\n"
              f"-------------------------------------------------")

    def getAdvDataLoader(
            self,
            progress=False,
            shuffle=False
    ):
        if os.path.exists(self.adv_path):
            data_dict = torch.load(self.adv_path)
        else:
            data_dict = generateAdvImage(
                classifier=self.classifier,
                attack_dataloder=self.attack_dataloder,
                attack_type=self.attack_type,
                eps=self.eps,
                progress=progress,
                savepath=self.adv_path
            )
        normal_data = data_dict["normal"]
        adv_data = data_dict["adv"]
        label = data_dict["label"]
        dataloder = DataLoader(TensorDataset(normal_data, adv_data, label), batch_size=self.batch_size, shuffle=shuffle)
        return dataloder


def testDirectActor():
    # seed = 1  # the seed for random function

    # attack_type = 'FGSM'
    # attack_type = 'PGD'
    attack_type = 'RandFGSM'
    # attack_type = 'HSJA'
    if attack_type == 'RandFGSM':
        eps = 0.015
    elif attack_type == "PGD":
        # eps = 0.01
        eps = 0.015
    else:
        # eps = 0.02
        eps = 0.03

    # classifier_name = "Resnet18"
    # classifier_name = "GoogleNet"
    # classifier_name = "ModelB"
    # classifier_name = "MSMDGANetCnn_wo_MaxPool"
    # classifier_name = "Tifs2019Cnn_wo_MaxPool"
    # classifier_name = "FVRASNet_wo_Maxpooling"
    # classifier_name = "LightweightDeepConvNN"

    directActorRec = DirectTrainActorReconstruct(classifier_name="ModelB", attack_type=attack_type, eps=eps)

    directActorRec.test(
        progress=True,
    )


if __name__ == "__main__":
    testDirectActor()

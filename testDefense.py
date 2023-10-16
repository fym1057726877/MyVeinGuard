import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from Classifier.trainClassifier import getDefinedClsModel
from MemoryDiffusion.encoderandmemory import EncoderAndMemory
from MemoryDiffusion.respace import SpacedDiffusion
from MemoryDiffusion.gaussiandiffusion import UNetModel
from utils import get_project_path
from Classifier.attackClassifier import generateAdvImage
from data.mydataset import testloader

# dataset_name = "Handvein"
# dataset_name = "Handvein3"
dataset_name = "Fingervein2"


def getAdvDataLoader(
        classifier=None,
        attack_dataloder=None,
        attack_type="fgsm",
        eps=0.03,
        progress=False,
        adv_path=None,
        batch_size=20,
        shuffle=False
):
    if adv_path is not None:
        data_dict = torch.load(adv_path)
    else:
        assert classifier is not None and attack_dataloder is not None
        data_dict = generateAdvImage(
            classifier=classifier,
            attack_dataloder=attack_dataloder,
            attack_type=attack_type,
            progress=progress,
            eps=eps,
        )
    normal_data = data_dict["normal"]
    adv_data = data_dict["adv"]
    label = data_dict["label"]
    dataloder = DataLoader(TensorDataset(normal_data, adv_data, label), batch_size=batch_size, shuffle=shuffle)
    return dataloder


class DirectTrainActorReconstruct:
    def __init__(self):
        super(DirectTrainActorReconstruct, self).__init__()
        self.device = "cuda"
        self.batch_size = 20
        self.testloader = testloader

        self.model_name = "encoder_memory"
        self.actor = EncoderAndMemory(feature_dims=4096, MEM_DIM=600)
        self.actor.load_state_dict(torch.load(os.path.join(get_project_path(), "pretrained", f"{self.model_name}.pth")))
        self.actor.to(self.device)

        self.generator = SpacedDiffusion(num_ddim_timesteps=100)
        self.unet = UNetModel(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 3, 4),
            num_res_blocks=2,
        ).to(self.device)
        self.unet_path = os.path.join(get_project_path(), "pretrained", "ddim_eps_64.pth")
        self.unet.load_state_dict(torch.load(self.unet_path))

        self.adv_path = os.path.join(get_project_path(), "data", "adv_imgs", "600_classes.pth")

    @torch.no_grad()
    def defense(self, img):
        z_tmp = self.actor(img)
        rec_img = self.generator.ddim_sample_loop(model=self.unet, shape=z_tmp.shape, noise=z_tmp, progress=True)
        return rec_img

    @torch.no_grad()
    def test(self, attack_type, eps, model_name, progress=False):
        classifier = getDefinedClsModel(
            dataset_name=dataset_name,
            model_name=model_name,
            device=self.device
        )
        classifier.load_state_dict(torch.load("path of classifier"))
        advDataloader = getAdvDataLoader(
            classifier=classifier,
            attack_dataloder=self.testloader,
            attack_type=attack_type,
            eps=eps,
            adv_path=self.adv_path,
            batch_size=20,
            shuffle=True
        )

        def accuracy(y, y1):
            return (y.max(dim=1)[1] == y1).sum()

        normal_acc, adv_acc, rec_acc, num = 0, 0, 0, 0
        total_num = len(advDataloader)

        iterObject = enumerate(advDataloader)
        if progress:
            iterObject = tqdm(iterObject, total=total_num)

        for i, (img, adv_img, label) in iterObject:
            img, adv_img, label = img.to(self.device), adv_img.to(self.device), label.to(self.device)
            y = classifier(img)
            normal_acc += accuracy(y, label)

            adv_y = classifier(adv_img)
            adv_acc += accuracy(adv_y, label)

            rec_img = self.defense(adv_img)
            rec_y = classifier(rec_img)
            rec_acc += accuracy(rec_y, label)
            num += label.size(0)

        print(f"-------------------------------------------------"
              f"test result:\n"
              f"NorAcc:{torch.true_divide(normal_acc, num).item():.6f}\n"
              f"AdvAcc:{torch.true_divide(adv_acc, num).item():.6f}\n"
              f"RecAcc:{torch.true_divide(rec_acc, num).item():.6f}\n"
              f"-------------------------------------------------")


def testDirectActor():
    # seed = 1  # the seed for random function

    attack_type = 'FGSM'
    # attack_type = 'PGD'
    # attack_type = 'RandFGSM'
    # attack_type = 'HSJA'
    if attack_type == 'RandFGSM':
        eps = 0.015
    elif attack_type == "PGD":
        # eps = 0.01
        eps = 0.015
    else:
        # eps = 0.02
        eps = 0.01

    # model_name = "Resnet18"
    # model_name = "GoogleNet"
    model_name = "ModelB"
    # model_name = "MSMDGANetCnn_wo_MaxPool"
    # model_name = "Tifs2019Cnn_wo_MaxPool"
    # model_name = "FVRASNet_wo_Maxpooling"
    # model_name = "LightweightDeepConvNN"

    directActorRec = DirectTrainActorReconstruct()

    directActorRec.test(
        attack_type=attack_type,
        eps=eps,
        model_name=model_name,
        progress=True,
    )


if __name__ == "__main__":
    testDirectActor()

import os
import time
import torch
from torch import nn
from tqdm import tqdm

from attacks.fsgm import GradientSignAttack
from attacks.pgd import PGDAttack
from classifier.resnet import resnet50
from dataset.fashion_mnist import get_fashion_mnist_dataloader
from utils import get_project_path

root_dir = get_project_path()

device = "cuda" if torch.cuda.is_available() else "cpu"

epsilons = [0.03, 0.09]
epsilon = epsilons[0]

MEM_DIM = 500
num_classes = 10
batch_size = 64


target_classifier = resnet50(num_classes=10, pretrained=True)
target_classifier.to(device)
target_classifier.eval()

attack_model = target_classifier
attack_model_str = "resnet50"

# Attack dictionary: {FGSM, BIM, PGD, CW}
attack_dict = {
    "fgsm": GradientSignAttack(
        attack_model,
        loss_fn=None,
        eps=epsilon,
        clip_min=0.0,
        clip_max=1.0,
        targeted=False
    ),

    "pgd": PGDAttack(
        attack_model,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        eps=epsilon,
        num_iter=40,
        iter_eps=0.01,
        rand_init=True,
        clip_min=0.0,
        clip_max=1.0,
        targeted=False
    ),
}

key = "fgsm"
attack = attack_dict[key]

print(f"==============================================\n"
      f"Attack  : {key}\n"
      f"Epsilon : {epsilon}\n"
      f"==============================================")

train_loader, test_loader = get_fashion_mnist_dataloader(batch_size=batch_size, shuffle=True)


# =============================================================================
# Preparing Data
# =============================================================================
def main(epsilon, attack, opt="train", contain_benign=False, save_failed=True):
    adv_imgs, adv_labs = [], []
    assert opt == "train" or opt == "test"
    if opt == "train":
        dataloader = train_loader
    else:
        dataloader = test_loader
    adv_dir = str(f"{root_dir}/dataset/attacked/fmnist/{opt}/{key}_eps_{epsilon}_{attack_model_str}")
    create_adv_dir(adv_dir, dataset_dirname="dataset")
    filename_adv_imgs = adv_dir + "/" + "images.pth"
    filename_adv_labs = adv_dir + "/" + "labels.pth"
    bar = tqdm(total=len(dataloader), desc="attacking", position=0)
    for batch_index, (imgs, labs) in enumerate(dataloader):
        imgs, labs = imgs.to(device), labs.to(device)
        if contain_benign:
            # ============= save benign imgs =============
            for img, lab in zip(imgs, labs):
                adv_imgs.append(img.to("cpu"))
                adv_labs.append(lab.to("cpu"))

        # =========== Adversarial ===========
        perturbed_image = attack.perturb(imgs, labs)
        # ===================================

        predict_label = target_classifier(perturbed_image).max(dim=1)[1]

        for img, lab, p in zip(perturbed_image, labs, predict_label):
            if save_failed:
                adv_imgs.append(img.to("cpu"))
                adv_labs.append(lab.to("cpu"))
            else:
                if p != lab:
                    adv_imgs.append(img.to("cpu"))
                    adv_labs.append(lab.to("cpu"))

        bar.update(1)

    torch.save(adv_imgs, filename_adv_imgs)
    torch.save(adv_labs, filename_adv_labs)
    print('==============================================')
    print(f'Data Prepared, Total examples - {len(adv_imgs)}')
    print('==============================================')


def create_adv_dir(target_path, dataset_dirname="dataset"):
    dataset_path = get_project_path() + "/" + dataset_dirname
    assert len(target_path) > len(dataset_path), "no dir can be created"
    if target_path[0:len(dataset_path)] != dataset_path:
        raise NotImplementedError("target_path must in the project")
    dir_list = target_path[len(dataset_path)+1:].split("/")
    create_path = dataset_path
    for dirname in dir_list:
        create_path = create_path + "/" + dirname
        if not os.path.exists(create_path):
            os.mkdir(create_path)


if __name__ == "__main__":
    start = time.time()
    print('Starting data generation...')

    time.sleep(1)

    main(epsilon, attack, opt="test", contain_benign=False, save_failed=True)

    end = time.time()
    total_time = int(end - start)
    minutes = total_time // 60
    seconds = total_time % 60
    print(f"Execution time: {minutes}min{seconds}s")
    print('==============================================')

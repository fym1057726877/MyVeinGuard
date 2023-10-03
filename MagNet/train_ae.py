# customize these object according to the path of your own data
import HandVeinDataset, HandVeinDataset3, getOriginalDatasetData
from torch.utils.data import DataLoader
from model import DenoisingAutoEncoder_1
from model import DenoisingAutoEncoder_2
import os


# 加载训练数据
# dataset_name = "Handvein"
# dataset_name = "Handvein3"
dataset_name = "Fingervein1"

batch_size = 16

lr = 1e-4

if dataset_name == "Fingervein1" or "Fingervein2":
    img_shape = (1, 64, 128)
else:
    img_shape = (1, 64, 64)

num_epochs = 100


def main():

    train_loader = DataLoader(
        getOriginalDatasetData(dataset_name, phase="Train"),
        batch_size=batch_size,
        shuffle=True
    )

    # train ae1
    print("train ae1")
    ae1 = DenoisingAutoEncoder_1(img_shape=img_shape)
    ae1.train(
        train_loader,
        dataset_name,
        lr=lr,
        v_noise=0.1,
        num_epochs=num_epochs
    )

    # train ae2
    print("train ae2")
    ae2 = DenoisingAutoEncoder_2(img_shape=img_shape)
    ae2.train(
        train_loader,
        dataset_name,
        lr=lr,
        v_noise=0.2,
        num_epochs=num_epochs
    )


if __name__ == "__main__":
    main()
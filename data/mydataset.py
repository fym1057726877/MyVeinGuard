import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from utils import get_project_path

class Vein600_128x128(Dataset):
    def __init__(self, path_dir=None, transform=None):
        super().__init__()
        if path_dir is None:
            path_dir = os.path.join(get_project_path(), "data", "600_128x128")
        self.path_dir = path_dir
        self.imgs = os.listdir(self.path_dir)
        self.transform = transform

    def __getitem__(self, index):  # 必须自己定义
        image_name = self.imgs[index]
        image_path = os.path.join(self.path_dir, image_name)
        image = Image.open(image_path).convert('L')
        label = int(image_name.split("_")[-1].split(".")[0])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):  # 必须自己定义
        return len(self.imgs)


class CASIA_PV200(Dataset):
    def __init__(self, path_dir=None, transform=None):
        super().__init__()
        if path_dir is None:
            path_dir = os.path.join(get_project_path(), "data", "600_128x128")
        self.path_dir = path_dir
        self.imgs = os.listdir(self.path_dir)
        self.transform = transform

    def __getitem__(self, index):  # 必须自己定义
        image_name = self.imgs[index]
        image_path = os.path.join(self.path_dir, image_name)
        image = Image.open(image_path).convert('L')
        label = int(image_name.split("_")[-1].split(".")[0])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):  # 必须自己定义
        return len(self.imgs)


class VERA_PV200(Dataset):
    def __init__(self, path_dir=None, transform=None):
        super().__init__()
        if path_dir is None:
            path_dir = os.path.join(get_project_path(), "data", "600_128x128")
        self.path_dir = path_dir
        self.imgs = os.listdir(self.path_dir)
        self.transform = transform

    def __getitem__(self, index):  # 必须自己定义
        image_name = self.imgs[index]
        image_path = os.path.join(self.path_dir, image_name)
        image = Image.open(image_path).convert('L')
        label = int(image_name.split("_")[-1].split(".")[0])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):  # 必须自己定义
        return len(self.imgs)


if __name__ == '__main__':
    transform = transforms.ToTensor()
    dataset = Vein600_128x128(transform=transform)
    dataloder = DataLoader(dataset, batch_size=100, shuffle=True)
    print(len(dataloder))
    x, y = next(iter(dataloder))
    print(x, x.shape)
    print(y, y.shape)
    plt.imshow(x[0].squeeze(0).numpy(), cmap="gray")
    plt.show()
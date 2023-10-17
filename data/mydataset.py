import os

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from utils import get_project_path

transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms.Normalize(mean=[0.485], std=[0.229])
])

transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485], std=[0.229, 0.224, 0.225]),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

normalize = transforms.Normalize(mean=[0.485], std=[0.229])


class Vein600_128x128(Dataset):
    def __init__(self, path_dir=None, transform=None):
        super().__init__()
        if path_dir is None:
            path_dir = os.path.join(get_project_path(), "data", "600_128x128")
        if transform is None:
            transform = transforms.ToTensor()
        self.path_dir = path_dir
        self.imgs = os.listdir(self.path_dir)
        self.transform = transform

    def __getitem__(self, index):  # 必须自己定义
        image_name = self.imgs[index]
        image_path = os.path.join(self.path_dir, image_name)
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path).convert('L')
        label = int(image_name.split("_")[0]) - 1
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):  # 必须自己定义
        return len(self.imgs)


def get_Vein600_128x128_Dataloader(train=True, batch_size=16, shuffle=False, transform=None):
    dataset = Vein600_128x128(transform=transform)
    num_classes = 600
    num_images_per_class = 20
    train_images_per_class = 15
    indices = []

    # 遍历每个类别
    for class_idx in range(num_classes):
        start_idx = class_idx * num_images_per_class
        split_index = start_idx + train_images_per_class
        end_idx = (class_idx + 1) * num_images_per_class
        if train:
            subset = Subset(dataset, range(start_idx, split_index))
        else:
            subset = Subset(dataset, range(split_index, end_idx))
        indices.extend(subset.indices)
    dataset = Subset(dataset, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


trainloader = get_Vein600_128x128_Dataloader(train=True, batch_size=20, shuffle=True, transform=transform_train)
testloader = get_Vein600_128x128_Dataloader(train=False, batch_size=20, shuffle=False, transform=transform_test)





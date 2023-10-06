import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from utils import get_project_path


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
        image = Image.open(image_path).convert('L')
        label = int(image_name.split("_")[0]) - 1
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):  # 必须自己定义
        return len(self.imgs)


def get_Vein600_128x128_Dataloader(batch_size=1, shuffle=False, transform=None):
    dataset = Vein600_128x128(transform=transform)
    if isinstance(batch_size, int):
        train_batch_size = test_batch_size = batch_size
    elif isinstance(batch_size, (list, tuple)):
        train_batch_size, test_batch_size = batch_size[0], batch_size[1]
    else:
        raise TypeError("the type of bacth_size must be int, list or tuple")

    if isinstance(shuffle, (int, bool)):
        train_shuffle = test_shuffle = bool(shuffle)
    elif isinstance(shuffle, (list, tuple)):
        train_shuffle, test_shuffle = bool(shuffle[0]), bool(shuffle[1])
    else:
        raise TypeError("the type of shuffle must be int, bool, list or tuple")

    num_classes = 600
    num_images_per_class = 20
    train_images_per_class = 15

    # 为测试集创建一个空列表
    train_indices = []
    test_indices = []

    # 遍历每个类别
    for class_idx in range(num_classes):
        # 获取当前类别的样本索引范围
        start_idx = class_idx * num_images_per_class
        split_index = start_idx + train_images_per_class
        end_idx = (class_idx + 1) * num_images_per_class

        # 为当前类别创建一个子集对象，并将其索引范围添加到测试集列表中
        train_subset = Subset(dataset, range(start_idx, split_index))
        train_indices.extend(train_subset.indices)

        test_subset = Subset(dataset, range(split_index, end_idx))
        test_indices.extend(test_subset.indices)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    trainloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle)
    testloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=test_shuffle)

    return trainloader, testloader


if __name__ == '__main__':
    get_Vein600_128x128_Dataloader(batch_size=50)
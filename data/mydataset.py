import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


class MyDataset(Dataset):
    def __init__(self, path_dir, transform=None):
        super().__init__()
        self.path_dir = path_dir
        self.imgs = os.listdir(self.path_dir)
        self.transform = transform

    def __getitem__(self, index):  # 必须自己定义
        image_name = self.imgs[index]
        image_path = os.path.join(self.path_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        label = int(image_name.split("_")[-1].split(".")[0])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):  # 必须自己定义
        return len(self.imgs)


if __name__ == '__main__':
    transform = transforms.ToTensor()
    dataset = MyDataset("./imgs", transform=transform)
    dataloder = DataLoader(dataset, batch_size=1, shuffle=False)
    x, y = next(iter(dataloder))
    print(x)
    print(y)
    plt.imshow(x.squeeze(0).permute(1, 2, 0).numpy())
    plt.show()
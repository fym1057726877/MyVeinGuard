import torch
import os
import matplotlib.pyplot as plt


def get_project_path(project_name='MyVeinGuard'):
    """
    :param project_name: 项目名称，如pythonProject
    :return: ******/project_name
    """
    # 获取当前所在文件的路径
    cur_path = os.path.abspath(os.path.dirname(__file__))

    # 获取根目录
    return cur_path[:cur_path.find(project_name)] + project_name


def draw_ori_and_recon_images16(images, recon_images):
    assert images.shape[0] >= 16
    images = images.cpu().squeeze(1).detach().numpy()
    recon_images = recon_images.cpu().squeeze(1).detach().numpy()
    fig = plt.figure()
    gs = fig.add_gridspec(4, 8)
    for i in range(4):
        for j in range(8):
            ax = fig.add_subplot(gs[i, j])
            if i < 2:
                idx = i * 4 + j
                ax.imshow(images[idx], cmap="gray")
            else:
                idx = (i - 2) * 4 + j
                ax.imshow(recon_images[idx], cmap="gray")
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_ori_noise_recon_images16(images, noise_imgs, recon_images):
    assert images.shape[0] == noise_imgs.shape[0] == recon_images.shape[0]
    assert images.shape[0] >= 16
    images = images.cpu().squeeze(1).detach().numpy()
    noise_imgs = noise_imgs.cpu().squeeze(1).detach().numpy()
    recon_images = recon_images.cpu().squeeze(1).detach().numpy()
    fig = plt.figure()
    gs = fig.add_gridspec(6, 8)
    for i in range(6):
        for j in range(8):
            ax = fig.add_subplot(gs[i, j])
            if i < 2:
                idx = i * 8 + j
                ax.imshow(images[idx], cmap="gray")
            elif 2 <= i < 4:
                idx = (i - 2) * 8 + j
                ax.imshow(noise_imgs[idx], cmap="gray")
            else:
                idx = (i - 4) * 8 + j
                ax.imshow(recon_images[idx], cmap="gray")
            ax.axis("off")
    plt.tight_layout()
    plt.show()



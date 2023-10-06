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


def draw_ori_and_recon_images32(images, recon_images):
    images = images.cpu().squeeze(1).detach().numpy()
    recon_images = recon_images.cpu().squeeze(1).detach().numpy()
    fig = plt.figure()
    gs = fig.add_gridspec(4, 8)
    for i in range(4):
        for j in range(8):
            ax = fig.add_subplot(gs[i, j])
            if i < 2:
                idx = i * 4 + j
                ax.imshow((images[idx] + 1) * 255 / 2, cmap="gray")
            else:
                idx = (i - 2) * 4 + j
                ax.imshow((recon_images[idx] + 1) * 255 / 2, cmap="gray")
            ax.axis("off")
    plt.tight_layout()
    plt.show()


# if __name__ == '__main__':
#     print(get_project_path())
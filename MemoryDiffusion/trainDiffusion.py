import os
import torch
from torch import optim
from gaussiandiffusion import GaussianDiffusion, UNetModel, ModelMeanType
from data.mydataset import get_Vein600_128x128_Dataloader
from tqdm import tqdm
from time import time
from utils import get_project_path, draw_ori_and_recon_images32

train_loader, test_loader = get_Vein600_128x128_Dataloader(batch_size=16, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(epochs):
    start = time()
    diffsuion = GaussianDiffusion(
        betas_schedule="linear",
        time_steps=1000,
        ddim_step=100,
        mean_type=ModelMeanType.START_X,
    )
    unet = UNetModel(
        in_channels=1,
        model_channels=64,
        out_channels=1,
        channel_mult=(1, 2, 2),
        attention_resolutions=[],
        num_res_blocks=2,
    ).to(device)
    unet.load_state_dict(torch.load(os.path.join(get_project_path(), "pretrained", "ddim.pth")))
    optimizer = optim.Adam(unet.parameters())

    for epoch in range(epochs):
        count = len(train_loader)
        epoch_loss = 0
        unet.train()
        for step, (images, _) in tqdm(enumerate(train_loader), desc=f"train step {epoch + 1}/{epochs}", total=count):
            optimizer.zero_grad()
            images = images.to(device)
            batch_size = images.shape[0]
            noise = torch.randn_like(images)
            t = diffsuion.get_rand_t(batch_size, device)
            loss = diffsuion.training_losses(model=unet, x_start=noise, t=t)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        torch.save(unet.state_dict(), os.path.join(get_project_path(), "pretrained", "ddim.pth"))

        test(diffsuion, unet)
        epoch_loss /= count
        print(f"Epoch:{epoch + 1}/{epochs}  Loss:{epoch_loss:.8f}")

    end = time()
    seconds = int(end - start)
    minutes = seconds // 60
    remain_second = seconds % 60
    print(f"time consumed: {minutes}min{remain_second}s")


def test(diff, unet):
    unet.eval()
    imgs, labels = next(iter(test_loader))
    imgs, labels = imgs.to(device), labels.to(device)
    noise = torch.randn_like(imgs)
    # show_ddim_results(ddim, model, imgs)
    final_sample = diff.ddim_sample_loop(unet, shape=imgs.shape, noise=imgs, progress=True)[0]
    draw_ori_and_recon_images32(imgs, final_sample)


# def test_class():
#     ddim, model = create_ddim_and_unet(device=device)
#     model.load_state_dict(torch.load("./pretrained/ddim_fmnist.pth"))
#     classifier = resnet50(pretrained=True).to(device)
#     model.eval()
#     classifier.eval()
#     total = 0
#     correct_ori = 0
#     correct_recon = 0
#
#     for batch_idx, (imgs, labels) in tqdm(enumerate(test_loader), desc='test step', total=len(test_loader)):
#         imgs, labels = imgs.to(device), labels.to(device)
#         out_ori = classifier(imgs)
#         predict_ori = torch.max(out_ori.data, dim=1)[1]
#         correct_ori += (predict_ori == labels).sum()
#
#         recon_imgs = ddim.ddim_sample_loop(model, shape=imgs.shape, noise=imgs, progress=False)
#         out_recon = classifier(recon_imgs)
#         predict_recon = torch.max(out_recon.data, dim=1)[1]
#         correct_recon += (predict_recon == labels).sum()
#
#         total += imgs.shape[0]
#
#     acc_ori_imgs = correct_ori / total
#     acc_recon_imgs = correct_recon / total
#     print(f"acc_ori_imgs:{acc_ori_imgs:.4f}({correct_ori}/{total})\n"
#           f"acc_recon_imgs:{acc_recon_imgs:.4f}({correct_recon}/{total})\n")
#
#
# def generate():
#     noise = torch.randn((64, 1, 28, 28), device=device)
#     imgs, labels = next(iter(test_loader))
#     imgs, labels = imgs.to(device), labels.to(device)
#     ddim, model = create_ddim_and_unet(device=device)
#     model.load_state_dict(torch.load("./pretrained/ddim_fmnist_eps.pth"))
#     generate_imgs = ddim.ddim_sample_loop(model=model, shape=noise.shape, noise=noise, progress=True, eta=1.)
#     draw_fashion_mnist_images16(generate_imgs)


if __name__ == "__main__":
    train(5)
    # test_class()
    # test()
    # generate()




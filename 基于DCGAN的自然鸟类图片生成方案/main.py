# Name: DCGAN_main
# Author: Reacubeth
# Time: 2021/5/28 19:47
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import requests
import tarfile
from tqdm import tqdm

from DCGAN import Discriminator, Generator

# 添加下载数据集的函数
def download_dataset():
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    target_path = "bird/CUB_200_2011.tgz"
    extracted_path = "bird"
    
    if os.path.exists('bird/CUB_200_2011'):
        print("数据集已存在，跳过下载")
        return
    
    os.makedirs('bird', exist_ok=True)
    
    print("正在下载CUB-200-2011数据集...")
    try:
        # 添加超时设置
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # 检查响应状态
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as file, tqdm(
            desc="下载进度",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=8192):  # 增加chunk_size
                if data:  # 过滤掉keep-alive新块
                    size = file.write(data)
                    pbar.update(size)
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)  # 清理未完成的下载文件
        return
    
    # 解压数据集
    print("正在解压数据集...")
    with tarfile.open(target_path) as tar:
        tar.extractall(path=extracted_path)
    
    # 删除压缩包
    os.remove(target_path)
    print("数据集准备完成！")

# 在主代码开始前调用下载函数
if __name__ == "__main__":
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # 下载并解压数据集
    download_dataset()

    path = 'bird/CUB_200_2011/'
    ROOT_TRAIN = path + 'images/'

    workers = 2
    batch_size = 128
    image_size = 128
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 200
    lr = 0.0002  # 0.0002
    beta1 = 0.5
    ngpu = 0  # 设置为 0 表示不使用 GPU

    dataset = dset.ImageFolder(root=ROOT_TRAIN,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

    # 设置设备为 CPU
    device = torch.device("cpu")

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    netG = Generator(ngpu).to(device)
    # 移除多 GPU 相关代码
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(ngpu).to(device)
    # 移除多 GPU 相关代码
    netD.apply(weights_init)
    print(netD)

    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    torch.save(netD, 'checkpoint/netD' + str(num_epochs) + '.pth')
    torch.save(netG, 'checkpoint/netG' + str(num_epochs) + '.pth')

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('gan_losses.pdf', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.savefig('fake_img.pdf', bbox_inches='tight')
    HTML(ani.to_jshtml())

    real_batch = next(iter(dataloader))

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('real_img.pdf', bbox_inches='tight')

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig('cmp_img.pdf', bbox_inches='tight')
    plt.show()
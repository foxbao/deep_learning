from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from diffusion_utilities import *
from torch.utils.tensorboard import SummaryWriter
from utils import *
from ViT import *
import math
from diffusion import Unet,Diffusion
from ddpm import DDPMSampler


def main():
    # hyperparameters

    # diffusion hyperparameters
    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02

    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available()
                        else torch.device('cpu'))
    n_feat = 64  # 64 hidden dimension feature
    n_cfeat = 5  # context vector is of size 5
    # height = 16  # 16x16 image
    # height = 256  # 16x16 image, don't forget to change transform_size in diffusion_utilities.py
    height = 32  # 16x16 image, don't forget to change transform_size in diffusion_utilities.py
    in_channels = 3
    save_dir = './weights/'

    # training hyperparameters
    batch_size = 64
    n_epoch = 2000
    lrate = 1e-3

    sampler=DDPMSampler(beta2,beta1,timesteps,device)

    writer = SummaryWriter('runs')
    # construct model
    nn_model = Diffusion(in_channels=in_channels,layout_channels=in_channels, n_feat=n_feat,
                        n_cfeat=n_cfeat, height=height).to(device)
    
    transform = get_transform(height)

    home_dir = os.path.expanduser('~')
    dataset = CustomDataset3(
        img_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/parking_generate_data"),
        img_names="data/parking_generate_data/data.txt",
        layout_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/parking_layout_data"),
        layout_names="data/parking_layout_data/data.txt",
        transform=transform,
        transform_layout=transform,
        null_context=False,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=1)

    val_dataset = CustomDataset3(
        img_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/val_parking_generate_data"),
        img_names="data/val_parking_generate_data/data.txt",
        layout_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/val_parking_layout_data"),
        layout_names="data/val_parking_layout_data/data.txt",
        transform=transform,
        transform_layout=transform,
        null_context=False,
    )

    val_batch_size = 4
    dataloader_val = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1
    )

    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    # helper function: perturbs an image to a specified noise level
    # training without context code
    # embedding_dim = 128  # 嵌入维度
    # time_embedding = TimeEmbedding(embedding_dim).to(device)
    # training without context code


    parameter_num = get_parameter_number(nn_model)
    print(parameter_num['Total'])

    nn_model.train()
    is_training = True
    if is_training:
        for ep in range(n_epoch):
            print(f"epoch {ep}")
            # linearly decay learning rate
            optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

            pbar = tqdm(dataloader, mininterval=2)
            tr_loss = 0
            for x, layout in pbar:  # x: images
                optim.zero_grad()
                x = x.to(device)

                layout = layout.to(device)

                # perturb data
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
                x_pert = sampler.perturb_input(x, t, noise)
                # x_pert = perturb_input(x, t, ab_t,noise)

                # use network to recover noise
                pred_noise = nn_model(
                    latent=x_pert, layout=layout, context=None,time=t/1.0)

                # loss is mean squared error between the predicted and true noise
                loss = F.mse_loss(pred_noise, noise)
                tr_loss += loss.item()
                loss.backward()
                optim.step()
            epoch_loss = tr_loss / len(dataloader)
            writer.add_scalar("Loss/train", epoch_loss, ep)
            print("loss:", loss.item())
            print("epoch_loss:", epoch_loss)
            # save model periodically
            if ep % 100 == 0 or ep == int(n_epoch - 1):
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
                print("saved model at " + save_dir + f"model_{ep}.pth")


    # load in model weights and set to eval mode
    # nn_model.load_state_dict(torch.load(
    #     f"{save_dir}/model_{1500}.pth", map_location=device))
    nn_model.load_state_dict(torch.load(
        f"{save_dir}/model_{n_epoch-1}.pth", map_location=device))
    nn_model.eval()
    print("Loaded in Model")


    for idx, (gt, layout) in enumerate(dataloader_val):
        gt = gt.to(device)
        layout = layout.to(device)
        samples, intermediate =sampler.sample_ddpm_context(layout.shape[0], nn_model,in_channels,height,timesteps,layout,device='cuda',save_rate=20)

        save_layout_sample_gt(
            layouts=layout, samples=samples, gts=gt, name=str(idx) + "_triple.jpg"
        )

    # visualize samples
    # plt.clf()
    # samples, intermediate_ddpm = sample_ddpm(32)


    # animation_ddpm = plot_sample(
    #     intermediate_ddpm, 32, 4, save_dir, "ani_run"+str(n_epoch-1), None, save=True)
    # HTML(animation_ddpm.to_jshtml())

if __name__ == "__main__":
    main()
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
from torch.utils.tensorboard import SummaryWriter
from diffusion import Diffusion
import os
from diffusion_utilities import *




def denoise_add_noise(b_t,a_t,ab_t,x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) /
            (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise


@torch.no_grad()
def sample_ddpm_context(n_sample, layout, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        # predict noise e_(x_t,t, ctx)
        eps = nn_model(samples, t, c=None, layout=None, layout_concate=layout)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

# sample using standard algorithm


@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, in_channels, height, height).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

def perturb_input(ab_t,x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

def train():
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
    in_channels = 3  # dont forget to modify cmap='gray'
    save_dir = './weights/'

    # training hyperparameters
    batch_size = 10
    n_epoch = 2000
    lrate = 1e-3


    # construct DDPM noise schedule
    b_t = (beta2 - beta1) * torch.linspace(0, 1,
                                        timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    writer = SummaryWriter('runs')
    
# construct model
    nn_model = Diffusion().to(device)

    transform_size = height
    transform = transforms.Compose([
        transforms.Resize([transform_size, transform_size]),
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,)),  # range [-1,1]
    ])
    
    home_dir = os.path.expanduser('~')
    dataset_train = CustomDataset3(
        img_dir=os.path.join(home_dir, "Downloads/parking2023/baojiali/park_generate/parking_generate_data"),
        img_names="data/parking_generate_data/data.txt",
        layout_dir=os.path.join(home_dir,"Downloads/parking2023/baojiali/park_generate/parking_layout_data"),
        layout_names="data/parking_layout_data/data.txt",
        transform=transform,
        null_context=False,
    )
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                            shuffle=True, num_workers=1)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)
    
    nn_model.train()
    
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        
        pbar = tqdm(dataloader_train, mininterval=2)
        for x, layout in pbar:   # x: images
            optim.zero_grad()
            x = x.to(device)
            layout = layout.to(device)
            # perturb data
            noise = torch.randn_like(x)

            t = torch.randint(1, timesteps + 1,size=())
            
            time_embedding = get_time_embedding(t).to(device)
            x_pert = perturb_input(ab_t,x, t, noise)
            
            # use network to recover noise
            pred_noise = nn_model(x=x_pert, context=None,time=time_embedding)
            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optim.step()
        writer.add_scalar("Loss/train", loss.item(), ep)
        print("loss:", loss.item())
        # save model periodically
        if ep % 100 == 0 or ep == int(n_epoch - 1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
            print("saved model at " + save_dir + f"model_{ep}.pth")

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)



def main():
    train()

if __name__ == '__main__':
    main()
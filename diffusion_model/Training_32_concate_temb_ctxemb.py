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


class SwitchSequential(nn.Sequential):
    def forward(self, x, context=None, time=None):
        for layer in self:
            if isinstance(layer, UnetResTime):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class LayoutEmbed(nn.Module):
    def __init__(self, in_channels, n_feat, h):
        super(LayoutEmbed, self).__init__()
        self.h = h
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.init_conv_layout = ResidualConvBlock(
            in_channels, n_feat, is_res=True)
        self.vitembedConcate = ViT(
            image_size=(n_feat, self.h, self.h), out_channels=4 * n_feat, patch_size=4, concate_mode=True
        )

    def forward(self, x, layout):
        x = self.init_conv(x)
        layout = self.init_conv_layout(layout)
        vembedConcate = self.vitembedConcate(layout)
        x = torch.concat((x, vembedConcate), dim=1)
        return x


class ContextUnet(nn.Module):
    # cfeat - context features
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28, hidden_layers_n=3):
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        # assume h == w. must be divisible by 4, so 28,24,20,16...
        self.h = height

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.init_conv_layout = ResidualConvBlock(
            in_channels, n_feat, is_res=True)
        # Initialize the down-sampling path of the U-Net with two levels

        # (Batch_Size, 3, Height , Width)->(Batch_Size, (1+4)*n_feat, Height , Width)
        self.layout_embed = LayoutEmbed(in_channels, n_feat, self.h)

        self.encoders = nn.ModuleList([
            # (Batch_Size, (1+4)*n_feat, Height , Width)->(Batch_Size, n_feat, Height/2, Width/2)
            SwitchSequential(UnetResTime(in_channels=(
                1+4)*n_feat, out_channels=n_feat, n_time=128), nn.MaxPool2d(2)),
            # (Batch_Size, n_feat, Height/2, Width/2)->(Batch_Size, 2*n_feat, Height/4, Width/4)
            SwitchSequential(UnetResTime(
                in_channels=n_feat, out_channels=2 * n_feat, n_time=128), nn.MaxPool2d(2)),
            # (Batch_Size, 2*n_feat, Height/4, Width/4)->(Batch_Size, 4*n_feat, Height/8, Width/8)
            SwitchSequential(UnetResTime(
                in_channels=2 * n_feat, out_channels=4 * n_feat, n_time=128), nn.MaxPool2d(2))
        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 4*n_feat, Height/8, Width/8)->(Batch_Size, 4*n_feat, Height/32, Width/32)
            nn.Sequential(nn.AvgPool2d((4)), nn.GELU()),
            nn.Sequential(
                nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, 4, 4),  # up-sample
                nn.GroupNorm(8, 4 * n_feat),  # normalize
                nn.ReLU(),
            )
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UnetResTime(8 * n_feat, 2*n_feat, n_time=128),
                             nn.ConvTranspose2d(2*n_feat, 2*n_feat, 2, 2)),
            SwitchSequential(UnetResTime(4 * n_feat, 1*n_feat, n_time=128),
                             nn.ConvTranspose2d(1*n_feat, 1*n_feat, 2, 2)),
            SwitchSequential(UnetResTime(2 * n_feat, 1*n_feat, n_time=128),
                             nn.ConvTranspose2d(1*n_feat, 1*n_feat, 2, 2)),
        ])

        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        self.vitembedConcate = ViT(
            image_size=(n_feat, self.h, self.h), out_channels=4 * n_feat, patch_size=4, concate_mode=True
        )

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample
            nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, 4, 4),  # up-sample
            nn.GroupNorm(8, 4 * n_feat),  # normalize
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            # nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),

            nn.Conv2d(5*n_feat+n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),  # normalize
            nn.ReLU(),
            # map to same number of channels as input
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None, layout_concate=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """

        x = self.layout_embed(x=x, layout=layout_concate)
        residue = x
        skip_connections = []
        for layers in self.encoders:
            x = layers(x=x, context=c,time=t)
            skip_connections.append(x)

        x = self.bottleneck(x)
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x=x, context=c,time=t)

        out = self.out(torch.cat((residue, x), 1))
        return out


# hyperparameters


# diffusion hyperparameters
timesteps = 50
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
batch_size = 8
n_epoch = 1000
lrate = 1e-3


# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1,
                                       timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

writer = SummaryWriter('runs')
# construct model
nn_model = ContextUnet(in_channels=in_channels, n_feat=n_feat,
                       n_cfeat=n_cfeat, height=height).to(device)

transform_size = height
transform = transforms.Compose([
    transforms.Resize([transform_size, transform_size]),
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,)),  # range [-1,1]
])

# # load dataset and construct optimizer
# dataset = CustomDataset2("data/jaffe", "data/jaffe/jaffe.txt", transform, null_context=True)
# dataset = CustomDataset2("data/parking_generate_data", "data/parking_generate_data/data.txt",transform, null_context=True)

home_dir = os.path.expanduser('~')
dataset = CustomDataset3(
    img_dir=os.path.join(
        home_dir, "Downloads/parking2023/baojiali/park_generate/parking_generate_data"),
    img_names="data/parking_generate_data/data.txt",
    layout_dir=os.path.join(
        home_dir, "Downloads/parking2023/baojiali/park_generate/parking_layout_data"),
    layout_names="data/parking_layout_data/data.txt",
    transform=transform,
    null_context=False,
)
# dataset = CustomDataset2("data/parking_layout_data", "data/parking_layout_data/data.txt",transform, null_context=True)

# load dataset and construct optimizer
# dataset = CustomDataset("./sprites_1788_16x16.npy", "./sprite_labels_nc_1788_16x16.npy", transform, null_context=False)

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
    null_context=False,
)

val_batch_size = 4
dataloader_val = DataLoader(
    val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1
)

optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

# helper function: perturbs an image to a specified noise level


def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise


# training without context code
embedding_dim = 128  # 嵌入维度
time_embedding = TimeEmbedding(embedding_dim).to(device)
# training without context code

is_training = True

nn_model.train()
is_training = True
if is_training:
    for ep in range(n_epoch):
        print(f"epoch {ep}")
        # linearly decay learning rate
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader, mininterval=2)

        for x, layout in pbar:  # x: images
            optim.zero_grad()
            x = x.to(device)

            layout = layout.to(device)

            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)

            time_emb = time_embedding(t/1.0)
            x_pert = perturb_input(x, t, noise)

            # use network to recover noise
            pred_noise = nn_model(
                x_pert, time_emb, c=None, layout_concate=layout)

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


# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)


def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) /
            (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

# sample using standard algorithm


@torch.no_grad()
def sample_ddpm_context(n_sample, layout, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f"sampling timestep {i:3d}", end="\r")

        # reshape time tensor
        # t = torch.tensor([i])[:, None, None, None].to(device)
        t = torch.tensor([i/1.0]).to(device)
        time_emb = time_embedding(t)
        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        # predict noise e_(x_t,t, ctx)
        eps = nn_model(samples, time_emb, c=None, layout_concate=layout)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, in_channels, height, height).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


# load in model weights and set to eval mode
# nn_model.load_state_dict(torch.load(
#     f"{save_dir}/model_{1500}.pth", map_location=device))
nn_model.load_state_dict(torch.load(
    f"{save_dir}/model_{500}.pth", map_location=device))
nn_model.eval()
print("Loaded in Model")


for idx, (gt, layout) in enumerate(dataloader_val):
    gt = gt.to(device)
    layout = layout.to(device)
    samples, intermediate = sample_ddpm_context(layout.shape[0], layout)
    # samples=torch.randn_like(layout)
    # save_images(layout, nrow=2, name=str(idx) + "_layout.jpg")
    # save_images(gt, nrow=2, name=str(idx) + "_gt.jpg")

    save_layout_sample_gt(
        layouts=layout, samples=samples, gts=gt, name=str(idx) + "_triple.jpg"
    )

# visualize samples
# plt.clf()
# samples, intermediate_ddpm = sample_ddpm(32)


# animation_ddpm = plot_sample(
#     intermediate_ddpm, 32, 4, save_dir, "ani_run"+str(n_epoch-1), None, save=True)
# HTML(animation_ddpm.to_jshtml())

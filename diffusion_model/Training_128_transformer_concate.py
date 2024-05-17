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
from ViT import *
from utils import *


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):  # cfeat - context features
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

        # self.down1 = UnetDown(n_feat, n_feat)
        # 1 is from the conv, 4 is from the Vit
        self.down1 = UnetDown((1+4)*n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)        #
        self.down4 = UnetDown(4 * n_feat, 8 * n_feat)
        self.down5 = UnetDown(8 * n_feat, 16 * n_feat)

        self.timeembed_down_1 = EmbedFC(1, 1*n_feat)
        self.timeembed_down_2 = EmbedFC(1, 2*n_feat)
        self.timeembed_down_3 = EmbedFC(1, 4*n_feat)
        self.timeembed_down_4 = EmbedFC(1, 8*n_feat)
        self.timeembed_down_5 = EmbedFC(1, 16*n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 16*n_feat)
        self.timeembed2 = EmbedFC(1, 8*n_feat)
        self.timeembed3 = EmbedFC(1, 4*n_feat)
        self.timeembed4 = EmbedFC(1, 2*n_feat)
        self.timeembed5 = EmbedFC(1, 1*n_feat)
        # self.timeembed6 = EmbedFC(1, 1*n_feat)

        self.contextembed1 = EmbedFC(n_cfeat, 16*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 8*n_feat)
        self.contextembed3 = EmbedFC(n_cfeat, 4*n_feat)
        self.contextembed4 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed5 = EmbedFC(n_cfeat, 1*n_feat)

        self.vitembedConcate = ViT(
            image_size=(n_feat, self.h, self.h), out_channels=4 * n_feat, patch_size=4, concate_mode=True
        )
        # self.contextembed6 = EmbedFC(n_cfeat, 1*n_feat)

        # self.imgembed1=EmbedImage(n_feat,16*n_feat)
        # self.imgembed2=EmbedImage(n_feat,8*n_feat)
        # self.imgembed3=EmbedImage(n_feat,4*n_feat)
        # self.imgembed4=EmbedImage(n_feat,2*n_feat)
        # self.imgembed5=EmbedImage(n_feat,1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample
            nn.ConvTranspose2d(16 * n_feat, 16 * n_feat, 4, 4),  # up-sample
            nn.GroupNorm(8, 16 * n_feat),  # normalize
            nn.ReLU(),
        )
        self.up1 = UnetUp(32 * n_feat, 8*n_feat)
        self.up2 = UnetUp(16 * n_feat, 4*n_feat)
        self.up3 = UnetUp(8 * n_feat, 2*n_feat)
        self.up4 = UnetUp(4 * n_feat, 1*n_feat)
        self.up5 = UnetUp(2 * n_feat, 1*n_feat)
        # self.up6 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            # 5 is from the concate, 1 is from the up
            nn.Conv2d((5+1) * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),  # normalize
            nn.ReLU(),
            # map to same number of channels as input
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None, layout=None, layout_concate=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        layout:(batch,n_feat,h,w) : context image
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        layout_concate = self.init_conv_layout(layout_concate)
        vembedConcate = self.vitembedConcate(layout_concate)
        x = self.init_conv(x)
        x = torch.concat((x, vembedConcate), dim=1)

        temb_down_1 = self.timeembed_down_1(t).view(-1, self.n_feat * 1, 1, 1)
        temb_down_2 = self.timeembed_down_2(t).view(-1, self.n_feat * 2, 1, 1)
        temb_down_3 = self.timeembed_down_3(t).view(-1, self.n_feat * 4, 1, 1)
        temb_down_4 = self.timeembed_down_4(t).view(-1, self.n_feat * 8, 1, 1)
        temb_down_5 = self.timeembed_down_5(t).view(-1, self.n_feat * 16, 1, 1)

        # pass the result through the down-sampling path
        down1 = self.down1(x)  # [10, 256, 8, 8]
        down2 = self.down2(down1+temb_down_1)  # [10, 256, 8, 8]
        down3 = self.down3(down2+temb_down_2)  # [10, 256, 4, 4]
        down4 = self.down4(down3+temb_down_3)
        down5 = self.down5(down4+temb_down_4)
        # down5 = self.down5(down4)
        # down6 = self.down6(down5)
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down5+temb_down_5)
        # hiddenvec2=self.to_vec(down3)
        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # embed context and timestep
        # (batch, 2*n_feat, 1,1)
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 16, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 16, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 8, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 8, 1, 1)
        cemb3 = self.contextembed3(c).view(-1, self.n_feat*4, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.n_feat*4, 1, 1)
        cemb4 = self.contextembed4(c).view(-1, self.n_feat*2, 1, 1)
        temb4 = self.timeembed4(t).view(-1, self.n_feat*2, 1, 1)
        cemb5 = self.contextembed5(c).view(-1, self.n_feat, 1, 1)
        temb5 = self.timeembed5(t).view(-1, self.n_feat, 1, 1)
        # cemb6 = self.contextembed6(c).view(-1, self.n_feat, 1, 1)
        # temb6 = self.timeembed6(t).view(-1, self.n_feat, 1, 1)
        # print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")
        if layout:
            layout = self.init_conv_layout(layout)
        # iemb1=self.imgembed1(layout).view(-1, self.n_feat * 16, 1, 1)
        # iemb2=self.imgembed2(layout).view(-1, self.n_feat * 8, 1, 1)
        # iemb3=self.imgembed3(layout).view(-1, self.n_feat * 4, 1, 1)
        # iemb4=self.imgembed4(layout).view(-1, self.n_feat * 2, 1, 1)
        # iemb5=self.imgembed5(layout).view(-1, self.n_feat * 1, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down5)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2 + temb2, down4)
        up4 = self.up3(cemb3*up3 + temb3, down3)
        up5 = self.up4(cemb4*up4 + temb4, down2)
        up6 = self.up5(cemb5*up5 + temb5, down1)
        # up7 = self.up6(cemb6*up6 + temb6, down1)
        out = self.out(torch.cat((up6, x), 1))
        return out

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
height = 128  # 16x16 image, don't forget to change transform_size in diffusion_utilities.py
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
dataset = CustomDataset3(
    img_dir="/home/baojiali/Downloads/parking2023/baojiali/park_generate/parking_generate_data",
    img_names="data/parking_generate_data/data.txt",
    layout_dir="/home/baojiali/Downloads/parking2023/baojiali/park_generate/parking_layout_data",
    layout_names="data/parking_layout_data/data.txt",
    transform=transform,
    null_context=False,
)
# dataset = CustomDataset2("data/parking_layout_data", "data/parking_layout_data/data.txt",transform, null_context=True)

# load dataset and construct optimizer
# dataset = CustomDataset("./sprites_1788_16x16.npy", "./sprite_labels_nc_1788_16x16.npy", transform, null_context=False)
val_dataset = CustomDataset3(
    img_dir="/home/baojiali/Downloads/parking2023/baojiali/park_generate/parking_generate_data",
    img_names="data/val_parking_generate_data/data.txt",
    layout_dir="/home/baojiali/Downloads/parking2023/baojiali/park_generate/parking_layout_data",
    layout_names="data/val_parking_layout_data/data.txt",
    transform=transform,
    null_context=False,
)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=1)

val_batch_size=5
dataloader_val = DataLoader(
    val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

# helper function: perturbs an image to a specified noise level


def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

# training without context code


# set into train mode
# nn_model.load_state_dict(torch.load(
#     f"{save_dir}/model_70.pth", map_location=device))
# nn_model.train()

is_training = False
if is_training:
    nn_model.train()
    for ep in range(n_epoch):
        print(f'epoch {ep}')

        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader, mininterval=2)

        for x, layout in pbar:   # x: images
            optim.zero_grad()
            x = x.to(device)
            layout = layout.to(device)
            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            x_pert = perturb_input(x, t, noise)

            # use network to recover noise
            pred_noise = nn_model(x_pert, t / timesteps,
                                  c=None, layout=None, layout_concate=layout)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optim.step()
        writer.add_scalar('Loss/train', loss.item(), ep)
        print("loss:", loss.item())
        # save model periodically
        if ep % 10 == 0 or ep == int(n_epoch-1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)


def denoise_add_noise(x, t, pred_noise, z=None):
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


# load in model weights and set to eval mode
nn_model.load_state_dict(torch.load(
    f"{save_dir}/model_{60}.pth", map_location=device))
nn_model.eval()
print("Loaded in Model")


for idx, (gt, layout) in enumerate(dataloader_val):
    gt = gt.to(device)
    layout = layout.to(device)
    samples, intermediate = sample_ddpm_context(layout.shape[0], layout)
    # show_images(layout)
    # show_images(samples)
    save_layout_sample_gt(
        layouts=layout, samples=samples, gts=gt, name=str(idx) + "_triple.jpg"
    )
    plt.clf()
    animation_ddpm = plot_sample(
        intermediate, val_batch_size, 2, save_dir, "ani_run_"+str(idx), None, save=True)
    # HTML(animation_ddpm.to_jshtml())
# visualize samples
plt.clf()
samples, intermediate_ddpm = sample_ddpm(32)


animation_ddpm = plot_sample(
    intermediate_ddpm, val_batch_size, 4, save_dir, "ani_run"+str(n_epoch-1), None, save=True)
HTML(animation_ddpm.to_jshtml())

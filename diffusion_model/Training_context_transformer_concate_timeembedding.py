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
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28,hidden_layers_n=3):  # cfeat - context features
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        # assume h == w. must be divisible by 4, so 28,24,20,16...
        self.h = height

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.init_conv_layout = ResidualConvBlock(in_channels, n_feat, is_res=True)
        # Initialize the down-sampling path of the U-Net with two levels
        
        
        self.encoders=nn.ModuleList([
            
        ])
        # self.down1 = UnetDown(n_feat, n_feat)
        # self.down=[]
        # self.down.append(UnetDown((1+4)*n_feat, n_feat))
        self.down1 = UnetDown((1+4)*n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2* n_feat)
        self.down3 = UnetDown(2* n_feat, 4 * n_feat)        # down1 #[10, 256, 8, 8]
        # self.down4 = UnetDown(n_feat, n_feat)    # down2 #[10, 256, 4,  4]
        # self.down5 = UnetDown(n_feat, 2 * n_feat)
        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        
        self.timeembed_down_1 = EmbedFC(320, 1*n_feat)
        self.timeembed_down_2 = EmbedFC(320, 2*n_feat)
        self.timeembed_down_3 = EmbedFC(320, 4*n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(320, 4*n_feat)
        self.timeembed2 = EmbedFC(320, 2*n_feat)
        self.timeembed3 = EmbedFC(320, 1*n_feat)
        # self.timeembed4 = EmbedFC(1, 1*n_feat)
        # self.timeembed5 = EmbedFC(1, 1*n_feat)
        # self.timeembed6 = EmbedFC(1, 1*n_feat)
        
        self.contextembed1 = EmbedFC(n_cfeat, 4*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed3 = EmbedFC(n_cfeat, 1*n_feat)
        # self.contextembed4 = EmbedFC(n_cfeat, 1*n_feat)
        # self.contextembed5 = EmbedFC(n_cfeat, 1*n_feat)
        # self.contextembed6 = EmbedFC(n_cfeat, 1*n_feat)
        
        self.vitembedConcate=ViT(
            image_size=(n_feat, self.h, self.h), out_channels=4 * n_feat, patch_size=4,concate_mode=True
        )


        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample
            nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, 4, 4),  # up-sample
            nn.GroupNorm(8, 4 * n_feat),  # normalize
            nn.ReLU(),
        )
        self.up1 = UnetUp(8 * n_feat, 2*n_feat)
        self.up2 = UnetUp(4 * n_feat, 1*n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        # self.up4 = UnetUp(2 * n_feat, n_feat)
        # self.up5 = UnetUp(2 * n_feat, n_feat)
        # self.up6 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            # nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            
            nn.Conv2d(5*n_feat+n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),  # normalize
            nn.ReLU(),
            # map to same number of channels as input
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None,layout_concate=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        layout_concate=self.init_conv_layout(layout_concate)
        vembedConcate=self.vitembedConcate(layout_concate)
        
        x = self.init_conv(x)
        x=torch.concat((x,vembedConcate),dim=1)
        temb_down_1 = self.timeembed_down_1(t).view(-1, self.n_feat * 1, 1, 1)
        temb_down_2 = self.timeembed_down_2(t).view(-1, self.n_feat * 2, 1, 1)
        temb_down_3 = self.timeembed_down_3(t).view(-1, self.n_feat * 4, 1, 1)
        # pass the result through the down-sampling path
        down1 = self.down1(x)  # [10, 256, 8, 8]
        down2 = self.down2(down1+temb_down_1)  # [10, 256, 8, 8]
        down3 = self.down3(down2+temb_down_2)  # [10, 256, 4, 4]
        # down4 = self.down4(down3)
        # down5 = self.down5(down4)
        # down6 = self.down6(down5)
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down3+temb_down_3)
        # hiddenvec2=self.to_vec(down3)
        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # embed context and timestep
        # (batch, 2*n_feat, 1,1)
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 4, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 4, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 2 , 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat *2, 1, 1)
        cemb3 = self.contextembed3(c).view(-1, self.n_feat, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.n_feat, 1, 1)
        # cemb4 = self.contextembed4(c).view(-1, self.n_feat, 1, 1)
        # temb4 = self.timeembed4(t).view(-1, self.n_feat, 1, 1)
        # cemb5 = self.contextembed5(c).view(-1, self.n_feat, 1, 1)
        # temb5 = self.timeembed5(t).view(-1, self.n_feat, 1, 1)
        # cemb6 = self.contextembed6(c).view(-1, self.n_feat, 1, 1)
        # temb6 = self.timeembed6(t).view(-1, self.n_feat, 1, 1)
        # print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")

        # vembedConcate=self.vitembedConcate(layout_concate).view(-1, self.n_feat * 4, 1, 1)
        # vembed1 = self.vitembed1(layout).view(-1, self.n_feat * 4, 1, 1)
        # vembed2 = self.vitembed2(layout).view(-1, self.n_feat * 2, 1, 1)
        # vembed3 = self.vitembed3(layout).view(-1, self.n_feat * 1, 1, 1)
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down3)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2 + temb2, down2)
        up4 = self.up3(cemb3*up3 + temb3, down1)
        # up5 = self.up4(cemb4*up4 + temb4, down2)
        # up6 = self.up5(cemb5*up5 + temb5, down1)
        # up7 = self.up6(cemb6*up6 + temb6, down1)
        out = self.out(torch.cat((up4, x), 1))
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

transform_size=height
transform = transforms.Compose([
    transforms.Resize([transform_size,transform_size]),
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,)),  # range [-1,1]
])

# # load dataset and construct optimizer
# dataset = CustomDataset2("data/jaffe", "data/jaffe/jaffe.txt", transform, null_context=True)
# dataset = CustomDataset2("data/parking_generate_data", "data/parking_generate_data/data.txt",transform, null_context=True)

home_dir = os.path.expanduser('~')
dataset = CustomDataset3(
    img_dir=os.path.join(home_dir, "Downloads/parking2023/baojiali/park_generate/parking_generate_data"),
    img_names="data/parking_generate_data/data.txt",
    layout_dir=os.path.join(home_dir,"Downloads/parking2023/baojiali/park_generate/parking_layout_data"),
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
    img_dir=os.path.join(home_dir,"Downloads/parking2023/baojiali/park_generate/val_parking_generate_data"),
    img_names="data/val_parking_generate_data/data.txt",
    layout_dir=os.path.join(home_dir,"Downloads/parking2023/baojiali/park_generate/val_parking_layout_data"),
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



class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb
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
            t = torch.randint(1, timesteps + 1, ()).to(device)
            
            time_embedding = get_time_embedding(t).to(device)
            x_pert = perturb_input(x, t, noise)

            # use network to recover noise
            pred_noise = nn_model(x_pert, time_embedding, c=None, layout_concate=layout)

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
        t = torch.tensor([i])[:, None, None, None].to(device)
        time_embedding = get_time_embedding(t).to(device)
        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        # predict noise e_(x_t,t, ctx)
        eps = nn_model(samples, time_embedding, c=None,layout_concate=layout)
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

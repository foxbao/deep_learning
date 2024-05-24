import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torch.utils.tensorboard import SummaryWriter
from utils import *
from tqdm import tqdm
from utils import *
from time import time
import os
from torch.utils.data import DataLoader
from diffusion_utilities import CustomDataset3, get_transform
from VAE import VAE, VAE_Encoder, VAE_Decoder
from diffusion import Diffusion
from ddpm import DDPMSampler


# def train(device, dataloader, encoder_VAE: VAE_Encoder, decoder_VAE: VAE_Decoder, nn_model: Diffusion, height,n_epoch=50):
#     save_dir = './weights/'
#     lrate = 1e-3
#     optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)
#     for ep in range(n_epoch):
#         print(f"epoch {ep}")
#         # linearly decay learning rate
#         optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

#         pbar = tqdm(dataloader, mininterval=2)
#         tr_loss = 0
#         for x, layout in pbar:  # x: images
#             optim.zero_grad()
#             x = x.to(device)
#             b,c,h,w=x.shape
#             layout = layout.to(device)
#             encoder_noise = torch.randn(size=(b,4,int(height),int(height)),device=device)

#             latent, mean, log_variance=encoder_VAE(x,encoder_noise)

#             # perturb data
#             noise = torch.randn_like(latent)
#             t = torch.randint(1, timesteps + 1, (latent.shape[0],)).to(device)
#             x_pert = sampler.perturb_input(latent, t, noise)

#             # use network to recover noise
#             pred_noise = nn_model(
#                 latent=x_pert, layout=layout, context=None, time=t/1.0)

#             # loss is mean squared error between the predicted and true noise
#             loss = F.mse_loss(pred_noise, noise)
#             tr_loss += loss.item()
#             loss.backward()
#             optim.step()
#         epoch_loss = tr_loss / len(dataloader)
#         writer.add_scalar("Loss/train", epoch_loss, ep)
#         print("loss:", loss.item())
#         print("epoch_loss:", epoch_loss)
#         # save model periodically
#         if ep % 10 == 0 or ep == int(n_epoch - 1):
#             if not os.path.exists(save_dir):
#                 os.mkdir(save_dir)
#             torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
#             print("saved model at " + save_dir + f"model_{ep}.pth")


def main():

    # hyperparameters

    # diffusion hyperparameters
    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02

    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else torch.device('cpu'))
    img_length = 256
    img_length = 256
    layout_length = int(img_length/8)
    batch_size = 32
    n_epoch = 100

    sampler = DDPMSampler(beta2, beta1, timesteps, device)

    save_dir = './weights/'
    transform = get_transform(img_length)
    transform_layout = get_transform(layout_length)

    home_dir = os.path.expanduser('~')
    dataset = CustomDataset3(
        img_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/parking_generate_data"),
        img_names="data/parking_generate_data/data.txt",
        layout_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/parking_layout_data"),
        layout_names="data/parking_layout_data/data.txt",
        transform=transform,
        transform_layout=transform_layout,
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
        transform_layout=transform_layout,
        null_context=False,
    )

    val_batch_size = 4
    dataloader_val = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1
    )

    model_VAE = VAE(device, height=img_length).to(device)
    model_VAE.load_state_dict(torch.load(
        f"{save_dir}/model_VAE_{199}.pth", map_location=device))
    encoder_VAE = model_VAE.encoder
    decoder_VAE = model_VAE.decoder

    latent_in_channels = 4
    layout_in_channels = 3
    n_feat = 64  # 64 hidden dimension feature
    n_cfeat = 5  # context vector is of size 5
    # 16x16 image, don't forget to change transform_size in diffusion_utilities.py
    latent_height = 32

    writer = SummaryWriter('runs')
    nn_model = Diffusion(latent_in_channels, layout_in_channels, n_feat,
                         n_cfeat, latent_height).to(device)
    lrate = 1e-3

    for name, param in encoder_VAE.named_parameters():
        param.requires_grad = False

    for name, param in decoder_VAE.named_parameters():
        param.requires_grad = False

    encoder_VAE.eval()
    decoder_VAE.eval()
    nn_model.train()
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    latents = []
    for ep in range(n_epoch):
        print(f"epoch {ep}")
        # linearly decay learning rate
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader, mininterval=2)
        tr_loss = 0
        for x, layout in pbar:  # x: images
            optim.zero_grad()
            x = x.to(device)
            b, c, h, w = x.shape
            layout = layout.to(device)
            encoder_noise=torch.zeros(size=(b, 4, int(latent_height), int(latent_height)), device=device)
            # encoder_noise = torch.randn(
            #     size=(b, 4, int(latent_height), int(latent_height)), device=device)
            # latent = x
            latent, mean, log_variance=encoder_VAE(x,encoder_noise)
            # latent_cpu=latent.detach().cpu().numpy()
            # latents.append(latent_cpu)

            # perturb data
            noise = torch.randn_like(latent)
            t = torch.randint(1, timesteps + 1, (latent.shape[0],)).to(device)
            x_pert = sampler.perturb_input(latent, t, noise)

            # use network to recover noise
            pred_noise = nn_model(
                latent=x_pert, layout=layout, context=None, time=t/1.0)

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
        if ep % 10 == 0 or ep == int(n_epoch - 1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
            print("saved model at " + save_dir + f"model_{ep}.pth")

    nn_model.load_state_dict(torch.load(
        f"{save_dir}/model_{n_epoch-1}.pth", map_location=device))
    nn_model.eval()
    print("Loaded in Model")

    # load the encoder of VAE

    for idx, (gt, layout) in enumerate(dataloader_val):
        gt = gt.to(device)
        layout = layout.to(device)

        encoder_noise = torch.randn(
            size=(b, 4, int(latent_height), int(latent_height)), device=device)
        latent, mean, log_variance = encoder_VAE(x, encoder_noise)

        output = encoder_VAE.sample(device)

        # samples, intermediate =sampler.sample_ddpm_context(layout.shape[0], nn_model,in_channels,height,timesteps,layout,device='cuda',save_rate=20)

        # save_layout_sample_gt(
        #     layouts=layout, samples=samples, gts=gt, name=str(idx) + "_triple.jpg"
        # )
    # encode the image to latent space

    # add noise to image

    # train on the latent space

    #

    pass


if __name__ == '__main__':
    main()

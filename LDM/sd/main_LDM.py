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


# def train(device, dataloader, encoder_VAE: VAE_Encoder, decoder_VAE: VAE_Decoder, nn_model: Diffusion, n_epoch=50):
#     lrate = 1e-3
#     encoder_VAE.eval()
#     decoder_VAE.eval()
#     nn_model.train()
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

#             layout = layout.to(device)

#             # perturb data
#             noise = torch.randn_like(x)
#             t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
#             x_pert = perturb_input(x, t, ab_t, noise)

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
#         if ep % 100 == 0 or ep == int(n_epoch - 1):
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
    batch_size = 32
    n_epoch = 2000
    
    sampler=DDPMSampler(beta2,beta1,timesteps,device)
    
    save_dir = './weights/'
    transform = get_transform(img_length)
    transform_layout=get_transform(int(img_length/8))
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

    model_VAE = VAE(device, height=img_length).to(device)
    model_VAE.load_state_dict(torch.load(
        f"{save_dir}/model_VAE_{199}.pth", map_location=device))
    encoder_VAE = model_VAE.encoder
    decoder_VAE = model_VAE.decoder

    latent_in_channels = 4
    latent_out_channels=latent_in_channels
    layout_in_channels=3
    n_feat = 64  # 64 hidden dimension feature
    n_cfeat = 5  # context vector is of size 5
    height = 32  # 16x16 image, don't forget to change transform_size in diffusion_utilities.py
    
    writer = SummaryWriter('runs')
    nn_model = Diffusion(latent_in_channels,layout_in_channels,n_feat,
                         n_cfeat,height).to(device)

    lrate = 1e-3
    
    
    for name, param in encoder_VAE.named_parameters():
        param.requires_grad = False
        
    for name, param in decoder_VAE.named_parameters():
        param.requires_grad = False
        
    encoder_VAE.eval()
    decoder_VAE.eval()
    nn_model.train()
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f"epoch {ep}")
        # linearly decay learning rate
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader, mininterval=2)
        tr_loss = 0
        for x, layout in pbar:  # x: images
            optim.zero_grad()
            x = x.to(device)
            b,c,h,w=x.shape
            layout = layout.to(device)
            encoder_noise = torch.randn(size=(b,4,int(height),int(height)),device=device)
            latent, mean, log_variance=encoder_VAE(x,encoder_noise)
            
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


    # load the encoder of VAE

    # encode the image to latent space

    # add noise to image

    # train on the latent space

    #

    pass


if __name__ == '__main__':
    main()

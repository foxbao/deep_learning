import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
from utils import *
from time import time
import os
import numpy as np

import cv2
import model_loader
from transformers import CLIPTokenizer
from diffusion import Diffusion
from ddpm import DDPMSampler
from VQVAE.model import VQVAE
from VQVAE.configs import get_cfg
from LDM.dataset import ParkFullDataset
from diffusion_utilities import save_layout_sample_gt


def train_generative_model(vqvae: VQVAE, model: Diffusion, dataset, device, ckpt_path):
    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02
    sampler = DDPMSampler(beta2, beta1, timesteps, device)

    n_epoch = 2000
    batch_size = 8
    lrate = 1e-3
    writer = SummaryWriter('runs')
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    vqvae.to(device)
    vqvae.eval()
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lrate)

    for param in vqvae.parameters():
        param.requires_grad = False

    for ep in range(n_epoch):
        print(f"epoch {ep}")
        # linearly decay learning rate
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader, mininterval=2)
        tr_loss = 0
        for batch_idx, (x, layout, prompt) in enumerate(pbar):  # x: images
            list_width = []
            for idx, p in enumerate(prompt):
                parts = p.split(':')
                width = parts[1]
                list_width.append(width)

            cond_context = None
            optim.zero_grad()
            x = x.to(device)
            layout = layout.to(device)
            with torch.no_grad():
                latent = vqvae.encode(x)

            # restore=vqvae.decode(latent)

            # # img=restore[0]
            # # img=img.permute(1, 2, 0)
            # # img = img * 255
            # # img = img.clip(0, 255)
            # # img = img.detach().cpu().numpy().astype(np.uint8)
            # # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # # cv2.imwrite(f'work_dirs/1111.jpg', img)
            noise = torch.randn_like(latent).to(device)

            t = torch.randint(1, timesteps + 1, (latent.shape[0],)).to(device)
            x_pert = sampler.perturb_input(latent, t, noise)

            # use network to recover noise
            pred_noise = model(
                latent=x_pert, layout=layout, context=cond_context, time=t/1.0)

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
            torch.save(model.state_dict(), save_dir + f"model_{ep}.pth")
            print("saved model at " + save_dir + f"model_{ep}.pth")


def sample_images(vqvae: VQVAE, model: Diffusion, device, latent_height, val_dataset):

    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02
    sampler = DDPMSampler(beta2, beta1, timesteps, device)

    val_batch_size = 4
    dataloader_val = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1
    )
    vqvae = vqvae.to(device)
    vqvae.eval()
    model = model.to(device)
    model.eval()

    for idx, (gt, layout, prompt) in enumerate(dataloader_val):
        list_width = []
        for _, p in enumerate(prompt):
            parts = p.split(':')
            width = parts[1]
            list_width.append(width)

    # Convert into a list of length Seq_Len=77
        # cond_tokens = tokenizer.batch_encode_plus(
        #     list_width, padding="max_length", max_length=77
        # ).input_ids
        # # (Batch_Size, Seq_Len)
        # cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        # # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        with torch.no_grad():
            # cond_context = clip(cond_tokens)
            cond_context = None
        gt = gt.to(device)
        layout = layout.to(device)
        samples, intermediate = sampler.sample_ddpm_context(
            layout.shape[0], model, vqvae.z_channels, latent_height, timesteps, layout, cond_context, device='cuda', save_rate=20)

        output = vqvae.decode(samples)

        # output = decoder_VAE(samples)
        save_layout_sample_gt(
            layouts=layout, samples=output, gts=gt, name=str(
                idx) + "_triple.jpg"
        )


def main():
    cfg = get_cfg(5)
    # diffusion hyperparameters

    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else torch.device('cpu'))
    img_shape = cfg['img_shape']
    # img_length = img_shape[0]
    latent_height = int(img_shape[1]/4)

    home_dir = os.path.expanduser('~')

    dataset = ParkFullDataset(
        img_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/parking_generate_data"),
        img_names="data/parking_generate_data/data.txt",
        layout_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/parking_layout_data"),
        layout_names="data/parking_layout_data/data.txt",
        text_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/parking_text_data"),
        text_names="data/parking_text_data/data.txt",
        use_layout=True,
        text_context=True
    )

    val_dataset = ParkFullDataset(
        img_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/val_parking_generate_data"),
        img_names="data/val_parking_generate_data/data.txt",
        layout_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/val_parking_layout_data"),
        layout_names="data/val_parking_layout_data/data.txt",
        text_dir=os.path.join(
            home_dir, "Downloads/parking2023/baojiali/park_generate/val_parking_text_data"),
        text_names="data/val_parking_text_data/data.txt",
        use_layout=True,
        text_context=True
    )

    vqvae = VQVAE(img_shape[0], cfg['dim'], cfg['n_embedding'])
    vqvae.load_state_dict(torch.load(cfg['vqvae_path']))

    latent_in_channels = 3
    layout_in_channels = 3
    n_feat = 64  # 64 hidden dimension feature
    n_cfeat = 5  # context vector is of size 5
    nn_model = Diffusion(latent_in_channels, layout_in_channels, n_feat,
                         n_cfeat, latent_height).to(device)

    # 3. Train Generative model
    train_generative_model(vqvae, nn_model, dataset,
                           device, ckpt_path=cfg['gen_model_path'])
    # 4. Sample VQVAE
    vqvae.load_state_dict(torch.load(cfg['vqvae_path']))
    nn_model.load_state_dict(torch.load(cfg['gen_model_path']))
    sample_images(vqvae, nn_model, device, latent_height, val_dataset)


if __name__ == '__main__':
    main()

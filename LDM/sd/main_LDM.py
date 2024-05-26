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
from diffusion_utilities import CustomDataset3, get_transform,save_layout_sample_gt
from VAE import VAE, VAE_Encoder, VAE_Decoder
import model_loader
from transformers import CLIPTokenizer
from diffusion import Diffusion
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
    img_length = 256
    latent_height=int(img_length/8)
    layout_length=latent_height
    batch_size = 8
    n_epoch = 2000

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
        text_dir=os.path.join(home_dir,"Downloads/parking2023/baojiali/park_generate/parking_text_data"),
        text_names="data/parking_text_data/data.txt",
        transform=transform,
        transform_layout=transform_layout,
        null_context=False,
        text_context=True
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
        text_dir=os.path.join(home_dir,"Downloads/parking2023/baojiali/park_generate/val_parking_text_data"),
        text_names="data/val_parking_text_data/data.txt",
        transform=transform,
        transform_layout=transform_layout,
        null_context=False,
        text_context=True
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
    img_in_channels=3
    n_feat = 64  # 64 hidden dimension feature
    n_cfeat = 5  # context vector is of size 5
    # 16x16 image, don't forget to change transform_size in diffusion_utilities.py

    writer = SummaryWriter('runs')
    nn_model = Diffusion(latent_in_channels, layout_in_channels, n_feat,
                         n_cfeat, latent_height).to(device)
    lrate = 1e-3
    
    tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
    model_file = "data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, device)
    # model_file = "../data/v1-5-pruned-emaonly.ckpt"
    # models = model_loader.preload_models_from_standard_weights(model_file, device)

    clip = models["clip"]
    clip.to(device)

    for name, param in encoder_VAE.named_parameters():
        param.requires_grad = False

    for name, param in decoder_VAE.named_parameters():
        param.requires_grad = False
    for name, param in clip.named_parameters():
        param.requires_grad = False

    encoder_VAE.eval()
    decoder_VAE.eval()
    clip.eval()
    nn_model.train()
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    latents = []
    train_mode=False
    if train_mode:
        for ep in range(n_epoch):
            print(f"epoch {ep}")
            # linearly decay learning rate
            optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

            pbar = tqdm(dataloader, mininterval=2)
            tr_loss = 0
            for x, layout,prompt in pbar:  # x: images
                list_width=[]
                for idx,p in enumerate(prompt):
                    parts = p.split(':')
                    width=parts[1]
                    list_width.append(width)
                    
            # Convert into a list of length Seq_Len=77
                cond_tokens = tokenizer.batch_encode_plus(
                    list_width, padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                with torch.no_grad():
                    cond_context = clip(cond_tokens)
                    
                    
                optim.zero_grad()
                x = x.to(device)
                b, c, h, w = x.shape
                layout = layout.to(device)
                # 注意，这里给encoder的noise要是0，以便固定encoded的值，否则训练不起来
                encoder_noise=torch.zeros(size=(b, 4, int(latent_height), int(latent_height)), device=device)
                with torch.no_grad():
                    latent, mean, log_variance=encoder_VAE(x,encoder_noise)
                # perturb data
                noise = torch.randn_like(latent)
                t = torch.randint(1, timesteps + 1, (latent.shape[0],)).to(device)
                x_pert = sampler.perturb_input(latent, t, noise)

                # use network to recover noise
                pred_noise = nn_model(
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
                torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
                print("saved model at " + save_dir + f"model_{ep}.pth")


    nn_model.load_state_dict(torch.load(
        f"{save_dir}/model_{190}.pth", map_location=device))
    nn_model.eval()
    print("Loaded in Model")

    # load the encoder of VAE

    for idx, (gt, layout,prompt) in enumerate(dataloader_val):
        list_width=[]
        for _,p in enumerate(prompt):
            parts = p.split(':')
            width=parts[1]
            list_width.append(width)
            
    # Convert into a list of length Seq_Len=77
        cond_tokens = tokenizer.batch_encode_plus(
            list_width, padding="max_length", max_length=77
        ).input_ids
        # (Batch_Size, Seq_Len)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        with torch.no_grad():
            cond_context = clip(cond_tokens)
        gt = gt.to(device)
        layout = layout.to(device)
        samples, intermediate =sampler.sample_ddpm_context(layout.shape[0], nn_model,latent_in_channels,latent_height,timesteps,layout,cond_context,device='cuda',save_rate=20)
        
        output = decoder_VAE(samples)
        save_layout_sample_gt(
            layouts=layout, samples=output, gts=gt, name=str(idx) + "_triple.jpg"
        )
    # encode the image to latent space

    # add noise to image

if __name__ == '__main__':
    main()

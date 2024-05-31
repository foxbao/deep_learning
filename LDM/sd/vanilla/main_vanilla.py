import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torch.utils.tensorboard import SummaryWriter
from utils import *
from tqdm import tqdm
from utils import *
from time import time
import os
from diffusion_utilities import CustomDataset3,get_transform
from torch.utils.data import DataLoader

from vanilla.vanilla_vae import VanillaVAE


def reconstruct(device, dataloader, model):
    model.eval()
    batch = next(iter(dataloader))
    x = batch[0].to(device)
    output = model(x)[0]
    output = output[0].detach().cpu()
    output=(output+1)/2
    # de-normalize
    # output=inverse_normalize(output)
    output=torch.clamp(output,0,1)

    input = batch[0][0].detach().cpu()
    input=(input+1)/2
    # de-normalize
    # input=inverse_normalize(input)
    # input=torch.clamp(input,0,1)

    combined = torch.cat((output, input), 1)
    img = ToPILImage()(combined)
    img.save('work_dirs/reconstruct.jpg')
    
def sample(device, model:VanillaVAE):
    model.eval()
    output = model.sample(num_samples=1,current_device=device)
    output = output[0].detach().cpu()
    output=(output+1)/2
    # output=inverse_normalize(output)
    output=torch.clamp(output,0,1)
    img = ToPILImage()(output)
    img.save('work_dirs/generate.jpg')

def train(device, dataloader, model: VanillaVAE,n_epoch=50):
    writer = SummaryWriter('runs')
    n_epoch = n_epoch
    lr = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr)
    begin_time = time()
    # train
    max_beta=10
    for i in range(n_epoch):
        optimizer.param_groups[0]["lr"] = lr * (1 - i / n_epoch)
        pbar = tqdm(dataloader, mininterval=2)
        tr_loss=0
        recons_loss=0
        kld_loss=0
        for x, layout,prompt in pbar:  # x: images
            x = x.to(device)
            results=model(x)
            beta=1+10*(i/n_epoch)
            train_loss = model.loss_function(*results,M_N=0.00025,beta=beta)
            loss=train_loss['loss']
            recon=train_loss['Reconstruction_Loss']
            kld=train_loss['KLD']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            kld_loss += kld.item()
            recons_loss += recon.item()
        epoch_loss = tr_loss / len(dataloader)
        kld_loss = kld_loss / len(dataloader)
        recons_loss = recons_loss / len(dataloader)
        
        writer.add_scalar("train/epoch_loss", epoch_loss, i)
        writer.add_scalar("train/kld_loss", kld_loss, i)
        writer.add_scalar("train/recons_loss", recons_loss, i)
        print("loss:", loss.item())
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f'epoch {i}: loss {epoch_loss} kl_loss {kld_loss} recon_loss {recons_loss} time {minute}:{second}')
        if i % 10 == 0 or i == int(n_epoch - 1):
            torch.save(model.state_dict(), f"weights/model_VAE_{i}.pth")
        if i % 10 == 0 or i == int(n_epoch - 1):
            torch.save(model.state_dict(), f"weights/model_VAE_{i}.pth")
            print("saved model at " + f"weights/model_VAE_{i}.pth")

def main():
    device = 'cuda:0'
    img_length=64
    batch_size=128
    in_channels=3
    latent_dim=512
    transform=get_transform(img_length)
    transform_layout=get_transform(int(img_length/8))
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
        use_layout=True,
        text_context=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=1)
    
    # dataloader = get_dataloader(root=os.path.join(current_work_dir,current_work_dir,'../data/parking_generate_data'),batch_size=batch_size,img_shape=(img_length,img_length))

    model = VanillaVAE(in_channels,latent_dim,img_length).to(device)
    n_epoch=500
    # model.load_state_dict(torch.load(
    #     f"weights/model_VAE_{330}.pth", map_location=device))
    train(device, dataloader, model,n_epoch)
    model.load_state_dict(torch.load(
        f"weights/model_VAE_{n_epoch-1}.pth", map_location=device))
    reconstruct(device, dataloader, model)
    sample(device, model)



if __name__ == '__main__':
    main()
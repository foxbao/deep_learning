from VAE import VAE
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torch.utils.tensorboard import SummaryWriter
from load_celebA import get_dataloader
from VAE import VAE
from utils import *
from tqdm import tqdm
from utils import *
from time import time
import os
from diffusion_utilities import CustomDataset3,get_transform
from torch.utils.data import DataLoader

def loss_fn(y, y_hat, mean, logvar,beta=1):
    kl_weight = 0.00025
    # print(mean.shape)
    # print(logvar.shape)
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), (1,2,3)), 0)
    loss = recons_loss + kl_loss * kl_weight*beta
    
    return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':kl_loss.detach()}

def train(device, dataloader, model: VAE,n_epoch=50):
    writer = SummaryWriter('runs')
    n_epoch = n_epoch
    lr = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr)
    begin_time = time()
    # train
    for i in range(n_epoch):
        optimizer.param_groups[0]["lr"] = lr * (1 - i / n_epoch)
        pbar = tqdm(dataloader, mininterval=2)
        tr_loss=0
        recons_loss=0
        kld_loss=0
        beta_max=40
        for x, layout,prompt in pbar:  # x: images
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            beta=1+beta_max*(i/n_epoch)
            train_loss = loss_fn(x, y_hat, mean, logvar,beta)
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
            print("saved model at " + f"weights/model_VAE_{i}.pth")

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


def generate(device, model:VAE):
    model.eval()
    output = model.sample(device)
    output = output[0].detach().cpu()
    output=(output+1)/2
    # output=inverse_normalize(output)
    output=torch.clamp(output,0,1)
    img = ToPILImage()(output)
    img.save('work_dirs/generate.jpg')

def main():
    device = 'cuda:0'
    img_length=256
    batch_size=4
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

    model = VAE(device,height=img_length).to(device)
    n_epoch=500
    # train(device, dataloader, model,n_epoch)
    model.load_state_dict(torch.load(
        f"weights/model_VAE_{250}.pth", map_location=device))
    reconstruct(device, dataloader, model)
    generate(device, model)

if __name__ == '__main__':
    main()
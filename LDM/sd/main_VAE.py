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

def loss_fn(y, y_hat, mean, logvar):
    kl_weight = 0.00025
    # print(mean.shape)
    # print(logvar.shape)
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), (1,2,3)), 0)
    loss = recons_loss + kl_loss * kl_weight
    return loss

def train(device, dataloader, model: VAE,n_epoch=50):
    writer = SummaryWriter('runs')
    n_epoch = n_epoch
    lr = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr)
    begin_time = time()
    # train
    for i in range(n_epoch):
        pbar = tqdm(dataloader, mininterval=2)
        tr_loss=0
        for x, layout in pbar:  # x: images
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        epoch_loss = tr_loss / len(dataloader)
        writer.add_scalar("Loss/train", epoch_loss, i)
        print("loss:", loss.item())
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f'epoch {i}: loss {epoch_loss} {minute}:{second}')
        if i % 10 == 0 or i == int(n_epoch - 1):
            torch.save(model.state_dict(), f"model_{i}.pth")

def reconstruct(device, dataloader, model):
    model.eval()
    batch = next(iter(dataloader))
    x = batch[0:1, ...].to(device)
    output = model(x)[0]
    output = output[0].detach().cpu()
    output=(output+1)/2
    # de-normalize
    # output=inverse_normalize(output)
    output=torch.clamp(output,0,1)

    input = batch[0].detach().cpu()
    input=(input+1)/2
    # de-normalize
    # input=inverse_normalize(input)
    # input=torch.clamp(input,0,1)

    combined = torch.cat((output, input), 1)
    img = ToPILImage()(combined)
    img.save('work_dirs/reconstruct.jpg')


def generate(device, model):
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
    img_length=128
    batch_size=32
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
        transform=transform,
        transform_layout=transform_layout,
        null_context=False,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=1)
    
    # dataloader = get_dataloader(root=os.path.join(current_work_dir,current_work_dir,'../data/parking_generate_data'),batch_size=batch_size,img_shape=(img_length,img_length))

    model = VAE(device,height=img_length).to(device)
    n_epoch=150
    train(device, dataloader, model,n_epoch)
    model.load_state_dict(torch.load(
        f"model_{n_epoch-1}.pth", map_location=device))
    reconstruct(device, dataloader, model)
    generate(device, model)

if __name__ == '__main__':
    main()
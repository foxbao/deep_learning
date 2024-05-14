from time import time

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from load_celebA import get_dataloader
from model import VAE
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
n_epochs = 10
kl_weight = 0.00025
lr = 0.005
writer = SummaryWriter('runs')

def loss_fn(y, y_hat, mean, logvar):
    # print(mean.shape)
    # print(logvar.shape)
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), dim=(1,2,3)), 0)
    
    # aaaa=-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), dim=(1,2,3)), 0
    loss = recons_loss + kl_loss * kl_weight
    return loss


def train(device:str, dataloader:DataLoader, model:VAE):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset_len = len(dataloader.dataset)

    begin_time = time()
    # train
    for ep in range(n_epochs):
        loss_sum = 0
        pbar = tqdm(dataloader, mininterval=2)
        for x in pbar:  # x: images
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        loss_sum /= dataset_len
        writer.add_scalar("Loss/train", loss_sum.item(), ep)
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f'epoch {ep}: loss {loss_sum} {minute}:{second}')
        torch.save(model.encoder.state_dict(),'encoder.pth')
        torch.save(model.decoder.state_dict(),'decoder.pth')
        torch.save(model.state_dict(), 'model.pth')
        if ep % 10 == 0 or ep == int(n_epochs - 1):
            torch.save(model.state_dict(), f"model_{ep}.pth")
            print("saved model at " + f"model_{ep}.pth")
            torch.save(model.state_dict(), 'model.pth')



def reconstruct(device, dataloader, model):
    model.eval()
    batch = next(iter(dataloader))
    x = batch[0:1, ...].to(device)

    output = model(x)[0]
    output = output[0].detach().cpu()
    # denormalize
    # output=inverse_normalize(output)
    # output=torch.clamp(output,0,1)

    # denormalize
    input = batch[0].detach().cpu()
    # input=inverse_normalize(input)
    # input=torch.clamp(input,0,1)

    combined = torch.cat((output, input), 1)
    img = ToPILImage()(combined)
    img.save('work_dirs/reconstruct.jpg')


def generate(device, model:VAE):
    model.eval()
    output = model.sample(device)
    output = output[0].detach().cpu()
    # output=inverse_normalize(output)
    # output=torch.clamp(output,0,1)
    img = ToPILImage()(output)
    img.save('work_dirs/generate.jpg')


def main():
    device = 'cuda:0'
    batch_size=50
    img_length=64
    # dataloader = get_dataloader("data/parking_generate_data",batch_size=batch_size,img_shape=(img_length,img_length))
    dataloader = get_dataloader(root='data/celebA/img_align_celeba',batch_size=batch_size,img_shape=(img_length,img_length))

    model = VAE().to(device)

    # If you obtain the ckpt, load it
    # model.load_state_dict(torch.load('model.pth', 'cuda:0'))

    # Choose the function
    train(device, dataloader, model)
    reconstruct(device, dataloader, model)
    generate(device, model)


if __name__ == '__main__':
    main()
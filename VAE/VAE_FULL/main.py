from model import VAE
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torch.utils.tensorboard import SummaryWriter
from load_celebA import get_dataloader
from model import VAE
from utils import *
from tqdm import tqdm
from utils import *
from time import time

writer = SummaryWriter('runs')
n_epochs = 10
kl_weight = 0.00025
lr = 0.005
def loss_fn(y, y_hat, mean, logvar):
    # print(mean.shape)
    # print(logvar.shape)
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), (1,2,3)), 0)
    loss = recons_loss + kl_loss * kl_weight
    return loss

def train(device, dataloader, model: VAE):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    begin_time = time()
    # train
    for i in range(n_epochs):
        pbar = tqdm(dataloader, mininterval=2)
        tr_loss=0
        for x in pbar:  # x: images
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
        torch.save(model.state_dict(), 'model.pth')

def main():
    device = 'cuda:0'
    img_length=64
    batch_size=50
    # dataloader = get_dataloader(root='data/parking_generate_data',batch_size=100,img_shape=(img_length,img_length))
    dataloader = get_dataloader(root='/home/baojiali/Downloads/deep_learning/VAE/data/celebA/img_align_celeba',batch_size=batch_size,img_shape=(img_length,img_length))
    model = VAE(device).to(device)
    train(device, dataloader, model)
    aaa=1

if __name__ == '__main__':
    main()
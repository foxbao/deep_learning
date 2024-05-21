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
import os





def loss_fn(y, y_hat, mean, logvar):
    kl_weight = 0.00025
    # print(mean.shape)
    # print(logvar.shape)
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), (1,2,3)), 0)
    loss = recons_loss + kl_loss * kl_weight
    return loss

def train(device, dataloader, model: VAE):
    writer = SummaryWriter('runs')
    n_epochs = 10
    lr = 0.005
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
    img_length=64
    batch_size=10
    # dataloader = get_dataloader(root='data/parking_generate_data',batch_size=100,img_shape=(img_length,img_length))
    current_work_dir = os.path.dirname(__file__)# 当前文件所在的目录
    dataloader = get_dataloader(root=os.path.join(current_work_dir,current_work_dir,'../data/celebA/img_align_celeba'),batch_size=batch_size,img_shape=(img_length,img_length))
    model = VAE(device,height=img_length).to(device)
    
    train(device, dataloader, model)
    model.load_state_dict(torch.load(
        f"model.pth", map_location=device))
    reconstruct(device, dataloader, model)
    generate(device, model)

if __name__ == '__main__':
    main()
import argparse
import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from VQVAE.configs import get_cfg
from VQVAE.dataset import get_dataloader
from VQVAE.model import VQVAE
# from VQVAE.pixelcnn_model import PixelCNNWithEmbedding
from tqdm import tqdm
USE_LMDB = False


def train_vqvae(model: VQVAE,
                img_shape=None,
                device='cuda',
                ckpt_path='VQVAE/model.pth',
                batch_size=64,
                dataset_type='MNIST',
                lr=1e-3,
                n_epochs=100,
                l_w_embedding=1,
                l_w_commitment=0.25):
    print('batch size:', batch_size)
    dataloader = get_dataloader(dataset_type,
                                batch_size,
                                img_shape=img_shape,
                                use_lmdb=USE_LMDB)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, mininterval=2)
        for x in pbar:
            current_batch_size = x.shape[0]
            x = x.to(device)

            x_hat, ze, zq = model(x)
            l_reconstruct = mse_loss(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)
            l_commitment = mse_loss(ze, zq.detach())
            loss = l_reconstruct + \
                l_w_embedding * l_embedding + l_w_commitment * l_commitment
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def train_generative_model(vqvae: VQVAE,
                           model,
                           img_shape=None,
                           device='cuda',
                           ckpt_path='dldemos/VQVAE/gen_model.pth',
                           dataset_type='MNIST',
                           batch_size=64,
                           n_epochs=50):
    print('batch size:', batch_size)
    dataloader = get_dataloader(dataset_type,
                                batch_size,
                                img_shape=img_shape,
                                use_lmdb=USE_LMDB)
    vqvae.to(device)
    vqvae.eval()
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, mininterval=2)
        for x in pbar:
            current_batch_size = x.shape[0]
            with torch.no_grad():
                x = x.to(device)
                x = vqvae.encode(x)

            predict_x = model(x)
            loss = loss_fn(predict_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def reconstruct(model, x, device, dataset_type='MNIST'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)
    
    n = x.shape[0]
    n1 = int(n**0.5)
    x_cat = torch.concat((x, x_hat), 3)
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    x_cat = ((x_cat.clip(-1, 1)+1)/2 * 255).cpu().numpy().astype(np.uint8)
    if dataset_type == 'CelebA' or dataset_type == 'CelebAHQ' or dataset_type == 'Park':
        x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'work_dirs/vqvae_reconstruct_{dataset_type}.jpg', x_cat)

if __name__ == '__main__':

    os.makedirs('work_dirs', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=5)
    parser.add_argument('-d', type=int, default=0)
    args = parser.parse_args()
    cfg = get_cfg(args.c)

    device = f'cuda:{args.d}'

    img_shape = cfg['img_shape']

    vqvae = VQVAE(img_shape[0], cfg['dim'], cfg['n_embedding'],cfg['z_channels'])

    # 1. Train VQVAE
    train_vqvae(vqvae,
                img_shape=(img_shape[1], img_shape[2]),
                device=device,
                ckpt_path=cfg['vqvae_path'],
                batch_size=cfg['batch_size'],
                dataset_type=cfg['dataset_type'],
                lr=cfg['lr'],
                n_epochs=cfg['n_epochs'],
                l_w_embedding=cfg['l_w_embedding'],
                l_w_commitment=cfg['l_w_commitment'])

    # 2. Test VQVAE by visualizaing reconstruction result
    vqvae.load_state_dict(torch.load(cfg['vqvae_path']))
    dataloader = get_dataloader(cfg['dataset_type'],
                                16,
                                img_shape=(img_shape[1], img_shape[2]))
    img = next(iter(dataloader)).to(device)
    reconstruct(vqvae, img, device, cfg['dataset_type'])

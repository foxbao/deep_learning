import torch
import torch.nn as nn
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from torchvision import transforms
from utils import *

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder=VAE_Encoder()
        self.decoder=VAE_Decoder()
    
    def forward(self, x):
        
        x,mean,log_variance=self.encoder(x)

        x=self.decoder(x)
        return x,mean,log_variance


    def sample(self, device='cuda'):
        z=torch.randn(1,4,8,8).to(device)
        z*= 0.18215
        x=self.decoder(z)
        # output = x[0].detach().cpu()
        return x
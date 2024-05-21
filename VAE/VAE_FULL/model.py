import torch
import torch.nn as nn
from encoder import *
from decoder import *

class VAE(nn.Module):
    def __init__(self,device):
        super(VAE, self).__init__()
        self.encoder=VAE_Encoder()
        self.decoder=VAE_Decoder()
        self.device=device
        
    def forward(self,x:torch.tensor):
        b,c,h,w=x.shape
        encoder_noise = torch.randn(size=(b,4,8,8),device=self.device)
        encoded, mean, log_variance = self.encoder(x,encoder_noise)
        
        decoded=self.decoder(encoded)
        return decoded,mean, log_variance
        
        
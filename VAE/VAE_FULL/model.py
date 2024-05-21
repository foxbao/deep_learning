import torch
import torch.nn as nn
from encoder import *
from decoder import *

class VAE(nn.Module):
    def __init__(self,device,height):
        super(VAE, self).__init__()
        self.encoder=VAE_Encoder()
        self.decoder=VAE_Decoder()
        self.device=device
        self.height=height
        self.latent_dim=int(height/8)
        
    def forward(self,x:torch.tensor):
        b,c,h,w=x.shape
        # VAE compress the input (b,c,h,w)->(b,4,h/8,w/8) latent
        encoder_noise = torch.randn(size=(b,4,self.latent_dim,self.latent_dim),device=self.device)
        encoded, mean, log_variance = self.encoder(x,encoder_noise)
        
        decoded=self.decoder(encoded)
        return decoded,mean, log_variance
    
    def sample(self, device='cuda'):
        # z = torch.randn(1, self.latent_dim).to(device)
        # x = self.decoder_projection(z)
        # x = torch.reshape(x, (-1, *self.decoder_input_chw))
        z=torch.randn(1,self.latent_dim,self.height/8,self.height/8)
        decoded = self.decoder(x)
        return decoded
        
        
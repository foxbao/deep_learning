import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp


class VQVAE(nn.Module):

    def __init__(self, input_dim, dim, n_embedding,z_channels=3):
        super().__init__()
        self.z_channels = z_channels
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
                                     ResidualBlock(dim), ResidualBlock(dim))
        self.encoder_conv_out = nn.Conv2d(dim, self.z_channels, kernel_size=3, padding=1)
        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        
        self.vq_embedding = nn.Embedding(n_embedding, self.z_channels)
        # self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)
        
        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, dim, kernel_size=3, padding=(1, 1))
        
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))
        self.n_downsample = 2

    def forward(self, x):
        # encode
        ze = self.encoder(x)
        
        ze=self.encoder_conv_out(ze)
        ze=self.pre_quant_conv(ze)

        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        # stop gradient
        decoder_input = ze + (zq - ze).detach()


        # decode
        decoder_input = self.post_quant_conv(decoder_input)
        decoder_input = self.decoder_conv_in(decoder_input)
        x_hat = self.decoder(decoder_input)
        return x_hat, ze, zq

    @torch.no_grad()
    def encode(self, x):
        ze = self.encoder(x)
        ze=self.encoder_conv_out(ze)
        ze=self.pre_quant_conv(ze)

        embedding = self.vq_embedding.weight.data

        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        return zq

    @torch.no_grad()
    def decode(self, zq):
        # zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        decoder_input = self.post_quant_conv(zq)
        decoder_input = self.decoder_conv_in(decoder_input)
        x_hat = self.decoder(decoder_input)
        return x_hat

    # Shape: [C, H, W]
    def get_latent_HW(self, input_shape):
        C, H, W = input_shape
        return (H // 2**self.n_downsample, W // 2**self.n_downsample)

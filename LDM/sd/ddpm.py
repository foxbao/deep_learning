import torch
import torch.nn.functional as F
import numpy as np


class DDPMSampler:
    def __init__(self, beta2, beta1, timesteps, device):
        # construct DDPM noise schedule
        self.b_t = (beta2 - beta1) * torch.linspace(0, 1,
                                                    timesteps + 1, device=device) + beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()
        self.ab_t[0] = 1

    def perturb_input(self, x, t, noise):
        return self.ab_t.sqrt()[t, None, None, None] * x + (1 - self.ab_t[t, None, None, None]) * noise

    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)

    def denoise_add_noise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) /
                (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        return mean + noise

    @torch.no_grad()
    def sample_ddpm_context(self, n_sample, nn_model, in_channels, height, timesteps, layout,context=None,LDM=False,device='cuda', save_rate=20):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, in_channels, height, height).to(device)
        # if LDM:
        #     samples*=0.18215*0.1
        # array to keep track of generated steps for plotting
        intermediate = []
        for i in range(timesteps, 0, -1):
            print(f"sampling timestep {i:3d}", end="\r")

            # reshape time tensor
            # t = torch.tensor([i])[:, None, None, None].to(device)
            t = torch.tensor([i/1.0]).to(device)
            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            # predict noise e_(x_t,t, ctx)
            eps = nn_model(samples, layout=layout,context=context, time=t)
            samples = self.denoise_add_noise(samples, i, eps, z)
            if i % save_rate == 0 or i == timesteps or i < 8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate

    @torch.no_grad()
    def sample_ddpm(self, n_sample, nn_model, in_channels, height, timesteps, device='cuda', save_rate=20):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, in_channels, height, height).to(device)

        # array to keep track of generated steps for plotting
        intermediate = []
        for i in range(timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i])[:, None, None, None].to(device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps = nn_model(samples, t)    # predict noise e_(x_t,t)
            samples = self.denoise_add_noise(samples, i, eps, z)
            if i % save_rate == 0 or i == timesteps or i < 8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate

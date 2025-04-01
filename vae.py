import torch
import numpy as np
import torch.nn as nn

class basic_VAE(nn.Module):
    def __init__(self, input_dim=1531, hidden_dim=500, latent_dim=200):
        super(basic_VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, 1)
            
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LeakyReLU(),
            )
     

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar


    def reparameterization(self, mean, logvar):
        epsilon = torch.randn_like(mean)     
        z = mean + logvar*epsilon
        return z


    def decode(self, x):
        return self.decoder(x)


    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    

def vae_loss(x, x_hat, mean, logvar):
    MSECriterion = nn.MSELoss(reduction='sum')
    d = x_hat.shape[1]
    var_dec = 1

    reconstruction_term = -MSECriterion(x, x_hat) / (2*var_dec)
    KLD = 0.5 * torch.sum((logvar**2)*d - d + torch.linalg.norm(mean)**2 - 2*logvar)
    ELBO = reconstruction_term - KLD

    return -ELBO
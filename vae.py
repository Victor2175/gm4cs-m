import torch
import torch.nn as nn

class basic_VAE(nn.Module):
    def __init__(self, input_dim=1531, hidden_dim=500, latent_dim=200):
        super(basic_VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

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
    

    def loss(self, x_hat, y, mean, logvar):
        MSECriterion = nn.MSELoss(reduction='sum')
        d = self.latent_dim
        var_dec = 1

        reconstruction_term = -MSECriterion(x_hat, y) / (2*var_dec)
        KLD = 0.5 * torch.sum((logvar**2)*d - d + torch.linalg.norm(mean)**2 - 2*d*logvar)
        ELBO = reconstruction_term - KLD

        return -ELBO
    

class VAE(nn.Module):
    def __init__(self, input_dim=44472, hidden_dim=500, latent_dim=100):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, 1)
            
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LeakyReLU()
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
    

    def loss(self, x, x_hat, mean, logvar):
        MSECriterion = nn.MSELoss(reduction='sum')
        d = self.latent_dim
        var_dec = 1

        reconstruction_term = -MSECriterion(x, x_hat) / (2*var_dec)
        KLD = 0.5 * torch.sum((logvar**2)*d - d + torch.linalg.norm(mean)**2 - 2*d*logvar)
        ELBO = reconstruction_term - KLD

        return -ELBO


    def sample(self, x, n_samples):
        samples = []
        mean, logvar = self.encode(x)
        for i in range(n_samples):
            z = self.reparameterization(mean, logvar)
            sample = self.decode(z)
            samples.append(sample)

        return samples
    

class CVAE(nn.Module):
    def __init__(self, mask, in_channels=34, hidden_dims=None, latent_dim=100):
        super(CVAE, self).__init__()
        self.mask = mask

        if hidden_dims is None:
            self.hidden_dims = [in_channels, 68, 136]
        else:
            self.hidden_dims.prepend(in_channels)
        self.latent_dim = latent_dim

        modules = []
        for i in range(len(self.hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i + 1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            ))
        
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(*modules)

        dummy = torch.randn(1, in_channels, 30, 72)
        with torch.no_grad():
            encoded_dummy = self.encoder(dummy)
            flattened_dim = encoded_dummy.numel()
            encoded_shape = encoded_dummy.shape[1:]
            print('encoded shape :', encoded_shape)
            print('flattened encoded :', flattened_dim)
        
        self.mean_layer = nn.Linear(flattened_dim, latent_dim)
        self.logvar_layer = nn.Linear(flattened_dim, 1)
             
        modules = []
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dims[i], self.hidden_dims[i + 1], kernel_size=3, stride=1, padding=1, output_padding=0),
                nn.LeakyReLU()
            ))

        self.decoder_input = nn.Linear(latent_dim, flattened_dim)
        self.unflatten = nn.Unflatten(1, encoded_shape)
        self.decoder = nn.Sequential(*modules)
        
        with torch.no_grad():
            decoded_dummy = self.decoder(encoded_dummy)
            decoded_shape = decoded_dummy.shape[1:]
            print('decoded shape :', decoded_shape)


    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar
    

    def reparameterization(self, mean, logvar):
        epsilon = torch.randn_like(mean)     
        z = mean + logvar*epsilon
        return z
    

    def decode(self, x):
        x = self.decoder_input(x)
        x = self.unflatten(x)
        x = self.decoder(x)[:, :, :30, :72]
        return x
    

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    

    def loss(self, x, x_hat, mean, logvar):
        MSECriterion = nn.MSELoss(reduction='sum')
        d = self.latent_dim
        var_dec = 1

        #x_cells = x[:, :, ~self.mask]
        #x_hat_cells = x_hat[:, :, ~self.mask]

        #reconstruction_term = -MSECriterion(x_cells, x_hat_cells) / (2*var_dec)
        #print('x shape:', x.shape)
        #print('x_hat shape:', x_hat.shape)
        reconstruction_term = -MSECriterion(x, x_hat) / (2*var_dec)
        KLD = 0.5 * torch.sum((logvar**2)*d - d + torch.linalg.norm(mean)**2 - 2*d*logvar)
        ELBO = reconstruction_term - KLD

        return -ELBO
        
import pickle as pkl
import numpy as np
import torch
from plots import *
from utils import *
from vae import *
import sys
import json

if __name__ == '__main__':
    print('Trying to read the data...')
    sys.stdout.flush()

    with open('/mydata/cope/mirco/data/ssp585_time_series.pkl', 'rb') as f:
        cope_data = pkl.load(f)

    print('Got it !')
    sys.stdout.flush()

    # Preprocess cope_data
    print('Preprocessing...')
    sys.stdout.flush()
    prune(cope_data, min_runs=5)
    cut_lat(cope_data, max_lat=60)
    downscale(cope_data)

    union_nan_mask = find_union_nan_mask(cope_data)
    inpute(cope_data, union_nan_mask)
    print('Done !\n')
    sys.stdout.flush()

    # Specify parameters
    epochs = 30
    lrs = [1e-3, 1.5e-3, 2e-3]
    hidden_dims = [500, 1000, 1500]
    latent_dims = [100, 200, 300]
    configurations = len(lrs)*len(hidden_dims)*len(latent_dims)

    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    sys.stdout.flush()

    ith = 1

    print('Begin cross validation...\n')
    sys.stdout.flush()

    all_mse_losses = {}
    all_nmse_losses = {}
    for lr in lrs:
        for hidden_dim in hidden_dims:
            for latent_dim in latent_dims:
                model = VAE(input_dim=44472, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
                mse_losses, nmse_losses = VAE_LOOCV(cope_data, union_nan_mask, model, epochs, batch_size, lr)
                sys.stdout.flush()

                all_mse_losses[f'vae_{lr}_{epochs}'] = mse_losses
                all_nmse_losses[f'vae_{lr}_{epochs}'] = nmse_losses
                
                print(f'Finished configuration {ith}/{configurations}')
                sys.stdout.flush()

                ith += 1

    print('\nFinished training !\n')
    sys.stdout.flush()

    with open('training_vae_mse_losses_05_05.txt', 'w') as f: 
        f.write(json.dumps(all_mse_losses, sort_keys=True, indent=2))

    with open('training_vae_nmse_losses_05_05.txt', 'w') as f:
        f.write(json.dumps(all_nmse_losses, sort_keys=True, indent=2))
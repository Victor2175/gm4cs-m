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

    #with open('/mydata/cope/mirco/data/ssp585_time_series.pkl', 'rb') as f:
    #    cope_data = pkl.load(f)
    with open('cope_data/ssp585_time_series.pkl', 'rb') as f:
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
    lr = 1e-3
    batch_size = 16
    epochs = 30
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    sys.stdout.flush()

    model = VAE(input_dim=44472, hidden_dim=500, latent_dim=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    ith = 1

    print('Begin cross validation...\n')
    sys.stdout.flush()

    losses = VAE_LOOCV(cope_data, union_nan_mask, model, epochs, batch_size, lr, verbose=True)

    print(f'Finished !')
    sys.stdout.flush()

    ith += 1

    print('\nFinished training !\n')
    sys.stdout.flush()

    with open('training_vae_losses.txt', 'w') as f: 
        f.write(json.dumps(losses, indent=2))
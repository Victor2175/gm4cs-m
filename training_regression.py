import pickle as pkl
import numpy as np
import torch
from plots import *
from utils import *
from regression import Ridge
import sys
import json

if __name__ == '__main__':
    with open('/mydata/cope/mirco/data/ssp585_time_series.pkl', 'rb') as f:
        cope_data = pkl.load(f)

    # Preprocess cope_data
    print('Preprocessing...')
    sys.stdout.flush()
    prune(cope_data, min_runs=5)
    downscale(cope_data)

    union_nan_mask = find_union_nan_mask(cope_data, downscale=True)
    inpute(cope_data, union_nan_mask)
    print('Done !\n')
    sys.stdout.flush()

    # Specify grid search parameters
    lambdas = np.logspace(-2, 2)
    ranks = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, None]
    best_loss = 1e6
    best_model = 'hein'

    print('Training...\n')
    sys.stdout.flush()
    results = {}
    for lambda_ in lambdas:
        for rank in ranks:
            machine_model = Ridge(lambda_=lambda_, rank=rank)
            losses = LOOCV(cope_data, union_nan_mask, machine_model, verbose=True)
            mean_loss = losses['MEAN']

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_model = f'ridge_{lambda_}_{rank}'

            print(f'Mean loss of model ridge_{lambda_}_{rank} : {mean_loss}')
            sys.stdout.flush()
            results[f'ridge_{lambda_}_{rank}'] = mean_loss

    results['best_model'] = best_model
    results['best_loss'] = best_loss
    print(f'Best model : {best_model}')
    print(f'Best loss : {best_loss}')
    sys.stdout.flush()

    with open('training_regression_results.txt', 'w') as f: 
        f.write(json.dumps(results, sort_keys=True, indent=2))
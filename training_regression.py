import pickle as pkl
import numpy as np
from plots import *
from utils import *
from regression import Ridge
from datetime import date
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

    # Specify grid search parameters
    lambdas = np.logspace(-1, 5, 20)
    ranks = [10, 50, 100, 300, 500, 700, None]
    configurations = len(lambdas)*len(ranks)
    ith = 1

    print('Training...\n')
    sys.stdout.flush()

    all_mse_losses = {}
    all_nmse_losses = {}
    for lambda_ in lambdas:
        for rank in ranks:
            machine_model = Ridge(lambda_=lambda_, rank=rank)
            mse_losses, nmse_losses = LOOCV(cope_data, union_nan_mask, machine_model)

            if rank is None:
                rank = np.linalg.matrix_rank(machine_model.W)

            all_mse_losses[f'ridge_{lambda_}_{rank}'] = mse_losses
            all_nmse_losses[f'ridge_{lambda_}_{rank}'] = nmse_losses

            print(f'Finished configuration {ith}/{configurations}')
            sys.stdout.flush()

            ith += 1

    print('\nFinished training !\n')
    sys.stdout.flush()

    training_date = date.today().strftime('%d_%m_%Y')
    with open(f'training_results/training_regression_mse_losses_{training_date}.txt', 'w') as f: 
        f.write(json.dumps(all_mse_losses, sort_keys=True, indent=2))

    with open(f'training_results/training_regression_nmse_losses_{training_date}.txt', 'w') as f:
        f.write(json.dumps(all_nmse_losses, sort_keys=True, indent=2))
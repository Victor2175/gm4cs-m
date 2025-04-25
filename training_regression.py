import pickle as pkl
import numpy as np
from plots import *
from utils import *
from regression import Ridge
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
    #best_mean_loss, best_median_loss = 1e6, 1e6
    #best_mean_model, best_median_model = 'hein', 'hein'
    configurations = len(lambdas)*len(ranks)
    ith = 1

    print('Training...\n')
    sys.stdout.flush()

    all_losses = {}
    for lambda_ in lambdas:
        for rank in ranks:
            machine_model = Ridge(lambda_=lambda_, rank=rank)
            losses = LOOCV(cope_data, union_nan_mask, machine_model)

            if rank is None:
                rank = np.linalg.matrix_rank(machine_model.W)

            all_losses[f'ridge_{lambda_}_{rank}'] = losses

            print(f'Finished configuration {ith}/{configurations}')
            sys.stdout.flush()

            ith += 1

    print('\nFinished training !\n')
    sys.stdout.flush()

    with open('training_regression_losses_21_04.txt', 'w') as f: 
        f.write(json.dumps(all_losses, sort_keys=True, indent=2))
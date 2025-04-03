import pickle as pkl
import numpy as np
import torch
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
    lambdas = np.logspace(-2, 3, 30)
    ranks = [10, 50, 100, 200, 400, 600, 800, None]
    best_mean_loss, best_median_loss = 1e6, 1e6
    best_mean_model, best_median_model = 'hein', 'hein'
    configurations = len(lambdas)*len(ranks)
    ith = 1

    print('Training...\n')
    sys.stdout.flush()

    results = {}
    for lambda_ in lambdas:
        for rank in ranks:
            machine_model = Ridge(lambda_=lambda_, rank=rank)
            losses = LOOCV(cope_data, union_nan_mask, machine_model)

            if rank is None:
                rank = np.linalg.matrix_rank(machine_model.W)

            mean_loss = sum(losses.values()) / len(losses)
            quantiles = np.quantile(np.array(list(losses.values())), [0.25, 0.5, 0.75])
            median_loss = quantiles[1]

            if mean_loss < best_mean_loss:
                best_mean_loss = mean_loss
                best_mean_model = f'ridge_{lambda_}_{rank}'

            if median_loss < best_median_loss:
                best_median_loss = median_loss
                best_median_model = f'ridge_{lambda_}_{rank}'

            results[f'ridge_{lambda_}_{rank}'] = {'mean_loss': mean_loss, 'Q1': quantiles[0], 'median': median_loss, 'Q3': quantiles[2]}

            print(f'Finished configuration {ith}/{configurations}')
            sys.stdout.flush()

            ith += 1

    print('\nFinished training !\n')
    sys.stdout.flush()

    results['best_mean_model'] = best_mean_model
    results['best_mean_loss'] = best_mean_loss

    print(f'Best mean model : {best_mean_model}')
    print(f'Best mean loss : {best_mean_loss}\n')

    results['best_median_model'] = best_median_model
    results['best_median_loss'] = best_median_loss

    print(f'Best median model : {best_median_model}')
    print(f'Best median loss : {best_median_loss}\n')
    sys.stdout.flush()

    with open('training_regression_results_adv.txt', 'w') as f: 
        f.write(json.dumps(results, sort_keys=True, indent=2))
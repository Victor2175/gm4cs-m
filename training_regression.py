import pickle as pkl
import numpy as np
import torch
from plots import *
from utils import *
from regression import Ridge
import json

if __name__ == '__main__':
    with open('/mydata/cope/mirco/data/ssp585_time_series.pkl', 'rb') as f:
        cope_data = pkl.load(f)

    # Preprocess cope_data
    prune(cope_data, min_runs=5)
    downscale(cope_data)

    union_nan_mask = find_union_nan_mask(cope_data, downscale=True)
    inpute(cope_data, union_nan_mask)

    # Specify grid search parameters
    alphas = np.logspace(-2, 2)
    ranks = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    best_loss = 1e6
    best_model = 'hein'

    results = {}
    for alpha in alphas:
        for rank in ranks:
            machine_model = Ridge(alpha=alpha, rank=rank)
            losses = LOOCV(cope_data, union_nan_mask, machine_model)
            mean_loss = losses['MEAN']

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_model = f'ridge_{alpha}_{rank}'

            print(f'Mean loss of model ridge_{alpha}_{rank} : {mean_loss}')
            results[f'ridge_{alpha}_{rank}'] = mean_loss

    results['best_model'] = best_model
    results['best_loss'] = best_loss

    with open('training_regression_results.txt', 'w') as f: 
        f.write(json.dumps(results, sort_keys=True, indent=2))
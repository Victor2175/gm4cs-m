import torch
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
from utils import *

class CopeDataset(Dataset):
    def __init__(self, data_path, downscale=True):
        with open(data_path, 'rb') as f:
            cope_data = pkl.load(f)

        prune(cope_data, min_runs=5)
        if downscale:
            downscale(cope_data)

        union_nan_mask = find_union_nan_mask(cope_data, downscale=downscale)
        inpute(cope_data, union_nan_mask)

        simulations = []
        
        for model in cope_data.keys():
            data, forced_responses, mean_grids, std_grid_timeseries = normalize_flatten_model(cope_data, model, union_nan_mask)
            for run, simulation in data.items():
                simulations.append(simulations)


    def __len__(self):
        

    def __getitem__(self, index):
        
import numpy as np
import copy


def normalize_pixel(dataset, model, pixel):
    """
    Normalizes the timeseries of the given model and pixel from the climate dataset and computes its associated mean forced response.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model 
    pixel (tuple of ints): The latitude and longitude

    Output:
    normalized timeseries (np.array): The normalized timeseries. Shape of (run, timestep)
    mean forced response (np.array): The mean forced response of the normalized timeseries. Shape of (timestep,)
    """
    timeserie_per_run = []
    for run in dataset[model].keys():
        timeserie = dataset[model][run][131:, pixel[0], pixel[1]]
        timeserie_per_run.append(timeserie)
        
    timeserie_per_run = np.array(timeserie_per_run)
    mean_timeserie = np.mean(timeserie_per_run, axis=0)
    mean = np.mean(mean_timeserie)

    std_timeserie = np.std(timeserie_per_run, axis=0)

    normalized_timeseries = (timeserie_per_run - mean) / std_timeserie
    mean_forced_response = np.mean(normalized_timeseries, axis=0)

    return normalized_timeseries, mean_forced_response


def normalize_model(dataset, model):
    """
    Normalizes the grids from 1980 to 2014 (end) of the given model from the climate dataset and computes its associated mean forced responses.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model 

    Output:
    normalized_grids (np.array): The normalized grids. Shape of (run, timestep, latitude, longitude)
    mean forced responses (np.array): The mean forced responses of the normalized grids. Shape of (timestep, latitude, longitude)
    """
    grid_timeserie_per_run = []
    for run in dataset[model].keys():
        grid_timeserie = dataset[model][run][131:, :, :]
        grid_timeserie_per_run.append(grid_timeserie)
        
    grid_timeserie_per_run = np.array(grid_timeserie_per_run)
    mean_grid_timeserie = np.mean(grid_timeserie_per_run, axis=0)
    mean_grid = np.mean(mean_grid_timeserie, axis=0)

    std_grid_timeserie = np.std(grid_timeserie_per_run, axis=0)

    normalized_grids = (grid_timeserie_per_run - mean_grid) / std_grid_timeserie
    mean_forced_responses = np.mean(normalized_grids, axis=0)

    return normalized_grids, mean_forced_responses

"""
def normalize2(dataset, model, mask):
    
    Normalizes the grids from 1980 to 2014 (end) of the given model from the climate dataset and computes its associated mean forced responses.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model 
    mask (np.array): A boolean mask for values that should not be used by the normalization and should not be output if flatten is true.
    flatten (boolean): If true, outputs the normalized grids flattened without values specified by the mask. Else, outputs the normalized grids in the same shape as in the dataset

    Output:
    normalized_grids (np.array): The normalized grids. Shape of (run, timestep, latitude x longitude) if flatten, else (run, timestep, latitude, longitude)
    mean forced responses (np.array): The mean forced responses of the normalized grids. Shape of (timestep, latitude x longitude) if flatten, else (timestep, latitude, longitude)
    
    flat_grid_timeserie_per_run = []
    runs = list(dataset[model].keys())
    for run in runs:
        grid_timeserie = dataset[model][run][131:, :, :]
        flat_grids = []
        for grid in grid_timeserie:
            flat_grids.append(from_grid_to_flat(grid, mask))
        flat_grid_timeserie_per_run.append(flat_grids)
        
    flat_grid_timeserie_per_run = np.array(flat_grid_timeserie_per_run)
    mean_flat_grid_timeserie = np.mean(flat_grid_timeserie_per_run, axis=0)
    mean_flat_grid = np.mean(mean_flat_grid_timeserie, axis=0)

    std_flat_grid_timeserie = np.std(flat_grid_timeserie_per_run, axis=0)

    normalized_flat_grids = (flat_grid_timeserie_per_run - mean_flat_grid) / std_flat_grid_timeserie
    mean_forced_flat_responses = np.mean(normalized_flat_grids, axis=0)

    normalized_grids_per_run = []
    for nfg in normalized_flat_grids:
        grid_timeserie = []
        for g in nfg:
            grid_timeserie.append(from_flat_to_grid(g, mask))
        normalized_grids_per_run.append(grid_timeserie)

    mean_forced_responses = []
    for mffr in mean_forced_flat_responses:
        mean_forced_responses.append(from_flat_to_grid(mffr, mask))

    normalized_grids_per_run = np.array(normalized_grids_per_run)
    mean_forced_responses = np.array(mean_forced_responses)

    normalized_grids = {k:v for (k,v) in zip(runs, normalized_grids_per_run)}

    return normalized_grids, mean_forced_responses"""


def force_normalize_flatten_model(dataset, model, mask, mean, std):
    flat_grid_timeserie_per_run = []
    runs = list(dataset[model].keys())
    for run in runs:
        grid_timeserie = dataset[model][run][131:, :, :]
        flat_grids = []
        for grid in grid_timeserie:
            flat_grids.append(from_grid_to_flat(grid, mask))
        flat_grid_timeserie_per_run.append(flat_grids)
        
    flat_grid_timeserie_per_run = np.array(flat_grid_timeserie_per_run)

    norm_flat_grids_per_run = (flat_grid_timeserie_per_run - mean) / std
    mean_flat_forced_responses = np.mean(norm_flat_grids_per_run, axis=0)

    norm_flat_grids = {k:v for (k,v) in zip(runs, norm_flat_grids_per_run)}

    return norm_flat_grids, mean_flat_forced_responses


def normalize_flatten_model(dataset, model, mask):
    """
    Normalizes the grids from 1980 to 2014 (end) of the given model from the climate dataset and 
    computes its associated mean forced responses. Outputs the flattened results with the mean and std.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model 
    mask (np.array): A boolean mask for values that should not be used by the normalization and should not be output if flatten is true.
    
    Output:
    normalized_flat_grids (np.array): The normalized flattened grids of shape (run, timestep, latitude x longitude)
    mean forced responses (np.array): The mean forced responses of the normalized grids. Shape of (timestep, latitude x longitude) if flatten, else (timestep, latitude, longitude)
    mean_flat_grid (np.array): ... Shape of (latitude x longitude,)
    std_flat_grid_timeserie (np.array): ... Shape of (timestep, latitude x longitude)
    """
    flat_grid_timeserie_per_run = []
    runs = list(dataset[model].keys())
    for run in runs:
        grid_timeserie = dataset[model][run][131:, :, :]
        flat_grids = []
        for grid in grid_timeserie:
            flat_grids.append(from_grid_to_flat(grid, mask))
        flat_grid_timeserie_per_run.append(flat_grids)
        
    flat_grid_timeserie_per_run = np.array(flat_grid_timeserie_per_run)
    mean_flat_grid_timeserie = np.mean(flat_grid_timeserie_per_run, axis=0)
    mean_flat_grid = np.mean(mean_flat_grid_timeserie, axis=0)

    std_flat_grid_timeserie = np.std(flat_grid_timeserie_per_run, axis=0)

    norm_flat_grids_per_run = (flat_grid_timeserie_per_run - mean_flat_grid) / std_flat_grid_timeserie
    mean_flat_forced_responses = np.mean(norm_flat_grids_per_run, axis=0)

    norm_flat_grids = {k:v for (k,v) in zip(runs, norm_flat_grids_per_run)}

    return norm_flat_grids, mean_flat_forced_responses, mean_flat_grid, std_flat_grid_timeserie


def normalize_flatten_dataset(dataset, mask):
    """
    Normalizes the grids from 1980 to 2014 (end) of the given model from the climate dataset and 
    computes its associated mean forced responses. Outputs the flattened results with the mean and std.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model 
    mask (np.array): A boolean mask for values that should not be used by the normalization and should not be output if flatten is true.
    
    Output:
    normalized_flat_grids (np.array): The normalized flattened grids of shape (run, timestep, latitude x longitude)
    mean forced responses (np.array): The mean forced responses of the normalized grids. Shape of (timestep, latitude x longitude) if flatten, else (timestep, latitude, longitude)
    mean_flat_grid (np.array): ... Shape of (latitude x longitude,)
    std_flat_grid_timeserie (np.array): ... Shape of (timestep, latitude x longitude)
    """
    flat_dataset = {}
    flat_forced_responses = {}
    mean_flat_grids = {}
    std_flat_grid_timeseries = {}

    models = list(dataset.keys())

    for model in models:
        norm_flat_grids, mean_flat_forced_responses, mean_flat_grid, std_flat_grid_timeserie = normalize_flatten_model(dataset, model, mask)
        
        flat_dataset[model] = norm_flat_grids
        flat_forced_responses[model] = mean_flat_forced_responses
        mean_flat_grids[model] = mean_flat_grid
        std_flat_grid_timeseries[model] = std_flat_grid_timeserie

    return flat_dataset, flat_forced_responses, mean_flat_grids, std_flat_grid_timeseries


def prune(dataset, min_runs=2):
    """
    Take off models from climate dataset that have less runs than min_runs.

    Keyword arguments:
    dataset (dict): The climate dataset
    min_runs (int, default=2): The minimum amount of runs a model should have

    Output:
    pruned_dataset (dict): The pruned dataset
    """
    bad_models = []
    for model in dataset.keys():
        if len(dataset[model]) < min_runs:
            bad_models.append(model)

    for bad_model in bad_models:
        dataset.pop(bad_model)


def find_union_nan_mask(data):
    union_nan_mask = np.zeros((72, 144))
    for model in data.keys():
        for run in data[model].keys():
            grids = data[model][run]
            for i in range(grids.shape[0]):
                nan_mask = np.isnan(grids[i, :, :])
                union_nan_mask = np.logical_or(union_nan_mask, nan_mask)
    
    return union_nan_mask


def inpute(data, mask, value=np.nan):
    for model in data.keys():
        for run in data[model].keys():
            grids = data[model][run]
            for i in range(grids.shape[0]):
                grids[i, :, :][mask] = value


def from_grid_to_flat(grid, mask):
    return grid[~mask]


def from_flat_to_grid(flat, mask):
    flat_idx = 0
    grid = []
    for nan_bool in np.nditer(mask):
        if nan_bool:
            grid.append(np.nan)
        else:
            grid.append(flat[flat_idx])
            flat_idx += 1

    grid = np.array(grid).reshape((72, 144))
    return grid
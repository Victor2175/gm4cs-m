import numpy as np


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


def normalize(dataset, model):
    """
    Normalizes the grids of the given model from the climate dataset and computes its associated mean forced responses.

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


def prune(dataset, min_runs=2):
    """
    Take off models from climate dataset that have less runs than min_runs.

    Keyword arguments:
    dataset (dict): The climate dataset
    min_runs (int, default=2): The minimum amount of runs a model should have

    Output:
    pruned_dataset (dict): The pruned dataset
    """
    pruned_dataset = {}
    for model in dataset.keys():
        if len(dataset[model]) >= min_runs:
            pruned_dataset[model] = dataset[model]

    return pruned_dataset
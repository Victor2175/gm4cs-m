import skimage as ski
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dataset import CopeDataset
from vae import *
import sys

def normalize_pixel(dataset, model, pixel):
    """
    Normalizes the timeseries of the given model and pixel from the climate dataset and computes its associated mean forced response.
    The mean is computed across runs and time while the std is computed across the runs. We then use the mean std.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model 
    pixel (tuple of ints): The latitude and longitude, respectively

    Output:
    runs_timeseries_normalized (np.array): The normalized timeseries. Shape of (run, timestep)
    mean_forced_responses (np.array): The mean forced response of the normalized timeseries. Shape of (timestep,)
    """
    runs_timeserie = []
    for run in dataset[model].keys():
        timeserie = dataset[model][run][131:, pixel[0], pixel[1]]
        runs_timeserie.append(timeserie)

    runs_timeserie = np.array(runs_timeserie)

    mean_timeserie = np.mean(runs_timeserie, axis=0)
    mean = np.mean(mean_timeserie)

    std_timeserie = np.std(runs_timeserie, axis=0)
    std = np.mean(std_timeserie)

    runs_timeseries_normalized = (runs_timeserie - mean) / std
    mean_forced_response = np.mean(runs_timeseries_normalized, axis=0)

    return runs_timeseries_normalized, mean_forced_response


def normalize_pixel_other(dataset, model, pixel):
    """
    Normalizes the timeseries of the given model and pixel from the climate dataset and computes its associated mean forced response.
    The mean and std are computed across runs. We then use the mean std.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model 
    pixel (tuple of ints): The latitude and longitude, respectively

    Output:
    runs_timeseries_normalized (np.array): The normalized timeseries. Shape of (run, timestep)
    mean_forced_responses (np.array): The mean forced response of the normalized timeseries. Shape of (timestep,)
    """
    runs_timeserie = []
    for run in dataset[model].keys():
        timeserie = dataset[model][run][131:, pixel[0], pixel[1]]
        runs_timeserie.append(timeserie)

    runs_timeserie = np.array(runs_timeserie)

    mean_timeserie = np.mean(runs_timeserie, axis=0)

    std_timeserie = np.std(runs_timeserie, axis=0)
    std = np.mean(std_timeserie)

    runs_timeseries_normalized = (runs_timeserie - mean_timeserie) / std
    mean_forced_response = np.mean(runs_timeseries_normalized, axis=0)

    return runs_timeseries_normalized, mean_forced_response


def normalize_model(dataset, model, mean_grid=None, std_grid=None):
    """
    Normalizes the grids from 1980 to 2014 (end) of the given model from the climate dataset and computes its associated mean forced responses.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model 
    mean_grid (np.array): The mean for the normalization. If None, it will be computed from the data. Shape of (latitude, longitude)
    std_grid (np.array): The std for the normalization. If None, it will be computed from the data. Shape of (latitude, longitude)

    Output:
    runs_with_timegrids_normalized (dict): The runs and their associated normalized timegrids. {'run': np.array with shape (timestep, latitude, longitude)} 
    mean_forced_responses (np.array): The mean forced responses of the normalized timegrids. Shape of (timestep, latitude, longitude)
    mean_grid (np.array): The mean grid used for the normalization. Shape of (latitude, longitude)
    std_grid (np.array): The std grid used for the normalization. Shape of (latitude, longitude)
    """
    runs = list(dataset[model].keys())
    runs_timegrids = []
    for run in dataset[model].keys():
        timegrid = dataset[model][run][131:, :, :]
        runs_timegrids.append(timegrid)
        
    runs_timegrids = np.array(runs_timegrids)

    if (mean_grid is None):
        mean_timegrid = np.nanmean(runs_timegrids, axis=0)
        mean_grid = np.nanmean(mean_timegrid, axis=0)

    if (std_grid is None):
        std_timegrid = np.nanstd(runs_timegrids, axis=0)
        std_grid = np.nanmean(std_timegrid, axis=0)

    runs_timegrids_normalized = (runs_timegrids - mean_grid) / std_grid

    runs_with_timegrids_normalized = {k:v for (k,v) in zip(runs, runs_timegrids_normalized)}
    mean_forced_responses = np.nanmean(runs_timegrids_normalized, axis=0)

    return runs_with_timegrids_normalized, mean_forced_responses, mean_grid, std_grid


def center_model(dataset, model):
    """
    Centers the grids from 1980 to 2014 (end) of the given model from the climate dataset and computes its associated mean forced responses.
    
    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model

    Output:
    runs_with_timegrids_centered (dict): The runs and their associated centered timegrids. {'run': np.array with shape (timestep, latitude, longitude)}
    mean_forced_responses (np.array): The mean forced responses of the centered timegrids. Shape of (timestep, latitude, longitude)
    """
    runs = list(dataset[model].keys())
    runs_timegrids = []
    for run in runs:
        timegrid = dataset[model][run][131:, :, :]
        runs_timegrids.append(timegrid)
        
    runs_timegrids = np.array(runs_timegrids)
        
    runs_mean_grid = np.mean(runs_timegrids, axis=1)
    runs_mean_grid = np.expand_dims(runs_mean_grid, axis=1)

    runs_timegrids_centered = runs_timegrids - runs_mean_grid

    runs_with_timegrids_centered = {k:v for (k,v) in zip(runs, runs_timegrids_centered)}
    mean_forced_responses = np.mean(runs_timegrids_centered, axis=0)

    return runs_with_timegrids_centered, mean_forced_responses

def center_flatten_model(dataset, mask, model):
    """
    Centers the grids from 1980 to 2014 (end) of the given model from the climate dataset,
    computes its associated mean forced responses and flattens the results using the provided mask.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)

    Output:
    runs_with_timegrids_centered (dict): The runs and their associated centered timegrids. {'run': np.array with shape (timestep, cells)}
    mean_forced_responses (np.array): The mean forced responses of the centered timegrids. Shape of (timestep, cells)

    """
    runs = list(dataset[model].keys())
    runs_timegrids = []
    for run in runs:
        timegrid = dataset[model][run][131:, :, :]
        flattened_timegrids = []
        for grid in timegrid:
            flattened_timegrids.append(from_grid_to_flat(grid, mask))
        runs_timegrids.append(flattened_timegrids)
        
    runs_timegrids = np.array(runs_timegrids)
    
    runs_mean_grid = np.mean(runs_timegrids, axis=1)
    runs_mean_grid = np.expand_dims(runs_mean_grid, axis=1)

    runs_timegrids_centered = runs_timegrids - runs_mean_grid

    runs_with_timegrids_centered = {k:v for (k,v) in zip(runs, runs_timegrids_centered)}
    mean_forced_responses = np.mean(runs_timegrids_centered, axis=0)

    return runs_with_timegrids_centered, mean_forced_responses


def force_normalize_flatten_model(dataset, model, mask, mean, std):
    """
    Forces the normalization of the grids from 1980 to 2014 (end) of the given model from the climate dataset,
    computes its associated mean forced responses and flattens the results using the provided mask.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)
    mean (np.array): The mean for the normalization
    std (np.array): The std for the normalization
    

    Output:
    runs_with_timegrids_normalized (dict): The runs and their associated normalized timegrids. {'run': np.array with shape (timestep, cells)}
    mean_forced_responses (np.array): The mean forced responses of the normalized timegrids. Shape of (timestep, cells)

    """
    runs = list(dataset[model].keys())
    runs_timegrids = []
    for run in runs:
        timegrid = dataset[model][run][131:, :, :]
        flattened_timegrids = []
        for grid in timegrid:
            flattened_timegrids.append(from_grid_to_flat(grid, mask))
        runs_timegrids.append(flattened_timegrids)
        
    runs_timegrids = np.array(runs_timegrids)

    runs_timegrids_normalized = (runs_timegrids - mean) / std

    runs_with_timegrids_normalized = {k:v for (k,v) in zip(runs, runs_timegrids_normalized)}
    mean_forced_responses = np.mean(runs_timegrids_normalized, axis=0)

    return runs_with_timegrids_normalized, mean_forced_responses


def normalize_flatten_model(dataset, model, mask, mean_grid=None, std_grid=None):
    """
    Normalizes the grids from 1980 to 2014 (end) of the given model from the climate dataset and 
    computes its associated mean forced responses. Outputs the flattened results with the mean and std.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model 
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)
    mean_grid (np.array): The mean for the normalization. If None, it will be computed from the data. Shape of (cells,)
    std_grid (np.array): The std for the normalization. If None, it will be computed from the data. Shape of (cells,)
    
    Output:
    runs_with_timegrids_normalized (np.array): The runs and their associated normalized timegrids. {'run': np.array with shape (timestep, cells)} 
    mean_forced_responses (np.array): The mean forced responses of the normalized timegrids. Shape of (timestep, cells)
    mean_grid (np.array): The mean grid used for the normalization. Shape of (cells,)
    std_grid (np.array): The std grid used for the normalization. Shape of (cells,)
    """
    runs = list(dataset[model].keys())
    runs_timegrids = []
    for run in runs:
        timegrid = dataset[model][run][131:, :, :]
        flattened_timegrids = []
        for grid in timegrid:
            flattened_timegrids.append(from_grid_to_flat(grid, mask))
        runs_timegrids.append(flattened_timegrids)
        
    runs_timegrids = np.array(runs_timegrids)

    if (mean_grid is None):
        mean_timegrid = np.mean(runs_timegrids, axis=0)
        mean_grid = np.mean(mean_timegrid, axis=0)

    if (std_grid is None):
        std_timegrid = np.std(runs_timegrids, axis=0)
        std_grid = np.mean(std_timegrid, axis=0)

    runs_timegrids_normalized = (runs_timegrids - mean_grid) / std_grid
    
    runs_with_timegrids_normalized = {k:v for (k,v) in zip(runs, runs_timegrids_normalized)}
    mean_forced_responses = np.mean(runs_timegrids_normalized, axis=0)

    return runs_with_timegrids_normalized, mean_forced_responses, mean_grid, std_grid


def normalize_dataset(dataset):
    """
    Normalizes the grids from 1980 to 2014 (end) for all models in the climate dataset and computes their associated mean forced responses.

    Keyword arguments:
    dataset (dict): The climate dataset

    Output:
    flat_dataset (dict): The models and their associated runs with normalized timegrids. {'model': {'run': np.array with shape (timestep, latitude, longitude)}}
    model_with_mean_forced_responses (dict): The models and their associated mean forced responses. {'model': np.array with shape (timestep, latitude, longitude)}
    model_with_mean_grid (dict): The models and their associated mean grid. {'model': np.array with shape (latitude, longitude)}
    model_with_std_grid (dict): The models and their associated std grid. {'model': np.array with shape (latitude, longitude)}
    """
    flat_dataset = {}
    model_with_mean_forced_responses = {}
    model_with_mean_grid = {}
    model_with_std_grid = {}

    models = list(dataset.keys())

    for model in models:
        runs_with_timegrids_normalized, mean_forced_responses, mean_grid, std_grid = normalize_model(dataset, model)
        
        flat_dataset[model] = runs_with_timegrids_normalized
        model_with_mean_forced_responses[model] = mean_forced_responses
        model_with_mean_grid[model] = mean_grid
        model_with_std_grid[model] = std_grid

    return flat_dataset, model_with_mean_forced_responses, model_with_mean_grid, model_with_std_grid


def normalize_flatten_dataset(dataset, mask):
    """
    Normalizes the grids from 1980 to 2014 (end) for all models in the climate dataset, 
    computes its associated mean forced responses and flattens the results using the provided mask.

    Keyword arguments:
    dataset (dict): The climate dataset
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)
    
    Output:
    flat_dataset (dict): The models and their associated runs with normalized timegrids. {'model': {'run': np.array with shape (timestep, cells)}}
    model_with_mean_forced_responses (dict): The models and their associated mean forced responses. {'model': np.array with shape (timestep, cells)}
    model_with_mean_grid (dict): The models and their associated mean grid. {'model': np.array with shape (cells,)}
    model_with_std_grid (dict): The models and their associated std grid. {'model': np.array with shape (cells,)}
    """
    flat_dataset = {}
    model_with_mean_forced_responses = {}
    model_with_mean_grid = {}
    model_with_std_grid = {}

    models = list(dataset.keys())

    for model in models:
        runs_with_timegrids_normalized, mean_forced_responses, mean_grid, std_grid = normalize_flatten_model(dataset, model, mask)
        
        flat_dataset[model] = runs_with_timegrids_normalized
        model_with_mean_forced_responses[model] = mean_forced_responses
        model_with_mean_grid[model] = mean_grid
        model_with_std_grid[model] = std_grid

    return flat_dataset, model_with_mean_forced_responses, model_with_mean_grid, model_with_std_grid


def prune(dataset, min_runs=2):
    """
    Take off models from climate dataset that have less runs than min_runs. 
    This operation is done in place.

    Keyword arguments:
    dataset (dict): The climate dataset
    min_runs (int, default=2): The minimum amount of runs a model should have
    """
    bad_models = []
    for model in dataset.keys():
        if len(dataset[model]) < min_runs:
            bad_models.append(model)

    for bad_model in bad_models:
        dataset.pop(bad_model)


def cut_lat(dataset, max_lat):
    """
    Cuts the latitude of the grids to the given max_lat.
    This operation is done in place.
    
    Keyword arguments:
    dataset (dict): The climate dataset
    max_lat (int): The maximum latitude to cut the grids to
    """
    dataset_copy = dataset.copy()

    for model in dataset_copy.keys():
        for run in dataset_copy[model].keys():
            cut_grids = dataset_copy[model][run][:, :max_lat, :]
                
            dataset[model][run] = cut_grids


def inpute(dataset, mask, value=np.nan):
    """
    Inputes the grids of the given data with the given value using the provided mask.
    This operation is done in place.

    Keyword arguments:
    dataet (dict): The climate dataset
    mask (np.array): A boolean mask that indicate cells to fill. Shape of (latitude, longitude)
    value (float, default=np.nan): The value to fill the cells with
    """
    for model in dataset.keys():
        for run in dataset[model].keys():
            grids = dataset[model][run]
            for i in range(grids.shape[0]):
                grids[i, :, :][mask] = value


def downscale(dataset):
    """
    Downscales the grids from 1980 to 2014 (end) of the given model from the climate dataset.
    This operation is done in place.

    Keyword arguments:
    dataset (dict): The climate dataset
    """
    for model in dataset.keys():
        for run in dataset[model].keys():
            dataset[model][run] = ski.transform.downscale_local_mean(dataset[model][run], (1,2,2))


def find_union_nan_mask(dataset):
    """
    Finds the smallest nan mask that contain all the nan configurations in the grids.

    Keyword arguments:
    dataset (dict): The climate dataset

    Outputs:
    union_nan_mask (np.array): The union of all the nan masks. Shape of (latitude, longitude)
    """
    init = False
    for model in dataset.keys():
        for run in dataset[model].keys():
            grids = dataset[model][run]
            for i in range(grids.shape[0]):
                
                if (not init):
                    union_nan_mask = np.zeros(grids[i, :, :].shape)
                    init = True
                
                nan_mask = np.isnan(grids[i, :, :])
                union_nan_mask = np.logical_or(union_nan_mask, nan_mask)
    
    return union_nan_mask


def from_grid_to_flat(grid, mask):
    """
    Flattens the grid and removes cells using the provided mask.

    Keyword arguments:
    grid (np.array): The grid to flatten. Shape of (latitude, longitude)
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)

    Outputs:
    flattened_grid (np.array): The flattened grid. Shape of (cells,)
    """
    flattened_grid = grid[~mask]
    return flattened_grid


def from_flat_to_grid(flat, mask):
    """
    Converts the flattened grid back to its original state.
    
    Keyword arguments:
    flat (np.array): The flattened grid. Shape of (cells,)
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)

    Outputs:
    grid (np.array): The original grid. Shape of (latitude, longitude)
    """
    flat_idx = 0
    grid = []
    for nan_bool in np.nditer(mask):
        if nan_bool:
            grid.append(np.nan)
        else:
            grid.append(flat[flat_idx])
            flat_idx += 1

    grid = np.array(grid).reshape(mask.shape)
    return grid


def from_flat_to_timegrid(flat, mask):
    """
    Converts the flattened timegrid back to its original state.
    
    Keyword arguments:
    flat (np.array): The flattened grid timeserie. Shape of (timestep*cells, )
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)

    Outputs:
    grid_timeserie (np.array): The original grid timeserie. Shape of (timestep, latitude, longitude)
    """
    flat_idx = 0

    grids = []
    for i in range(34):
        grid = []
        for nan_bool in np.nditer(mask):
            if nan_bool:
                grid.append(np.nan)
            else:
                grid.append(flat[flat_idx])
                flat_idx += 1

        grid = np.array(grid).reshape(mask.shape)
        grids.append(grid)

    return np.array(grids)


def fill_grid_timeserie(grid_timeserie, mask, method='mean', fill_value=None):
    filled_grid_timeserie = []
    if method == 'mean':
        for grid in grid_timeserie:
            grid[mask] = np.nanmean(grid)
            filled_grid_timeserie.append(grid)
    elif method == 'value':
        if fill_value is None:
            raise ValueError("fill_value must be provided when method is 'value'")
        
        for grid in grid_timeserie:
            grid[mask] = fill_value
            filled_grid_timeserie.append(grid)

    return np.array(filled_grid_timeserie)


def extract_per_grid(cope_data, mask, r=[-10, 10], eval_model=None):
    """
    Normalizes and extracts the data and their associated mean forced responses into grids.

    Keyword arguments:
    cope_data (dict): The climate dataset
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)
    r (list of floats): The range of values to consider for the grids. Default is [-10, 10]
    eval_model (string, optional): The testing model. If None, the first model in cope_data will be used.

    Outputs:
    X_train (np.array): The training grids. Shape of (samples, cells)
    y_train (np.array): The mean forced responses for the training grids. Shape of (samples, cells)
    X_test (np.array): The testing grids. Shape of (samples, cells)
    y_test (np.array): The mean forced responses for the testing grids. Shape of (samples, cells)
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    if eval_model is None:
        eval_model = list(cope_data.keys())[0]  # Default to the first model if none is provided

    train_models = [model for model in cope_data.keys() if model != eval_model]

    for model in train_models:
        normalized_grids, mean_forced_responses, _, _ = normalize_flatten_model(cope_data, model, mask)
        for run, grid_timeserie in normalized_grids.items():
            for grid, mean_forced_response in zip(grid_timeserie, mean_forced_responses):
                if (r[0] < grid.min() and grid.max() < r[1]):
                    X_train.append(grid)
                    y_train.append(mean_forced_response)

    normalized_grids, mean_forced_responses = center_flatten_model(cope_data, mask, eval_model)
    for run, grid_timeserie in normalized_grids.items():
        for grid, mean_forced_response in zip(grid_timeserie, mean_forced_responses):
            if (r[0] < grid.min() and grid.max() < r[1]):
                X_test.append(grid)
                y_test.append(mean_forced_response)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, y_train, X_test, y_test


def extract_per_timeserie(cope_data, mask, r=[-10, 10], eval_model=None):
    """
    Extracts the data and their associated mean forced responses into timeseries.

    Keyword arguments:
    cope_data (dict): The climate dataset
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)
    r (list of floats): The range of values to consider for the grids. Default is [-10, 10]
    eval_model (string, optional): The testing model. If None, the first model in cope_data will be used.

    Outputs:
    X_train (np.array): The training timeseries. Shape of (samples, 34*cells)
    y_train (np.array): The mean forced responses for the training timeseries. Shape of (samples, 34*cells)
    X_test (np.array): The testing timeseries. Shape of (samples, 34*cells)
    y_test (np.array): The mean forced responses for the testing timeseries. Shape of (samples, 34*cells)
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    if eval_model is None:
        eval_model = list(cope_data.keys())[0]  # Default to the first model if none is provided

    train_models = [model for model in cope_data.keys() if model != eval_model]

    for model in train_models:
        normalized_grids, mean_forced_response, _, _ = normalize_flatten_model(cope_data, model, mask)
        flattened_mean_forced_response = mean_forced_response.flatten()
        for run, grid_timeserie in normalized_grids.items():
            flattened_grid_timeserie = grid_timeserie.flatten()
            if (r[0] < flattened_grid_timeserie.min() and flattened_grid_timeserie.max() < r[1]):
                X_train.append(flattened_grid_timeserie)
                y_train.append(flattened_mean_forced_response)

    runs_with_timegrids_centered, mean_forced_responses = center_flatten_model(cope_data, mask, eval_model)
    flat_mean_forced_response = mean_forced_responses.flatten()

    X_test, y_test = [], []
    for run, timegrid in runs_with_timegrids_centered.items():
        flat_timegrid = timegrid.flatten()
        if (r[0] < flat_timegrid.min() and flat_timegrid.max() < r[1]):
            X_test.append(flat_timegrid)
            y_test.append(flat_mean_forced_response)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, y_train, X_test, y_test


def extract_per_images(cope_data, mask, r=[-10, 10], eval_model=None):
    """
    Extracts the data and their associated mean forced responses into images by filling the nan values with the mean of the grid.
    
    Keyword arguments:
    cope_data (dict): The climate dataset
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)
    r (list of floats): The range of values to consider for the grids. Default is [-10, 10]
    eval_model (string, optional): The testing model. If None, the first model in cope_data will be used.

    Outputs:
    X_train (np.array): The training images. Shape of (samples, 34, latitude, longitude)
    y_train (np.array): The mean forced responses for the training images. Shape of (samples, 34, latitude, longitude)
    X_test (np.array): The testing images. Shape of (samples, 34, latitude, longitude)
    y_test (np.array): The mean forced responses for the testing images. Shape of (samples, 34, latitude, longitude)
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    if eval_model is None:
        eval_model = list(cope_data.keys())[0]  # Default to the first model if none is provided

    train_models = [model for model in cope_data.keys() if model != eval_model]
    
    for model in train_models:
        normalized_grids, mean_forced_responses, _, _ = normalize_model(cope_data, model)
        filled_mean_forced_responses = fill_grid_timeserie(mean_forced_responses, mask, method='mean')
        for run, grid_timeserie in normalized_grids.items():
            if (r[0] < np.nanmin(grid_timeserie) and np.nanmax(grid_timeserie) < r[1]):
                filled_grid_timeserie = fill_grid_timeserie(grid_timeserie, mask, method='mean')

                X_train.append(filled_grid_timeserie)
                y_train.append(filled_mean_forced_responses)
    
    runs_with_timegrids_centered, mean_forced_response = center_model(cope_data, eval_model)
    filled_mean_forced_response = fill_grid_timeserie(mean_forced_response, mask)
    
    X_test, y_test = [], []
    for run, timegrid in runs_with_timegrids_centered.items():
        if (r[0] < np.nanmin(timegrid) and np.nanmax(timegrid) < r[1]):
            filled_timegrid = fill_grid_timeserie(timegrid, mask, method='mean')

            X_test.append(filled_timegrid)
            y_test.append(filled_mean_forced_response)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, y_train, X_test, y_test


def get_confidence_interval_grid_timeserie(samples):
    """
    Computes the confidence interval of the given samples.

    Keyword arguments:
    samples (np.array): The samples to compute the confidence interval for. Shape of (n_samples, timestep, *)

    Outputs:
    conf_int (np.array): The confidence interval of the samples. Shape of (timestep, *)
    """
    samples = np.array(samples)
    std = np.std(samples, axis=0)
    conf_int = 1.96 * std / np.sqrt(samples.shape[0])

    return conf_int


def LOOCV(dataset, mask, machine_model):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) for the given dataset using the provided ridge regression model.
    For each model in the dataset, it trains the machine model on all other models and evaluates it on the left-out model.
    The mean squared error (MSE) and normalized MSE (NMSE) are computed for each evaluation.

    Keyword arguments:
    dataset (dict): The climate dataset
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)
    machine_model (object): The ridge regression model to use for training and evaluation

    Outputs:
    test_mse_losses (dict): The MSE losses for each evaluation model.
    test_nmse_losses (dict): The normalized MSE losses for each evaluation model. 
    """
    models = list(dataset.keys())
    n_models = len(models)
    ith_model = 1

    test_mse_losses = {}
    test_nmse_losses = {}

    for eval_model in models:
        X_train, y_train, _, _ = extract_per_grid(dataset, mask, eval_model=eval_model)
        X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)

        machine_model.fit(X_train, y_train)

        runs_with_timegrids_centered, mean_forced_responses = center_flatten_model(dataset, mask, eval_model)

        runs_loss = []
        runs_loss_with_std = []

        for run, timegrid in runs_with_timegrids_centered.items():
            X_test, y_test = [], []
            for grid, mean_forced_response in zip(timegrid, mean_forced_responses):
                X_test.append(grid)
                y_test.append(mean_forced_response)

            X_test, y_test = np.array(X_test), np.array(y_test)
            X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

            y_hat = machine_model.predict(X_test)
            temp = (y_hat - y_test)**2
            #run_loss = torch.mean(temp).item()
            run_loss = torch.sum(temp).item()
            runs_loss.append(run_loss)

            var = torch.var(y_test, axis=0)
            temp /= var
            #run_loss_with_std = torch.mean(temp).item()
            run_loss_with_std = torch.sum(temp).item()
            runs_loss_with_std.append(run_loss_with_std)
            
        loss = sum(runs_loss) / len(runs_loss)
        loss_with_std = sum(runs_loss_with_std) / len(runs_loss_with_std)

        print(f"[{ith_model}/{n_models}] For eval model {eval_model}, \t MSE is {round(loss, 2)}, \t NMSE is {round(loss_with_std, 2)}")
        sys.stdout.flush()
        
        test_mse_losses[eval_model] = loss
        test_nmse_losses[eval_model] = loss_with_std

        ith_model += 1

    mean_test_loss = sum(test_mse_losses.values()) / len(test_mse_losses)
    mean_test_loss_with_std = sum(test_nmse_losses.values()) / len(test_nmse_losses)
    
    print(f"The mean MSE is {round(mean_test_loss, 2)}, the mean NMSE is {round(mean_test_loss_with_std, 2)}\n")
    sys.stdout.flush()

    return test_mse_losses, test_nmse_losses


def VAE_LOOCV(dataset, mask, model_configs, training_configs, verbose=False):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) for the given dataset using a Variational Autoencoder (VAE).
    For each model in the dataset, it trains the VAE on all other models and evaluates it on the left-out model.
    The mean squared error (MSE) and normalized MSE (NMSE) are computed for each evaluation.

    Keyword arguments:
    dataset (dict): The climate dataset
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)
    model_configs (dict): The configurations for the VAE model, including hidden and latent dimensions
    training_configs (dict): The configurations for training, including device, epochs, batch size, and learning rate
    verbose (bool, default=False): If True, prints the progress of the training

    Outputs:
    test_mse_losses (dict): The MSE losses for each evaluation model.
    test_nmse_losses (dict): The normalized MSE losses for each evaluation model.
    """
    models = list(dataset.keys())
    n_models = len(models)
    ith_model = 1

    hidden_dim, latent_dim = list(model_configs.values())
    device, epochs, batch_size, lr = list(training_configs.values())

    test_mse_losses = {}
    test_nmse_losses = {}

    for eval_model in models:
        machine_model = VAE(input_dim=44472, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
        
        X_train, y_train, _, _ = extract_per_timeserie(dataset, mask, eval_model=eval_model)

        runs_with_timegrids_centered, mean_forced_responses = center_flatten_model(dataset, mask, eval_model)
        flat_mean_forced_response = mean_forced_responses.flatten()

        test_var = torch.var(torch.tensor(mean_forced_responses), dim=0).to(device)
        test_var = torch.cat([test_var] * 34, dim=0)

        X_test, y_test = [], []
        for run, timegrid in runs_with_timegrids_centered.items():
            flat_timegrid = timegrid.flatten()

            X_test.append(flat_timegrid)
            y_test.append(flat_mean_forced_response)

        X_test, y_test = np.array(X_test), np.array(y_test)

        train_dataset = CopeDataset(samples=X_train, labels=y_train)
        trained_machine_model = train_a_vae(machine_model, train_dataset, device, epochs, batch_size, lr, verbose)

        test_dataset = CopeDataset(samples=X_test, labels=y_test)
        loss, loss_with_std = eval_vae(trained_machine_model, test_dataset, device, batch_size, test_var)
        
        print(f"[{ith_model}/{n_models}] For eval model {eval_model}, \t MSE is {round(loss, 2)}, \t NMSE is {round(loss_with_std, 2)}")
        sys.stdout.flush()
        
        test_mse_losses[eval_model] = loss
        test_nmse_losses[eval_model] = loss_with_std

        ith_model += 1

    mean_test_loss = sum(test_mse_losses.values()) / len(test_mse_losses)
    mean_test_loss_with_std = sum(test_nmse_losses.values()) / len(test_nmse_losses)

    
    print(f"The mean MSE is {round(mean_test_loss, 2)}, the mean NMSE is {round(mean_test_loss_with_std, 2)}\n")
    sys.stdout.flush()

    return test_mse_losses, test_nmse_losses


def CVAE_LOOCV(dataset, mask, model_configs, training_configs, verbose=False):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) for the given dataset using a Convolutional Variational Autoencoder (CVAE).
    For each model in the dataset, it trains the CVAE on all other models and evaluates it on the left-out model.
    The mean squared error (MSE) and normalized MSE (NMSE) are computed for each evaluation.

    Keyword arguments:
    dataset (dict): The climate dataset
    mask (np.array): A boolean mask that indicate cells that should be ignored (such as nans). Shape of (latitude, longitude)
    model_configs (dict): The configurations for the CVAE model, including input channels, hidden dimensions, and latent dimension
    training_configs (dict): The configurations for training, including device, epochs, batch size, and learning rate
    verbose (bool, default=False): If True, prints the progress of the training

    Outputs:
    test_mse_losses (dict): The MSE losses for each evaluation model.
    test_nmse_losses (dict): The normalized MSE losses for each evaluation model.
    """
    models = list(dataset.keys())
    n_models = len(models)
    ith_model = 1

    in_channels, hidden_dims, latent_dim = list(model_configs.values())
    device, epochs, batch_size, lr = list(training_configs.values())

    test_mse_losses = {}
    test_nmse_losses = {}

    for eval_model in models:
        machine_model = CVAE(mask=mask, in_channels=in_channels, hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)
        
        X_train, y_train, _, _ = extract_per_images(dataset, mask, eval_model=eval_model)

        runs_with_timegrids_centered, mean_forced_response = center_model(dataset, eval_model)
        filled_mean_forced_response = fill_grid_timeserie(mean_forced_response, mask)

        test_var = torch.var(torch.tensor(filled_mean_forced_response), dim=0).to(device)
        test_var = torch.stack([test_var] * 34)

        X_test, y_test = [], []
        for run, timegrid in runs_with_timegrids_centered.items():
            filled_timegrid = fill_grid_timeserie(timegrid, mask)

            X_test.append(filled_timegrid)
            y_test.append(filled_mean_forced_response)

        X_test, y_test = np.array(X_test), np.array(y_test)

        train_dataset = CopeDataset(samples=X_train, labels=y_train)
        trained_machine_model = train_a_vae(machine_model, train_dataset, device, epochs, batch_size, lr, verbose)

        test_dataset = CopeDataset(samples=X_test, labels=y_test)
        loss, loss_with_std = eval_cvae(trained_machine_model, test_dataset, mask, device, batch_size, test_var)
        
        print(f"[{ith_model}/{n_models}] For eval model {eval_model}, \t MSE is {round(loss, 2)}, \t NMSE is {round(loss_with_std, 2)}")
        sys.stdout.flush()
        
        test_mse_losses[eval_model] = loss
        test_nmse_losses[eval_model] = loss_with_std

        ith_model += 1

    mean_test_loss = sum(test_mse_losses.values()) / len(test_mse_losses)
    mean_test_loss_with_std = sum(test_nmse_losses.values()) / len(test_nmse_losses)

    print(f"The mean MSE is {round(mean_test_loss, 2)}, the mean NMSE is {round(mean_test_loss_with_std, 2)}\n")
    sys.stdout.flush()

    return test_mse_losses, test_nmse_losses


def train_a_vae(model, train_dataset, device, epochs, batch_size, lr, verbose=False):
    """
    Trains the given VAE model on the provided training dataset.
    
    Keyword arguments:
    model (torch.nn.Module): The VAE model to train
    train_dataset (torch.utils.data.Dataset): The training dataset
    device (torch.device): The device to train the model on (CPU or GPU)
    epochs (int): The number of epochs to train the model
    batch_size (int): The batch size for training
    lr (float): The learning rate for the optimizer
    verbose (bool, default=False): If True, prints the progress of the training
    
    Returns:
    model (torch.nn.Module): The trained VAE model
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            x_hat, mean, var = model(x)
            loss = model.loss(x_hat, y, mean, var)
            
            overall_loss += loss.item()
        
            loss.backward()
            optimizer.step()
        
        if verbose:
            print("Epoch", epoch + 1, "complete !", "\tAverage training loss: ", overall_loss / (batch_idx*batch_size))
            sys.stdout.flush()

    if verbose:
        print("\nModel training complete !\n")
        sys.stdout.flush()

    return model


def eval_vae(trained_model, test_dataset, device, batch_size, test_var):
    """
    Evaluates the trained VAE model on the provided test dataset.

    Keyword arguments:
    trained_model (torch.nn.Module): The trained VAE model
    test_dataset (torch.utils.data.Dataset): The test dataset
    device (torch.device): The device to evaluate the model on (CPU or GPU)
    batch_size (int): The batch size for evaluation
    test_var (torch.Tensor): The variance of the test dataset, used for computation of NMSE

    Returns:
    loss (float): The mean squared error (MSE) loss of the model on the test dataset
    loss_with_std (float): The normalized mean squared error (NMSE) loss of the model on the test dataset
    """
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    trained_model.eval()

    batch_losses = []
    batch_losses_with_std = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)

            x_hat, _, _ = trained_model(x)
            
            temp = (x_hat - y)**2
            #loss = torch.mean(temp).item()
            loss = torch.mean(temp, dim=0).sum().item()
            batch_losses.append(loss)

            temp /= test_var
            #loss_with_std = torch.mean(temp).item()
            loss_with_std = torch.mean(temp, dim=0).sum().item()
            batch_losses_with_std.append(loss_with_std)

    loss = sum(batch_losses) / len(batch_losses)
    loss_with_std = sum(batch_losses_with_std) / len(batch_losses_with_std)

    return loss, loss_with_std


def eval_cvae(trained_model, test_dataset, mask, device, batch_size, test_var):
    """
    Evaluates the trained CVAE model on the provided test dataset.

    Keyword arguments:
    trained_model (torch.nn.Module): The trained CVAE model
    test_dataset (torch.utils.data.Dataset): The test dataset
    mask (np.array): A boolean mask that indicates cells to ignore (such as nans). Shape of (latitude, longitude)
    device (torch.device): The device to evaluate the model on (CPU or GPU)
    batch_size (int): The batch size for evaluation
    test_var (torch.Tensor): The variance of the test dataset, used for computation of NMSE

    Returns:
    loss (float): The mean squared error (MSE) loss of the model on the test dataset
    loss_with_std (float): The normalized mean squared error (NMSE) loss of the model on the test dataset
    """
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    trained_model.eval()

    batch_losses = []
    batch_losses_with_std = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)

            x_hat, _, _ = trained_model(x)
            
            temp = (x_hat[:, :, ~mask] - y[:, :, ~mask])**2
            #loss = torch.mean(temp).item()
            loss = torch.mean(temp, dim=0).sum().item()
            batch_losses.append(loss)

            temp /= test_var[:, ~mask]
            #loss_with_std = torch.mean(temp).item()
            loss_with_std = torch.mean(temp, dim=0).sum().item()
            batch_losses_with_std.append(loss_with_std)

    loss = sum(batch_losses) / len(batch_losses)
    loss_with_std = sum(batch_losses_with_std) / len(batch_losses_with_std)

    return loss, loss_with_std
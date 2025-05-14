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


def normalize_model(dataset, model):
    """
    Normalizes the grids from 1980 to 2014 (end) of the given model from the climate dataset and computes its associated mean forced responses.

    Keyword arguments:
    dataset (dict): The climate dataset
    model (string): The model 

    Output:
    runs_with_timegrids_normalized (dict): The runs and their associated normalized timegrids. {'run': np.array with shape (timestep, latitude, longitude)} 
    mean_forced_responses (np.array): The mean forced responses of the normalized timegrids. Shape of (timestep, latitude, longitude)
    """
    runs = list(dataset[model].keys())
    runs_timegrids = []
    for run in dataset[model].keys():
        timegrid = dataset[model][run][131:, :, :]
        runs_timegrids.append(timegrid)
        
    runs_timegrids = np.array(runs_timegrids)
    mean_timegrid = np.nanmean(runs_timegrids, axis=0)
    mean_grid = np.nanmean(mean_timegrid, axis=0)

    std_timegrid = np.nanstd(runs_timegrids, axis=0)
    std_grid = np.nanmean(std_timegrid, axis=0)

    runs_timegrids_normalized = (runs_timegrids - mean_grid) / std_grid

    runs_with_timegrids_normalized = {k:v for (k,v) in zip(runs, runs_timegrids_normalized)}
    mean_forced_responses = np.nanmean(runs_timegrids_normalized, axis=0)

    return runs_with_timegrids_normalized, mean_forced_responses


def center_flatten_model(dataset, model, mask):
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
    Downscales the grids from 1980 to 2014 (end) of the given model from the climate dataset
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


def from_flat_to_grid_timeserie(flat, mask):
    """
    Converts the flattened grid timeserie back to its original state.
    
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


def extract_per_grid(cope_data, mask, r=[-10, 10]):
    X, Y = [], []
    for model in cope_data.keys():
        normalized_grids, mean_forced_responses, _, _ = normalize_flatten_model(cope_data, model, mask)
        for run, grid_timeserie in normalized_grids.items():
            for grid, mean_forced_response in zip(grid_timeserie, mean_forced_responses):
                if (r[0] < grid.min() and grid.max() < r[1]):
                    X.append(grid)
                    Y.append(mean_forced_response)

    X, Y = np.array(X), np.array(Y)
    return X, Y


def extract_per_timeserie(cope_data, mask, r=[-10, 10]):
    X, Y = [], []
    for model in cope_data.keys():
        normalized_grids, mean_forced_response, _, _ = normalize_flatten_model(cope_data, model, mask)
        flattened_mean_forced_response = mean_forced_response.flatten()
        for run, grid_timeserie in normalized_grids.items():
            flattened_grid_timeserie = grid_timeserie.flatten()
            
            if (r[0] < flattened_grid_timeserie.min() and flattened_grid_timeserie.max() < r[1]):
                X.append(flattened_grid_timeserie)
                Y.append(flattened_mean_forced_response)

    X, Y = np.array(X), np.array(Y)
    return X, Y


def extract_images(cope_data, mask, r=[-10, 10]):
    X, Y = [], []
    for model in cope_data.keys():
        normalized_grids, mean_forced_responses = normalize_model(cope_data, model)
        filled_mean_forced_responses = fill_grid_timeserie(mean_forced_responses, mask, method='mean')
        for run, grid_timeserie in normalized_grids.items():
            if (r[0] < np.nanmin(grid_timeserie) and np.nanmax(grid_timeserie) < r[1]):
                filled_grid_timeserie = fill_grid_timeserie(grid_timeserie, mask, method='mean')

                X.append(filled_grid_timeserie)
                Y.append(filled_mean_forced_responses)

    X, Y = np.array(X), np.array(Y)          
    return X, Y


def get_confidence_interval_grid_timeserie(samples):
    #samples: np.array of shape (n_samples, 34, 30, 72)
    samples = np.array(samples)
    std = np.std(samples, axis=0)
    conf_int = 1.96 * std / np.sqrt(samples.shape[0])

    return conf_int

"""
def LOOCV(dataset, mask, machine_model, verbose=False):
    models = list(dataset.keys())
    n_models = len(models)
    ith_model = 1

    test_losses = {}

    flat_dataset, model_with_mean_forced_responses, _, _ = normalize_flatten_dataset(dataset, mask)

    for eval_model in models:
        train_models = [model for model in models if model != eval_model]
        
        X_train, y_train = [], []

        for train_model in train_models:
            for run, timegrid in flat_dataset[train_model].items():
                for grid, mean_forced_response in zip(timegrid, model_with_mean_forced_responses[train_model]):
                    X_train.append(grid)
                    y_train.append(mean_forced_response)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)

        runs_with_timegrids_centered, mean_forced_responses = center_flatten_model(dataset, eval_model, mask)

        X_test, y_test = [], []
        for run, timegrid in runs_with_timegrids_centered.items():
            for grid, mean_forced_response in zip(timegrid, mean_forced_responses):
                X_test.append(grid)
                y_test.append(mean_forced_response)

        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

        machine_model.fit(X_train, y_train)
        y_hat = machine_model.predict(X_test)
        criterion = torch.nn.MSELoss()
        test_var = torch.var(y_test)

        #loss = torch.sqrt(criterion(y_hat, y_test)).item()
        loss = (criterion(y_hat, y_test) / test_var).item()

        if verbose:
            print(f"[{ith_model}/{n_models}] The NMSE for model {eval_model} is {round(loss, 2)}")
        
        test_losses[eval_model] = loss
        ith_model += 1

    mean_test_loss = sum(test_losses.values()) / len(test_losses)

    if verbose:
        print(f"The mean RMSE is {round(mean_test_loss, 2)}")

    return test_losses
"""

def LOOCV(dataset, mask, machine_model, verbose=False):
    models = list(dataset.keys())
    n_models = len(models)
    ith_model = 1

    test_mse_losses = {}
    test_nmse_losses = {}

    flat_dataset, model_with_mean_forced_responses, _, _ = normalize_flatten_dataset(dataset, mask)

    for eval_model in models:
        train_models = [model for model in models if model != eval_model]
        
        X_train, y_train = [], []
        for train_model in train_models:
            for run, timegrid in flat_dataset[train_model].items():
                for grid, mean_forced_response in zip(timegrid, model_with_mean_forced_responses[train_model]):
                    X_train.append(grid)
                    y_train.append(mean_forced_response)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)

        machine_model.fit(X_train, y_train)

        runs_with_timegrids_centered, mean_forced_responses = center_flatten_model(dataset, eval_model, mask)

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
            run_loss = torch.sum(temp).item()
            runs_loss.append(run_loss)

            var = torch.var(y_test, axis=0)
            temp /= var
            run_loss_with_std = torch.sum(temp).item()
            runs_loss_with_std.append(run_loss_with_std)
            
        loss = sum(runs_loss) / len(runs_loss)
        loss_with_std = sum(runs_loss_with_std) / len(runs_loss_with_std)

        if verbose:
            print(f"[{ith_model}/{n_models}] For eval model {eval_model}, \t MSE is {round(loss, 2)}, \t NMSE is {round(loss_with_std, 2)}")
        
        test_mse_losses[eval_model] = loss
        test_nmse_losses[eval_model] = loss_with_std

        ith_model += 1

    mean_test_loss = sum(test_mse_losses.values()) / len(test_mse_losses)
    mean_test_loss_with_std = sum(test_nmse_losses.values()) / len(test_nmse_losses)

    if verbose:
        print(f"The mean MSE is {round(mean_test_loss, 2)}, the mean NMSE is {round(mean_test_loss_with_std, 2)}")

    return test_mse_losses, test_nmse_losses

"""
def VAE_LOOCV(dataset, mask, machine_model, epochs, batch_size, lr, verbose=False):
    models = list(dataset.keys())
    n_models = len(models)
    ith_model = 1

    test_losses = {}

    flat_dataset, flat_forced_responses, mean_flat_grids, std_flat_grids = normalize_flatten_dataset(dataset, mask)

    for eval_model in models:
        train_models = [model for model in models if model != eval_model]
        
        X_train, y_train = [], []
        for train_model in train_models:
            flattened_forced_response = flat_forced_responses[train_model].flatten()
            for run, flat_grid_timeserie in flat_dataset[train_model].items():
                flattened_grid_timeserie = flat_grid_timeserie.flatten()
            
                X_train.append(flattened_grid_timeserie)
                y_train.append(flattened_forced_response)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train, y_train = torch.tensor(X_train).to(torch.float32), torch.tensor(y_train).to(torch.float32)

        norm_flat_grids, norm_flat_forced_responses = center_flatten_model(dataset, eval_model, mask)

        X_test, y_test = [], []
        flattened_forced_response = norm_flat_forced_responses.flatten()
        for run, flat_grid_timeserie in norm_flat_grids.items():
            flattened_grid_timeserie = flat_grid_timeserie.flatten()
            
            X_test.append(flattened_grid_timeserie)
            y_test.append(flattened_forced_response)

        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test, y_test = torch.tensor(X_test).to(torch.float32), torch.tensor(y_test).to(torch.float32)

        train_dataset = CopeDataset(samples=X_train, labels=y_train)
        test_dataset = CopeDataset(samples=X_test, labels=y_test)
        
        trained_machine_model = train_vae(machine_model, train_dataset, epochs, batch_size, lr)
        
        loss = eval_vae(trained_machine_model, test_dataset)

        if verbose:
            print(f"[{ith_model}/{n_models}] The NMSE for model {eval_model} is {round(loss, 3)} \n")
        
        test_losses[eval_model] = loss
        ith_model += 1

    mean_test_loss = sum(test_losses.values()) / len(test_losses)

    if verbose:
        print(f"The mean RMSE is {round(mean_test_loss, 3)} \n")

    return test_losses
"""

def VAE_LOOCV(dataset, mask, model_configs, training_configs, verbose=False):
    models = list(dataset.keys())
    n_models = len(models)
    ith_model = 1

    hidden_dim, latent_dim = list(model_configs.values())
    device, epochs, batch_size, lr = list(training_configs.values())

    test_mse_losses = {}
    test_nmse_losses = {}

    flat_dataset, model_with_mean_forced_responses, _, _ = normalize_flatten_dataset(dataset, mask)

    for eval_model in models:
        train_models = [model for model in models if model != eval_model]
        machine_model = VAE(input_dim=44472, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
        
        X_train, y_train = [], []
        for train_model in train_models:
            flat_mean_forced_response = model_with_mean_forced_responses[train_model].flatten()
            for run, timegrid in flat_dataset[train_model].items():
                flat_timegrid = timegrid.flatten()
            
                X_train.append(flat_timegrid)
                y_train.append(flat_mean_forced_response)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train, y_train = torch.tensor(X_train).to(torch.float32), torch.tensor(y_train).to(torch.float32)

        runs_with_timegrids_centered, mean_forced_responses = center_flatten_model(dataset, eval_model, mask)
        flat_mean_forced_response = mean_forced_responses.flatten()

        test_var = torch.var(torch.tensor(mean_forced_responses), dim=0).to(device)
        test_var = torch.cat([test_var] * 34, dim=0)

        X_test, y_test = [], []
        for run, timegrid in runs_with_timegrids_centered.items():
            flat_timegrid = timegrid.flatten()

            X_test.append(flat_timegrid)
            y_test.append(flat_mean_forced_response)

        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test, y_test = torch.tensor(X_test).to(torch.float32), torch.tensor(y_test).to(torch.float32)

        train_dataset = CopeDataset(samples=X_train, labels=y_train)
        test_dataset = CopeDataset(samples=X_test, labels=y_test)

        trained_machine_model = train_vae(machine_model, train_dataset, device, epochs, batch_size, lr)
        loss, loss_with_std = eval_vae(trained_machine_model, test_dataset, device, test_var)
        
        if verbose:
            print(f"[{ith_model}/{n_models}] For eval model {eval_model}, \t MSE is {round(loss, 2)}, \t NMSE is {round(loss_with_std, 2)}")
        
        test_mse_losses[eval_model] = loss
        test_nmse_losses[eval_model] = loss_with_std

        ith_model += 1

    mean_test_loss = sum(test_mse_losses.values()) / len(test_mse_losses)
    mean_test_loss_with_std = sum(test_nmse_losses.values()) / len(test_nmse_losses)

    if verbose:
        print(f"The mean MSE is {round(mean_test_loss, 2)}, the mean NMSE is {round(mean_test_loss_with_std, 2)}")

    return test_mse_losses, test_nmse_losses


def train_vae(model, train_dataset, device, epochs, batch_size, lr):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        print("Epoch", epoch + 1, "complete !", "\tAverage training loss: ", overall_loss / (batch_idx*batch_size))
        sys.stdout.flush()

    return model


def eval_vae(trained_model, test_dataset, device, test_var):
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    trained_model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)

            x_hat, _, _ = trained_model(x)
            
            temp = (x_hat - y)**2
            losses = temp.sum(dim=1)
            loss = torch.mean(losses).item()

            temp /= test_var
            losses_with_std = temp.sum(dim=1)
            loss_with_std = torch.mean(losses_with_std).item()

    return loss, loss_with_std
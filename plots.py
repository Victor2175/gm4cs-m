import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import seaborn as sns
from utils import normalize_pixel, normalize_pixel_other, from_flat_to_grid


def plot_histogram_runs(dataset):
    """
    Plots a histogram of the number of runs per model in the dataset.

    Keyword arguments:
    dataset (dict): The climate dataset
    """
    runs_per_model = [len(dataset[mod]) for mod in dataset.keys()]
    total_number_of_runs = sum(runs_per_model)

    plt.figure(figsize=(18, 4))
    plt.title(f"Number of runs per model, total = {total_number_of_runs}")
    plt.bar(height=runs_per_model, x=dataset.keys(), width=0.5)
    plt.xticks(rotation=90)
    plt.show()


def plot_timeseries(dataset, model, pixel):
    normalized_timeseries, mean_forced_response = normalize_pixel(dataset, model, pixel)

    for t in normalized_timeseries:
        plt.plot(t, color='blue', linewidth=0.5)

    plt.plot(mean_forced_response, color='red')
    plt.title(f'Timeseries of model {model} for pixel {pixel}')
    plt.ylabel('SST anomalies')
    plt.show()


def plot_timeseries_other(dataset, model, pixel):
    normalized_timeseries, mean_forced_response = normalize_pixel_other(dataset, model, pixel)

    for t in normalized_timeseries:
        plt.plot(t, color='blue', linewidth=0.5)

    plt.plot(mean_forced_response, color='red')
    plt.title(f'Timeseries of model {model} for pixel {pixel}')
    plt.ylabel('SST anomalies')
    plt.show()


def plot_heatmap_years(dataset, model, run, years=[2000, 2005, 2010]):
    for y in years:
        if y < 1980 or 2013 < y: 
            raise ValueError('The year range provided is not in [1980, 2013]')

    simulation = dataset[model][run][131:, :, :]
    f, axs = plt.subplots(1, len(years), sharex=True, sharey=True, figsize=(8*len(years), 6))

    for i in range(len(years)):
        cells = np.flip(simulation[years[i] - 1980, :, :], 0)
        sns.heatmap(cells, ax=axs[i])


def plot_heatmap(dataset, model, run, year=2010):
    if year < 1980 or 2013 < year: 
        raise ValueError('The year provided is not in [1980, 2013]')
        
    simulation = dataset[model][run][131:, :, :]
    grid = np.flip(simulation[year - 1980, :, :], 0)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(grid, cmap='coolwarm', linewidths=0.5)


def plot_grid(grid, union_nan_mask, flip=True):
    if len(grid.shape) != 2:
        grid = from_flat_to_grid(grid, union_nan_mask)
    
    if flip:
        grid = np.flip(grid, 0)
    
    abs_max = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)))

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(grid, cmap='coolwarm', vmin=-abs_max, vmax=abs_max, linewidths=0.5)


def plot_duo_grids(x_hat, y, union_nan_mask, flip=True):
    if len(x_hat.shape) != 2:
        x_hat = from_flat_to_grid(x_hat, union_nan_mask)

    if len(y.shape) != 2:
        y = from_flat_to_grid(y, union_nan_mask)

    if flip:
        x_hat = np.flip(x_hat, 0)
        y = np.flip(y, 0)

    fig, ax = plt.subplots(1, 2, figsize=(26, 8))

    abs_max = max(abs(np.nanmin(x_hat)), abs(np.nanmax(x_hat)), abs(np.nanmin(y)), abs(np.nanmax(y)))

    sns.heatmap(x_hat, ax=ax[0], cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Intensity'}, cbar=False, xticklabels=False, yticklabels=False, vmin=-abs_max, vmax=abs_max)
    ax[0].set_title("Predicted forced response", fontsize=20)

    sns.heatmap(y, ax=ax[1], cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Intensity'}, xticklabels=False, yticklabels=False,  vmin=-abs_max, vmax=abs_max)
    ax[1].set_title("Actual forced response", fontsize=20)

    plt.tight_layout()
    plt.show()


def plot_trio_grids(x, x_hat, y, union_nan_mask, flip=True):
    if len(x.shape) != 2:
        x = from_flat_to_grid(x, union_nan_mask)

    if len(x_hat.shape) != 2:
        x_hat = from_flat_to_grid(x_hat, union_nan_mask)

    if len(y.shape) != 2:
        y = from_flat_to_grid(y, union_nan_mask)

    if flip:
        x = np.flip(x, 0)
        x_hat = np.flip(x_hat, 0)
        y = np.flip(y, 0)

    fig, ax = plt.subplots(1, 3, figsize=(32, 8))

    abs_max = max(abs(np.nanmin(x)), abs(np.nanmin(x)), abs(np.nanmin(x_hat)), abs(np.nanmax(x_hat)), abs(np.nanmin(y)), abs(np.nanmax(y)))

    sns.heatmap(x, ax=ax[0], cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Standard temp.'}, cbar=False, xticklabels=False, yticklabels=False, vmin=-abs_max, vmax=abs_max)
    ax[0].set_title("Datapoint", fontsize=20)

    sns.heatmap(x_hat, ax=ax[1], cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Intensity'}, cbar=False, xticklabels=False, yticklabels=False, vmin=-abs_max, vmax=abs_max)
    ax[1].set_title("Predicted forced response", fontsize=20)

    sns.heatmap(y, ax=ax[2], cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Intensity'}, xticklabels=False, yticklabels=False,  vmin=-abs_max, vmax=abs_max)
    ax[2].set_title("Actual forced response", fontsize=20)

    plt.tight_layout()
    plt.show()


def get_animated_timeserie(timeserie, union_nan_mask, flip=True, flattened=True):
    timeserie = np.array(timeserie)

    abs_max = max(abs(np.nanmin(timeserie)), abs(np.nanmax(timeserie)))

    if flattened:
        timeserie = np.reshape(timeserie, (34, -1))

    fig = plt.figure()

    def animate(i):
        grid = timeserie[i]

        if len(grid.shape) != 2:
            grid = from_flat_to_grid(grid, union_nan_mask)

        if flip:
            grid = np.flip(grid, 0)

        plt.clf()
        res = sns.heatmap(grid, cmap='coolwarm', linewidths=0.5, vmin=-abs_max, vmax=abs_max, xticklabels=False, yticklabels=False, cbar_kws={'label': 'Intensity'})
        res.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        res.axhline(y=grid.shape[0], color='k', linewidth=2, alpha=0.5)
        res.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        res.axvline(x=grid.shape[1], color='k', linewidth=2, alpha=0.5)
        plt.title(f"Time step {i} ({1980+i})")

    animation = ani.FuncAnimation(fig, animate, frames=34, interval=1000, repeat=True)
    return animation


def get_duo_animated_timeserie(left_timeserie, right_timeserie, union_nan_mask, left_title='x_hat (predicted)', right_title='y (ground truth)', flip=True, flattened=True):
    abs_max = max(abs(np.nanmin(left_timeserie)), abs(np.nanmax(left_timeserie)), abs(np.nanmin(right_timeserie)), abs(np.nanmax(right_timeserie)))

    if flattened:
        left_timeserie = np.reshape(left_timeserie, (34, -1))
        right_timeserie = np.reshape(right_timeserie, (34, -1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18, 6))

    # Create a colorbar axis
    cbar_ax = fig.add_axes([0.49, 0.15, 0.02, 0.7])

    def animate(i): 
        ax1.clear()  # Clear the left axis
        ax2.clear()  # Clear the right axis

        x_grid = left_timeserie[i]
        y_grid = right_timeserie[i]
    
        if len(x_grid.shape) != 2:
            x_grid = from_flat_to_grid(x_grid, union_nan_mask)

        if len(y_grid.shape) != 2:
            y_grid = from_flat_to_grid(y_grid, union_nan_mask)
       
        if flip:
            x_grid = np.flip(x_grid, 0)
            y_grid = np.flip(y_grid, 0)
        
        # Plot heatmaps
        left = sns.heatmap(x_grid, ax=ax1, cmap='coolwarm', linewidths=0.5, cbar=True, square=False, vmin=-abs_max, vmax=abs_max, xticklabels=False, yticklabels=False, cbar_ax=cbar_ax)
        left.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        left.axhline(y=x_grid.shape[0], color='k', linewidth=2, alpha=0.5)
        left.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        left.axvline(x=x_grid.shape[1], color='k', linewidth=2, alpha=0.5)
        ax1.set_title(left_title, fontsize=15)

        right = sns.heatmap(y_grid, ax=ax2, cmap='coolwarm', linewidths=0.5, cbar=True, square=False, vmin=-abs_max, vmax=abs_max, xticklabels=False, yticklabels=False, cbar_ax=cbar_ax)
        right.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        right.axhline(y=y_grid.shape[0], color='k', linewidth=2, alpha=0.5)
        right.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        right.axvline(x=y_grid.shape[1], color='k', linewidth=2, alpha=0.5)
        ax2.set_title(right_title, fontsize=15)

        plt.title(f"Time step {i} ({1980+i})", x=0.5, y=1.08, fontsize=20)

    animation = ani.FuncAnimation(fig, animate, frames=34, interval=1000, repeat=True)
    return animation


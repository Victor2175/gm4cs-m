import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import normalize_pixel


def plot_histogram_runs(dataset):
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


def plot_heatmap_years(dataset, model, run, years=[2000, 2005, 2010]):
    for y in years:
        if y < 1980 or 2013 < y: 
            raise ValueError('The year range provided is not in [1980, 2013]')

    simulation = dataset[model][run][131:, :, :]
    f, axs = plt.subplots(1, len(years), sharex=True, sharey=True, figsize=(8*len(years), 6))

    for i in range(len(years)):
        cells = np.flip(simulation[years[i] - 1980, :, :], 0)
        sns.heatmap(cells, ax=axs[i])


def plot_heat_map(dataset, model, run, year=2010):
    if year < 1980 or 2013 < year: 
        raise ValueError('The year provided is not in [1980, 2013]')
        
    simulation = dataset[model][run][131:, :, :]
    grid = np.flip(simulation[year - 1980, :, :], 0)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(grid, cmap='coolwarm', linewidths=0.5)
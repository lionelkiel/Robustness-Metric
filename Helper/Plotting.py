import matplotlib.pyplot as plt
import numpy as np
from Helper.ImportDatasetsFairness import networks

# Added xlabel input, might destroy some things?
def Boxplots(plots, ylabel=None, xlabel=None, hline='N', xrot=False, ylim=None, figsize=(1, 1),
             col=['black', 'red', 'blue', 'green'], legend_loc='upper right', title=None, means=False, save=None, dpi=800):
    plotlabels = list(plots.keys())
    w, h = *figsize,
    fig, ax = plt.subplots(figsize=(6.4 * w, 4.8 * h))
    if ylim:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if hline != 'N':
        plt.axhline(hline, linestyle='--')
    if title:
        fig.suptitle(title, y=1)

    n_dicts = len(plots)
    if type(plots[plotlabels[0]]) != dict:
        n_dicts = 1
    boxes = []
    for i in range(n_dicts):
        c = col[i]
        if n_dicts == 1:
            labels, data = [*zip(*plots.items())]
        else:
            labels, data = [*zip(*plots[plotlabels[i]].items())]
        data = [np.array(dat)[~np.isnan(np.array(dat))] for dat in data]
        pos = np.arange(0, len(labels)) * n_dicts + 1 + i
        tempbox = ax.boxplot(data, positions=pos, showmeans=means, patch_artist=True,
                             boxprops=dict(facecolor='white', color=c),
                             capprops=dict(color=c),
                             whiskerprops=dict(color=c),
                             flierprops=dict(color=c, markeredgecolor=c)
                             )
        boxes.append(tempbox)
    pos = np.arange(0, len(labels)) * n_dicts + 1 + (n_dicts - 1) / 2
    ax.set_xticks(pos)
    if xrot:
        ax.set_xticklabels(labels, rotation=45, ha='right')
    else:
        ax.set_xticklabels(labels)
    if n_dicts != 1:
        ax.legend([box["boxes"][0] for box in boxes], plotlabels, loc=legend_loc)
    if save:
        plt.savefig(save, dpi=dpi)
    plt.show()
    
## Evaluation plots

def plot_uncertainty_size(quantiles_networks, names=False, lens=np.arange(10, 850)):
    """
    Plots the mean uncertainty and 95% confidence intervals of the uncertainty size for different methods and network.
    Parameters:
    - quantiles_networks (dict): A dictionary containing the quantiles data for each network and method.
    - names (list, optional): A list of names for each method. Defaults to False.
    - lens (numpy.ndarray, optional): An array of sample lengths. Defaults to np.arange(10, 850).
    Returns:
    - fig_list (list): A list of figures.
    - ax_list (list): A list of axes.
    """
    networks = quantiles_networks.keys()
    
    fig_list = []
    ax_list = []
    
    # Determine global y-axis limits
    global_max = float('-inf')
    
    for network in networks:
        for method_data in quantiles_networks[network]:
            uncertainty_size = method_data[:, :, 1] - method_data[:, :, 0]
            mean_uncertainty = np.mean(uncertainty_size, axis=0)
            conf_int = np.percentile(uncertainty_size, [2.5, 97.5], method='nearest', axis=0)
            global_max = max(global_max, np.max(conf_int[1]))
    
    for network in networks:
        fig, ax = plt.subplots()
        
        for method_index, method_data in enumerate(quantiles_networks[network]):
            uncertainty_size = method_data[:, :, 1] - method_data[:, :, 0]
            mean_uncertainty = np.mean(uncertainty_size, axis=0)
            conf_int = np.percentile(uncertainty_size, [2.5, 97.5], method='nearest', axis=0)
            
            # Plot mean uncertainty
            label = f'Method {method_index} - {names[method_index]}' if names else f'Method {method_index}'
            ax.plot(lens, mean_uncertainty, label=label)
            
            # Plot confidence intervals
            ax.fill_between(lens, conf_int[0], conf_int[1], alpha=0.2)
        
        # Set y-axis limits
        ax.set_ylim(0, global_max)
        
        # Legend
        ax.legend()
        # Labels
        ax.set_xlabel('Length of sample')
        ax.set_ylabel('Uncertainty size')
        ax.set_title(f'Network: {network} - Uncertainty (of quantile) size mean and 95% CI over length of sample')

        fig_list.append(fig)
        ax_list.append(ax)
    
    return fig_list, ax_list
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
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

def plot_uncertainty_size(quantiles_networks, names=False, log = False, lens=np.arange(10, 850), confidence_quantiles = [0.025, 0.975], quantile_method = 'nearest'):
    """
    Plots the mean uncertainty and 95% confidence intervals of the uncertainty size for different methods and network.
    Parameters:
    - quantiles_networks (dict): A dictionary containing the quantiles data for each network and method.
    - names (list, optional): A list of names for each method. Defaults to False.
    - lens (numpy.ndarray, optional): An array of sample lengths. Defaults to np.arange(10, 850).
    - confidence_quantiles (list, optional): A list of 2 confidence quantiles. Defaults to [0.025, 0.975].
    - quantile_method (str, optional): The method used to calculate quantiles. Defaults to 'nearest'.
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
        methods = quantiles_networks[network]
        if not isinstance(methods, list):
            methods = [methods]
        for method_data in methods:
            if not log:
                method_data = np.exp(method_data)
            uncertainty_size = method_data[:, :, 1] - method_data[:, :, 0]
            mean_uncertainty = np.mean(uncertainty_size, axis=0)
            conf_int = np.quantile(uncertainty_size, confidence_quantiles, method=quantile_method, axis=0)
            global_max = max(global_max, np.max(conf_int[1]))
    
    for network in networks:
        fig, ax = plt.subplots()
        
        methods = quantiles_networks[network]
        if not isinstance(methods, list):
            methods = [methods]
        for method_index, method_data in enumerate(methods):
            if not log:
                method_data = np.exp(method_data)
            uncertainty_size = method_data[:, :, 1] - method_data[:, :, 0]
            mean_uncertainty = np.mean(uncertainty_size, axis=0)
            conf_int = np.quantile(uncertainty_size, confidence_quantiles, method=quantile_method, axis=0)
            
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
        if log:
            ax.set_ylabel('Uncertainty size in log critical epsilon values')
        else:
            ax.set_ylabel('Uncertainty size in critical epsilon values')
        ax.set_title(f'Network: {network} - Mean of uncertainty and 95% CI of quantile, over length of sample')

        fig_list.append(fig)
        ax_list.append(ax)
    
    return fig_list, ax_list

def plot_uncertainty_distribution(quantiles_networks, plot_type = 'mean', names=False, log = False, lens=np.arange(10, 850), confidence_quantiles = [0.025, 0.975], quantile_method = 'nearest', all_in_one = False):
    """
    Plots the uncertainty distribution for given quantiles of networks.
    Parameters:
    - quantiles_networks (dict): A dictionary containing the quantiles of networks.
    - plot_type (str): The type of plot to generate. Default is 'mean'.
    - names (bool): Whether to include method names in the plot labels. Default is False.
    - log (bool): Whether to plot the y-axis on a logarithmic scale. Default is False.
    - lens (numpy.ndarray): An array of sample lengths. Default is np.arange(10, 850).
    - confidence_quantiles (list): A list of confidence quantiles for the uncertainty. Default is [0.025, 0.975].
    - quantile_method (str): The method to compute quantiles. Default is 'nearest'.
    - all_in_one (bool): Whether to plot all networks in a single plot. Default is False.
    Returns:
    - fig_list (list): A list of matplotlib Figure objects.
    - ax_list (list): A list of matplotlib Axes objects.
    """
    
    
    plot_data = {}
    
    fig_list = []
    ax_list = []
    
    # Determine global y-axis limits
    global_max = float('-inf')
    global_min = float('inf')
    
    for network in networks:
        plot_data[network] = []
        methods = quantiles_networks[network]
        if not isinstance(methods, list):
            methods = [methods]
        
        for method_index, method_data in enumerate(methods):
            if not log:
                method_data = np.exp(method_data)
            
            if plot_type == 'mean':
                mean_left = np.mean(method_data[:, :, 0], axis=0)
                mean_right = np.mean(method_data[:, :, 1], axis=0)
                plot_data[network].append((mean_left, mean_right))
                global_max = max(global_max, np.max(mean_right))
                global_min = min(global_min, np.min(mean_left))
            
            # 95% confidence interval for the uncertainty
            if 'conf' in plot_type:
                conf_int_left = np.quantile(method_data[:, :, 0], confidence_quantiles[0], method=quantile_method, axis=0)
                conf_int_right = np.quantile(method_data[:, :, 1], confidence_quantiles[1], method=quantile_method, axis=0)
                plot_data[network].append((conf_int_left, conf_int_right))
                global_max = max(global_max, np.max(conf_int_right))
                global_min = min(global_min, np.min(conf_int_left))
    
    if all_in_one:
        fig, ax = plt.subplots()
    
    for network in networks:
        if not all_in_one:
            fig, ax = plt.subplots()
        for method_index, (left, right) in enumerate(plot_data[network]):
            # Plot confidence intervals
            label = f'Method {method_index} - {names[method_index]}' if names else f'Method {method_index}'
            if all_in_one:
                label = f'{network} - {label}'
            
            ax.fill_between(lens, left, right, alpha=0.2, label=label)
        
        
        ax.legend()
        
        # Set y-axis limits
        ax.set_ylim(global_min, global_max)
        
        ax.set_xlabel('Length of sample')
        if log:
            ax.set_ylabel('Log critical epsilon value')
        else:
            ax.set_ylabel('Critical epsilon value')
        
        if plot_type == 'mean':
            ax.set_title(f'{network} - Mean left and right side of quantile, over length of sample')
        elif 'conf' in plot_type:
            ax.set_title(f'{network} - 95% CI left and right side of quantile, over length of sample')
        
        if all_in_one:
            ax.set_title(f'All networks - {plot_type} uncertainty distribution')
        
        fig_list.append(fig)
        ax_list.append(ax)
    
    return fig_list, ax_list

def plot_binary(quantiles_networks, names=False, lens=np.arange(10, 850)):
    """
    Plots binary comparison between different networks. The plot shows the percentage of when network 1 has a better robustness than network 2, when network 2 has a better robustness than network 1 and when the uncertainties overlap.
    Args:
        quantiles_networks (dict): A dictionary containing quantiles data for different networks.
        names (bool, optional): Whether to include method names in the plot titles. Defaults to False.
        lens (numpy.ndarray, optional): Array of sample lengths. Defaults to np.arange(10, 850).
    Returns:
        tuple: A tuple containing lists of figures and axes.
    """
    
    
    plot_data = {}
    fig_list = []
    ax_list = []
    
    plot_data = {}
    n_samples = quantiles_networks[networks[0]][0].shape[0]
    
    for (net1, net2) in combinations(networks, 2):
        plot_data[(net1, net2)] = []
        methods1 = quantiles_networks[net1]
        methods2 = quantiles_networks[net2]
        if not isinstance(methods1, list):
            methods1 = [methods1]
        if not isinstance(methods2, list):
            methods2 = [methods2]
        
        # Determine how often uncertainties overlap and how often one wins over the other
        for method_index, (method_data1, method_data2) in enumerate(zip(methods1, methods2)):
            left_quantile_net1 = method_data1[:, :, 0]
            right_quantile_net1 = method_data1[:, :, 1]
            left_quantile_network2 = method_data2[:, :, 0]
            right_quantile_net2 = method_data2[:, :, 1]
            
            # Check when network 1 has a better robustness than network 2
            net1_wins = np.sum(right_quantile_net1 <= left_quantile_network2, axis=0) / n_samples
            
            # Check when network 2 has a better robustness than network 1
            net2_wins = np.sum(right_quantile_net2 <= left_quantile_net1, axis=0) / n_samples
            
            # Check if the uncertainties overlap
            overlap = np.sum((left_quantile_net1 < right_quantile_net2) & (left_quantile_network2 < right_quantile_net1), axis=0) / n_samples

            plot_data[(net1, net2)].append((net1_wins, net2_wins, overlap))
        
        # subplots horizontally per method
        n_methods = len(plot_data[(net1, net2)])
        fig, ax = plt.subplots(1, n_methods, figsize=(6 * n_methods, 4))
        if n_methods == 1:
            ax = [ax]
        
        for method_index, (net1_wins, net2_wins, overlap) in enumerate(plot_data[(net1, net2)]):
            # Plot the percentages of when network 1 wins, network 2 wins and when they overlap
            ax[method_index].plot(lens, net1_wins, label=f'{net1} wins')
            ax[method_index].plot(lens, net2_wins, label=f'{net2} wins')
            ax[method_index].plot(lens, overlap, label='Overlap')
            
            # Labels and titles
            ax[method_index].set_xlabel('Length of sample')
            ax[method_index].set_ylabel('Ratio')
            title = f'Method {method_index} - {names[method_index]}' if names else f'Method {method_index}'
            ax[method_index].set_title(title)
            ax[method_index].legend()

        fig.suptitle(f'{net1} vs {net2} - Binary comparison of uncertainties')
        fig_list.append(fig)
        ax_list.append(ax)
    
    return fig_list, ax_list
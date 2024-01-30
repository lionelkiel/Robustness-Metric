import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

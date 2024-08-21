import numpy as np

from scipy.special import comb
from Helper.ImportDatasetsFairness import df_epsilon_crit

def binomial(n, p, x):
    '''
    :param n: number of trials
    :param p: probability of success, value of a (quantile)
    :param x: number of successes

    :return: probability of x successes
    '''

    return comb(n, x) * (p ** x) * ((1 - p) ** (n - x))

def binomial_bounds(n, p, alpha):
    '''
    :param n: number of trials
    :param p: probability of success, value of a (quantile)
    :param alpha: confidence interval

    :return: lower and upper bound of confidence interval
    '''
    probs = np.arange(0, n + 1)
    probs = binomial(n, p, probs)

    # take sum of probabilities until we reach alpha/2
    cumulated_probs = np.cumsum(probs)
    lower_index = np.where(cumulated_probs <= alpha / 2)[0][-1] # we want [ <= alpha/2 ][ >= 1-alpha ][ <= alpha/2 ]
    upper_index = np.where(cumulated_probs >= 1 - alpha / 2)[0][0] # this way the confidence interval is at least 1-alpha
    
    return lower_index, upper_index

def get_quantile(dat, sigma, verbose=False):
    '''
    :param dat: numpy array of data
    :param sigma: quantile

    :return: confidence interval for sigma quantile given the data
    '''

    n = len(dat)

    # We sort the critical epsilons
    order_statistics = np.sort(dat)
    # We use the order statistics to estimate the sigma quantile
    index = int(n * sigma) + 1  # As given by David et al. 2003 (Order Statistics)
    lower_index, upper_index = binomial_bounds(n, sigma, 0.05)
    
    if verbose:
        print(f"Indexes: {index}, {lower_index}, {upper_index}")
    
    return order_statistics[index], order_statistics[lower_index], order_statistics[upper_index]

def get_quantile_fairness(network, sigma):
    '''
    :param network: name of network
    :param sigma: quantile

    :return: confidence interval for sigma quantile given the entire fairness dataset, cannot define data in this one
    '''

    # Take all critical epsilons of the test set and put into numpy array
    df_for_network = df_epsilon_crit[df_epsilon_crit['network'] == network]
    df_for_network = df_for_network[df_for_network['ds'] == 'test']
    df_for_network = df_for_network.dropna() # remove nans
    crit_epsilons = df_for_network['Epsilon'].to_numpy()
    
    return get_quantile(crit_epsilons, sigma)
import numpy as np
from tqdm import tqdm
from scipy.special import comb
from Helper.ImportDatasetsFairness import df_epsilon_crit, log_crit_epsilons_network, networks
from scipy.stats import nct, norm


def binomial(n, p, x):
    '''
    :param n: number of trials
    :param p: probability of success, value of a (quantile)
    :param x: number of successes

    :return: probability of x successes
    '''

    return comb(n, x) * (p ** x) * ((1 - p) ** (n - x))

def binomial_exact(n, p, x):
    '''    
    :param n: number of trials
    :param p: probability of success, value of a (quantile)
    :param x: number of successes

    :return: probability of x successes
    '''

    return np.array([comb(n, xi, exact=True) for xi in x]) * (p ** x) * ((1 - p) ** (n - x))

def calculate_confidence_interval(cumulated_probs, alpha):
    '''
    :param cumulated_probs: array of cumulative probabilities
    :param alpha: confidence interval

    :return: lower and upper bound of confidence interval
    '''
    # take sum of probabilities until we reach alpha/2
    lower_index = np.where(cumulated_probs <= alpha / 2)[0][-1] if cumulated_probs[0] <= alpha / 2 else 0 # if the first value is smaller than alpha/2, we take it
    upper_index = np.where(cumulated_probs >= 1 - alpha / 2)[0][0] if cumulated_probs[-1] >= 1 - alpha / 2 else -1 # if the last value is bigger than 1-alpha/2, we take it
    
    return lower_index, upper_index


### Non-parametric method

def binomial_bounds(n, p, alpha):
    '''
    DEPRECATED, Implemented directly into get_quantile function
    
    :param n: number of trials
    :param p: probability of success, value of a (quantile)
    :param alpha: confidence interval

    :return: lower and upper bound of confidence interval
    '''
    probs = np.arange(0, n + 1)
    probs = binomial(n, p, probs)

    cumulated_probs = np.cumsum(probs)
    return calculate_confidence_interval(cumulated_probs, alpha)

def get_quantile(dat, sigma, verbose=False):
    '''
    DEPRECATED, more efficient version implemented in get_quantile_nonparam. This function is only meant for one sample.
    A sample in this case means, take n arbitrary crit_epsilons from the test set.
    So multiple samples means we take n arbitrary crit_epsilons multiple times from the test set, see sample_from_data function.
    
    :param dat: numpy array of data, one sample
    :param sigma: quantile

    :return: nonparametric confidence interval for sigma quantile given the data
    '''

    n = len(dat)

    # We sort the critical epsilons
    order_statistics = np.sort(dat)
    # We use the order statistics to estimate the sigma quantile
    index = int(n * sigma) + 1  # As given by David et al. 2003 (Order Statistics)
    probabilities = np.arange(0, n + 1)
    probabilities = binomial(n, sigma, probabilities)

    cumulated_probs = np.cumsum(probabilities)
    lower_index, upper_index = calculate_confidence_interval(cumulated_probs, 0.05)
    
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


### Parametric methods

def cdf_order_statistic_normal(index, n, x, mean, std, exact=False):
    r'''
    DEPRECATED, too inneficient. The distribution of the order statistic is given by F_{X_{(i)}}(x) = \sum_{j=i}^{n} {n \choose j} F(x)^j (1-F(x))^{n-j} where F(x) is the cdf of the distribution of the random variable X.
    Since we are dealing with the normal distribution, we can use the cdf of the normal distribution to calculate the cdf of the order statistic.
    
    :param index: the index of the order statistic
    :param n: number of samples
    :param x: value we input into the cdf
    :param mean: mean of the normal distribution
    :param std: standard deviation of the normal distribution
    :param exact: whether to use exact binomial calculation or not (optional)
    
    :return: cdf of the order statistic    
    '''
    
    probabilities = norm.cdf(x, loc=mean, scale=std)
    indices = np.arange(index, n+1)
    
    if exact:
        cdf = np.sum(binomial_exact(n, probabilities[:,None], indices), axis=1)
    else:
        cdf = np.sum(binomial(n, probabilities[:,None], indices), axis=1)
    
    return cdf

def get_quantile_normal_orderdistr(dat, sigma, mean, std, exact=False, verbose=False, x = False):
    '''
     DEPRECATED, too inneficient. Use get_quantile_normal_tdistr instead.
    
    :param dat: numpy array of data
    :param sigma: quantile
    :param mean: mean of the normal distribution
    :param std: standard deviation of the normal distribution
    :param exact: whether to use exact binomial calculation or not (optional)
    :param verbose: whether to print indexes or not (optional)
    :param x: values to calculate the cdf for (optional), if not given, it will be calculated from linspace of min and max of data with 10000 points

    :return: confidence interval for sigma quantile given the data
    '''

    n = len(dat)

    # We sort the critical epsilons
    order_statistics = np.sort(dat)
    # We use the order statistics to estimate the sigma quantile
    index = int(n * sigma) + 1  # As given by David et al. 2003 (Order Statistics)
    
    if x == False:
        lower_limit = np.min(dat)
        upper_limit = np.max(dat)
        x = np.linspace(lower_limit, upper_limit, 10000)

    cdf = cdf_order_statistic_normal(index, n, x, mean, std, exact)

    lower_index, upper_index = calculate_confidence_interval(cdf, 0.05)
    
    if verbose:
        print(f"Indexes: {index}, {lower_index}, {upper_index}")
    
    return order_statistics[index], x[lower_index], x[upper_index]

def get_quantile_normal_tdistr(dat, sigma, mean, std, verbose=False, alpha=0.05):
    '''
    This method uses the non-central t-distribution to calculate the confidence interval for the sigma quantile.
    
    :param dat: numpy array of data
    :param sigma: quantile
    :param mean: mean of the normal distribution
    :param std: standard deviation of the normal distribution
    :param verbose: whether to print indexes or not (optional)
    
    :return: confidence interval for sigma quantile given the data
    '''

    n = len(dat)

    noncentrality = -np.sqrt(n)*norm.ppf(sigma)
    tl = nct.ppf(1-alpha/2, n-1, noncentrality)
    t2 = nct.ppf(alpha/2, n-1, noncentrality)
    
    return mean - np.array([tl,t2])*std/np.sqrt(n)

### Run on the data

def sample_from_data(data, length_sample, n_samples, with_replacements = False):
    '''
    Sample from data with or without replacements. It is possible that a sample can repeat.
    A sample in this case means, take n arbitrary crit_epsilons from the test set.
    So multiple samples means we take n arbitrary crit_epsilons multiple times from the test set, see sample_from_data function.
    
    Parameters:
    data: numpy array
    length_sample: int indicating the length of each sample
    n_samples: int indicating the number of samples to take
    with_replacements: bool indicating whether to sample with or without replacements
    
    Returns:
    samples: numpy array of shape (n_samples, length_sample)
    '''
    samples = np.array([np.random.choice(data, length_sample, replace=with_replacements) for i in range(n_samples)])
    return samples

def calculate_quantiles(methods, n_samples = 1000, lens = np.arange(10, 850), networks = networks):
    
    '''
        Calculate quantiles for given methods, number of samples, lengths, and networks.
        
        Parameters:
            methods (list): List of methods to calculate quantiles. Methods have to return a numpy array of shape (n_samples, 2).
            n_samples (int, optional): Number of samples. Defaults to 1000.
            lens (numpy.ndarray, optional): Array of lengths. Defaults to np.arange(10, 850).
            networks (list, optional): List of networks. Defaults to all networks.
        Returns:
            dict: A dictionary containing quantiles for each network and method.
    '''
    quantiles_networks = {}
    
    NoErr = True
    for network in networks:
        # Initialize the dictionary
        quantiles_networks[network] = [np.zeros((n_samples, len(lens), 2)) for method in methods]
        
        print(f'----------------- Network: {network} -----------------')
        data = log_crit_epsilons_network[network]
        for i, length in tqdm(enumerate(lens), total=len(lens)):
            if length > np.shape(data)[0]:
                if NoErr:
                    print('Error: length of the data is smaller than the length of the sample')
                    NoErr = False
                continue
            
            # Run methods
            samples = sample_from_data(data, length, n_samples)
            for k, method in enumerate(methods):
                quantiles_networks[network][k][:, i] = method(samples)

        NoErr = True

    return quantiles_networks

def intervals_quantiles_nonparam(samples, sigma, verbose=False, normal_approx=False, method = 'linear', alpha=0.05):
    '''
        Calculate the confidence intervals for a given set of samples using nonparametric quantile estimation.
        With normal_approx = False it does the same as get_quantile function but much more efficient for calculating on many samples.
        
        Parameters:
        - samples: numpy.ndarray
            The samples for which to calculate the confidence intervals. The shape of the array should be (n_samples, length_of_a_sample).
        - sigma: float
            The quantile to estimate, ranging from 0 to 1.
        - verbose: bool, optional
            Whether to print additional information. Default is False.
        - normal_approx: bool, optional
            Whether to use the normal approximation method. Default is False.
        - method: str, optional
            The method to use for quantile estimation. Default is 'linear'.
        - alpha: float, optional
            The confidence level, ranging from 0 to 1. Default is 0.05.
        Returns:
        - intervals: numpy.ndarray
            The calculated confidence intervals.

    
    '''

    n = samples.shape[1]
    
    if not normal_approx:
        # Classic method with binomial
        probabilities = np.arange(0, n + 1)
        probabilities = binomial(n, sigma, probabilities)

        cumulated_probs = np.cumsum(probabilities)
        lower_index, upper_index = calculate_confidence_interval(cumulated_probs, alpha)
        
        if verbose:
            print(f"Indexes: {lower_index}, {upper_index}")
        
        order_statistics = np.sort(samples, axis=1)
        
        intervals = np.vstack([order_statistics[:, lower_index], order_statistics[:, upper_index]]).swapaxes(0, 1)
        
    else:  
        # Normal approximation of the binomial
        mean = n * sigma
        std = np.sqrt(n * sigma * (1 - sigma))
        
        # Calculate the lower and upper quantiles
        lower_quantile = norm.ppf(alpha/2, mean, std) / n
        upper_quantile = norm.ppf(1-alpha/2, mean, std) / n
        
        # Clip the values to be within the range [0, 1]
        lower_quantile = np.clip(lower_quantile, 0, 1)
        upper_quantile = np.clip(upper_quantile, 0, 1)
        
        intervals = np.quantile(samples, [lower_quantile, upper_quantile], axis=1, method=method).swapaxes(0, 1)
    
    return intervals

def intervals_quantiles_normal_tdistr(samples, sigma, alpha=0.05):
    """
    Calculate the confidence intervals for the quantiles with a normal distribution using the t-distribution.
    Parameters:
    - samples (ndarray): Array of shape (m, n) containing m samples of size n.
    - sigma (float): The desired quantile level.
    - alpha (float, optional): The significance level. Default is 0.05.
    Returns:
    - ndarray: Array of shape (m, 2) containing the lower and upper confidence intervals for each sample.
    """
    
    n = samples.shape[1]
    noncentrality = -np.sqrt(n)*norm.ppf(sigma)
    
    tl = nct.ppf(1-alpha/2, n-1, noncentrality)
    t2 = nct.ppf(alpha/2, n-1, noncentrality)
    
    means = np.mean(samples, axis=1)
    stds = np.std(samples, axis=1)
    
    return (means - np.array([[tl],[t2]])*stds/np.sqrt(n)).swapaxes(0,1)
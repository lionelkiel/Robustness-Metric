import numpy as np

from scipy.special import comb
from Helper.ImportDatasetsFairness import df_epsilon_crit
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
    lower_index = np.where(cumulated_probs <= alpha / 2)[0][-1] if cumulated_probs[0] <= alpha / 2 else 0
    upper_index = np.where(cumulated_probs >= 1 - alpha / 2)[0][0] if cumulated_probs[-1] >= 1 - alpha / 2 else -1
    
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
    :param dat: numpy array of data
    :param sigma: quantile

    :return: confidence interval for sigma quantile given the data
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
    '''
    The distribution of the order statistic is given by F_{X_{(i)}}(x) = \sum_{j=i}^{n} {n \choose j} F(x)^j (1-F(x))^{n-j} where F(x) is the cdf of the distribution of the random variable X.
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

def get_quantile_normal_method_1(dat, sigma, mean, std, exact=False, verbose=False, x = False):
    '''
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

def get_quantile_normal_method_2(dat, sigma, mean, std, verbose=False, alpha=0.05):
    '''
    :param dat: numpy array of data
    :param sigma: quantile
    :param mean: mean of the normal distribution
    :param std: standard deviation of the normal distribution
    :param verbose: whether to print indexes or not (optional)
    
    :return: confidence interval for sigma quantile given the data
    
    This method uses the non-central t-distribution to calculate the confidence interval for the sigma quantile.
    '''

    n = len(dat)

    noncentrality = -np.sqrt(n)*norm.ppf(sigma)
    tl = nct.ppf(1-alpha/2, n-1, noncentrality)
    t2 = nct.ppf(alpha/2, n-1, noncentrality)
    
    return mean - np.array([tl,t2])*std/np.sqrt(n)
import numpy as np
from tqdm import tqdm
from Helper.ImportDatasetsFairness import df_epsilon_crit, log_crit_epsilons_network, networks
from scipy.special import comb

# get confidence interval with binomial distribution
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
    lower_index = np.where(cumulated_probs <= alpha / 2)[0][-1] + 1
    upper_index = np.where(cumulated_probs >= 1 - alpha / 2)[0][0]

    return lower_index, upper_index

def get_quantile(network, sigma):
    '''
    :param network: name of network
    :param sigma: quantile

    :return: confidence interval for sigma quantile
    '''

    # Take all critical epsilons of the test set and put into numpy array
    df_for_network = df_epsilon_crit[df_epsilon_crit['network'] == network]
    df_for_network = df_for_network[df_for_network['ds'] == 'test']
    crit_epsilons = df_for_network['Epsilon'].to_numpy()
    n = len(crit_epsilons)

    # We sort the critical epsilons
    order_statistics = np.sort(crit_epsilons)
    # We use the order statistics to estimate the sigma quantile
    index = int(n * sigma) + 1  # As given by David et al. 1986
    lower_index, upper_index = binomial_bounds(n, sigma, 0.05)
    return order_statistics[index], order_statistics[lower_index], order_statistics[upper_index]


def get_metrics(final_distribution, a_bins, upper_confidence, lower_confidence):
    bin_size = a_bins[1] - a_bins[0]

    # Metric 1, binary
    lower_bound_area = a_bins[torch.where(final_distribution != 0)][0]
    upper_bound_area = a_bins[torch.where(final_distribution != 0)][-1] + bin_size

    if lower_bound_area < upper_confidence <= upper_bound_area:
        metric_1 = 1
    elif lower_bound_area <= lower_confidence < upper_bound_area:
        metric_1 = 1
    elif lower_confidence <= lower_bound_area and upper_confidence >= upper_bound_area:
        metric_1 = 1
    else:
        metric_1 = 0

    # Metric 2, probability given to the area
    if metric_1 == 1:
        lower_bound_index = torch.where(a_bins >= lower_confidence)[0]
        lower_bound_index = lower_bound_index[0]

        upper_bound_index = torch.where(a_bins + bin_size <= upper_confidence)[
            0]  # we don't include bins who's right side is larger than the quantile
        upper_bound_index = upper_bound_index[-1]

        metric_2 = torch.sum(final_distribution[lower_bound_index:upper_bound_index + 1]).item()

    else:
        metric_2 = 0

    # Metric 3, distance to P_max
    indices = torch.where(final_distribution == torch.max(final_distribution))
    Pmax = torch.mean(a_bins[indices]) + bin_size  # we take the right side of the bin
    if lower_confidence <= Pmax <= upper_confidence:
        metric_3 = 0
    else:
        metric_3 = torch.min(torch.abs(Pmax - upper_confidence), torch.abs(Pmax - lower_confidence)).item()

    # Metric 4, width
    metric_4 = (upper_bound_area - lower_bound_area).item()

    return metric_1, metric_2, metric_3, metric_4
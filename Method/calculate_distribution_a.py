import numpy as np
import torch
from torch.distributions.log_normal import LogNormal


def log_normal_cdf(x, mu, s):
    """
    Log normal cdf, we use this in the integral.

    Note that mu is in the log normal distribution thus it is not the mean. Mean is exp(mu + s^2/2).
    """
    # catch error. check for any s <= 0
    if torch.any(s <= 0):
        raise ValueError("s must be larger than 0")

    return 0.5 * (1 + torch.erf((torch.log(x) - mu) / (s * np.sqrt(2))))

def log_normal_pdf(x, mu, s):
    """
    Log normal pdf.
    """
    # catch error. check for any s <= 0
    if torch.any(s <= 0):
        raise ValueError("s must be larger than 0")

    return torch.exp(-0.5 * ((torch.log(x) - mu) / s) ** 2) / (x * s * np.sqrt(2 * np.pi))

def integrand(thetas, D):
    """
    value under integrand without prior, so integral( val * prior )
    currently prior is uniform so it doesn't matter, it would also not matter if it werent as then we just sample using MCMC and it would turn out the same.
    Note that errors are not caught here, they are caught in distribution_a.

    :param thetas: Sampled parameters to try, 2x... tensor, first row is mus, second row is sigmas
    :param D: Data of epsilons, 2xN tensor, first row is lower bounds, second row is upper bounds
    :return: Likelihoods (val's under integral), a tensor of size thetas.shape[1:], for each theta we have a value
    """
    # For broadcasting of thetas and D we add a dimension to thetas at the end
    thetas = thetas[..., None]
    # thetas is a 2x... tensor, first row is mus, second row is sigmas
    # D is a 2xN tensor, first row is lower bounds, second row is upper bounds
    vals = log_normal_cdf(D[1],thetas[0],thetas[1]) - log_normal_cdf(D[0],thetas[0],thetas[1])
    # Likelihoods is a tensor of size thetas.shape[1:], for each theta we have a value
    # We aggregate over the last dimension (size N) which is the dimension of D
    Likelihoods = vals.prod(dim=-1)
    return Likelihoods

def common_constant(val):
    """
    Common constant when inverting log normal cdf

    :param val: usually P(x<x0) = val
    """
    return torch.erfinv(2 * val - 1)

def calculate_mu_upper(a, sigma, p = 0.01, max_val = 0.4):
    """
    DEPRECATED: did not work as well as expected

    Calculate the upper bound of mu, this bound is given by the constraint that we know P(x > 0.4) <= 0.01
    """
    K = common_constant(1-torch.tensor([p]))/common_constant(sigma)
    return (K*torch.log(a) - torch.log(torch.tensor([max_val])))/(K-1)

def calculate_mu_and_s_from_mode(a, mode, sigma):
    """
    Calculate mu and s from mode and a.

    Mode of log normal is given by exp(mu - s^2), together with the value of a this gives us two equations.
    The answer has two possible solutions although one of them seems to be very peaked and thus we discard it.
    """
    # catch error, if there is an a that is larger or equal to mode, then we can't find a solution
    if torch.any(a >= mode):
        raise ValueError('a is larger or equal to mode')

    # Also catch error, if mode is larger than the upper bound of the mode, then we can't find a solution
    if torch.any(mode >= upper_bound_mode(a, sigma)):
        raise ValueError('mode is larger or equal to upper bound')

    K = common_constant(sigma)
    C1 = torch.log(mode)*2*K**2 + torch.log(a)**2
    C2 = 2*K**2+2*torch.log(a)
    mu1 = (C2 + torch.sqrt(C2**2 - 4*C1))/2
    # We discard mu2 as it gives veary peaked distributions
    #mu2 = (C2 - torch.sqrt(C2**2 - 4*C1))/2
    s1 = torch.sqrt(mu1 - torch.log(mode))
    # Some rounding errors occur making mu2 - torch.log(mode) a tiny bit negative, we thus set it to the absolute value
    #ind = torch.where(mu2 - torch.log(mode) < 0)
    #s2 = torch.sqrt(torch.abs(mu2 - torch.log(mode)))
    return torch.stack([mu1,s1])

def calculate_mu_and_s_from_median(a, median, sigma):
    """
    Calculate mu and s from median and a.

    Median of log normal is given by exp(mu), we also are given an a, this gives us two equations.
    """
    # catch error, if there is an a that is larger or equal to median, then we can't find a solution
    if torch.any(a >= median):
        raise ValueError('a is larger or equal to median')

    K = common_constant(sigma)
    mu = torch.log(median)
    s = (torch.log(a) - mu) / (K * np.sqrt(2))
    return torch.stack([mu, s])

def upper_bound_mode(a, sigma):
    """
    The upper bound is given by the fact that C2**2 - 4*C1 > 0. The lower bound is given by the fact that mu > log(a) which works out to be mode > a.
    """
    K = common_constant(sigma)
    return a * torch.exp((K**2)/2)

# We perform monte carlo integration for the integral
def distribution_a(a, D, sigma = 0.05, n_samples = 1000):
    """
    We want to find the probability of a given D, to forego computing the integral we use monte carlo integration.
    Since our prior is unigorm for now we just sample uniformly. NOTE: it will be normalized over the range of a which has been given as input

    :param a: a values to query, LxM tensor
    :param D: Data of epsilons, 2xN tensor, first row is lower bounds, second row is upper bounds
    :return:
    """
    # catch errors
    # sigma has to be between 0 and 1
    if sigma <= 0 or sigma >= 1:
        raise ValueError("sigma has to be between 0 and 1")
    # number of samples has to be an integer larger than 0
    if n_samples <= 0 or not isinstance(n_samples, int):
        raise ValueError("n_samples has to be an integer larger than 0")
    # a has to be a tensor
    if not isinstance(a, torch.Tensor):
        raise ValueError("a has to be a tensor")
    # D has to be a 2xN tensor
    if not isinstance(D, torch.Tensor) or D.shape[0] != 2:
        raise ValueError("D has to be a 2xN tensor")
    # values in a have to be larger than 0
    if torch.any(a <= 0):
        raise ValueError("a has to be larger than 0")

    # Turn floats into tensors
    sigma = torch.tensor([sigma])

    # Bounds of the medians
    lower_bound_median = a + 0.00001  # We add a small value to a as median > a
    # mu_upper = torch.exp(calculate_mu_upper(a_samples, sigma))
    upper_bound_median = 0.4
    # We sample medians uniformly
    medians = torch.rand(n_samples, *a.shape) * (upper_bound_median - lower_bound_median) + lower_bound_median
    # We calculate mu and s from the median
    thetas = calculate_mu_and_s_from_median(a, medians, sigma)

    # Monte carlo integration
    integral_approximation = torch.mean(integrand(thetas, D), dim=[0,1])

    # Normalization
    return integral_approximation / integral_approximation.sum()
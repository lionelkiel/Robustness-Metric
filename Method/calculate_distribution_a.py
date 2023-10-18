import numpy as np
import torch
from torch.distributions.log_normal import LogNormal


def log_normal_cdf(x, mu, s):
    """
    Log normal cdf, we use this in the integral
    """
    return 0.5 * (1 + torch.erf((torch.log(x) - mu) / (s * np.sqrt(2))))

def integrand(thetas, D):
    """
    value under integrand without prior, so integral( val * prior )
    currently prior is uniform so it doesn't matter, it would also not matter if it werent as then we just sample using MCMC and it would turn out the same

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
    Calculate the upper bound of mu, this bound is given by the constraint that we know P(x > 0.4) <= 0.01
    """
    K = common_constant(1-torch.tensor([p]))/common_constant(sigma)
    return (K*torch.log(a) - torch.log(torch.tensor([max_val])))/(K-1)


def calculate_s(a, mu, sigma):
    return (torch.log(a) - mu) / (common_constant(sigma) * np.sqrt(2))


# We perform monte carlo integration for the integral
def distribution_a(a, D, sigma = 0.05, n_samples = 1000):
    """
    We want to find the probability of a given D, to forego computing the integral we use monte carlo integration.
    Since our prior is unigorm for now we just sample uniformly. NOTE: it will be normalized over the range of a which has been given as input

    :param a: a values to query, LxM tensor
    :param D: Data of epsilons, 2xN tensor, first row is lower bounds, second row is upper bounds
    :return:
    """
    # Turn floats into tensors
    sigma = torch.tensor([sigma])

    # Median of log normal is exp(mu), thus we want to sample exp(mu) uniformly from 0 to 0.4
    # We have seen that mu > log(a) otherwise we cannot get cdf(a) = sigma, thus that is our lower bound
    # We give a constraint that P(x > 0.4) <= 0.01, this gives us an upper bound on mu
    mu_lower = torch.log(a + 0.00001) # We add a small value to a to avoid s = 0
    mu_upper = calculate_mu_upper(a, sigma)
    mus = torch.rand(n_samples, *a.shape) * (mu_upper - mu_lower) + mu_lower

    ss = calculate_s(a, sigma, mus)
    thetas = torch.stack([mus, ss])

    # Monte carlo integration
    integral_approximation = torch.mean(integrand(thetas, D), dim=[0,1])

    # Normalization
    return integral_approximation / integral_approximation.sum()
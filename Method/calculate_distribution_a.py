import numpy as np
import torch
from torch.distributions.log_normal import LogNormal


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
    distributions = LogNormal(thetas[0], thetas[1])
    # D is a 2xN tensor, first row is lower bounds, second row is upper bounds
    vals = distributions.cdf(D[0]) - distributions.cdf(D[1])
    # Likelihoods is a tensor of size thetas.shape[1:], for each theta we have a value
    # We aggregate over the last dimension (size N) which is the dimension of D
    Likelihoods = vals.prod(dim=-1)
    return Likelihoods


def calculate_s(a, sigma, mu):
    return (torch.log(a) - mu) / (torch.erfinv(2 * torch.tensor([sigma]) - 1) * np.sqrt(2))


# We perform monte carlo integration for the integral
def distribution_a(a, D, sigma, n_samples):
    """
    We want to find the probability of a given D, to forego computing the integral we use monte carlo integration.
    Since our prior is unigorm for now we just sample uniformly. NOTE: it will be normalized over the range of a which has been given as input

    :param a: a values to query, tensor of size M
    :param D: Data of epsilons, 2xN tensor, first row is lower bounds, second row is upper bounds
    :return:
    """
    # Median of log normal is exp(mu), thus we want to sample exp(mu) uniformly from 0 to 0.4
    # We have seen that mu > log(a) otherwise we cannot get cdf(a) = sigma, thus that is our lower bound
    mu_upper = np.log(0.4)
    mus = torch.rand(n_samples, a.shape[0]) * (mu_upper - torch.log(a + 0.001)) + torch.log(a + 0.001)

    ss = calculate_s(a, sigma, mus)
    thetas = torch.stack([mus, ss])

    # Monte carlo integration
    integral_approximation = torch.mean(integrand(thetas, D), dim=0)

    # Normalization
    return integral_approximation / integral_approximation.sum()
import pickle
import os
import numpy as np
from Helper.ImportDatasetsFairness import networks

file_path = 'Datasets/quantiles_networks.pkl'

# Check if the file exists
if os.path.exists(file_path):
    print('Loading...')
    # Load the dictionary from the file
    with open(file_path, 'rb') as file:
        quantiles_networks = pickle.load(file)
    print('Loaded!')
else:
    print('File not found...')

# Ensure quantiles_networks[network] is a list
for network in networks:
    if isinstance(quantiles_networks[network], tuple):
        quantiles_networks[network] = list(quantiles_networks[network])
    
    # Modify the elements
    if quantiles_networks[network][0].shape[1] > 840:
        quantiles_networks[network][0] = quantiles_networks[network][0][:, :840]
        quantiles_networks[network][1] = quantiles_networks[network][1][:, :840]
    
    # # Convert back to tuple if necessary
    # quantiles_networks[network] = tuple(quantiles_networks[network])

lens = np.arange(10, 850)
n_samples = quantiles_networks[networks[0]][0].shape[0]
import pickle
import numpy as np

# Load the processed dataset
with open('qm9_sample.sparsedataset', 'rb') as f:
    dataset = pickle.load(f)

# Print the keys in the dataset to understand its structure
print("Keys in the dataset:", dataset.keys())

# Assuming the dataset contains keys like 'data', 'data_A', 'data_X', and 'smiles'
# Extracting the relevant data
data = dataset['data']  # List of molecule objects
data_A = dataset['data_A']  # Adjacency matrices
data_X = dataset['data_X']  # Feature matrices
smiles = dataset['smiles']  # SMILES representations

# Select the first molecule (for example)
index = 0

# Print the SMILES representation of the molecule
print(f"SMILES: {smiles[index]}")

# Print the adjacency matrix
print("Adjacency Matrix:")
print(data_A[index])

# Print the feature matrix
print("Feature Matrix:")
print(data_X[index])

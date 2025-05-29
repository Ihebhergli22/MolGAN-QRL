import pandas as pd

# Load qm9_5k_with_mol_id.csv to get the subset of SMILES strings and their mol_ids
subset_df = pd.read_csv('qm9_5k_with_mol_id.csv')

# Create a set of SMILES strings for faster lookup
smiles_set = set(subset_df['smiles'])

# Find SMILES strings in qm9_5k.smi that do not have a matching mol_id
missing_mol_ids = [smi for smi in qm9_5k_smi if smi not in smiles_set]

# Display the number of missing mol_ids
len(missing_mol_ids), missing_mol_ids[:5]  # Displaying first 5 missing SMILES for inspection

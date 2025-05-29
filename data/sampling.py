import pandas as pd
import random

# Load the original qm9.csv
qm9_df = pd.read_csv('qm9.csv')

# Randomly select 5k molecules
subset_df = qm9_df.sample(n=5000, random_state=42)

# Save the selected SMILES strings to new_qm9_5k.smi
subset_df['smiles'].to_csv('qm9_sample.smi', index=False, header=False)

# Get the selected mol_id for further processing
selected_mol_ids = set(subset_df['mol_id'])

# Display the first few rows of the selected subset
print(subset_df.head())


from rdkit import Chem

# Load the gdb9.sdf file
suppl = Chem.SDMolSupplier('gdb9.sdf')

# Filter the molecules that match the selected mol_id
filtered_mols = [mol for mol in suppl if mol and mol.GetProp('_Name') in selected_mol_ids]

# Write the filtered molecules to a new SDF file
w = Chem.SDWriter('gdb9_sample.sdf')
for mol in filtered_mols:
    w.write(mol)
w.close()




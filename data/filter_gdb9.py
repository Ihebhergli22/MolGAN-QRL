from rdkit import Chem
import pandas as pd

# Load qm9_5k_with_mol_id.csv to get the subset of SMILES strings and their mol_ids
subset_df = pd.read_csv('qm9_5k_with_mol_id.csv')

# Create a set of SMILES strings for faster lookup
smiles_set = set(subset_df['smiles'])

# Create a dictionary to map SMILES to mol_id
smi_to_mol_id = dict(zip(subset_df['smiles'], subset_df['mol_id']))

# Load the gdb9.sdf file
suppl = Chem.SDMolSupplier('gdb9.sdf')

# Filter the molecules
filtered_mols = [mol for mol in suppl if mol and Chem.MolToSmiles(mol) in smiles_set]

# Add mol_id as a property to each molecule
for mol in filtered_mols:
    smiles = Chem.MolToSmiles(mol)
    mol.SetProp('mol_id', smi_to_mol_id[smiles])

# Write the filtered molecules to a new SDF file
w = Chem.SDWriter('gdb9_5k.sdf')
for mol in filtered_mols:
    w.write(mol)
w.close()



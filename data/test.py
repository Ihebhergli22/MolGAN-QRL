import pandas as pd

# Read the qm9.csv file
df_qm9 = pd.read_csv('qm9.csv')

# Read the qm9_sample.smi file
with open('qm9.smi', 'r') as file:
    smiles_sample = file.readlines()

# Strip newlines and other whitespace characters
smiles_sample = [smile.strip() for smile in smiles_sample]

# Create a dictionary for quick lookup of mol_id by SMILES
smiles_to_mol_id = {row['smiles']: row['mol_id'] for _, row in df_qm9.iterrows()}

# Create a new list to hold the SMILES strings with their corresponding mol_id
smiles_with_ids = []
for smile in smiles_sample:
    mol_id = smiles_to_mol_id.get(smile)
    if mol_id is not None:
        smiles_with_ids.append(f"{smile} {mol_id}")

# Write the new qm9_sample_id.smi file
with open('qm9_id.smi', 'w') as file:
    file.write('\n'.join(smiles_with_ids))

print("File qm9_id.smi has been created.")

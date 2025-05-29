import pandas as pd

# Read the qm9.csv file
df_qm9 = pd.read_csv('qm9.csv')

# Extract the 'smiles' column
smiles_list = df_qm9['smiles']

# Write the SMILES strings to qm9.smi file
with open('qm9.smi', 'w') as file:
    for smile in smiles_list:
        file.write(smile + '\n')

print("File qm9.smi has been created.")

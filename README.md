# 🧪 MolGAN-QRL: Quantum-Enhanced Reinforcement Learning for Molecular Graph Generation

This repository provides an extended implementation of **MolGAN** (Molecular Generative Adversarial Networks) with a **Hybrid Quantum-Classical Reward Algorithm (HRA)** designed to address mode collapse during molecular graph generation.

> **MolGAN-QRL replaces the original reward network in MolGAN with a hybrid model combining a Variational Quantum Circuit (VQC) and a Neural Network (NN).**

---

## 📚 Background

MolGAN is a deep generative model that generates molecular graphs directly using GANs and reinforcement learning. Our work builds on the original model by incorporating quantum-enhanced reward computation.

- 📄 Original Paper (MolGAN):  
  De Cao & Kipf, *MolGAN: An implicit generative model for small molecular graphs*  
  DOI: [10.48550/arXiv.1805.11973](https://doi.org/10.48550/arXiv.1805.11973)

- 💻 Original TensorFlow Code:  
  [https://github.com/nicola-decao/MolGAN](https://github.com/nicola-decao/MolGAN)

- 💻 PyTorch Reference Implementation Used:  
  [https://github.com/kfzyqin/Implementation-MolGAN-PyTorch](https://github.com/kfzyqin/Implementation-MolGAN-PyTorch)

---

## 📂 Repository Structure

```plaintext
MolGAN-QRL-GitHub/
├── molgan/           # Modified MolGAN implementation
├── molgan_qrl/       # Hybrid reward with VQC + NN
├── data/             # Preprocessed QM9 data + SMILES + SDF
├── environment.yml   # Conda environment setup
├── .gitignore
└── README.md

## 🧬 Dataset: QM9

The model is trained on the **QM9 molecular dataset**, provided in two chemical formats:

- **SMILES**: `data/qm9.smi`
- **SDF**: `data/gdb9.sdf`

Additional files include `gdb9.csv`, `gdb9_sample.sdf`, etc.

> Dataset transformation and preprocessing is handled by  
> `data/sparse_molecular_dataset.py`

---

## ⚙️ Environment Setup

Use the provided conda file to create the environment:

```bash
conda env create -f environment.yml
conda activate molgan-pt

## 🚀 Running the Code

Both versions of the model can be launched using the same entrypoint:

### ▶️ To run **MolGAN** (baseline):
```bash
python molgan/main_gan.py

### ▶️ To run **MolGAN-QRL** (quantum-enhanced):
```bash
python molgan_qrl/main_gan.py

The hybrid quantum-classical reward is implemented in:

```bash
molgan_qrl/circuit.py

## 🧠 Hybrid Reward Algorithm

The original reward network in MolGAN is replaced by a **Hybrid Reward Algorithm (HRA)** that combines:

- A **Variational Quantum Circuit (VQC)** for encoding quantum features
- A **Neural Network** for post-processing

This aims to enhance molecular diversity and reduce mode collapse in the generated molecular graphs.

---

## 🤝 Acknowledgements

- [Nicola De Cao](https://github.com/nicola-decao) for the original MolGAN implementation
- [kfzyqin](https://github.com/kfzyqin) for the PyTorch base code

---

## 📜 License

This project is released under the **MIT License**.

---

## 🔬 Citation

If you use this work in your research, please cite the original MolGAN paper and feel free to reference this repository once the corresponding MolGAN-QRL publication is out.


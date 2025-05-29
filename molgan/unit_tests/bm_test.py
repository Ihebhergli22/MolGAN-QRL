import unittest
from solver_qgan import Solver
from data.sparse_molecular_dataset import SparseMolecularDataset



class Config:
    def __init__(self):
        self.batch_size = 32
        self.d_conv_dim = [[128, 64], 128, [128, 64]]
        self.d_lr = 0.0001
        self.dropout = 0.0
        self.g_conv_dim = [128, 256, 512]
        self.g_lr = 0.0001
        self.img_dir_path = 'unit_test/images'
        self.lambda_cls = 1
        self.lambda_gp = 10.0
        self.lambda_rec = 10
        self.lambda_wgan = 0.5
        self.log_dir_path = 'unit_test/logs'
        self.log_step = 1
        self.lr_update_step = 1000
        self.mode = 'train'
        self.model_dir_path = 'unit_test/models'
        self.model_save_step = 1
        self.mol_data_dir = 'data/gdb9_9nodes.sparsedataset'
        self.n_critic = 5
        self.num_epochs = 50
        self.num_workers = 1
        self.post_method = 'softmax'
        self.resume_epoch = None
        self.sample_step = 1000
        self.test_epochs = 100
        self.z_dim = 8


class TestSolverGAN(unittest.TestCase):
    def setUp(self):
        # Initialize with the complete config
        config = Config()
        self.solver = Solver(config)

    def test_build_model_initialization(self):
        """Test that all model components are correctly initialized."""
        self.assertIsNotNone(self.solver.G, "Generator should be initialized.")
        self.assertIsNotNone(self.solver.D, "Discriminator should be initialized.")
        self.assertIsNotNone(self.solver.V, "HybridModel should be initialized.")

if __name__ == '__main__':
    unittest.main()








'''
import unittest
from solver_gan import Solver
from main_gan import main 
from args import get_GAN_config
from util_dir.utils_io import get_date_postfix
import os
import logging 
from data.sparse_molecular_dataset import SparseMolecularDataset

# Mocking the configuration settings typically loaded from args.py
class Config:
    def __init__(self):
        self.mol_data_dir = 'data/gdb9_9nodes.sparsedataset'  # Specify the correct path to your molecular data
        self.data = SparseMolecularDataset()
        self.data.load(self.mol_data_dir)
        
        self.lambda_wgan = 0.5
        self.lambda_gp = 10.0
        self.lambda_rec = 10.0
        self.g_lr = 1e-4
        self.d_lr = 1e-4
        self.n_critic = 5
        self.num_epochs = 10
        self.batch_size = 32
        self.log_step = 1
        self.z_dim = 100
        self.g_conv_dim = [64, 128, 256]
        self.d_conv_dim = [64, 128, 256]
        self.device = 'cpu'  # or 'cuda' if GPU is available
        self.post_method = 'soft_gumbel'  # Ensure this is correct as per your model requirements
        self.dropout = 0.  # Add any other configurations used in Solver
        self.resume_epoch = None 
        self.mode = 'train'
        self.saving_dir = '../exp_results/GAN/' 
        self.log_dir_path = os.path.join(self.saving_dir, 'log_dir')
        self.model_dir_path = os.path.join(self.saving_dir, 'model_dir')
        self.img_dir_path = os.path.join(self.saving_dir, 'img_dir')
        self.model_save_step = 1. 
        self.m_dim = self.data.atom_num_types  # Placeholder, set appropriately
        self.b_dim = self.data.bond_num_types
        

# Test case for the Solver with complete model initialization
class TestSolverGAN(unittest.TestCase):
    def setUp(self):
        config = Config()
        self.solver = Solver(config)  # Assumes Solver can handle this config directly

    def test_build_model_initialization(self):
        """Test that all components are correctly initialized."""
        self.solver.build_model()
        self.assertIsNotNone(self.solver.G, "Generator should not be None")
        self.assertIsNotNone(self.solver.D, "Discriminator should not be None")
        self.assertIsNotNone(self.solver.V, "Quantum model (V) should not be None")

if __name__ == '__main__':
    unittest.main()
'''
import torch
import unittest
from unittest.mock import MagicMock, patch
from solver_gan import Solver  # Ensure this import matches the actual path and name

class TestSolverGAN(unittest.TestCase):
    def setUp(self):
        # Configuration for initializing Solver
        config = MagicMock()
        config.mol_data_dir = 'data/gdb9_9nodes.sparsedataset'  # As specified in your args.py
        config.z_dim = 8
        config.g_conv_dim = [128, 256, 512]  # Example: multilayer generator convolution dimensions
        config.d_conv_dim = [[128, 64], 128, [128, 64]]  # Discriminator convolution dimensions
        config.lambda_wgan = 0.5
        config.lambda_gp = 10.0
        config.g_lr = 0.001
        config.d_lr = 0.001
        config.dropout = 0.0
        config.n_critic = 5
        config.num_epochs = 150
        config.batch_size = 32
        config.mode = 'train'
        config.post_method = 'soft_gumbel'

        # Initialize the solver with the mocked configuration
        self.solver = Solver(config)
        self.solver.data = MagicMock()
        self.solver.quantum_model = MagicMock()  # Mock the quantum model
        self.solver.reward = MagicMock(return_value=torch.tensor([1.0, 2.0]))

    def test_reward_integration(self):
        # Prepare mock data for nodes and edges
        nodes_hat = torch.randn(1, 9)  # Example tensor for nodes
        edges_hat = torch.randn(1, 9, 9)  # Example tensor for edges
        method = 'soft_gumbel'  # As per your args.py

        # Mock the quantum model to return a fixed reward
        self.solver.quantum_model.return_value = torch.tensor([0.5, 0.5], dtype=torch.float32)

        # Call get_reward to calculate
        rewards = self.solver.get_reward(nodes_hat, edges_hat, method)

        # Define expected rewards for the test, here we expect classical and quantum rewards to add up
        expected_rewards = torch.tensor([1.5, 2.5])  # Example expected rewards

        # Assert that the computed rewards match expected rewards
        self.assertTrue(torch.allclose(rewards, expected_rewards))

if __name__ == '__main__':
    unittest.main()

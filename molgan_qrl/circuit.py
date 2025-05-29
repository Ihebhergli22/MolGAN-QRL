import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn

# Define the quantum device
dev = qml.device('default.qubit', wires=9)

class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        # Initialize quantum device and circuit parameters
        self.weights = torch.nn.Parameter(torch.randn(3, 9, 3), requires_grad=True)
        dev = qml.device('default.qubit', wires=9)
        self.qnode = qml.QNode(self.quantum_circuit, dev)

    def quantum_circuit(self, features, weights):
        # Define the structure of the quantum circuit
        qml.templates.AngleEmbedding(features, wires=range(9))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(9))
        return [qml.expval(qml.PauliZ(i)) for i in range(9)]

    def forward(self, x):
        results = []
        for input_tensor in x:
            detached_input = input_tensor.clone().detach().requires_grad_(False).to(torch.float64)
            detached_weights = self.weights.detach().numpy()
            # Ensure tensor sizes are correct
            if len(detached_input.shape) != 1 or detached_input.shape[0] != 9:
                detached_input = torch.flatten(detached_input)[:9]  # Ensure the input is of the correct shape
            if detached_weights.shape != (3, 9, 3):
                detached_weights = detached_weights.reshape((3, 9, 3))  # Correct weights shape
            try:
                quantum_output = self.qnode(detached_input, detached_weights)
                results.append(quantum_output)
            except Exception as e:
                print(f"Error processing quantum node: {e}")
                print(f"Input shape: {detached_input.shape}, Weights shape: {detached_weights.shape}")
                continue

        return torch.tensor(np.array(results), dtype=torch.float32)

    
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.quantum_layer = QuantumLayer()
        self.linear = nn.Linear(9, 1)  # Example linear layer for further processing


    def forward(self, x):
        x = self.quantum_layer(x)
        x = self.linear(x)
        return x
"""equivalent to MyTest as of 11 July 2025"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam as SGD
import numpy as np


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(4, 3)
        self.l2 = nn.Linear(3, 3)
        self.l3 = nn.Linear(3, 2)

        self.optim = SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.tanh(self.l3(x))
        return x

    def update_parameters(self, X, Y):
        assert len(X) == len(Y)
        pred = self.forward(X)
        loss = F.mse_loss(pred, Y, reduction="sum")
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        total_loss = loss.item()
        return total_loss


def make_samples(N_SAMPLES):
    N_ACTIONS = 100
    N_OUTPUTS = 2
    N_INPUTS = 4
    action_samples = np.random.uniform(-1.0, 1.0, (N_ACTIONS, N_OUTPUTS))
    X = np.random.uniform(-10.0, 10.0, (N_SAMPLES, N_INPUTS))
    Y = np.empty((N_SAMPLES, N_OUTPUTS))
    for i in range(N_SAMPLES):
        best = 99999.9
        best_j = 0
        for j in range(N_ACTIONS):
            curr = np.linalg.norm(X[i, 0:2] - X[i, 2:4] + action_samples[j])
            if curr < best:
                best_j = j
        Y[i] = action_samples[best_j]
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def main():
    mlp = MLP()
    X, Y = make_samples(10000)
    print("training...")
    for i in range(1000):
        loss = mlp.update_parameters(X, Y)
        if i % 100 == 0:
            print(f"Epoch: {i} Loss: {loss}")
    print("done training")


if __name__ == "__main__":
    main()

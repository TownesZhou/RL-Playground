import random
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal


class PolicyNet(nn.Module):
    """
        Simple Multilayer Perceptron for Policy Gradient with Gaussian policies

        There is a neural network for calculating mean actions  miu_theta(s),
            and a separate neural network for calculating the log standard deviations sigma_theta(s).
        These two neural networks should share some common layers as the encoding network.

        layer_sizes should be a dictionary containing the following key-value entries:
        layer_sizes = {
                        "encoding" : (a list representing encoding network layer sizes)
                        "mean" : (a list representing mean network layer sizes)
                        "std" : (a list representing log standard deviation network layer sizes)
                      }
        Note:
            1) the last entry of "encoding" and the first entry of "mean" and that of "std" should
               be the same.
            2) the last entry of "mean" and that of "std" should be the same, representing the number
               of dimension of the action space.

    """

    def __init__(self, layer_sizes, action_lim):
        super(PolicyNet, self).__init__()

        encoding_layer_sizes = layer_sizes['encoding']
        mean_layer_sizes = layer_sizes['mean']
        std_layer_sizes = layer_sizes['std']

        # Record the action space dimension
        self.k = mean_layer_sizes[-1]

        # Action value upper and lower bound
        self.action_lim = action_lim

        # Store the network layers in a ModuleList
        self.encoding_layers = nn.ModuleList()
        self.mean_layers = nn.ModuleList()
        self.std_layers = nn.ModuleList()

        # Construct and register layers
        input_size = encoding_layer_sizes[0]
        for output_size in encoding_layer_sizes[1:]:
            self.encoding_layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

        input_size = mean_layer_sizes[0]
        for output_size in mean_layer_sizes[1:]:
            self.mean_layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

        input_size = mean_layer_sizes[0]
        for output_size in std_layer_sizes[1:]:
            self.std_layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

        self.Elu = nn.ELU()
        self.tanh = nn.Tanh()


    def forward(self, x):
        # Forward propagation
        x_encoding = x
        for encoding_layer in self.encoding_layers:
            x_encoding = self.Elu(encoding_layer(x_encoding))

        x_mean = x_encoding
        for mean_layer in self.mean_layers[:-1]:
            x_mean = self.Elu(mean_layer(x_mean))
        x_mean = self.tanh(self.mean_layers[-1](x_mean)) * self.action_lim

        x_log_std = x_encoding
        for std_layer in self.std_layers[:-1]:
            x_log_std = self.Elu(std_layer(x_log_std))
        x_log_std = self.std_layers[-1](x_log_std)

        # Take the exponential of x_log_std to calculate the standard deviations
        x_std = torch.exp(x_log_std)

        # If the model is in evaluation mode, use values of mean action as the optimal action
        # Else, return an action sampled from N(x_mean, x_std) and the log of its probability
        if not self.training:
            return x_mean.squeeze(0)
        else:
            # Instantiate a Multivariate Normal distribution that can be used to sample action
            #   and compute the action log-probabilities

            # Covariance matrix is a diagonal matrix with x_std as the diagonal entries
            # m = MultivariateNormal(x_mean, covariance_matrix=torch.diag(x_std).unsqueeze(0))

            m = []
            action = None
            log_prob = None
            for i in range(self.k):
                m.append(Normal(x_mean[0][i], x_std[0][i]))
                if action is None:
                    action = m[-1].sample().unsqueeze(0)
                    log_prob = m[-1].log_prob(action)
                else:
                    action = torch.cat([action, m[-1].sample().unsqueeze(0)])
                    log_prob += m[-1].log_prob(action[-1])


            return action, log_prob


def optimize_model(policy_net, batch_log_prob, batch_rewards, optimizer, GAMMA=0.999, device='cuda'):
    """ Optimize the model for one step"""

    # Obtain batch size
    batch_size = len(batch_log_prob)

    # Calculate weight
    # Simple Policy Gradient: Use trajectory Reward To Go
    batch_weight = []
    for rewards in batch_rewards:
        n = rewards.shape[0]
        rtg = torch.zeros(n, device=device)
        for i in reversed(range(n)):
            rtg[i] = rewards[i] + (GAMMA * rtg[i+1] if i + 1 < n else 0)
        batch_weight.append(rtg)

    # Calculate grad-prob-log
    loss = None
    for i in range(batch_size):
        if loss is None:
            loss =  - torch.sum(batch_log_prob[i] * batch_weight[i])
        else:
            loss += - torch.sum(batch_log_prob[i] * batch_weight[i])

    loss = loss / torch.tensor(batch_size, device=device)
    # Gradient Ascent
    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

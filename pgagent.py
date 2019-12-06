import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

import Neurosmash


class EpisodeMemory(object):
    """Implements memory that is used as Episode Memory to store action
       probabilities and rewards for Policy Gradient.

    Attributes:
        capacity = [int] the maximum number of elements in the memory
        memory   = [list] the actual collection of memories
        position = [int] the index at which a new element can be stored
    """

    def __init__(self, capacity):
        """Initializes the memory with a given capacity.

        Args:
            capacity = [int] the maximum number of elements in the memory
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def reset(self):
        """Unload the elements in the memory and reset the pointer."""
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a new element.

        Args:
            see environment
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args

        self.position = (self.position + 1) % self.capacity

    def take_all(self):
        """Take all available elements.

        Returns [list]:
            All the available elements in list format, transposed.
        """
        return list(zip(*self.memory))

    def __len__(self):
        """Returns current number of elements in the memory."""
        return len(self.memory)


class MLP(nn.Module):
    """Implements a Multi Layer Perceptron.

    Attributes:
        fc1     = [nn.Module] input fully connected layer
        dropout = [nn.Module] dropout layer for generalization
        fc2     = [nn.Module] output fully connected layer
    """

    def __init__(self, n_in, n_hidden, n_out, p=0.5):
        """
        Initializes the MLP.

        Args:
            n_in     = [int] number of input units
            n_hidden = [int] number of hidden units
            n_out    = [int] number of output units
            p        = [float] probability for dropout layer
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden, bias=False)
        self.dropout = nn.Dropout(p=p)
        self.fc2 = nn.Linear(n_hidden, n_out, bias=False)

    def forward(self, x):
        h = self.dropout(F.relu(self.fc1(x)))
        y = self.fc2(h)
        return y


class PGAgent(Neurosmash.Agent):
    """Implements a Neural Policy Gradient Agent.

    Attributes:
        model     = [nn.Module] model that is updated every episode
        optimizer = [Optimizer] optimizer used to train the model
        y         = [float] the gamma parameter for computing Q values
        memory    = [EpisodeMemory] stores (action probability, reward) pairs
    """

    def __init__(self, n_obs, n_actions):
        # setup the neural network
        self.model = MLP(n_obs, 128, n_actions)

        # setup an optimizer
        self.optimizer = Adam(self.model.parameters(), lr=0.01)

        # set Q learning parameters
        self.y = 0.99  # gamma

        # setup Episode Memory
        self.memory = EpisodeMemory(512)

    def step(self, end, reward, state):
        """Agent determines the action based on the current state and model.

        Args:
            state = [object] current state of the environment

        Returns [int]:
            Action in the range [0, n_actions).
        """
        # apply neural network to get action distribution
        x = torch.tensor(state)
        action_probs = F.softmax(self.model(x), dim=-1)

        # sample from action distribution to get action
        action_distr = Categorical(action_probs)
        action = action_distr.sample()

        return action

    def train(self, end, action, old_state, reward, new_state):
        """Trains agent based on four things: the old state, the action
           performed in the old state, the reward received after doing
           that action in the old state and the resulting new state.

        Args:
           end       = [bool] whether the episode has finished
           action    = [int] action as int in the range [0, n_actions)
           old_state = [object] previous state of the environment
           reward    = [int] reward received after doing action in old_state
           new_state = [object] the state of the environment after doing
                                action in old_state
        """
        # apply neural network to get action distribution
        x = torch.tensor(old_state)
        action_probs = F.softmax(self.model(x), dim=-1)

        # compute log probability of sampled action
        action_log_prob = action_probs[action].log()

        # add action log probability and reward to memory
        self.memory.push(action_log_prob, reward)

        if not end:  # episode is not yet done
            return

        # get all the log probabilities and rewards of this episode
        minibatch = self.memory.take_all()

        # unpack minibatch
        action_log_prob_batch = torch.stack(minibatch[0])
        reward_batch = minibatch[1]

        # compute Q values for each step in the episode
        Q_values = torch.zeros(len(reward_batch))
        Q_values[-1] = reward_batch[-1]
        for i in range(len(Q_values) - 1, 0, -1):
            Q_values[i - 1] = reward_batch[i - 1] + self.y * Q_values[i]

        # compute loss
        loss = -1 * torch.sum(action_log_prob_batch * Q_values)

        # update model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # reset Episode Memory
        self.memory.reset()

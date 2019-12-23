import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import Neurosmash


class ReplayMemory(object):
    """Implements memory that is used as Replay Memory to increase
       Q learning stability.

    Attributes:
        capacity    = [int] the maximum number of elements in the memory
        memory      = [list] the actual collection of memories
        position    = [int] the index at which a new element can be stored
        n           = [int] number of transitions used for N-step DDQN
        transitions = [list] list of self.n transitions for N-step DDQN
    """

    def __init__(self, capacity, n):
        """Initializes the memory with a given capacity.

        Args:
            capacity = [int] the maximum number of elements in the memory
            n        = [int] number of transitions used for N-step DDQN
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

        # set parameters for N-step DQN
        self.n = n
        self.transitions = []

    def push(self, end, action, old_state, reward, new_state):
        """Aggregates state transitions until `self.n` transitions have
           been aggregated in the `self.transitions` list. Then it adds a
           transition list to the memory each time step.
           If the end of the episode has been reached, then it adds
           `self.n` transition lists, where the ending transition replaces
           the next latest transition for each list. So if we have
           transitions 5, 6, 7, and 8, where 8 is the transition where the
           episode has ended, then it adds the lists [5, 6, 7, 8],
           [6, 7, 8, 8], [7, 8, 8, 8], and [8, 8, 8, 8]. If the episode
           has not yet ended, then it only adds the [5, 6, 7, 8] list.
           The transitions are reset at the end of an episode.

        Args:
            end       = [bool] whether the episode has finished
            action    = [int] action as number in the range [0, num_actions)
            old_state = [object] previous state of the environment
            reward    = [int] reward received after doing action in old_state
            new_state = [object] the state of the environment after doing
                                 action in old_state
        """
        # determine whether n transitions are aggregated this time step
        n_transitions = len(self.transitions) > (self.n - 1) * 4
        transition = [end, action, old_state, reward, new_state]
        if not n_transitions:
            # add latest transition to incomplete transition list
            self.transitions[-1:] = transition
        else:
            # add extra transition lists at end of episode
            for _ in range(1 + end * (self.n - 1)):
                # add latest transition to almost complete transition list
                self.transitions[-1:] = transition

                # add latest transition list to memory
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = self.transitions

                # move pointer to next element and remove oldest transition
                self.position = (self.position + 1) % self.capacity
                self.transitions = self.transitions[-(self.n * 4 - 3):]

        if end:  # reset transition list
            self.transitions = []

    def sample(self, batch_size):
        """Take a random sample of the available elements.

        Args:
            batch_size = [int] the number of elements to be sampled

        Returns [list]:
            The batch_size sampled elements from the memory, transposed.
        """
        elements = random.sample(self.memory, batch_size)
        return list(zip(*elements))

    def __len__(self):
        """Returns current number of elements in the memory."""
        return len(self.memory)


class DDQN(nn.Module):
    """Implements a Dueling Deep Q Network.

    Attributes:
        fc1a    = [nn.Module] input linear layer to hidden advantage layer
        fc1v    = [nn.Module] input linear layer to hidden value layer
        dropout = [nn.Module] Dropout layer to decrease overfitting
        fc2a    = [nn.Module] output linear layer from hidden advantage layer
        fc2v    = [nn.Module] output linear layer from hidden value layer
    """

    def __init__(self, num_in, num_hidden, num_out,
                 p=0.5, device=torch.device('cpu')):
        """Initializes the DDQN.

        Args:
            num_in     = [int] number of input units
            num_hidden = [int] number of hidden units in linear layers
                               between input and advantage layer and
                               between input and value layer
            num_out    = [int] number of output units
            p          = [float] probability for Dropout layers
            device     = [torch.device] device to put the model and data on
        """
        super(DDQN, self).__init__()
        self.fc1a = nn.Linear(num_in, num_hidden, bias=False)
        self.fc1v = nn.Linear(num_in, num_hidden, bias=False)
        self.dropout = nn.Dropout(p=p)
        self.fc2a = nn.Linear(num_hidden, num_out, bias=False)
        self.fc2v = nn.Linear(num_hidden, 1, bias=False)

        self.to(device)

    def forward(self, x):
        ha = self.dropout(F.relu(self.fc1a(x)))
        hv = self.dropout(F.relu(self.fc1v(x)))

        a = self.fc2a(ha)
        v = self.fc2v(hv)

        return v + a - a.mean(dim=-1, keepdim=True)


class QAgent(Neurosmash.Agent):
    """Implements a Neural Q Agent.

    Attributes:
        policy_net = [nn.Module] DDQN that is updated every transition
        target_net = [nn.Module] DDQN that is used to retrieve the Q values
        device     = [torch.device] device to put the model and data on
        optimizer  = [Optimizer] optimizer used to train the model
        criterion  = [nn.Module] the loss function of the model
        y          = [float] the gamma parameter for Q learning
        memory     = [ReplayMemory] memory to randomly sample state transitions
        batch_size = [int] the batch size in one update step
        num_update = [int] number of steps before target network is updated
        num_steps  = [int] the number of steps the agent has performed
        n          = [int] number of transitions used for N-step DDQN
    """

    def __init__(self, num_obs, num_actions, device=torch.device('cpu')):
        """Initializes the agent.

        Args:
            num_obs     = [int] number of elements in DDQN input vector
            num_actions = [int] number of possible actions in environment
            device      = [torch.device] device to put the model and data on
        """
        super(QAgent, self).__init__()

        # setup the policy and target neural networks
        self.policy_network = DDQN(num_obs, 128, num_actions, device=device)
        self.target_network = DDQN(num_obs, 128, num_actions, device=device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.device = device

        # setup an optimizer
        self.optimizer = Adam(self.policy_network.parameters(), lr=0.001)

        # setup the loss function as Mean Square Error
        self.criterion = nn.MSELoss()

        # set Q learning parameters
        self.y = 0.99  # gamma

        # set parameters for N-step DQN
        self.n = 16

        # setup Replay Memory
        self.memory = ReplayMemory(2048, self.n)
        self.batch_size = 32

        # set target network updating parameters
        self.num_update = 64
        self.num_steps = 0

    def step(self, end, reward, state):
        """The agent selects action given the current state and target network.

        Args:
            end    = [bool] whether the episode has finished
            reward = [int] reward received after doing previous action
            state  = [torch.Tensor] current state of the environment

        Returns [int]:
            The action encoded as a number in the range [0, num_actions).
        """
        # apply q-learning neural network to get q-value estimation
        Q = self.policy_network(state)

        # choose an action by greedily picking from Q table
        action = torch.argmax(Q)

        return action

    def train(self, end, action, old_state, reward, new_state):
        """Trains agent based on four things: the old state, the action
           performed in the old state, the reward received after doing
           that action in the old state and the resulting new state.

        Args:
            end       = [bool] whether the episode has finished
            action    = [int] action as number in the range [0, num_actions)
            old_state = [object] previous state of the environment
            reward    = [int] reward received after doing action in old_state
            new_state = [object] the state of the environment after doing
                                 action in old_state
        """
        self.memory.push(end, action, old_state, reward, new_state)

        if len(self.memory) < self.batch_size:
            return

        # sample state transition lists
        minibatch = self.memory.sample(self.batch_size)

        # unpack minibatch
        ends_batch = torch.tensor(minibatch[:-1:4],
                                  dtype=torch.float,
                                  device=self.device).t().unsqueeze(2)
        action_batch = torch.tensor(minibatch[1],
                                    device=self.device).unsqueeze(1)
        old_state_batch = torch.stack(minibatch[2]).to(self.device)
        rewards_batch = torch.tensor(minibatch[3:-1:4],
                                     dtype=torch.float,
                                     device=self.device).t().unsqueeze(2)
        new_state_batch = torch.stack(minibatch[-1]).to(self.device)

        # compute predicted Q values on old states
        Q_pred = self.policy_network(old_state_batch).gather(1, action_batch)

        # get target network Q value on new state based on policy network action
        actions = self.policy_network(new_state_batch).argmax(1, keepdim=True)
        Q_target = self.target_network(new_state_batch).gather(1, actions)

        # compute what the predicted Q values should have been
        for i in range(self.n)[::-1]:
            Q_target = rewards_batch[:, i] + \
                       (1 - ends_batch[:, i]) * self.y * Q_target

        # compute the loss as MSE between predicted and target Q values
        loss = self.criterion(Q_pred, Q_target)

        # update model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network periodically
        self.num_steps += 1
        if self.num_steps % self.num_update == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            print(f'loss: {loss}')

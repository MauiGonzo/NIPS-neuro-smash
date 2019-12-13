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
            action    = [int] action as number in the range [0, env.n_action)
            old_state = [object] previous state of the environment
            reward    = [int] reward received after doing action in old_state
            new_state = [object] the state of the environment after doing
                                action in old_state
        """
        # determine whether n transitions are aggregated this time step
        n_transitions = len(self.transitions) > (self.n - 1) * 4
        if not n_transitions:
            # add latest transition to incomplete transitions
            self.transitions[-1:] = [end, action, old_state, reward, new_state]
        else:
            # add extra transitions at end of episode
            for _ in range(1 + end * (self.n - 1)):
                # add latest transition to almost complete transitions
                self.transitions[-1:] = [end, action, old_state, reward, new_state]

                # add latest n transitions to memory
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = self.transitions

                # move pointer to next element and remove oldest transition
                self.position = (self.position + 1) % self.capacity
                self.transitions = self.transitions[-(self.n * 4 - 3):]

        if end:
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

    def __init__(self, n_in, n_hidden, n_out, p=0.5):
        """Initializes the DDQN.

        Args:
            n_in     = [int] number of input units
            n_hidden = [int] number of hidden units in linear layers
                             between input and advantage layer and
                             between input and value layer
            n_out    = [int] number of output units
            p        = [float] probability for Dropout layers
        """
        super(DDQN, self).__init__()
        self.fc1a = nn.Linear(n_in, n_hidden, bias=False)
        self.fc1v = nn.Linear(n_in, n_hidden, bias=False)
        self.dropout = nn.Dropout(p=p)
        self.fc2a = nn.Linear(n_hidden, n_out, bias=False)
        self.fc2v = nn.Linear(n_hidden, 1, bias=False)

    def forward(self, x):
        ha = self.dropout(F.relu(self.fc1a(x)))
        hv = self.dropout(F.relu(self.fc1v(x)))

        a = self.fc2a(ha)
        v = self.fc2v(hv)

        return v + a - a.mean()


class QAgent(Neurosmash.Agent):
    """Implements a Neural Q Agent.

    Attributes:
        policy_net = [nn.Module] DDQN that is updated every transition
        target_net = [nn.Module] DDQN that is used to retrieve the Q values
        n_obs      = [int] number of elements in DDQN input vector
        optimizer  = [Optimizer] optimizer used to train the model
        criterion  = [nn.Module] the loss of the model
        y          = [float] the gamma parameter for Q learning
        memory     = [ReplayMemory] memory to randomly sample state transitions
        batch_size = [int] the batch size in one update step
        n_update   = [int] number of steps before target network is updated
        n_steps    = [int] the number of steps the agent has performed
        n          = [int] number of transitions used for N-step DDQN
    """

    def __init__(self, n_obs, n_actions):
        """Initializes the agent.

        Args:
            n_obs     = [int] number of elements in state vector
            n_actions = [int] number of possible actions in environment
        """
        # setup the policy and target neural networks
        self.policy_network = DDQN(n_obs, 64, n_actions)
        self.target_network = DDQN(n_obs, 64, n_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.n_obs = n_obs

        # setup an optimizer
        self.optimizer = Adam(self.policy_network.parameters(), lr=4e-5)

        # setup the loss function as Mean Square Error
        self.criterion = nn.MSELoss()

        # set Q learning parameters
        self.y = 0.99  # gamma

        # set parameters for N-step DQN
        self.n = 4

        # setup Replay Memory
        self.memory = ReplayMemory(2048, self.n)
        self.batch_size = 32

        # set target network updating parameters
        self.n_update = 128
        self.n_steps = 0

    def step(self, end, reward, state):
        """The agent selects action given the current state and target network.

        Args:
            end    = [bool] whether the episode has finished
            reward = [int] reward received after doing previous action
            state = [torch.Tensor] current state of the environment

        Returns [int]:
            The action encoded as a number in the range [0, n_actions).
        """
        # apply q-learning neural network to get q-value estimation
        Q = self.policy_network(state)

        # choose an action by greedily picking from Q table
        action = torch.argmax(Q)

        return action

    def train(self, end, action, old_state, reward, new_state):
        """
        Trains the agent based on last action and state and new reward and state.

        Args:
            end       = [bool] whether the episode has finished
            action    = [int] action as number in the range [0, env.n_action)
            old_state = [object] previous state of the environment
            reward    = [int] reward received after doing action in old_state
            new_state = [object] the state of the environment after doing
                                action in old_state
        """
        self.memory.push(end, action, old_state, reward, new_state)

        if len(self.memory) < self.batch_size:
            return

        # sample state transitions
        minibatch = self.memory.sample(self.batch_size)

        # unpack minibatch
        end_batch = torch.zeros((self.batch_size, self.n, 1)).float()
        action_batch = torch.zeros((self.batch_size, self.n, 1)).long()
        old_states_batch = torch.zeros((self.batch_size, self.n_obs, self.n))
        reward_batch = torch.zeros((self.batch_size, self.n, 1))
        new_state_batch = torch.stack(minibatch[-1])
        for i in range(self.n):
            end_batch[:, i] = torch.tensor(minibatch[i * 4]).unsqueeze(1)
            action_batch[:, i] = torch.tensor(minibatch[i * 4 + 1]).unsqueeze(1)
            old_states_batch[:, :, i] = torch.stack(minibatch[i * 4 + 2])
            reward_batch[:, i] = torch.tensor(minibatch[i * 4 + 3]).unsqueeze(1)

        # compute predicted Q values on old states
        Q_pred = self.policy_network(old_states_batch[:, :, 0]).gather(1, action_batch[:, 0])

        # compute target network Q values on new states based on policy network actions
        new_actions = self.policy_network(new_state_batch).argmax(1, keepdim=True)
        Q_new = self.target_network(new_state_batch).gather(1, new_actions)
        for i in range(self.n - 1, 0, -1):
            Q_new = reward_batch[:, i] + self.y * Q_new

        if end:
            i = 2

        # compute what the predicted Q values should have been
        Q_target = reward_batch[:, 0] + (1 - end_batch[:, 0]) * self.y * Q_new

        # compute the loss as MSE between predicted and target Q values
        loss = self.criterion(Q_pred, Q_target)

        # update model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network periodically
        self.n_steps += 1
        if self.n_steps % self.n_update == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

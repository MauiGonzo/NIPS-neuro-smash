import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SPPLayer(nn.Module):
    """Neural network layer for spatial pyramid pooling.

    Attributes:
        num_levels = [int] height of the pyramid
        pool_type  = [str] either 'max_pool' or 'avg_pool'

    Source:
        https://gist.github.com/erogol/a324cc054a3cdc30e278461da9f1a05e#file-spp_net-py
    """

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


class ConvNet(nn.Module):
    """Implements a deep Convolutional Neural Network.

    Attributes:
        environment = [Environment] the environment in which the agent acts
        transformer = [Transformer] object that transforms images
        conv1       = [nn.Module] first convolutional layer
        bn1         = [nn.Module] first batch normalization layer
        conv2       = [nn.Module] second convolutional layer
        bn2         = [nn.Module] second batch normalization layer
        conv3       = [nn.Module] third convolutional layer
        bn3         = [nn.Module] third batch normalization layer
        spp         = [nn.Module] spatial pyramid pooling layer
        fc1         = [nn.Module] first fully connected layer
        dropout     = [nn.Module] dropout layer for generalization
        fc2         = [nn.Module] second fully connected layer
    """

    def __init__(self, environment, transformer, device=torch.device('cpu')):
        """Initializes the Convolutional Neural Network.

        Args:
            environment = [Environment] the environment in which the agent acts
            transformer = [Transformer] object that transforms images
            device      = [torch.device] device to put the neural network on
        """
        super(ConvNet, self).__init__()
        self.environment = environment
        self.transformer = transformer

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.spp = SPPLayer(3, pool_type='max_pool')
        self.fc1 = nn.Linear(3840, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 2)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        input_is_1d = type(x) is list
        if input_is_1d:  # convert state list to tensor
            x = self.transformer.perspective(self.environment.state2image(x))
            x = np.asarray(x, 'f').transpose(2, 0, 1)
            x = torch.tensor(x, device=self.device).unsqueeze(0)

        h = F.max_pool2d(F.relu(self.conv1(x)), 3, stride=2)
        h = self.bn1(h)
        h = F.max_pool2d(F.relu(self.conv2(h)), 3, stride=2)
        h = self.bn2(h)
        h = F.relu(self.conv3(h))
        h = self.bn3(h)
        h = self.spp(h)
        h = self.dropout(F.relu(self.fc1(h)))
        y = self.fc2(h)

        if input_is_1d:  # convert output to a location
            y = y.squeeze().to(torch.device('cpu')).data.numpy()

        return y

class TwoCNNsLocator(object):
    """Determines x and y pixel coordinates of agents with Convoluational
       Neural Networks.

    Attributes:
        cnn_red    = [nn.Module] CNN that determines location of red agent
        cnn_blue   = [nn.Module] CNN that determines location of blue agent
    """

    def __init__(self, environment, transformer,
                 models_dir='models/', device=torch.device('cpu')):
        """Initializes CNNs.

        Args:
            environment = [Environment] the environment in which the agent acts
            transformer = [Transformer] object that transforms images
            models_dir  = [str] directory where the CNNs are stored
            device      = [torch.device] device to put the neural network on
        """
        # initialize CNN for red agent
        self.cnn_red = ConvNet(environment, transformer, device=device)
        self.cnn_red.load_state_dict(torch.load(f'{models_dir}cnn_red.pt'))
        self.cnn_red.eval()

        # initialize CNN for blue agent
        self.cnn_blue = ConvNet(environment, transformer, device=device)
        self.cnn_blue.load_state_dict(torch.load(f'{models_dir}cnn_blue.pt'))
        self.cnn_blue.eval()

    def get_locations(self, state_img):
        """Determine the x and y pixel coordinates of the red and blue agents.

        Args:
            state_img = [ndarray] current state of environment as NumPy ndarray

        Returns [(x_red, y_red), (x_blue, y_blue)]:
            The x and y pixel coordinates of the red and blue agents
            as a tuple of NumPy ndarrays.
        """
        loc_red = self.cnn_red(state_img)
        loc_blue = self.cnn_blue(state_img)

        return loc_red, loc_blue

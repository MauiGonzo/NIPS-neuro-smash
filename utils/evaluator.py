import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.transformer import Transformer
from cnn import ConvNet
import Neurosmash


def plot_locations(environment, transformer, state, xy_red, xy_blue):
    """Plots the given locations of the agents on the
       given state image of the environment.

    Args:
        environment = [Environment] the environment in which the agent acts
        transformer = [Transformer] object that transforms images
        state       = [[int]] current state of the environment
        xy_red      = [int] x and y pixel coordinates of red agent
        xy_blue     = [int] x and y pixel coordinates of blue agent
    """
    image = transformer.perspective(environment.state2image(state))

    ax.clear()
    ax.imshow(image, extent=(0.5, transformer.size + 0.5,
                             transformer.size + 0.5, 0.5))
    ax.scatter([xy_red[0], xy_blue[0]], [xy_red[1], xy_blue[1]], c=['r', 'b'])
    ax.set_title('Predicted locations of agents')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, transformer.size + 1)
    ax.set_ylim(transformer.size + 1, 0)
    plt.pause(0.001)


def evaluate_locations(cnn_red, cnn_blue, num_episodes=10):
    """Evaluate the convolutional neural network on the real environment.

    Args:
        cnn_red      = [nn.Module] the convolutional neural network that
                                   predicts the location of the red agent
        cnn_blue     = [nn.Module] the convolutional neural network that
                                   predicts the location of the blue agent
        num_episodes = [int] number of episodes to evaluate
    """
    while num_episodes > 0:
        end, reward, state = environment.reset()
        num_episodes -= 1

        while not end:
            action = agent.step(end, reward, state)
            end, reward, state = environment.step(action)

            xy_red = cnn_red(state)
            xy_blue = cnn_blue(state)
            plot_locations(transformer, environment, state, xy_red, xy_blue)


def plot_rewards(rewards, window=10):
    """Plot the given rewards as the average reward in a window of episodes.

    Args:
        rewards = [[float]] list of sum of rewards for each episode
        window  = [int] number of episode rewards to take the average of

    Returns [float]:
        Latest smoothed total episode reward.
    """
    if len(rewards) < window:
        return 0

    rewards_smoothed = []
    for i in range(len(rewards) - window + 1):
        rewards_smoothed.append(np.mean(rewards[i:i + window]))

    ax.clear()
    ax.plot(np.arange(window, len(rewards) + 1), rewards_smoothed)
    ax.set_title('Smoothed episode rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Average reward in previous {window} episodes')
    plt.pause(0.001)

    return rewards_smoothed[-1]


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


if __name__ == '__main__':
    data_dir = '../data/'
    models_dir = '../models/'
    size = 64
    timescale = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    environment = Neurosmash.Environment(size=size, timescale=timescale)
    agent = Neurosmash.Agent()

    transformer = Transformer(size, bg_file_name=
                                      f'{data_dir}background_transposed_64.png')

    cnn_red = ConvNet(environment, transformer, device)
    cnn_red.load_state_dict(torch.load(f'{models_dir}cnn_red.pt'))
    cnn_red.eval()

    cnn_blue = ConvNet(environment, transformer, device)
    cnn_blue.load_state_dict(torch.load(f'{models_dir}cnn_blue.pt'))
    cnn_blue.eval()

    evaluate_locations(cnn_red, cnn_blue)

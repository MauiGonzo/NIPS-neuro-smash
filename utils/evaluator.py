import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.transformer import Transformer
from locators.cnn import TwoCNNsLocator
from locators.agent_locator import AgentLocator
import Neurosmash


# settings for pyplot
matplotlib.use('tkagg')
plt.ion()

# empty figures for location prediction
fig_locations = None
ax_locations = None
state_img = None
red_location = None
blue_location = None

# empty figures for rewards
fig_rewards = None
ax_rewards = None
rewards_plot = None


def plot_locations(environment, transformer,
                   state, xy_red, xy_blue, perspective):
    """Plots the given locations of the agents on the
       given state image of the environment.

    Args:
        environment = [Environment] the environment in which the agent acts
        transformer = [Transformer] object that transforms images
        state       = [[int]] current state of the environment
        xy_red      = [int] x and y pixel coordinates of red agent
        xy_blue     = [int] x and y pixel coordinates of blue agent
        perspective = [bool] whether to perspective transform the state image
    """
    global fig_locations
    global ax_locations
    global state_img
    global red_location
    global blue_location

    image = environment.state2image(state)
    if perspective:
        image = transformer.perspective(image)

    if not fig_locations:
        fig_locations = plt.figure()
        ax_locations = fig_locations.add_subplot(1, 1, 1)
        ax_locations.set_title('Predicted locations of agents')
        ax_locations.set_xlabel('x')
        ax_locations.set_ylabel('y')
        ax_locations.set_xlim(0, transformer.size + 1)
        ax_locations.set_ylim(transformer.size + 1, 0)
        state_img = ax_locations.imshow(np.zeros((1, 1, 3)))
        state_img.set_extent((0.5, transformer.size + 0.5,
                              transformer.size + 0.5, 0.5))
        red_location = ax_locations.scatter(0, 0, c='r')
        blue_location = ax_locations.scatter(0, 0, c='b')

    state_img.set_data(image)
    red_location.set_offsets(xy_red)
    blue_location.set_offsets(xy_blue)
    plt.draw()
    plt.pause(0.001)


def plot_rewards(rewards, window=10):
    """Plot the given rewards as the average reward in a window of episodes.

    Args:
        rewards = [[float]] list of sum of rewards for each episode
        window  = [int] number of episode rewards to take the average of
    """
    global fig_rewards
    global ax_rewards
    global rewards_plot

    if len(rewards) < window:
        return

    rewards_smoothed = []
    for i in range(len(rewards) - window + 1):
        rewards_smoothed.append(np.mean(rewards[i:i + window]))

    if not fig_rewards:
        fig_rewards = plt.figure()
        ax_rewards = fig_rewards.add_subplot(1, 1, 1)
        ax_rewards.set_title('Smoothed episode rewards')
        ax_rewards.set_xlabel('Episode')
        ax_rewards.set_ylabel(f'Average reward in previous {window} episodes')
        rewards_plot = ax_rewards.plot(0)[0]

    ax_rewards.set_xlim(1, len(rewards))
    ax_rewards.set_ylim(min(rewards), max(rewards))
    rewards_plot.set_data(np.arange(window, len(rewards) + 1), rewards_smoothed)
    plt.draw()
    plt.pause(0.001)


def evaluate_locations(num_episodes=10):
    """Evaluate the agent locator on the real environment.

    Args:
        num_episodes  = [int] number of episodes to evaluate
    """
    rewards = []
    for _ in range(num_episodes):
        end, reward, state = environment.reset()
        rewards.append(reward)

        while not end:
            action = agent.step(end, reward, state)
            end, reward, state = environment.step(action)

            xy_red, xy_blue = agent_locator.get_locations(state)
            plot_locations(environment, transformer, state,
                           xy_red, xy_blue, agent_locator.perspective)

            rewards[-1] += reward
        plot_rewards(rewards, window=2)


if __name__ == '__main__':
    data_dir = '../data/'
    models_dir = '../models/'
    size = 64
    timescale = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    environment = Neurosmash.Environment(size=size, timescale=timescale)
    agent = Neurosmash.Agent()

    transformer = Transformer(
        size,
        bg_file_name=f'{data_dir}background_transposed_64.png'
    )

    agent_locator = TwoCNNsLocator(environment, transformer, models_dir, device)
    evaluate_locations()

    # in locators.agent_locator, set background file name from this file
    agent_locator = AgentLocator(cooldown_time=0, minimum_agent_area=4,
                                 minimum_agent_area_overlap=3)
    evaluate_locations()

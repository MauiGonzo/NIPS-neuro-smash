#!/usr/bin/python
#
# neurosmasher trainer
# Niels, Roel, Maurice
#
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

import Neurosmash
from agent_locator import AgentLocator
from pg_agent import PGAgent
from q_agent import QAgent
from utils import Aggregator

def neatagent():
    # do stuff
    pass


def get_locations(state_img: list):
    """Determines the agents' locations in pixel coordinates
       given the state image.

    Args:
        state_img = [[int]] current state of the environment

    Returns [((x_red, y_red), (x_blue, y_blue))]:
        The x and y pixel coordinate locations of the red and blue agents.
    """
    # get (size, size, 3) np.ndarray of current state
    state_img = np.asarray(environment.state2image(state_img), np.uint8, 'f')

    # update agents' locations and retrieve said locations
    locator.update_agent_locations(state_img)
    loc_red = locator.red_agent.pos
    loc_blue = locator.blue_agent.pos

    return loc_red, loc_blue


def plot_locations(state_img, xy_red, xy_blue):
    """Plots the given locations of the agents on the state image.

    Args:
        state_img = [[int]] current state of the environment
        xy_red    = [(int,int)] x and y pixel coordinates of red agent
        xy_blue   = [(int,int)] x and y pixel coordinates of blue agent
    """
    # get PIL image of current state
    state_img = environment.state2image(state_img)

    # plot agents' locations
    ax.clear()
    ax.imshow(state_img, extent=(0.5, size + 0.5, size + 0.5, 0.5))
    ax.scatter([xy_red[0], xy_blue[0]], [xy_red[1], xy_blue[1]], c=['r', 'b'])
    ax.set_title('Predicted locations of agents')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, size + 1)
    ax.set_ylim(size + 1, 0)
    fig.canvas.draw()
    plt.pause(0.0001)


def run_agent(env, agent, epsilon_start=0.95, epsilon_decay=0.995,
              epsilon_min=0.005, n_episodes=1000, train=True):
    """Function to run the agent for a given number of episodes.

    Args:
        env           = [Neurosmash.Environment] the environment in which
                                                 the agent lives and acts
        agent         = [Neurosmash.Agent] the agent that determines the
                                           action, given the current state
        epsilon_start = [float] starting value for exploration/
                                exploitation parameter
        epsilon_decay = [float] number signifying how fast epsilon decreases
        epsilon_min   = [float] minimal value for the epsilon parameter
        n_episodes    = [int] number of episodes the agent will learn for
        train         = [bool] whether to update the agent's model's parameters
    """
    R = []  # keep track of rewards
    epsilon = epsilon_start  # initialize epsilon
    for i_episode in range(n_episodes):  # loop over episodes
        # add element to rewards list
        R.append(0)

        # reset environment, create first observation
        end, reward, state_img = env.reset()
        old_loc_red, old_loc_blue = get_locations(state_img)
        old_state = aggregator.aggregate(old_loc_red, old_loc_red,
                                         old_loc_blue, old_loc_blue)
        while not end:
            # choose an action
            if random.random() < epsilon:
                # random actions to explore environment (exploration)
                action = random.randrange(environment.n_actions)
            else:
                # strictly follow currently learned behaviour (exploitation)
                action = agent.step(end, reward, old_state)

            for _ in range(locator.cooldown_time + 1):
                # do action, get reward and a new observation for the next round
                end, reward, state_img = env.step(action)
                new_loc_red, new_loc_blue = get_locations(state_img)
                plot_locations(state_img, new_loc_red, new_loc_blue)

            new_state = aggregator.aggregate(old_loc_red, new_loc_red,
                                             old_loc_blue, new_loc_blue)

            if train:  # adjust agent model's parameters (training step)
                agent.train(end, action, old_state, reward, new_state)

            # update state and locations variables
            old_state = new_state
            old_loc_red, old_loc_blue = new_loc_red, new_loc_blue

            # track the rewards
            R[i_episode] += reward

            # decay epsilon parameter
            epsilon = max(epsilon_min, epsilon * epsilon_decay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agents', nargs='+', type=str, help='agent names to train, choose <PG, NEAT, Q, random>')
    args = parser.parse_args()

    size = 64
    locator = AgentLocator(cooldown_time=1,
                           minimum_agent_area=10,
                           minimum_agent_area_overlap=4,
                           blue_marker_thresh=2,
                           red_marker_thresh=2)
    aggregator = Aggregator(size, 0.02*(locator.cooldown_time + 1))
    environment = Neurosmash.Environment(size=size, timescale=10)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.show()
    fig.canvas.draw()

    for ag in args.agents:
        if ag == 'PG':
            print('Processing agent: {}'.format(ag))
            pg_agent = PGAgent(aggregator.n_obs, environment.n_actions)
            run_agent(environment, pg_agent)
        elif ag == 'NEAT':
            print('Processing agent: {}'.format(ag))
            # do stuff
        elif ag == 'Q':
            print('Processing agent: {}'.format(ag))
            q_agent = QAgent(aggregator.n_obs, environment.n_actions)
            run_agent(environment, q_agent)
        elif ag == 'random':
            print('Processing agent: {}'.format(ag))
            random_agent = Neurosmash.Agent()
            run_agent(environment, random_agent)
        else: # current agent not known
            print('unknonw agent   : {}'.format(ag))

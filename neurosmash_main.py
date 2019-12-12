#!/usr/bin/python
#
# neurosmasher trainer
# Niels, Roel, Maurice
#
import argparse
import random

import numpy as np

import Neurosmash
from agent_locator import AgentLocator
from pg_agent import PGAgent
from utils import Aggregator

def neatagent():
    # do stuff
    pass

def get_locations(state_img: list):
    state_img = np.array(state_img, np.uint8).reshape(size, size, 3)
    locator.update_agent_locations(state_img)
    loc_red = locator.red_agent.pos
    loc_blue = locator.blue_agent.pos

    return loc_red, loc_blue


def run_agent(env, agent, epsilon_start=0.95, epsilon_decay=0.995,
              epsilon_min=0.005, n_episodes=1000, train=True):
    """Function to run the agent for a given number of episodes.

    Args:
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
        old_state = aggregator.aggregate(old_loc_red[0], old_loc_red[1],
                                         old_loc_red[0], old_loc_red[1],
                                         old_loc_blue[0], old_loc_red[1],
                                         old_loc_blue[0], old_loc_red[1])
        while not end:
            # choose an action
            if random.random() < epsilon:
                # random actions to explore environment (exploration)
                action = random.randrange(environment.n_actions)
            else:
                # strictly follow currently learned behaviour (exploitation)
                action = agent.step(end, reward, old_state)

            for _ in range(locator.cooldown_time + 1):
                # do action, get reward, and a new observation for the next round
                end, reward, state_img = env.step(action)
                new_loc_red, new_loc_blue = get_locations(state_img)

            new_state = aggregator.aggregate(old_loc_red[0], old_loc_red[1],
                                             new_loc_red[0], new_loc_red[1],
                                             old_loc_blue[0], old_loc_red[1],
                                             new_loc_blue[0], new_loc_red[1])
            print(new_state)

            if train:  # adjust agent model's parameters (training step)
                agent.train(end, action, old_state, reward, new_state)

            # update state variable
            old_state = new_state
            old_loc_red, old_loc_blue = new_loc_red, new_loc_blue

            # track the rewards
            R[i_episode] += reward

            # decay epsilon parameter
            epsilon = max(epsilon_min, epsilon * epsilon_decay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agents', nargs='+', type=str, help='agent names to train, choose <PG, NEAT, random>')
    args = parser.parse_args()

    size = 64
    locator = AgentLocator(cooldown_time=0,
                           minimum_agent_area=10,
                           minimum_agent_area_overlap=4,
                           blue_marker_thresh=2,
                           red_marker_thresh=2)
    aggregator = Aggregator(size, 0.02*(locator.cooldown_time + 1))
    environment = Neurosmash.Environment(size=size)

    for ag in args.agents:
        if ag == 'PG':
            print('Processing agent: {}'.format(ag))
            pg_agent = PGAgent(aggregator.n_obs, environment.n_actions)
            run_agent(environment, pg_agent)
        elif ag == 'NEAT':
            print('Processing agent: {}'.format(ag))
            # do stuff
        elif ag == 'random':
            print('Processing agent: {}'.format(ag))
            random_agent = Neurosmash.Agent()
            run_agent(environment, random_agent)
        else: # current agent not known
            print('unknonw agent   : {}'.format(ag))

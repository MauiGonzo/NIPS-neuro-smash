#!/usr/bin/python
#
# neurosmasher trainer
# Niels, Roel, Maurice
import logging

import matplotlib
matplotlib.use('tkagg')
#
import argparse
import platform
import subprocess
from typing import List, Dict, Type

import matplotlib.pyplot as plt
import numpy as np

import Neurosmash
from agent_locator import AgentLocator
from neat_agent import NeatAgent

ip = "127.0.0.1"  # Ip address that the TCP/IP interface listens to
port = 13000  # Port number that the TCP/IP interface listens to
size = 64  # Please check the Updates section above for more details
timescale = 1  # Please check the Updates section above for more details


class NeurosmashRunner(object):
    def __init__(self, args):
        self.args = args

    def run_round(self, agent: Neurosmash.Agent):
        """
        Run a single round of Neurosmash, given the agent `agent`

        Parameters
        ----------
        agent : Neurosmash.Agent
            The agent to query at each simulation step
        """
        end, reward, state = env.reset()
        state_img = state_to_ndarray(state)

        al = AgentLocator(cooldown_time=0, minimum_agent_area=4, minimum_agent_area_overlap=3, save_difficult=self.args.save_difficult,
                          save_difficult_prob=self.args.save_diff_prob)
        al.update_agent_locations(state_img)
        blue, red = al.blue_agent.pos, al.red_agent.pos
        if self.args.b_plot:
            mpl_img = plt.imshow(state_img)
            ax = plt.gca()
            mpl_blue = ax.scatter(blue[1], blue[0], c='b')
            mpl_red = ax.scatter(red[1], red[0], c='r')
            plt.ion()

        logging.info('\nStarting new round...')

        # timout = 700
        timeout = 2500

        steps = 0
        while end == 0 and steps < timeout:
            steps += 1
            action = agent.step(end, reward, state, al.blue_agent.pos, al.red_agent.pos)
            end, reward, state = env.step(action)
            # print(end, reward)
            state_img = state_to_ndarray(state)
            al.update_agent_locations(state_img)
            blue, red = al.blue_agent.pos, al.red_agent.pos
            if args.b_plot:
                mpl_img.set_data(state_img)
                mpl_blue.set_offsets(blue[::-1])
                mpl_red.set_offsets(red[::-1])
                plt.draw()
                plt.pause(0.0001)

        if args.b_plot:
            plt.close()
        print(steps, reward > 0.1)
        return steps, reward > 0.1


def state_to_ndarray(state: List[int]) -> np.ndarray:
    """

    Parameters
    ----------
    state : List[int]
        The environment state in the form of a 1D integer list

    Returns
    -------
        np.ndarray
            The state as a numpy array of shape (size, size, channels=3)

    """
    return np.array(state).reshape(size, size, 3)


agents_types: Dict[str, Type[Neurosmash.Agent]] = {
    'NEAT': NeatAgent,
    'RAND': Neurosmash.Agent
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str, choices=list(agents_types.keys()),
                        help='agent names to train, choose <PG, NEAT>')
    parser.add_argument('--plot-positions', dest='b_plot', action='store_true',
                        help='whether the detected agent positions should be plotted')
    parser.add_argument('--eval-only', dest='eval_only', action='store_true')
    parser.add_argument('--save-difficult', dest='save_difficult', action='store_true')
    parser.add_argument('-save-diff-prob', dest='save_diff_prob', type=float)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')

    for agent_type in agents_types.values():
        agent_type.define_args(parser)

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if platform.system() == 'Darwin':
        p = subprocess.Popen(['open', 'Mac.app'], stderr=subprocess.PIPE)
        while p.poll() is None:
            pass

    connected = False
    while not connected:
        # if platform.system() in ['Darwin'] and p.poll() is None:
        #     logging.critical('Failed to run Neurosmash')
        #     sys.exit(1)

        try:
            env = Neurosmash.Environment(size=size, timescale=10)
            connected = True
        except ConnectionRefusedError:
            pass

    runner = NeurosmashRunner(args)
    agent = agents_types[args.agent](runner, args)
    agent.run()

    # end, reward, state = env.reset()
    # state_img = state_to_ndarray(state)
    # # blue, red = detect_positions(state_img, None, None)
    # al = AgentLocator()
    # al.update_agent_locations(state_img)
    # blue, red = al.blue_agent.pos, al.red_agent.pos
    # if args.b_plot:
    #     mpl_img = plt.imshow(state_img)
    #     ax = plt.gca()
    #     mpl_blue = ax.scatter(blue[1], blue[0], c='b')
    #     mpl_red = ax.scatter(red[1], red[0], c='r')
    #     plt.ion()

    # while not agent.is_finished():
    #     action = agent.step(end, reward, state)
    #     end, reward, state = env.step(action)
    #     state_img = state_to_ndarray(state)
    #     al.update_agent_locations(state_img)
    #     blue, red = al.blue_agent.pos, al.red_agent.pos
    #     if args.b_plot:
    #         mpl_img.set_data(state_img)
    #         mpl_blue.set_offsets(blue[::-1])
    #         mpl_red.set_offsets(red[::-1])
    #         plt.draw()
    #         plt.pause(0.0001)

    if platform.system() in ['Darwin']:
        p.terminate()

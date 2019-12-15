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
import utils
import platform
import subprocess
from neat_agent import NeatAgent
import matplotlib.pyplot as plt
from agent_locator import AgentLocator
from typing import List, Dict, Type
import numpy as np
import logging
import sys

ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 13000       # Port number that the TCP/IP interface listens to
size       = 64         # Please check the Updates section above for more details
timescale  = 1           # Please check the Updates section above for more details

def init_neurosmash(size=768):
    agent = Neurosmash.Agent()
    environment = Neurosmash.Environment(size=size)
    return agent, environment

def pgagent():
    # do stuff
    # pgagent, pgenvironment = init_neurosmash()
    # yxredbluetorchtensor = locations(pgagent, pgenvironment)
    # pgmetrics = utils.Aggregate(yxredbluetorchtensor)
    pass

# def neatagent():
#     # do stuff
#     pass

class NeurosmashRunner(object):
    def __init__(self, args):
        self.args = args
    
    def run_round(self, agent: Neurosmash.Agent):
        print('aaaaah')
        end, reward, state = env.reset()
        print('aaah2')
        state_img = state_to_ndarray(state)

        al = AgentLocator(cooldown_time=1, minimum_agent_area=10, minimum_agent_area_overlap=4, blue_marker_thresh=2, red_marker_thresh=2)
        al.update_agent_locations(state_img)
        blue, red = al.blue_agent.pos, al.red_agent.pos
        if self.args.b_plot:
            mpl_img = plt.imshow(state_img)
            ax = plt.gca()
            mpl_blue = ax.scatter(blue[1], blue[0], c='b')
            mpl_red = ax.scatter(red[1], red[0], c='r')
            plt.ion()
        
        steps = 0
        while end == 0 and steps < 460:
            steps += 1
            action = agent.step(end, reward, state, blue, red)
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
        print(steps, reward > 0.1)
        return steps, reward > 0.1
        

def state_to_ndarray(state: List) -> np.ndarray:
    return np.array(state).reshape(size, size, 3)

agents_types: Dict[str, Type[Neurosmash.Agent]] = {
    'NEAT': NeatAgent, 
    'RAND': Neurosmash.Agent
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str, choices=list(agents_types.keys()), help='agent names to train, choose <PG, NEAT>')
    parser.add_argument('--plot-positions', dest='b_plot', action='store_true', help='whether the detected agent positions should be plotted')
    args = parser.parse_args()
    
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
    agent = agents_types[args.agent](runner)
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

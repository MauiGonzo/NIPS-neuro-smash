import argparse

import numpy as np
import socket

from tqdm import tqdm
from PIL import Image


class Agent:
    def __init__(self, neurosmash_runner, args):
        self.neurosmash_runner = neurosmash_runner
        self.args = args

    def step(self, end, reward, state, blue_hist, red_hist):
        # return 0 # nothing
        # return 1 # left
        # return 2 # right
        return   0 # random

    def run(self):
        for i in tqdm(range(self.neurosmash_runner.args.rounds)):
            self.neurosmash_runner.run_round(self)

    def train(self, end, action, old_state, reward, new_state):
        pass

    @classmethod
    def define_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('-rounds', type=int, required=False)

class Environment:
    def __init__(self, ip = "127.0.0.1", port = 13000, size = 768, timescale = 1):
        self.client     = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip         = ip
        self.port       = port
        self.size       = size
        self.timescale  = timescale

        self.client.connect((ip, port))

        self.n_actions = 3

    def reset(self):
        self._send(1, 0)
        return self._receive()

    def step(self, action):
        self._send(2, action)
        return self._receive()

    def state2image(self, state):
        return Image.fromarray(np.array(state, np.uint8).reshape(self.size, self.size, 3))

    def _receive(self):
        # Kudos to Jan for the socket.MSG_WAITALL fix!
        data = self.client.recv(2 + 3 * self.size ** 2, socket.MSG_WAITALL)
        end = data[0]
        if end:
            reward = data[1]
        else:
            reward = 0.1
        state = [data[i] for i in range(2, len(data))]

        return end, reward, state

    def _send(self, action, command):
        self.client.send(bytes([action, command]))

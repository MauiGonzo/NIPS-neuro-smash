import socket

import numpy as np
from PIL import Image


class Agent:
    def __init__(self):
        pass

    def step(self, end, reward, state):
        # return 0  # nothing
        return 1  # left
        # return 2  # right
        # return 3  # random

    def train(self, end, action, old_state, reward, new_state):
        pass


class Environment:
    def __init__(self, ip="127.0.0.1", port=13000, size=768, timescale=1):
        self.size = size
        self.num_actions = 3

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((ip, port))

    def reset(self):
        self._send(1, 0)
        return self._receive()

    def step(self, action):
        self._send(2, action)
        return self._receive()

    def state2image(self, state):
        state = np.array(state, np.uint8).reshape((self.size, self.size, 3))
        return Image.fromarray(state)

    def _receive(self):
        # Kudos to Jan for the socket.MSG_WAITALL fix!
        data = self.client.recv(2 + 3 * self.size ** 2, socket.MSG_WAITALL)
        end = data[0]
        if end:
            reward = (data[1] - 5) * 2
        else:
            reward = 0.1
        state = [data[i] for i in range(2, len(data))]

        return end, reward, state

    def _send(self, action, command):
        self.client.send(bytes([action, command]))

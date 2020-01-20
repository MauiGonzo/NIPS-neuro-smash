import math

import Neurosmash


class ChaseAgent(Neurosmash.Agent):
    """Implements an agent with hardcoded rules.

    Attributes:
        aggregator = [Aggregator] object to compute angles between locations
    """

    def __init__(self, aggregator):
        """Initializes the agent.

        Args:
            aggregator = [Aggregator] object to compute angles between locations
        """
        super(ChaseAgent, self).__init__()

        self.aggregator = aggregator

    def step(self, end, reward, state):
        """Agent determines the action based on the current state.

        Args:
            end    = [bool] whether the episode has finished
            reward = [int] reward received after doing previous action
            state  = [[int]]] current state of the environment

        Returns [int]:
            Action in the range [0, num_actions).
        """
        # get angle between old red and new red and between new red and new blue
        red_direction = state[self.aggregator.old_red_new_red_dir_idx]
        blue_direction = state[self.aggregator.new_red_new_blue_dir_idx]

        # compute the difference of the angles in the range (-math.pi, math.pi]
        direction_difference = red_direction - blue_direction
        if direction_difference > math.pi:
            direction_difference -= math.pi*2
        elif direction_difference < -math.pi:
            direction_difference += math.pi*2

        # determine action
        if math.pi*1/4 >= direction_difference >= -math.pi*1/4:
            # follow blue agent if it's in front of red agent
            return 0
        else:  # otherwise, turn left
            return 1

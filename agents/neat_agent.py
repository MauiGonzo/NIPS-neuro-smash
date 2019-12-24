import os
import re
import glob

import neat
import numpy as np

from utils.dataloader import natural_key
import Neurosmash


class NeatAgent(Neurosmash.Agent):
    """Implements a Neuroevolution of Augmenting Topologies Agent.

    Attributes:
        run_agent = [function] function that runs episodes in the environment
        net       = [RecurrentNetwork] ANN that implements agent's policy
        p         = [Population] NEAT population in the current generation
    """

    def __init__(self, model_dir, run_agent):
        """Initializes the agent. Either a brand new agent, or an agent
           recovered from a NEAT checkpoint.

        Args:
            model_dir = [str] directory where NEAT model will be stored
            run_agent = [function] function to run episodes in the environment
        """
        super(NeatAgent, self).__init__()

        # set function to run the agent
        self.run_agent = run_agent

        # sets member that will be the ANN that implements the agent's policy
        self.net = None

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        prefix = f'{model_dir}neat-checkpoint-'
        checkpoints = glob.glob(f'{prefix}*')
        if not checkpoints:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 'agents/config-feedforward')

            # create population, which is the top-level object for a NEAT run
            self.p = neat.Population(config)
        else:
            # create population, which is the top-level object for a NEAT run
            checkpoints = sorted(checkpoints, key=natural_key)
            last_checkpoint = checkpoints[-1]
            self.p = neat.Checkpointer.restore_checkpoint(last_checkpoint)

        # add a stdout reporter to show progress in the terminal
        self.p.add_reporter(neat.StdOutReporter(True))
        self.p.add_reporter(neat.StatisticsReporter())
        self.p.add_reporter(neat.Checkpointer(5, filename_prefix=prefix))

    def run(self):
        """Run the agent in the NEAT framework for 100 generations."""
        self.p.run(self.eval_fitness, 100)

    def eval_fitness(self, genomes, config):
        """Fitness function that computes the fitness for each genome.

        Args:
            genomes = [[(int, Genome)]] network genomes in current generation
            config  = [Config] configuration of the neural network
        """
        for _, genome in genomes:
            # create neural network policy given genome
            self.net = neat.nn.RecurrentNetwork.create(genome, config)

            # run some episodes and get the number of steps and wins
            num_steps, wins = self.run_agent(self, num_episodes=7,
                                             epsilon_start=0, epsilon_min=0)

            # compute and set the fitness of the network
            fitness_list = [self._fitness(*f) for f in zip(num_steps, wins)]
            genome.fitness = np.mean(fitness_list)
            print(f'fitness: {genome.fitness}')

    def step(self, end, reward, state):
        """The agent selects action given the current state and network.

        Args:
            end    = [bool] whether the episode has finished
            reward = [int] reward received after doing previous action
            state  = [torch.Tensor] current state of the environment

        Returns [int]:
            The action encoded as a number in the range [0, num_actions).
        """
        state = state.cpu().data.numpy()
        return np.argmax(self.net.activate(state))

    @staticmethod
    def _fitness(num_steps, won):
        """Compute the fitness of an agent during an episode.

        Args:
            num_steps = [int] number of steps in the episode
            won       = [bool] whether the red agent won in the episode

        Returns [float]:
            Fitness of the agent for the episode.
        """
        if won:
            return -num_steps
        elif num_steps < 700:
            return -700 * 5 + num_steps
        else:
            return -700 * 3.3

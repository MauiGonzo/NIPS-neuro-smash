import argparse
import os
import re
from typing import List

import neat
import numpy as np
from six import iteritems

from Neurosmash import Agent


class NeatAgent(Agent):
    def __init__(self, neurosmash_runner, args):
        """

        Parameters
        ----------
        neurosmash_runner:
        """
        Agent.__init__(self, neurosmash_runner, args)
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')

        agent_dir_exists = os.path.isdir(f'agents/{self.args.name}')

        if not agent_dir_exists or len(list(os.scandir(f'agents/{self.args.name}'))) == 0:
            if not agent_dir_exists:
                os.mkdir(f'agents/{self.args.name}')

            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)

            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)
        else:
            last_checkpoint = max([int(re.findall(r'neat-checkpoint-(\d+)', f.name)[0]) for f in os.scandir(f'agents/{self.args.name}') if
                     re.match(r'neat-checkpoint-\d+', f.name)])
            p = neat.Checkpointer.restore_checkpoint(f'agents/{self.args.name}/neat-checkpoint-{last_checkpoint}')

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5, filename_prefix=f'agents/{self.args.name}/neat-checkpoint-'))
        self.p = p

    @classmethod
    def define_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('-name', type=str)

    def run(self):
        if not self.args.eval:
            self.p.run(self.eval_fitness, 100)
        else:
            best_won = 0

            for genome_id, genome in list(iteritems(self.p.population)):
                self.net = neat.nn.RecurrentNetwork.create(genome, self.p.config)
                avg_steps, avg_won = 0, 0

                for i in range(self.args.rounds):
                    steps, won = self.neurosmash_runner.run_round(self)
                    avg_steps += steps
                    avg_won += won

                avg_steps /= self.args.rounds
                avg_won /= self.args.rounds
                if best_won > avg_won:
                    best_won = avg_won
                print(f'Avg steps: {avg_steps}')
                print(f'Avg won: {avg_won}')
            print(f'Best avg won: {best_won}')

    def eval_fitness(self, genomes, config):
        for genome_id, genome in genomes:
            self.net = neat.nn.RecurrentNetwork.create(genome, config)
            fitnesses = []
            for i in range(7):
                steps, won = self.neurosmash_runner.run_round(self)
                if won:
                    f = -steps
                else:
                    if steps < 700:
                        f = -700*5+steps
                    else:
                        f = -700*3.3

                fitnesses.append(f)

            genome.fitness = np.mean(fitnesses)
            print(genome.fitness)

    def step(self, end: bool, reward: float, state: List[int], blue: np.ndarray, red: np.ndarray):
        # hist_indices = [-1]
        # blue_poses = [blue[i] if len(blue) >= -i else blue[-1] for i in hist_indices]
        # red_poses = [red[i] if len(red) >= -i else red[-1] for i in hist_indices]

        # blue_inp_x, blue_inp_y = zip(*blue_poses)
        # red_inp_x, red_inp_y = zip(*red_poses)
        #
        # inp = np.concatenate([blue_inp_x, blue_inp_y, red_inp_x, red_inp_y])
        inp = np.array([blue[0], blue[1], red[0], red[1]])/64
        # print(inp)
        return np.argmax(self.net.activate(inp))

from Neurosmash import Agent
from typing import List
import neat
import os
import numpy as np

class NeatAgent(Agent):
    def __init__(self, neurosmash_runner):
        Agent.__init__(self, neurosmash_runner)
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))
        self.p = p
    
    def run(self):
        self.p.run(self.eval_fitness, 100)
    
    def eval_fitness(self, genomes, config):
        for genome_id, genome in genomes:
            self.net = neat.nn.FeedForwardNetwork.create(genome, config)
            steps, won = self.neurosmash_runner.run_round(self)
            fitnesses = []
            for i in range(3):
                fitnesses.append(-steps if won else steps*0.25-460*2)
            
            genome.fitness = min(fitnesses)

    def step(self, end: bool, reward: float, state: List[int], blue: np.ndarray, red: np.ndarray):
        return np.argmax(self.net.activate(list(blue) + list(red)))
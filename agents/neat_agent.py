import os
import re

import neat
import numpy as np

import Neurosmash


class NeatAgent(Neurosmash.Agent):
    """Implements a Neuroevolution of Augmenting Topologies Agent.

    Attributes:
        environment   = [Environment] the environment in which the agent acts
        agent_locator = [object] determines x and y pixel coordinates of agents
        transformer   = [Transformer] object that transforms images
        aggregator    = [Aggregator] object that aggregates a number of stats
                                     given the positions of the agents
        run_agent     = [fn] function that runs an episode in the environment
        net           = [RecurrentNetwork] ANN that implements agent's policy
        p             = [Population] NEAT population in the current generation
    """

    def __init__(self, environment, agent_locator, transformer, aggregator,
                 model_dir, run_agent):
        """Initializes the agent.

        Args:
            environment   = [Environment] environment in which the agents act
            agent_locator = [object] determines pixel coordinates of agents
            transformer   = [Transformer] object that transforms images
            aggregator    = [Aggregator] object that aggregates a number of
                                         stats given the positions of the agents
            model_dir    = [str] directory where NEAT model will be stored
            run_agent     = [fn] function that runs a round in the environment
        """
        super(NeatAgent, self).__init__()

        self.environment = environment
        self.agent_locator = agent_locator
        self.transformer = transformer
        self.aggregator = aggregator
        self.run_agent = run_agent

        # sets member that will be the ANN that implements the agent's policy
        self.net = None

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        if not list(os.scandir(model_dir)):
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 'agents/config-feedforward')

            # create population, which is the top-level object for a NEAT run
            self.p = neat.Population(config)
        else:
            checkpoints = []
            for f in os.scandir(model_dir):
                if re.match(r'neat-checkpoint-\d+', f.name):
                    match = re.search(r'neat-checkpoint-(\d+)', f.name).group()
                    checkpoints.append(int(match))

            # create population, which is the top-level object for a NEAT run
            last_checkpoint = max(checkpoints)
            self.p = neat.Checkpointer.restore_checkpoint(
                f'{model_dir}neat-checkpoint-{last_checkpoint}'
            )

        # add a stdout reporter to show progress in the terminal
        self.p.add_reporter(neat.StdOutReporter(True))
        self.p.add_reporter(neat.StatisticsReporter())
        prefix = f'{model_dir}neat-checkpoint-'
        self.p.add_reporter(neat.Checkpointer(5, filename_prefix=prefix))

    def run(self):
        self.p.run(self.eval_fitness, 100)

    def eval_fitness(self, genomes, config):
        for genome_id, genome in genomes:
            self.net = neat.nn.RecurrentNetwork.create(genome, config)
            fitness_list = []
            for i in range(7):
                num_steps, won = self.run_agent(
                    self.environment, self, self.agent_locator,
                    self.transformer, self.aggregator,
                    epsilon_start=0, epsilon_min=0, num_episodes=1
                )
                if won:
                    f = -num_steps
                else:
                    if num_steps < 700:
                        f = -700*5+num_steps
                    else:
                        f = -700*3.3

                fitness_list.append(f)

            genome.fitness = np.mean(fitness_list)
            print(f'fitness: {genome.fitness}')

    def step(self, end, reward, state):
        state = state.data.numpy()
        return np.argmax(self.net.activate(state))

    def train(self, end, action, old_state, reward, new_state):
        pass

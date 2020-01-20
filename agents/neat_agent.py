import os
import glob

from pyneat.activations import ActivationSigmoid, ActivationSigmoidAdjusted, ActivationReLU
from pyneat.attributes import FloatAttribute, ChoiceAttribute
from pyneat.neat import Population
from pyneat.genome import Config
from pyneat.graph import RecurrentNetwork
from pyneat.persistence import save_population_checkpoint, load_population_checkpoint
import numpy as np

from utils.aggregator import AggregationType
from utils.dataloader import natural_key
from scipy.stats import norm, gamma
from statsmodels.stats.proportion import proportion_confint

import Neurosmash


class NeatAgent(Neurosmash.Agent):
    """Implements a Neuroevolution of Augmenting Topologies Agent.

    Attributes:
        run_agent = [function] function that runs episodes in the environment
        net       = [RecurrentNetwork] ANN that implements agent's policy
        p         = [Population] NEAT population in the current generation
    """

    aggregation_type_config_mapping = {
        AggregationType.FULL: 'config-feedforward',
        AggregationType.POS_ONLY: 'config-feedforward-pos-only'
    }

    def __init__(self, model_dir: str, run_agent, aggregation_type: AggregationType, timeout: int, evaluate: bool):
        """Initializes the agent. Either a brand new agent, or an agent
           recovered from a NEAT checkpoint.

        Args:
            model_dir = [str] directory where NEAT model will be stored
            run_agent = [function] function to run episodes in the environment
        """
        super(NeatAgent, self).__init__()

        # set function to run the agent
        self.run_agent = run_agent

        self.timeout = timeout
        self.evaluate = evaluate

        # sets member that will be the ANN that implements the agent's policy
        self.net = None

        self.model_dir = model_dir

        try:
            self.population = load_population_checkpoint(self.model_dir, self.eval_fitness)
        except:
            config = Config()
            config.bias_attribute = FloatAttribute(norm(loc=0, scale=1), norm(loc=0, scale=0.42), 0.7, 0.1)
            config.weight_attribute = FloatAttribute(norm(loc=0, scale=1), norm(loc=0, scale=0.42), 0.8, 0.1)
            config.compatibility_disjoint_coefficient = 1.0
            config.compatibility_threshold = 2.2
            config.output_activations = ChoiceAttribute({
                ActivationSigmoid(): 0.5,
                ActivationSigmoidAdjusted(): 0.5,
            }, 0.0)
            config.hidden_activations = ChoiceAttribute({
                ActivationSigmoid(): 0.25,
                ActivationSigmoidAdjusted(): 0.25,
                ActivationReLU(): 0.5
            }, 0.01)
            self.population = Population(38, 4 if aggregation_type == AggregationType.POS_ONLY else 18, 3, config, self.eval_fitness, log_generations=True)

        # prefix = os.path.join(model_dir, 'neat-checkpoint-')
        # checkpoints = glob.glob(f'{prefix}*')
        # if not checkpoints:
        #     config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        #                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
        #                          os.path.join('agents', NeatAgent.aggregation_type_config_mapping[aggregation_type]))
        #
        #     # create population, which is the top-level object for a NEAT run
        #     self.p = neat.Population(config)
        # else:
        #     # create population, which is the top-level object for a NEAT run
        #     checkpoints = sorted(checkpoints, key=natural_key)
        #     last_checkpoint = checkpoints[-1]
        #     self.p = neat.Checkpointer.restore_checkpoint(last_checkpoint)
        #
        # # add a stdout reporter to show progress in the terminal
        # self.p.add_reporter(neat.StdOutReporter(True))
        # self.p.add_reporter(neat.StatisticsReporter())
        # self.p.add_reporter(neat.Checkpointer(5, filename_prefix=prefix))

    def run(self):
        """Run the agent in the NEAT framework for 100 generations."""
        if self.evaluate:
            nets = []
            for g in self.population.population:
                nets += [RecurrentNetwork.from_genome(g)]

            best_net = None
            best_ind = 0
            best_wins = 0

            for i, net in enumerate(nets):
                self.net = net
                # self.net.render()
                num_steps, wins = [], []
                n = 50
                for j in range(n):
                    self.net.reset()
                    num_step, win = self.run_agent(self, num_episodes=1,
                                                   epsilon_start=0, epsilon_min=0)
                    num_steps += num_step
                    wins += win

                    if j >= 4 and proportion_confint(sum(wins), j + 1, method='beta')[1] < 0.78:
                        break

                if best_net is None or sum(wins) > best_wins:
                    best_net = net
                    best_ind = i
                    best_wins = sum(wins)
                    print(f'New best: {best_wins}, from {i}')
                print(num_steps)
                print(wins)

            print(best_ind)

            self.net = best_net
            # self.net = RecurrentNetwork.from_genome(self.population.population[25])
            num_steps, wins = self.run_agent(self, num_episodes=1000,
                                             epsilon_start=0, epsilon_min=0)
            print(num_steps)
            print(wins)
        else:
            for i in range(self.population.generation, self.population.generation+25):
                self.population.step()
                save_population_checkpoint(self.model_dir, i, self.population)


    def eval_fitness(self, genome):
        """Fitness function that computes the fitness for each genome.

        Args:
            genomes = [[(int, Genome)]] network genomes in current generation
            config  = [Config] configuration of the neural network
        """
        # create neural network policy given genome
        self.net = RecurrentNetwork.from_genome(genome)

        # run some episodes and get the number of steps and wins
        num_steps, wins = [], []
        n = 25
        for i in range(n):
            self.net.reset()
            num_step, win = self.run_agent(self, num_episodes=1,
                                             epsilon_start=0, epsilon_min=0)
            num_steps += num_step
            wins += win

            if i >= 4 and proportion_confint(sum(wins), i+1, method='beta')[1] < 0.67:
                break

        # compute and set the fitness of the network
        fitness_list = [self._fitness(n_step, win) for n_step, win in zip(num_steps, wins)]
        fitness = np.mean(fitness_list)
        print(f'fitness: {fitness}')
        return fitness

    def step(self, end, reward, state):
        """The agent selects action given the current state and network.

        Args:
            end    = [bool] whether the episode has finished
            reward = [int] reward received after doing previous action
            state  = [torch.Tensor] current state of the environment

        Returns [int]:
            The action encoded as a number in the range [0, num_actions).
        """
        state = state.cpu().data.numpy()/64
        return np.argmax(self.net.predict(state))

    def _fitness(self, num_steps, won):
        """Compute the fitness of an agent during an episode.

        Args:
            num_steps = [int] number of steps in the episode
            won       = [bool] whether the red agent won in the episode

        Returns [float]:
            Fitness of the agent for the episode.
        """
        if won:
            return -num_steps
        elif num_steps < self.timeout:
            return -self.timeout * 5 + num_steps
        else:
            return -self.timeout * 3.3

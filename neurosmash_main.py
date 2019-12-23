import argparse
import random

import torch

from agents.chase_agent import ChaseAgent
from agents.neat_agent import NeatAgent
from agents.pg_agent import PGAgent
from agents.q_agent import QAgent
from locators.agent_locator import AgentLocator
from locators.cnn import TwoCNNsLocator
from utils.transformer import Transformer
from utils.aggregator import Aggregator
from utils.evaluator import plot_locations, plot_rewards
import Neurosmash

ip = '127.0.0.1'  # IP address that the TCP/IP interface listens to
port = 13000  # port number that the TCP/IP interface listens to
size = 64  # width and height pixels of environment
timescale = 10  # number of timesteps per action
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = 'data/'
models_dir = 'models/'


def run_agent(env, agent, agent_locator, transformer, aggregator,
              epsilon_start=0.95, epsilon_decay=0.995, epsilon_min=0.005,
              num_episodes=1000, train=True):
    """Function to run the agent for a given number of episodes.

    Args:
        env           = [Environment] environment in which the agent acts
        agent         = [Agent] agent that selects action given current state
        agent_locator = [object] determines x and y pixel coordinates of agents
        transformer   = [Transformer] object that transforms images
        aggregator    = [Aggregator] object that aggregates a number of stats
                                     given the positions of the agents
        epsilon_start = [float] starting value for exploration/
                                exploitation parameter
        epsilon_decay = [float] number signifying how fast epsilon decreases
        epsilon_min   = [float] minimal value for the epsilon parameter
        num_episodes  = [int] number of episodes the agent will learn for
        train         = [bool] whether to update the agent's model's parameters

    Returns [(int, bool)]:
        Number of steps in the last episode and whether the red agent was
        victorious in said episode.
    """
    rewards = []
    epsilon = epsilon_start  # initialize epsilon
    for i_episode in range(num_episodes):
        # reset environment, create first observation
        end, reward, state_img = env.reset()
        old_loc_red, old_loc_blue = agent_locator.get_locations(state_img)
        old_state = aggregator.aggregate(old_loc_red, old_loc_red,
                                         old_loc_blue, old_loc_blue)

        rewards.append(reward)
        num_steps = 0
        while not end:
            # choose an action
            if random.random() < epsilon:
                # random actions to explore environment (exploration)
                action = random.randrange(env.num_actions)
            else:
                # strictly follow currently learned behaviour (exploitation)
                action = agent.step(end, reward, old_state)

            # do action, get reward and a new observation for the next round
            end, reward, state_img = env.step(action)
            new_loc_red, new_loc_blue = agent_locator.get_locations(state_img)
            new_state = aggregator.aggregate(old_loc_red, new_loc_red,
                                             old_loc_blue, new_loc_blue)

            if args.p_plot:  # plot predicted locations of agents
                plot_locations(env, transformer, state_img, new_loc_red,
                               new_loc_blue, agent_locator.perspective)

            if train:  # adjust agent model's parameters (training step)
                agent.train(end, action, old_state, reward, new_state)

            # bookkeeping
            old_state = new_state
            old_loc_red, old_loc_blue = new_loc_red, new_loc_blue
            rewards[-1] += reward
            num_steps += 1

            # decay epsilon parameter
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # determine whether the agent lost or won
        if reward == -10:
            print('--- LOSER ---')
        elif reward == 10:
            print('--- WINNER ---')

        if args.r_plot:
            plot_rewards(rewards)

    return num_steps, reward == 10


if __name__ == '__main__':
    # process the command options
    parser = argparse.ArgumentParser()
    agents = ['PG', 'PG_run', 'Q', 'Q_run', 'NEAT', 'random', 'chase']
    parser.add_argument('agent', type=str, choices=agents,
                        help=f'agent names to train, choose {agents}')
    parser.add_argument('agent_locator', type=str, choices=['TwoCNN', 'Vision'],
                        help='finds agent positions, choose <TwoCNN, Vision>')
    parser.add_argument('--plot-positions', dest='p_plot', action='store_true',
                        help='plot the positions of the agents')
    parser.add_argument('--plot-rewards', dest='r_plot', action='store_true',
                        help='plot the smoothed rewards of the episodes')
    parser.add_argument('--save-difficult', dest='save_difficult',
                        action='store_true')
    parser.add_argument('-save-diff-prob', dest='save_diff_prob', type=float)
    args = parser.parse_args()

    # connect to environment via TCP/IP
    connected = False
    while not connected:
        try:
            environment = Neurosmash.Environment(size=size, timescale=timescale)
            connected = True
        except ConnectionRefusedError:
            pass

    # initialize the transformer, aggregator, and agent locator
    transformer = Transformer(size, bg_file_name=f'{data_dir}background_64.png')
    aggregator = Aggregator(size, device=device)
    if args.agent_locator == 'TwoCNN':
        agent_locator = TwoCNNsLocator(environment, transformer,
                                       models_dir=models_dir, device=device)
    else:  # args.agent_locator == 'Vision'
        agent_locator = AgentLocator(cooldown_time=0, minimum_agent_area=4,
                                     minimum_agent_area_overlap=3,
                                     save_difficult=args.save_difficult,
                                     save_difficult_prob=args.save_diff_prob)

    # run, train, or process the picked agent
    if args.agent == 'PG':
        print('Training Policy Gradient agent')
        pg_agent = PGAgent(aggregator.num_obs, environment.num_actions,
                           device=device)
        run_agent(environment, pg_agent, agent_locator,
                  transformer, aggregator)
        torch.save(pg_agent.model.state_dict(), f'{models_dir}mlp.pt')
    elif args.agent == 'PG_run':
        print('Running Policy Gradient agent')
        pg_agent = PGAgent(aggregator.num_obs, environment.num_actions,
                           device=device)
        pg_agent.model.load_state_dict(torch.load(f'{models_dir}mlp.pt'))
        pg_agent.model.eval()
        run_agent(environment, pg_agent, agent_locator,
                  transformer, aggregator,
                  epsilon_start=0, epsilon_min=0, train=False)
    elif args.agent == 'Q':
        print('Training Q-learning agent')
        q_agent = QAgent(aggregator.num_obs, environment.num_actions,
                         device=device)
        run_agent(environment, q_agent, agent_locator,
                  transformer, aggregator)
        torch.save(q_agent.policy_network.state_dict(),
                   f'{models_dir}ddqn.pt')
    elif args.agent == 'Q_run':
        print('Running Q-learning agent')
        q_agent = QAgent(aggregator.num_obs, environment.num_actions,
                         device=device)
        q_agent.policy_network.load_state_dict(
                torch.load(f'{models_dir}ddqn.pt'))
        q_agent.policy_network.eval()
        run_agent(environment, q_agent, agent_locator,
                  transformer, aggregator,
                  epsilon_start=0, epsilon_min=0, train=False)
    elif args.agent == 'NEAT':
        print('Processing NEAT agent')
        aggregator = Aggregator(size, device=torch.device('cpu'))
        neat_agent = NeatAgent(environment, agent_locator, transformer,
                               aggregator, f'{models_dir}NEAT/', run_agent)
        neat_agent.run()
    elif args.agent == 'chase':
        print('Running chase agent')
        chase_agent = ChaseAgent(aggregator)
        run_agent(environment, chase_agent, agent_locator,
                  transformer, aggregator,
                  epsilon_start=0, epsilon_min=0, train=False)
    else:  # args.agent == 'random'
        print('Running random agent')
        random_agent = Neurosmash.Agent()
        run_agent(environment, random_agent, agent_locator,
                  transformer, aggregator,
                  epsilon_start=0, epsilon_min=0, train=False)

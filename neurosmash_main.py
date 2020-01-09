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
import pandas as pd
import numpy as np

ip = '127.0.0.1'  # IP address that the TCP/IP interface listens to
port = 13000  # port number that the TCP/IP interface listens to
size = 64  # width and height pixels of environment
timescale = 10  # number of timesteps per action
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = 'data/'
models_dir = 'models/'


def run_agent(agent, num_episodes=1000, train=True,
              epsilon_start=0.95, epsilon_decay=0.995, epsilon_min=0.005):
    """Function to run the agent for a given number of episodes.

    Args:
        agent         = [Agent] agent that selects action given current state
        num_episodes  = [int] number of episodes the agent will learn for
        train         = [bool] whether to update the agent's model's parameters
        epsilon_start = [float] starting value for exploration/
                                exploitation parameter
        epsilon_decay = [float] number signifying how fast epsilon decreases
        epsilon_min   = [float] minimal value for the epsilon parameter

    Returns [([int], [bool])]:
        For each episode, the number of steps in the episode and whether
        the red agent was victorious.
    """
    rewards = [0]*num_episodes
    num_steps = [0]*num_episodes
    wins = [0]*num_episodes

    epsilon = epsilon_start  # initialize epsilon
    for i_episode in range(num_episodes):
        # reset environment, create first observation
        end, reward, state_img = environment.reset()
        if hasattr(agent_locator, 'reset'):
            agent_locator.reset()
        old_loc_red, old_loc_blue = agent_locator.get_locations(state_img)
        old_state = aggregator.aggregate(old_loc_red, old_loc_red,
                                         old_loc_blue, old_loc_blue)
        while not end:
            # choose an action
            if random.random() < epsilon:
                # random actions to explore environment (exploration)
                action = random.randrange(environment.num_actions)
            else:
                # strictly follow currently learned behaviour (exploitation)
                action = agent.step(end, reward, old_state)

            # do action, get reward and a new observation for the next round
            end, reward, state_img = environment.step(action)
            new_loc_red, new_loc_blue = agent_locator.get_locations(state_img)
            new_state = aggregator.aggregate(old_loc_red, new_loc_red,
                                             old_loc_blue, new_loc_blue)

            if args.p_plot:  # plot predicted locations of agents
                plot_locations(environment, transformer, state_img, new_loc_red,
                               new_loc_blue, agent_locator.perspective)

            if train:  # adjust agent model's parameters (training step)
                agent.train(end, action, old_state, reward, new_state)

            # bookkeeping
            old_state = new_state
            old_loc_red, old_loc_blue = new_loc_red, new_loc_blue
            rewards[i_episode] += reward
            num_steps[i_episode] += 1

            if num_steps[i_episode] > 2500:  # bug in environment
                break

            # decay epsilon parameter
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # determine whether the agent lost or won
        wins[i_episode] = reward == 10
        print('--- WINNER ---' if wins[i_episode] else '--- LOSER ---')
        print(f'Number of steps: {num_steps[i_episode]}')
        print(f'Reward: {rewards[i_episode]}')


        if args.r_plot:
            plot_rewards(rewards[:i_episode + 1])

    data = {'win': np.asarray(wins), 'num_steps': np.asarray(num_steps), 'rewards': np.asarray(rewards)}
    df = pd.DataFrame(data)
    df.to_csv('current_result.csv')
    return num_steps, wins


if __name__ == '__main__':
    # combinations of options for agent and agent locator
    agents = ['PG', 'PG_run', 'Q', 'Q_run', 'NEAT', 'NEAT_run', 'left', 'chase']
    agent_locators = ['TwoCNNs', 'Vision']

    # process the command options
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str, choices=agents,
                        help=f'agent to train or run')
    parser.add_argument('agent_locator', type=str, choices=agent_locators,
                        help='determines positions of agents')
    parser.add_argument('--plot-positions', dest='p_plot', action='store_true',
                        help='plot the positions of the agents')
    parser.add_argument('--plot-rewards', dest='r_plot', action='store_true',
                        help='plot the smoothed rewards of the episodes')
    parser.add_argument('--save-difficult', dest='save_difficult',
                        action='store_true')
    parser.add_argument('-save-diff-prob', dest='save_diff_prob', type=float)
    args = parser.parse_args()

    # connect to environment via TCP/IP
    environment = None
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
    if args.agent_locator == 'TwoCNNs':
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
        run_agent(pg_agent)
        torch.save(pg_agent.model.state_dict(), f'{models_dir}mlp.pt')
    elif args.agent == 'PG_run':
        print('Running Policy Gradient agent')
        pg_agent = PGAgent(aggregator.num_obs, environment.num_actions,
                           device=device)
        pg_agent.model.load_state_dict(torch.load(f'{models_dir}mlp.pt'))
        pg_agent.model.eval()
        run_agent(pg_agent, train=False, epsilon_start=0, epsilon_min=0)
    elif args.agent == 'Q':
        print('Training Q-learning agent')
        q_agent = QAgent(aggregator.num_obs, environment.num_actions,
                         device=device)
        run_agent(q_agent)
        torch.save(q_agent.policy_network.state_dict(), f'{models_dir}ddqn.pt')
    elif args.agent == 'Q_run':
        print('Running Q-learning agent')
        q_agent = QAgent(aggregator.num_obs, environment.num_actions,
                         device=device)
        q_agent.policy_network.load_state_dict(torch.load(f'{models_dir}ddqn.pt'))
        q_agent.policy_network.eval()
        run_agent(q_agent, train=False, epsilon_start=0, epsilon_min=0)
    elif args.agent == 'NEAT':
        print('Training NEAT agent')
        neat_agent = NeatAgent(f'{models_dir}NEAT_new/', run_agent)
        neat_agent.run()
    elif args.agent == 'NEAT_run':
        print('Running NEAT agent')
        neat_agent = NeatAgent(f'{models_dir}NEAT/', run_agent)
        run_agent(neat_agent, epsilon_start=0, epsilon_min=0)
    elif args.agent == 'chase':
        print('Running chase agent')
        chase_agent = ChaseAgent(aggregator)
        run_agent(chase_agent, epsilon_start=0, epsilon_min=0)
    else:  # args.agent == 'left'
        print('Running left agent')
        left_agent = Neurosmash.Agent()
        run_agent(left_agent, epsilon_start=0, epsilon_min=0)

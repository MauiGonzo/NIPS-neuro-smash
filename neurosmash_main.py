#!/usr/bin/python
#
# neurosmasher trainer
# Niels, Roel, Maurice
#
import argparse
import Neurosmash
import utils

def init_neurosmash(size=768):
    ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
    port       = 13000       # Port number that the TCP/IP interface listens to
    # size       = 768         # Please check the Updates section above for more details
    timescale  = 1           # Please check the Updates section above for more details
    agent = Neurosmash.Agent()
    environment = Neurosmash.Environment()
    return agent, environment

def pgagent():
    # do stuff
    # pgagent, pgenvironment = init_neurosmash()
    # yxredbluetorchtensor = locations(pgagent, pgenvironment)
    # pgmetrics = utils.Aggregate(yxredbluetorchtensor)
    break

def neatagent():
    # do stuff
    break




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agents', nargs='+', type=str, help='agent names to train, choose <PG, NEAT>')
    args = parser.parse_args()

    for ag in args.agents:
        if ag == 'PG':
            print('Processing agent: {}'.format(ag))
            # do stuff -> DO run pgagent(), that is
            # find locations
            # input for agent is output utils(locations)
        elif ag == 'NEAT':
            print('Processing agent: {}'.format(ag))
            # do stuff
        else: # current agent not known
            print('unknonw agent   : {}'.format(ag))

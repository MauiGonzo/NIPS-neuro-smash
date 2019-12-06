#!/usr/bin/python
#
# neurosmasher trainer
# Niels, Roel, Maurice
#
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agents', nargs='+', type=str, help='agent names to train, choose <PG, NEAT>')
    args = parser.parse_args()

    for ag in args.agents:
        if ag == 'PG':
            print('Processing agent: {}'.format(ag))
            # do stuff
        elif ag == 'NEAT':
            print('Processing agent: {}'.format(ag))
            # do stuff
        else: # current agent not known
            print('unknonw agent   : {}'.format(ag))
from __future__ import division


import numpy as np
import gym

import logging
import pickle

from frozen_lake_qlearning import max_action

logging.basicConfig()
logger = logging.getLogger('FrozenLake-v0')
logger.setLevel(logging.INFO)


def play(env, q_table):

    curr_state = env.reset()
    game_over = False

    action, _ = max_action(q_table[curr_state])

    while not game_over:
        env.render()
        new_state, _, game_over, info = env.step(action)
        curr_state = new_state
        action,_ = max_action(q_table[curr_state])        
        
        raw_input('Press ENTER for next step')

if __name__ == '__main__':

    
    params = {}
    params['model_path'] = './trained_models/q_table_v1'

    env = gym.make('FrozenLake-v0')
    q_table = pickle.load(open(params['model_path'],'rb'))

    play(env,q_table)

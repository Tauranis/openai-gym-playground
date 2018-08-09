from __future__ import division

from matplotlib import pyplot as plt

import numpy as np
import itertools
import math

import logging
import pickle

logging.basicConfig()
logger = logging.getLogger('FrozenLake-v0')
logger.setLevel(logging.INFO)

import gym
from gym.envs.registration import register

id2a = {
    0:'L',
    1:'D',
    2:'R',
    3:'U'
}

def max_action(q_s):
    """ Return the tuple (action, q_value) for the action which returns the highest q_value

        Args:

        q_s : list
            list of q_values for a given state

        Returns:

        Tuple (action, q_value)

    """
    a = np.argmax(q_s)
    return (a, q_s[a])


def random_action(action, all_actions, t, epsilon=0.5):
    """ Choose a random action given epsilon-greedy approach
    """
    
    if np.random.random() < (epsilon / t):
        return np.random.choice(all_actions, 1)[0]
    else:
        return action


def exp_decay(lr, decay_rate, step, decay_step):
    """ Apply exponential decay for learning rate

        Args:

        lr (float): learning rate

        decay_rate (float): decay rate

        step (int): step

        decay_step (int): 

        Returns:

        Decayed learning rate

    """

    return lr * (math.pow(decay_rate, step / decay_step))


def extract_Vs(q_table, n_states):
    """ Extract V(s) from Q-Table
    """    
    V = [-1] * n_states
    P = [-1] * n_states
    for s in range(n_states):
        a,V[s] = max_action(q_table[s])

        P[s] = id2a[a]
    return V,P


def train(params, env):

    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    lr_count = np.ones((env.observation_space.n, env.action_space.n))

    all_actions = range(env.action_space.n)
    logger.info("All actions: {}".format(all_actions))

    t = 1

    delta = []
    for e in range(params['total_episodes']):

        if e % 100 == 0:
            logger.info("Episode {}".format(e))
            t += 1e-2
            

        curr_state = env.reset()
        game_over = False

        biggest_change = 0
        step = 0
        while not game_over:

            action, _ = max_action(q_table[curr_state])    
            action = random_action(action, all_actions,
                                   t, params['epsilon'])
            

            new_state, reward, game_over, info = env.step(action)

            lr_count[curr_state][action] += 5e-2
            lr = exp_decay(params['lr'], params['decay_rate'],
                              lr_count[curr_state][action], params['decay_steps'])                        
            
            _, max_qval = max_action(q_table[new_state])

            old_sa = q_table[curr_state][action]

            q_table[curr_state][action] = q_table[curr_state][action] + lr * \
                (reward + params['gamma'] *
                 max_qval - q_table[curr_state][action])

            change = np.abs(old_sa - q_table[curr_state][action])
            biggest_change = max(biggest_change, change)

            curr_state = new_state

            # Avoid infinite loop
            step+=1
            if step >= params['max_steps_per_episode']:
                game_over = True
                
        
        delta.append(biggest_change)        

    return q_table, delta


if __name__ == '__main__':

    params = {}
    params['epsilon'] = 0.5
    params['lr'] = 0.05
    params['decay_rate'] = 0.9
    params['decay_steps'] = 50
    params['gamma'] = 0.9
    params['total_episodes'] = 20000
    params['max_steps_per_episode'] = 100
    params['model_path'] = './trained_models/q_table_v1'

    
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78,  # optimum = .8196
    )

    env = gym.make('FrozenLakeNotSlippery-v0')

    Q_table, delta = train(params, env)
    V_s, policy = extract_Vs(Q_table, env.observation_space.n)

    
    logger.info('V(s)')
    for i in range(0,env.observation_space.n,4):
        logger.info( np.round(V_s[i:(i+4)],3) )

    logger.info('Policy')
    for i in range(0,env.observation_space.n,4):
        logger.info(policy[i:(i+4)])

            


    pickle.dump(Q_table,open(params['model_path'],'wb'))

    plt.plot(delta)
    plt.show()

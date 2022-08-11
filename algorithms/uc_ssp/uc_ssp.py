import numpy as np
from typing import Tuple
import gym
import gym_maze
import gym.spaces as spaces

class Policy:
    def __init__(self, nstates, nactions):
        self.policy = np.zeros((nstates,), dtype=int)
        pass

    def __call__(self, state: np.ndarray) -> int:
        raise NotImplemented




def state_to_idx(state: np.ndarray) -> int:
    '''return state 1D representation'''
    raise NotImplemented


def idx_to_state(idx: int) -> np.ndarray:
    '''return state 2D coordinate representation'''
    raise NotImplemented


def evi_ssp(k: int, j: int) -> Tuple[Policy, int]:
    raise NotImplemented


if __name__ == "__main__":
    env = gym.make("maze-random-10x10-plus-v0")
    '''
    Defining the environment related constants
    '''
    N_STATES = (env.observation_space.high+1)[0]**2
    N_ACTIONS = env.action_space.n

    '''
    UC-SSP algorithm variables
    '''
    R = np.zeros(shape=(N_STATES,N_ACTIONS)) # empirical accumulated reward
    N_k = np.zeros(shape=(N_STATES,N_ACTIONS)) # state-action counter for episode k
    G_kj = 0 # number of attemps in phase 2 of episode k
    K = 200 # num


    ''' RUN ALGORITHM '''
    s = env.reset()
    s_idx = state_to_idx(s)
    t = 1 # total env steps

    for k in range(1,K+1):
        j = 0 # num attemps of phase 2 in episode k
        done = False

        while not done:
            t_kj = t # timestep of last j atemp (unless j=0)
            nu_k = np.zeros_like(N_k) # state-action counter for
            G_kj += j
            pi, H = evi_ssp(k,j)

            while t <= t_kj + H and not done:
                a = pi(s)
                s_, r, done, info = env.step(a)
                R[s,a] += r
                # TODO: need to transfrom rewards to positive costs in range [0,1] with 0 cost for goal state
                s_idx_ = state_to_idx(s_)
                nu_k[s_idx,a] += 1
                t += 1

            if not done: # switch to phase 2 if goal not reached after H steps
                N_k += nu_k
                j += 1
        # if done:
        N_k += nu_k



import gym
import numpy as np

from typing import Tuple
import gym.spaces as spaces
from maze_env import gym_maze


class Policy:
    def __init__(self, n_states, n_actions):
        self.policy = np.zeros((n_states,), dtype=int)
        pass

    def __call__(self, state: np.ndarray) -> int:
        raise NotImplemented


def state_to_idx(state: np.ndarray) -> int:
    """return state 1D representation"""
    raise NotImplemented


def idx_to_state(idx: int) -> np.ndarray:
    """return state 2D coordinate representation"""
    raise NotImplemented


class BellmanCost:
    def __init__(self, n_states, n_actions):
        self.cost = np.zeros((n_states, n_actions))

    def set_cost(self, state, action, cost):
        if self.cost[state][action] == 0:
            self.cost[state][action] = cost
        else:
            # already set this cost - make sure this is a deterministic cost MDP
            assert self.cost[state][action] == cost

    def get_cost(self, state, action, j):
        if j != 0:
            return 1

        return self.cost[state][action]


class UC_SSP:
    def __init__(self, c_min: float, c_max: float, env: gym.Env):
        self.c_min = c_min
        self.c_max = c_max
        self.env = env

        # Defining the environment related constants
        self.n_states = (env.observation_space.high + 1)[0] ** 2
        self.n_actions = env.action_space.n

    def bellman_operator(values: np.ndarray, n_actions: int, j: int) ->np.ndarray:
        new_values = np.zeros_like(values)
        for state in range(len(values)):
            min_cost_action = np.inf
            for action in range(n_actions):
                cost = BellmanCost.get_cost(state, action, j)


    def evi_ssp(k: int, j: int, t_kj: int, G_kj: int, n_states: int) -> Tuple[Policy, int]:
        if j == 0:
            epsilon_kj = c_min / 2*t_kj
            gamma_kj = 1 / np.sqrt(k)
        else:
            epsilon_kj = 1 / 2*t_kj
            gamma_kj = 1 / np.sqrt(G_kj)
        # estimate MDP
        m = 0
        v = np.zeros(n_states)
        next_v = bellman_operator(v)

    def run(self):
        """
        UC-SSP algorithm variables
        """
        DELTA = 0.1  # confidence
        R = np.zeros(shape=(N_STATES, N_ACTIONS), dtype=np.float32)  # empirical accumulated reward
        N_k = np.zeros(shape=(N_STATES, N_ACTIONS), dtype=np.int)  # state-action counter for episode k
        G_kj = 0  # number of attempts in phase 2 of episode k
        K = 100  # num episode
        P_counts = np.zeros(shape=(N_STATES, N_ACTIONS, N_STATES), dtype=np.int)  # empirical (s,a,s') transition counts

        ''' RUN ALGORITHM '''
        s = env.reset()
        s_idx = state_to_idx(s)
        t = 1  # total env steps

        for k in range(1, K + 1):
            j = 0  # num attempts of phase 2 in episode k
            done = False

            while not done:
                t_kj = t  # timestep of last j attempt (unless j=0)
                nu_k = np.zeros_like(N_k)  # state-action counter for
                G_kj += j
                pi, H = evi_ssp(k, j, t_kj, G_kj, N_STATES)

                while t <= t_kj + H and not done:
                    a = pi(s)
                    s_, r, done, info = env.step(a)
                    R[s_idx, a] += r  # empirical accumulated reward
                    # TODO: need to transform rewards to positive costs in range [0,1] with 0 cost for goal state
                    s_idx_ = state_to_idx(s_)
                    nu_k[s_idx, a] += 1
                    P_counts[s_idx, a, s_idx_] += 1  # add transition count
                    s_idx = s_idx_  # t <-- t+1
                    t += 1

                if not done:  # switch to phase 2 if goal not reached after H steps
                    N_k += nu_k
                    j += 1
            # if done:
            N_k += nu_k


if __name__ == "__main__":
    c_min = 0.1
    c_max = 1
    env = gym.make("maze-random-10x10-plus-v0")
    algorithm = UC_SSP(c_min=c_min, c_max=c_max, env=env)
    algorithm.run()


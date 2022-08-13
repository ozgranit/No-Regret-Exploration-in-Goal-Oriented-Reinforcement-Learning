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
    def __init__(self, min_cost: float, max_cost: float, env: gym.Env):
        self.c_min = min_cost
        self.c_max = max_cost
        self.env = env

        # Defining the environment related constants
        self.n_states = (env.observation_space.high + 1)[0] ** 2
        self.n_actions = env.action_space.n
        self.bellman_cost = BellmanCost(self.n_states, self.n_actions)

    def bellman_operator(self, values: np.ndarray, j: int) -> np.ndarray:

        new_values = np.zeros_like(values)
        for state in range(self.n_states):

            min_cost = np.inf
            for action in range(self.n_actions):
                cost = self.bellman_cost.get_cost(state, action, j)

                min_expected_val = np.inf
                for p in confidence_set(state, action):
                    expected_val = sum([p[state][action][y]*values[y] for y in range(self.n_states)])
                    if expected_val < min_expected_val:
                        min_expected_val = expected_val

                cost += min_expected_val
                if cost < min_cost:
                    min_cost = cost

            new_values[state] = min_cost

        return new_values

    def evi_ssp(self, k: int, j: int, t_kj: int, G_kj: int, n_states: int) -> Tuple[Policy, int]:
        if j == 0:
            epsilon_kj = c_min / 2*t_kj
            gamma_kj = 1 / np.sqrt(k)
        else:
            epsilon_kj = 1 / 2*t_kj
            gamma_kj = 1 / np.sqrt(G_kj)
        # estimate MDP
        m = 0
        v = np.zeros(n_states)
        next_v = self.bellman_operator(v)

    def run(self):
        """
        UC-SSP algorithm variables
        """
        DELTA = 0.1  # confidence
        # empirical accumulated cost
        C = np.zeros(shape=(self.n_states, self.n_actions), dtype=np.float32)
        # state-action counter for episode k
        N_k = np.zeros(shape=(self.n_states, self.n_actions), dtype=np.int)
        G_kj = 0  # number of attempts in phase 2 of episode k
        K = 100  # num episode
        # empirical (s,a,s') transition counts
        P_counts = np.zeros(shape=(self.n_states, self.n_actions, self.n_states), dtype=np.int)

        ''' RUN ALGORITHM '''
        s = self.env.reset()
        s_idx = state_to_idx(s)
        t = 1  # total env steps

        for k in range(1, K + 1):
            j = 0  # num attempts of phase 2 in episode k
            done = False

            while not done:
                t_kj = t  # time-step of last j attempt (unless j=0)
                nu_k = np.zeros_like(N_k)  # state-action counter for
                G_kj += j
                pi, H = self.evi_ssp(k, j, t_kj, G_kj, self.n_states)

                while t <= t_kj + H and not done:
                    a = pi(s)
                    s_, r, done, info = self.env.step(a)
                    C[s_idx, a] += r  # empirical accumulated reward
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
    maze_env = gym.make("maze-random-10x10-plus-v0")
    algorithm = UC_SSP(min_cost=c_min, max_cost=c_max, env=maze_env)
    algorithm.run()


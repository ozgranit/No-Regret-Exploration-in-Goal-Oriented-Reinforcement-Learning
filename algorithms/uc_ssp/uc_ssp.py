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

        # UC-SSP algorithm variables
        self.DELTA = 0.1  # confidence
        # empirical accumulated cost
        self.C = np.zeros(shape=(self.n_states, self.n_actions), dtype=np.float32)
        # state-action counter for episode k
        self.N_k = np.zeros(shape=(self.n_states, self.n_actions), dtype=np.int)
        self.G_kj = 0  # number of attempts in phase 2 of episode k
        self.K = 100  # num episode
        # empirical (s,a,s') transition counts
        self.P_counts = np.zeros(shape=(self.n_states, self.n_actions, self.n_states), dtype=np.int)

    def inner_minimization(self, p_sa_hat, confidence_bound_p_sa, rank):
        """
        Find the best local transition p(.|s, a) within the plausible set of transitions as bounded by the confidence bound for some state action pair.
        Arg:
            p_sa_hat : (n_states)-shaped float array. MLE estimate for p(.|s, a).
            confidence_bound_p_sa : scalar. The confidence bound for p(.|s, a) in L1-norm.
            rank : (n_states)-shaped int array. The sorted list of states in Ascending order of value.
        Return:
            (n_states)-shaped float array. The optimistic transition p(.|s, a).
        """

        p_sa = np.array(p_sa_hat)
        p_sa[rank[0]] = min(1, p_sa_hat[rank[0]] + confidence_bound_p_sa / 2)
        rank_dup = list(rank)
        last = rank_dup.pop()
        # Reduce until it is a distribution (equal to one within numerical tolerance)
        while sum(p_sa) > 1 + 1e-9:
            # print('inner', last, p_sa)
            p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
            last = rank_dup.pop()
        # print('p_sa', p_sa)
        return p_sa

    def confidence_set_p(self, state, action, values):
        # Sort the states by their values in Ascending order
        rank = np.argsort(values)
        p_sa_hat = self.P_counts[state][action]  # vector of size S
        # TODO: fill
        confidence_bound_p_sa = None

        p = self.inner_minimization(p_sa_hat, confidence_bound_p_sa, rank)

        return p

    def bellman_operator(self, values: np.ndarray, j: int) -> np.ndarray:
        """as defined in Eq. 4 in the article"""

        new_values = np.zeros_like(values)
        for state in range(self.n_states):

            min_cost = np.inf
            for action in range(self.n_actions):
                cost = self.bellman_cost.get_cost(state, action, j)

                p = self.confidence_set_p(state, action, values)
                expected_val = sum([p[state][action][y]*values[y] for y in range(self.n_states)])
                cost += expected_val
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
        """ RUN ALGORITHM """

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


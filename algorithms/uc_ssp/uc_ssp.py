import gym
import numpy as np

from typing import Tuple


class Policy:
    def __init__(self, n_states, n_actions):
        self.map = np.zeros((n_states,), dtype=int)

    def __call__(self, state_idx: int) -> int:
        return self.map[state_idx]



def state_transform(grid_size: np.ndarray):
    def state_to_idx(state: np.ndarray) -> int:
        """return state 1D representation"""
        idx = state[1] * grid_size[0] + state[0]
        return idx

    def idx_to_state(idx: int) -> np.ndarray:
        """return state 2D coordinate representation"""
        x = idx % grid_size[0]
        y = np.floor(idx / grid_size[0])
        return np.array([x, y])

    return state_to_idx, idx_to_state


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
    def __init__(self, min_cost: float, max_cost: float, confidence: float, env: gym.Env):
        """
        UC-SSP
        """
        # algorithm params
        self.c_min = min_cost
        self.c_max = max_cost
        self.delta = confidence

        # Defining the environment related constants
        self.env = env
        self.n_states = (env.observation_space.high + 1)[0] ** 2
        self.n_actions = env.action_space.n
        self.bellman_cost = BellmanCost(self.n_states, self.n_actions)

        # state-action counter for episode k
        self.N_k = np.zeros(shape=(self.n_states, self.n_actions), dtype=np.int)
        self.K = 100  # num episode
        # empirical (s,a,s') transition counts
        self.P_counts = np.zeros(shape=(self.n_states, self.n_actions, self.n_states), dtype=np.int)
        self.policy = Policy(self.n_states, self.n_actions)

    @staticmethod
    def inner_minimization(p_sa_hat, beta, rank):
        """
        Find the best local transition p(.|s, a) within the plausible set of transitions
        as bounded by the confidence bound for some state action pair.
        Arg:
            p_sa_hat : (n_states)-shaped float array. MLE estimate for p(.|s, a).
            beta : scalar. The confidence bound for p(.|s, a) in L1-norm.
            rank : (n_states)-shaped int array. The sorted list of states in Ascending order of value.
        Return:
            (n_states)-shaped float array. The optimistic transition p(.|s, a).
        """
        p_sa = np.array(p_sa_hat)
        # TODO: not sure if '/ 2' is required
        p_sa[rank[0]] = min(1, p_sa_hat[rank[0]] + beta / 2)
        rank_dup = list(rank)
        last = rank_dup.pop()
        # Reduce until it is a distribution (equal to one within numerical tolerance)
        while sum(p_sa) > 1 + 1e-9:
            p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
            last = rank_dup.pop()

        assert np.linalg.norm((p_sa - p_sa_hat), ord=1) <= beta

        return p_sa

    def confidence_set_p(self, values, p_sa_hat, beta_sa):
        # Sort the states by their values in Ascending order
        rank = np.argsort(values)
        p = self.inner_minimization(p_sa_hat, beta_sa, rank)

        return p

    def bellman_operator(self, values: np.ndarray, j: int, p_hat: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """as defined in Eq. 4 in the article"""

        new_values = np.zeros_like(values)
        for state in range(self.n_states):

            min_cost = np.inf
            for action in range(self.n_actions):
                cost = self.bellman_cost.get_cost(state, action, j)

                # get best p in confidence set
                p_sa_hat = p_hat[state][action]  # vector of size S
                beta_sa = beta[state, action]
                p_sa_tilde = self.confidence_set_p(values, p_sa_hat, beta_sa)

                expected_val = sum([p_sa_tilde[y]*values[y] for y in range(self.n_states)])
                cost += expected_val
                if cost < min_cost:
                    min_cost = cost
                    # In parallel, update the policy:
                    self.policy.map[state] = action

            new_values[state] = min_cost

        return new_values

    def evi_ssp(self, k: int, j: int, t_kj: int, G_kj: int) -> Tuple[Policy, int]:
        if j == 0:
            epsilon_kj = c_min / 2*t_kj
            gamma_kj = 1 / np.sqrt(k)
        else:
            epsilon_kj = 1 / 2*t_kj
            gamma_kj = 1 / np.sqrt(G_kj)
        # estimate MDP
        # compute p_hat estimates:
        N_k_ = np.maximum(1, self.N_k)  # 'N_k_plus'
        beta = np.sqrt(
            (8 * self.n_states * np.log(2 * self.n_actions * N_k_ / self.delta))
            / N_k_)  # bound for norm_1(|p^ - p~|)
        p_hat = self.P_counts / N_k_.reshape((self.n_states, self.n_actions, 1))

        m = 0
        v = np.zeros(self.n_states)
        next_v = self.bellman_operator(v, j, p_hat, beta)
        # TODO: value iteration while loop 'till convergence
        # TODO: compute p_tilde optimistic transition model
        # TODO: compute Q_tilde the transition matrix of pi_tilde
        # TODO: compute H
        H = 20
        return self.policy, H

    def run(self):
        """ RUN ALGORITHM """
        G_kj = 0  # number of attempts in phase 2 of episode k
        t = 1  # total env steps

        s = self.env.reset()
        s_idx = _1D_state(s)


        for k in range(1, self.K + 1):
            j = 0  # num attempts of phase 2 in episode k
            done = False

            while not done:
                t_kj = t  # time-step of last j attempt (unless j=0)
                nu_k = np.zeros_like(self.N_k)  # state-action counter
                G_kj += j
                pi, H = self.evi_ssp(k, j, t_kj, G_kj)

                while t <= t_kj + H and not done:
                    a = pi(s_idx)
                    s_, c, done, info = self.env.step(a)
                    # TODO: make sure that it is also correct for case of reaching the goal state
                    self.bellman_cost.set_cost(s_idx, a, c)
                    s_idx_ = _1D_state(s_)
                    nu_k[s_idx, a] += 1
                    self.P_counts[s_idx, a, s_idx_] += 1  # add transition count
                    s_idx = s_idx_  # t <-- t+1
                    t += 1

                if not done:  # switch to phase 2 if goal not reached after H steps
                    self.N_k += nu_k
                    j += 1
            # if done:
            self.N_k += nu_k

        return pi


if __name__ == "__main__":
    # algorithm related parameters:
    c_min = 0.1
    c_max = 0.1
    DELTA = 0.1

    maze_env = gym.make("maze-random-10x10-plus-v0")
    _1D_state, _2D_state = state_transform(maze_env.observation_space.high + 1)
    goal_state = _1D_state(maze_env.maze_view.goal)

    algorithm = UC_SSP(min_cost=c_min,
                       max_cost=c_max,
                       confidence=DELTA,
                       env=maze_env)
    pi = algorithm.run()
    # TODO: plot pi. compare to optimal pi

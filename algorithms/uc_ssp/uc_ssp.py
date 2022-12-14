import gym
import pickle
import pathlib
import gym_maze
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from util import plot_policy, plot_values, get_env_features
from gym_frozen_lake import stochastic_env, deterministic_env


class Policy:
    def __init__(self, n_states):
        self.map = np.zeros((n_states,), dtype=int)

    def __call__(self, state_idx: int) -> int:
        return int(self.map[state_idx])


class BellmanCost:
    def __init__(self, costs: np.ndarray, goal_state: int):
        self.cost = costs
        self.goal = goal_state

    def set_cost(self, state, action, cost):
        if self.cost[state][action] == 0:
            self.cost[state][action] = cost
        else:
            # already set this cost - make sure this is a deterministic cost MDP
            assert np.abs(self.cost[state][action] - cost) < 1e-8

    def get_cost(self, state: int, action: int, j: int) -> float:
        if CLASSIC_VALUE_ITERATION:
            j = 0
        if j != 0:
            if state != self.goal:
                return 1.0

        return self.cost[state][action]


class UC_SSP:
    def __init__(self, min_cost: float, max_cost: float, confidence: float,
                 all_states: np.ndarray, non_goal_states: np.ndarray, n_actions: int,
                 goal: int, costs: np.ndarray, K: int):
        """
        UC-SSP
        """
        # algorithm params
        self.c_min = min_cost
        self.c_max = max_cost
        self.delta = confidence

        # Defining the environment related constants
        self.n_states = len(all_states)
        self.n_actions = n_actions
        self.costs = costs
        self.all_states = all_states  # S'
        self.non_goal_states = non_goal_states  # S (excluding s_goal)
        self.goal = goal  # goal_state
        self.bellman_cost = BellmanCost(self.costs, self.goal)

        # state-action counter for episode k
        self.N_k = np.zeros(shape=(self.n_states, self.n_actions), dtype=int)
        self.K = K  # num episode
        # empirical (s,a,s') transition counts
        # set uniform probability for any (s,a)
        self.P_counts = np.zeros(shape=(self.n_states, self.n_actions, self.n_states), dtype=int)
        # set p(s_goal|s_goal,a)=1 for any a
        self.P_counts[self.goal, :, self.goal] = 1
        # p~ optimistic model
        self.p_tilde = np.zeros_like(self.P_counts, dtype=np.float32)

        self.policy = Policy(self.n_states)

    def compute_Q(self):
        """compute the transition matrix of pi~ in the optimistic model p~ for any (s,s')"""
        S = len(self.non_goal_states)
        Q = np.zeros(shape=(S, S), dtype=np.float32)
        for s in self.non_goal_states:
            for s_ in self.non_goal_states:
                a = self.policy(s)
                Q[s, s_] = self.p_tilde[s, a, s_]
        return Q

    @staticmethod
    def compute_H(Q, gamma):
        # compute the infinite matrix norm. We assume only positive values in Q
        Q_inf_norm = np.max(np.sum(Q, axis=1))
        n = 1
        while Q_inf_norm > gamma or n == 1:
            Q = np.matmul(Q, Q)
            Q_inf_norm = np.max(np.sum(Q, axis=1))
            n += 1
        return n

    @staticmethod
    def inner_minimization(p_sa_hat, beta, rank) -> np.ndarray:
        """
        Find the best local transition p(.|s, a) within the plausible set of transitions
        as bounded by the beta bound for some state action pair.
        As described in 'Near-optimal Regret Bounds for Reinforcement Learning', T. Jaksch
        Arg:
            p_sa_hat : (n_states)-shaped float array. MLE estimate for p(.|s, a).
            beta : scalar. The confidence bound for p(.|s, a) in L1-norm.
            rank : (n_states)-shaped int array. The sorted list of states in Ascending order of value.
        Return:
            (n_states)-shaped float array. The optimistic transition p(.|s, a).
        """
        if CLASSIC_VALUE_ITERATION:
            return p_sa_hat
        p_sa = np.array(p_sa_hat)
        p_sa[rank[0]] = min(1, p_sa_hat[rank[0]] + beta)
        rank_dup = list(rank)
        last = rank_dup.pop()
        # Reduce until it is a proper distribution (equal to one within numerical tolerance)
        while sum(p_sa) > 1 + 1e-9:
            p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
            last = rank_dup.pop()

        if IMPROVED_UC_SSP:
            # scale up if started with lower value
            if np.sum(p_sa) < 1:
                p_sa *= 1/np.sum(p_sa)

        if abs(np.sum(p_sa) - 1) > 1e-9:
            raise AssertionError(f"probability vector sum should be 1 and not {np.round(np.sum(p_sa), 2)}")
        # if np.linalg.norm((p_sa - p_sa_hat), ord=1) > beta:
        #     raise AssertionError(f"optimistic p is out of set bounds by "
        #                          f"{np.linalg.norm((p_sa - p_sa_hat), ord=1) - beta:.4f}.\n"
        #                          f"beta is {beta:.4f}")

        return p_sa

    def get_beta(self, state, action):
        n_k_plus = max(1, self.N_k[state][action])
        inner_log = (2 * self.n_actions * n_k_plus) / self.delta
        numerator = 8 * self.n_states * np.log(inner_log)

        beta = np.sqrt(numerator / n_k_plus)

        if IMPROVED_UC_SSP:
            beta /= 1000  # if not scaled down, won't work for sure

        return beta

    def get_p_hat(self, state, action):
        n_k_plus = max(1, self.N_k[state][action])
        p_hat = self.P_counts[state][action] / n_k_plus

        return p_hat

    def bellman_operator(self, values: np.ndarray, j: int) -> np.ndarray:
        """as defined in Eq. 4 in the article"""
        new_values = np.zeros_like(values)
        # Sort the values by their values in ascending order
        rank = np.argsort(values)

        for state in self.all_states:

            min_cost = np.inf
            for action in range(self.n_actions):
                cost = self.bellman_cost.get_cost(state, action, j)
                # estimate MDP. compute p_hat estimates and beta:
                p_sa_hat = self.get_p_hat(state, action)  # vector of size S
                beta_sa = self.get_beta(state, action)

                # get best p in confidence set
                if state == self.goal:  # p(.|s_goal,a) = 1_hot
                    p_sa_tilde = p_sa_hat
                else:  # s != s_goal, take inner product minimization
                    p_sa_tilde = self.inner_minimization(p_sa_hat, beta_sa, rank)

                # update optimistic model p~
                self.p_tilde[state, action] = p_sa_tilde

                assert np.abs(np.sum(p_sa_tilde) - 1) < 1e-9 or np.sum(p_sa_tilde) == 0

                expected_val = sum([p_sa_tilde[y] * values[y] for y in self.all_states])
                Qsa = cost + expected_val
                if Qsa < min_cost:
                    min_cost = Qsa
                    # In parallel, update the policy:
                    self.policy.map[state] = action

            new_values[state] = min_cost

            assert new_values[self.goal] == 0

        return new_values

    def evi_ssp(self, k: int, j: int, t_kj: int, G_kj: int) -> Tuple[Policy, int]:
        """
        EVI algorithm as described in
        'Near-optimal Regret Bounds for Reinforcement Learning' T. Jaksch
        with modifications for UC-SSP as described according to the paper
        """
        if j == 0:
            epsilon_kj = c_min / (2 * t_kj)
            gamma_kj = 1 / np.sqrt(k)
        else:
            epsilon_kj = 1 / (2 * t_kj)
            gamma_kj = 1 / np.sqrt(G_kj)

        # TODO: note how we initialize v
        v = np.zeros(self.n_states)
        # v.fill(0.1)
        # v[self.goal] = 0  # exclude the goal state

        next_v = self.bellman_operator(v, j)
        # value iteration step:
        while np.max(np.abs(next_v - v)) > epsilon_kj:
            v = next_v
            next_v = self.bellman_operator(v, j)

        # p~ and pi~ are updated during the value iteration in backend
        Q_tilde = self.compute_Q()
        H = self.compute_H(Q_tilde, gamma_kj)

        if k % EPISODES == 0:
            # save some plots
            save_path = pathlib.Path(__file__).parent.resolve()
            algo_label = get_algo_label()
            plot_values(next_v.reshape(grid_size, grid_size), save_path / f'{algo_label}_values.png')

            state_count = np.array([np.sum(self.P_counts[s]) for s in self.all_states])
            plot_values(state_count.reshape(grid_size, grid_size), save_path / f'{algo_label}_state_count.png')

        return self.policy, H

    def run(self):

        G_k_0 = 0  # number of attempts in phase 2 of episode k
        t = 1  # total env steps
        cost_log = []

        for k in range(1, self.K + 1):
            j = 0  # num attempts of phase 2 in episode k
            episode_cost = 0
            s = env.reset()
            state_idx = to_1D(s)
            if RENDER:
                env.render()
            # the environment returns done=True if s_==goal_state
            while not state_idx == self.goal:
                t_kj = t  # time-step of last j attempt (unless j=0)
                nu_k = np.zeros_like(self.N_k)  # state-action counter
                G_kj = G_k_0 + j
                pi, H = self.evi_ssp(k, j, t_kj, G_kj)

                while t <= t_kj + H and state_idx != self.goal: # work in phase 1
                    action = pi(state_idx)
                    next_state, cost, _, _ = env.step(action)
                    episode_cost += cost
                    # assuming known cost function
                    self.bellman_cost.set_cost(state_idx, action, cost)  # for debugging purpose
                    next_state_idx = to_1D(next_state)
                    nu_k[state_idx, action] += 1
                    self.P_counts[state_idx, action, next_state_idx] += 1  # add transition count
                    state_idx = next_state_idx
                    t += 1

                    if RENDER:
                        env.render()
                # switch to phase 2 if goal not reached after H steps
                if not state_idx == self.goal:
                    self.N_k += nu_k
                    j += 1
            # if s==s_goal:
            self.N_k += nu_k
            G_k_0 = G_kj
            cost_log.append(episode_cost)
            print(f'Episode: {k}, Cost: {episode_cost:.1f}')

        # plot cost
        plt.clf()
        plt.plot(cost_log)
        plt.ylabel('Cumulative cost')
        plt.xlabel('Episode')
        plt.show()

        return pi, cost_log


def save_policy(opt_pi):

    save_path = pathlib.Path(__file__).parent.resolve()
    file_name = 'frozen_lake_opt_policy.pkl'
    # file_name = 'maze_5x5_opt_policy.pkl'

    with open(save_path / file_name, 'wb') as file:
        pickle.dump(opt_pi, file)


def load_policy():

    save_path = pathlib.Path(__file__).parent.resolve()
    file_name = 'frozen_lake_opt_policy.pkl'
    # file_name = 'maze_5x5_opt_policy.pkl'

    with open(save_path / file_name, 'rb') as file:
        opt_pi = pickle.load(file)

    return opt_pi


def run_policy(opt_pi, env, episodes):

    cost_log = []

    for episode in range(episodes):
        episode_cost = 0
        done = False
        s = env.reset()
        state_idx = to_1D(s)

        while not done:
            action = opt_pi(state_idx)
            next_state, cost, done, _ = env.step(action)
            episode_cost += cost

            next_state_idx = to_1D(next_state)
            state_idx = next_state_idx

        cost_log.append(episode_cost)

    return cost_log


def get_algo_label():
    if CLASSIC_VALUE_ITERATION:
        return "Classic Value Iteration"

    elif ORIG_UC_SSP:
        return "UC_SSP"

    elif IMPROVED_UC_SSP:
        return "Modified UC_SSP"


if __name__ == "__main__":

    # decide what type of algorithm
    CLASSIC_VALUE_ITERATION = False
    ORIG_UC_SSP = False
    IMPROVED_UC_SSP = True
    # pick only 1
    assert [CLASSIC_VALUE_ITERATION, ORIG_UC_SSP, IMPROVED_UC_SSP].count(True) == 1

    # algorithm related parameters:
    DELTA = 0.9
    EPISODES = 1000
    RENDER = False

    # uncomment the right option:
    ENV_NAME = 'frozen_lake'
    # ENV_NAME = 'maze'

    if ENV_NAME == 'frozen_lake':
        env = stochastic_env
        # env = deterministic_env
    elif ENV_NAME == 'maze':
        # env = gym.make("maze-random-10x10-plus-v0")
        env = gym.make("maze-v0", enable_render=RENDER)  # 5x5
        # env = gym.make("maze-sample-3x3-v0")
    else:
        raise ValueError("Unknown env name")

    # env features
    c_min, c_max, costs, non_goal_states, states, \
    goal_state, grid_size, n_actions, n_states, \
    to_1D, to_2D = get_env_features(env)
    algorithm = UC_SSP(min_cost=c_min,
                       max_cost=c_max,
                       confidence=DELTA,
                       all_states=states,
                       non_goal_states=non_goal_states,
                       n_actions=n_actions,
                       goal=goal_state,
                       costs=costs,
                       K=EPISODES)

    pi, cost_log = algorithm.run()

    save_path = pathlib.Path(__file__).parent.resolve()
    algo_label = get_algo_label()
    plot_policy(pi.map.reshape(grid_size, grid_size), save_path / f'{algo_label}_policy.png')

    # compare to optimal
    opt_pi = load_policy()
    opt_cost_log = run_policy(opt_pi, env, EPISODES)

    regret = np.sum(cost_log) - np.sum(opt_cost_log)
    plt.clf()
    plt.plot(opt_cost_log, label=f"opt policy, overall regret={regret: .2f}")
    plt.plot(cost_log, label=get_algo_label())
    plt.legend()

    file_name = f'opt_vs_{algo_label}.png'
    plt.savefig(save_path / file_name)

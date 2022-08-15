import gym
import gym_maze
import numpy as np

from typing import Tuple


class Policy:
    def __init__(self, n_states, n_actions):
        self.map = np.zeros((n_states,), dtype=np.int)

    def __call__(self, state_idx: int) -> int:
        return int(self.map[state_idx])


def state_transform(grid_size: np.ndarray):
    def state_to_idx(state: np.ndarray) -> int:
        """return state 1D representation"""
        idx = state[1] * grid_size[0] + state[0]
        return int(idx)

    def idx_to_state(idx: int) -> np.ndarray:
        """return state 2D coordinate representation"""
        x = idx % grid_size[0]
        y = np.floor(idx / grid_size[0])
        return np.array([x, y], dtype=np.int)

    return state_to_idx, idx_to_state


class BellmanCost:
    def __init__(self, costs: np.ndarray, goal_state: int):
        self.cost = costs
        self.goal = goal_state

    def set_cost(self, state, action, cost):

        # TODO: not required as we assume known costs. consider removing
        if self.cost[state][action] == 0:
            self.cost[state][action] = cost
        else:
            # already set this cost - make sure this is a deterministic cost MDP
            assert self.cost[state][action] == cost

    def get_cost(self, state: int, action: int, j: int) -> float:
        if j != 0:
            if state != self.goal:
                return 1.0

        return self.cost[state][action]


class UC_SSP:
    def __init__(self, min_cost: float, max_cost: float, confidence: float,
                 state_space_: np.ndarray, state_space: np.ndarray, n_actions: int,
                 goal: int, costs: np.ndarray, K: int):
        """
        UC-SSP
        """
        # algorithm params
        self.c_min = min_cost
        self.c_max = max_cost
        self.delta = confidence

        # Defining the environment related constants
        self.n_states = len(state_space_)
        self.n_actions = n_actions
        self.costs = costs
        self.states_ = state_space_ # S'
        self.states = state_space # S (excluding s_goal)
        self.goal = goal # goal_state
        self.bellman_cost = BellmanCost(self.costs, self.goal)

        # state-action counter for episode k
        self.N_k = np.zeros(shape=(self.n_states, self.n_actions), dtype=np.int)
        self.K = K  # num episode
        # empirical (s,a,s') transition counts
        self.P_counts = np.zeros(shape=(self.n_states, self.n_actions, self.n_states), dtype=np.int)
        # set p(s_goal|s_goal,a)=1 for any a
        self.P_counts[self.goal,:,self.goal] = 1 # this ensures that we get prob 1 as needed.
        # p~ optimistic model
        self.p_tilde = np.zeros_like(self.P_counts, dtype=np.float32)

        self.policy = Policy(self.n_states, self.n_actions)

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
        p_sa = np.array(p_sa_hat)
        p_sa[rank[0]] = min(1, p_sa_hat[rank[0]] + beta)
        rank_dup = list(rank)
        last = rank_dup.pop()
        # Reduce until it is a proper distribution (equal to one within numerical tolerance)
        while sum(p_sa) > 1 + 1e-9:
            p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
            last = rank_dup.pop()

        assert np.linalg.norm((p_sa - p_sa_hat), ord=1) <= beta

        return p_sa


    def bellman_operator(self, values: np.ndarray, j: int, p_hat: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """as defined in Eq. 4 in the article"""
        new_values = np.zeros_like(values)
        # Sort the values by their values in ascending order
        rank = np.argsort(values)

        for state in range(self.n_states):

            min_cost = np.inf
            for action in range(self.n_actions):
                cost = self.bellman_cost.get_cost(state, action, j)
                # get best p in confidence set
                p_sa_hat = p_hat[state][action]  # vector of size S
                beta_sa = beta[state][action]

                # if state == self.goal: # p(.|s_goal,a) = 1_hot
                #     p_sa_tilde = p_sa_hat
                # else: # s != s_goal, take inner product minimization
                #     p_sa_tilde = self.inner_minimization(p_sa_hat, beta_sa, rank)

                p_sa_tilde = self.inner_minimization(p_sa_hat, beta_sa, rank)

                # update optimistic model p~
                self.p_tilde[state,action] = p_sa_tilde

                assert np.abs(np.sum(p_sa_tilde)-1) < 1e-9

                # TODO: check if we should really run over all states in S'
                expected_val = sum([p_sa_tilde[y]*values[y] for y in self.states_])
                Qsa = cost + expected_val
                if Qsa < min_cost:
                    min_cost = Qsa
                    # In parallel, update the policy:
                    self.policy.map[state] = action

            # todo: ?
            new_values[state] = min_cost

            assert new_values[self.goal] == 0

        return new_values

    def compute_Q(self):
        """compute the transition matrix of pi~ in the optimistic model p~ for any (s,s')"""
        S = len(self.states)
        Q = np.zeros(shape=(S, S), dtype=np.float32)
        for s in self.states:
            for s_ in self.states:
                a = self.policy(s)
                Q[s,s_] = self.p_tilde[s,a,s_]
        return Q

    def compute_H(self, Q, gamma):
        # compute the infinite matrix norm. We assume only positive values in Q
        Q_inf_norm = np.max(np.sum(Q, axis=1))
        n = 1
        while Q_inf_norm > gamma or n == 1:
            Q = np.matmul(Q,Q)
            Q_inf_norm = np.max(np.sum(Q, axis=1))
            n += 1
        return n

    def evi_ssp(self, k: int, j: int, t_kj: int, G_kj: int) -> Tuple[Policy, int]:
        if j == 0:
            epsilon_kj = c_min / (2*t_kj)
            gamma_kj = 1 / np.sqrt(k)
        else:
            epsilon_kj = 1 / (2*t_kj)
            gamma_kj = 1 / np.sqrt(G_kj)
        # estimate MDP. compute p_hat estimates and beta:
        N_k_ = np.maximum(1, self.N_k)  # 'N_k_plus'
        beta = np.sqrt(
            (8 * self.n_states * np.log(2 * self.n_actions * N_k_ / self.delta))
            / N_k_)  # bound for norm_1(|p^ - p~|)
        p_hat = self.P_counts / N_k_.reshape((self.n_states, self.n_actions, 1))

        # assert np.sum(p_hat, axis=2)

        v = np.zeros(self.n_states)
        next_v = self.bellman_operator(v, j, p_hat, beta)
        dv_norm = np.max(next_v-v)
        # value iteration step:
        while dv_norm > epsilon_kj:
            v = next_v
            next_v = self.bellman_operator(v, j, p_hat, beta)
            dv_norm = np.max(next_v - v)
        # p~ and pi~ are updated during the value iteration in backend
        Q_tilde = self.compute_Q()
        H = self.compute_H(Q_tilde, gamma_kj)
        return self.policy, H

    def run(self):
        """ RUN ALGORITHM """
        G_kj = 0  # number of attempts in phase 2 of episode k
        t = 1  # total env steps

        for k in range(1, self.K + 1):
            j = 0  # num attempts of phase 2 in episode k
            s = env.reset()
            s_idx = _1D_state(s)
            if RENDER_MAZE:
                env.render()
            # the environment returns done=True if s_==goal_state
            while not s_idx==self.goal:
                t_kj = t  # time-step of last j attempt (unless j=0)
                nu_k = np.zeros_like(self.N_k)  # state-action counter
                G_kj += j
                pi, H = self.evi_ssp(k, j, t_kj, G_kj)
                
                while t <= t_kj + H and not s_idx==self.goal:
                    a = pi(s_idx)
                    s_, c, _, _ = env.step(a)
                    # assuming known costs
                    # self.bellman_cost.set_cost(s_idx, a, c)
                    s_idx_ = _1D_state(s_)
                    nu_k[s_idx, a] += 1
                    self.P_counts[s_idx, a, s_idx_] += 1  # add transition count
                    s_idx = s_idx_  # t <-- t+1
                    t += 1

                    if RENDER_MAZE:
                        env.render()

                if not s_idx==self.goal:  # switch to phase 2 if goal not reached after H steps
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
    EPISODES = 100

    # maze_env = gym.make("maze-random-10x10-plus-v0")
    env = gym.make("maze-v0")
    RENDER_MAZE = True
    # util function
    _1D_state, _2D_state = state_transform(env.observation_space.high + 1)
    # env features
    GOAL_STATE = _1D_state(env.maze_view.goal)
    N_STATES = (env.observation_space.high + 1)[0] ** 2
    N_ACTIONS = env.action_space.n
    S_ = np.arange(N_STATES)
    S = np.delete(S_, GOAL_STATE)
    # assume we know the costs in advance:
    COSTS = np.zeros(shape=(N_STATES, N_ACTIONS), dtype=np.float32)
    # c(s,a)=const for any (s,a) in SxA
    COSTS.fill(0.1)
    # set c(s_goal,a)=0 for any a
    COSTS[GOAL_STATE,:] = 0

    algorithm = UC_SSP(min_cost=c_min,
                       max_cost=c_max,
                       confidence=DELTA,
                       state_space_=S_,
                       state_space=S,
                       n_actions=N_ACTIONS,
                       goal=GOAL_STATE,
                       costs=COSTS,
                       K=EPISODES)
    pi = algorithm.run()
    # TODO: plot pi. compare to optimal pi

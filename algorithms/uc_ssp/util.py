import numpy as np
import matplotlib.pyplot as plt



def plot_policy(policy):
    vec_map = {0: [0, 1], 1: [0, -1], 2: [1, 0], 3: [-1, 0]}
    x = np.arange(policy.shape[0])
    y = np.arange(policy.shape[1])

    X,Y = np.meshgrid(x,-y)

    policy_ = policy.reshape((-1,))
    u = [vec_map[a][0] for a in policy_]
    v = [vec_map[a][1] for a in policy_]
    u = np.array(u).reshape(policy.shape)
    v = np.array(v).reshape(policy.shape)
    v[-1][-1]=0 # goal state

    fig, ax = plt.subplots(figsize=(4,4))

    ax.quiver(X+0.5, Y-0.5, u, v, scale=10, width=0.02, headwidth=3 ,headlength=5.5)

    ax.xaxis.set_ticks(np.arange(len(x)))
    ax.yaxis.set_ticks(np.arange(-len(y), 1))
    ax.tick_params(axis='both', labelsize=1)

    ax.set_aspect('equal')
    ax.set_xlim(0,len(x))
    ax.set_ylim(-len(y),0)
    ax.grid()
    plt.show()


def state_transform(grid_size=None):
    """ util functions """
    def state_to_idx(state) -> int:
        """return state 1D representation"""
        if grid_size is None:
            return state
        idx = state[1] * grid_size[0] + state[0]
        return int(idx)

    def idx_to_state(idx: int) -> np.ndarray:
        """return state 2D coordinate representation"""
        x = idx % grid_size[0]
        y = np.floor(idx / grid_size[0])
        return np.array([x, y], dtype=int)

    return state_to_idx, idx_to_state


def get_env_features(env):
    if env.metadata['name'] == 'maze':
        c_min = 0.1
        c_max = 0.1
        # env features
        to_1D, to_2D = state_transform(env.observation_space.high + 1)
        goal_state = to_1D(env.maze_view.goal)
        grid_size = (env.observation_space.high + 1)[0]
        n_states = grid_size ** 2
        n_actions = env.action_space.n
        states = np.arange(n_states)
        non_goal_states = np.delete(states, goal_state)
        # assume we know the costs in advance:
        costs = np.zeros(shape=(n_states, n_actions), dtype=np.float32)
        # c(s,a)=const for any (s,a) in SxA
        costs.fill(c_min)
        # set c(s_goal,a)=0 for any a
        costs[goal_state, :] = 0.0
        # util function

    if env.metadata['name'] == 'frozen_lake':
        c_min = 0.1
        c_max = 0.4
        goal_state = env.observation_space.n - 1
        grid_size = int(np.sqrt(env.observation_space.n))
        n_states = env.nS
        n_actions = env.nA
        states = np.arange(n_states)
        non_goal_states = np.delete(states, goal_state)
        # assume we know the costs in advance:
        costs = env.get_costs()
        # dummy function
        to_1D, to_2D = state_transform()

    return c_min, c_max, costs, non_goal_states, states, \
           goal_state, grid_size, n_actions, n_states, \
           to_1D, to_2D


def plot_values(values):
    fig, ax = plt.subplots()
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(values)

    for (i, j), z in np.ndenumerate(values):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

    plt.show()

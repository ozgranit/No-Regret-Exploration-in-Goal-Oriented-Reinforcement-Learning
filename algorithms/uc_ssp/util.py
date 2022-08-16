import numpy as np
import matplotlib.pyplot as plt

vec_map = {0: [0, 1], 1: [0, -1], 2: [1, 0] , 3: [-1, 0]}

def plot_policy(policy):
    x = np.arange(policy.shape[0])
    y = np.arange(policy.shape[1])

    X,Y = np.meshgrid(x,-y)

    policy_ = policy.reshape((-1,))
    u = [vec_map[a][0] for a in policy_]
    v = [vec_map[a][1] for a in policy_]
    u = np.array(u).reshape(policy.shape)
    v = np.array(v).reshape(policy.shape)
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
# No-Regret-Exploration-in-Goal-Oriented-Reinforcement-Learning
Implementation of the algorithm described in “No-regret Exploration in Goal-Oriented Reinforcement Learning”, Tarbouriech et al., 2019.

We perform the experiments on two different grid-world gym style environments:
- [Maze](https://github.com/MattChanTK/gym-maze)
- [Frozen Lake](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/) - a modified version where the game doesn't ends when reaching a hole.

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- Pyglet (OpenGL 3D graphics)
- The provided packages

Install the experiment environments by running the following:

```
cd fozen_lake_env
pip install -e .
cd ..
cd maze_env
pip install -e .
```

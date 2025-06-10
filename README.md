# Blokus Reinforcement Learning

This repository implements a small experiment using reinforcement learning for the board game Blokus. The environment is based on `gymnasium` and supports masked actions.

## Multi-agent training

The new script `agent/multiagent_selfplay.py` demonstrates how two independent PPO agents can be trained against each other using self-play. Each agent alternates as the opponent for the other and the weights are updated in turns.

Run the script as a normal Python module once the dependencies from `pyproject.toml` are installed.

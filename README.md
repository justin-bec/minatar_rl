# Deep Reinforcement Learning on MinAtar

A complete study of reinforcement learning algorithms applied to MinAtar, a suite of lightweight Atari-style environments. Implemented as part of the MVA (Mathématiques, Vision, Apprentissage) program.

## Overview

This project explores and compares classical and deep RL approaches on the MinAtar Breakout environment (10×10 pixel grid), then evaluates generalization to other MinAtar games (Asterix, Freeway, Space Invaders).

## What's Inside

**Part 1 – Environment Analysis**
Detailed study of the state space (~10¹² theoretical states), action space (3 actions), reward structure, episode dynamics, and sources of stochasticity (random ball spawn, sticky actions with probability 0.1).

**Part 2 – Tabular Q-Learning (TD(0))**
Hand-crafted state representation (paddle position, ball position, ball direction) reducing the state space to 4,000 states. Trained with ε-greedy exploration and linear decay of both α and ε over 100,000 episodes. Achieves a mean reward of ~30, outperforming both the random and heuristic baselines.

**Part 3 – Deep Q-Network (DQN)**
Raw pixel input processed by a shallow CNN (one conv layer + two FC layers). Trained with Experience Replay (buffer size 50,000) and a Target Network (synced every 1,000 updates). Achieves a mean reward of ~10 after 300,000 steps.

**Part 4 – Policy Interpretation**
Qualitative analysis of agent behavior (heatmaps, failure case visualization), showing that the tabular agent learns to *predict* ball trajectories rather than simply follow the ball — a more effective strategy.

**Part 5 – Performance Improvements**
- Hyperparameter sensitivity analysis: γ ∈ {0.9, 0.95, 0.99}, various α schedules
- Exploration strategy comparison: ε-greedy vs. Boltzmann (softmax) exploration with temperature decay

**Part 6 – Generalization**
Direct transfer and retraining experiments across Asterix, Freeway, and Space Invaders. The DQN architecture generalizes across games; the tabular agent does not transfer (state features are Breakout-specific).

## Key Results

| Agent | Mean Reward | Training Time |
|---|---|---|
| Random Policy | ~0.7 | — |
| Heuristic Policy | ~4.5 | — |
| Tabular Q-Learning | ~30 | ~3 min |
| DQN | ~15 | ~20 min |

Tabular Q-Learning significantly outperforms DQN on this task — the small, structured state space makes hand-crafted features more effective than learned representations from raw pixels.

## Setup

```bash
pip install numpy gymnasium minatar torch tqdm matplotlib seaborn
```

Then open the notebook:

```bash
jupyter notebook minatar_rl.ipynb
```

A GPU is not required but will speed up the DQN training sections.

## Tech Stack

- Python, PyTorch, NumPy
- MinAtar, Gymnasium
- Matplotlib, Seaborn

# Advanced ML – Assignment 1

GitHub user: **ishan-monie**  |  Repository: **advanced-ml-hw1**

This repository holds all the code for Assignment 1. It is built one question at a time.

## What is here so far
| Folder | Assignment part | Status |
|---|---|---|
| `q1_shuttle/` | Question 2: Value and Policy Iteration (the shuttle problem) | done |
| `q3_policy_gradient/` | Question 3: Policy Gradient Part 1 (CartPole done, Pong not started) | in progress |
| `q4_baselines/` | Policy Gradient Part 2 (baselines) | not started |

## How to run the code

**1. Install what is needed** (Python 3.12 was used):
```
pip install numpy matplotlib
```
(Later parts also need `torch`, `gymnasium` and `ale-py`; these are added when those parts are done.)

**2. Shuttle problem** – open a terminal in the `q1_shuttle` folder:
```
python run_q1.py     # solves the problem 3 ways and saves the 3 plots into figures/
python test_q1.py    # runs the 32 edge-case / correctness checks (prints PASS or FAIL for each)
```
Or open `shuttle_walkthrough.ipynb` in Jupyter / VS Code to see the graphs one by one with a short description of each part.

## What each shuttle file does
- `shuttle.py` – defines the problem and the three methods (enumeration, value iteration, policy iteration).
- `run_q1.py` – runs the methods and makes the plots: `figures/a_enumeration.png`, `b_value_iteration.png`, `c_policy_iteration.png`.
- `shuttle_walkthrough.ipynb` – a notebook that goes through (a), (b) and (c) and shows each graph.
- `test_q1.py` – the checks. Each one prints what it tests and the evidence.

## Policy gradient, CartPole (`q3_policy_gradient/`)
Open a terminal in the `q3_policy_gradient` folder. This part also needs `pip install torch "gymnasium[classic-control]"`.
```
python run_cartpole.py      # trains the base model (1500 games), plays 500 test games, saves the graphs in figures/
python sweep_cartpole.py    # optional: the learning-rate comparison (3 rates x 3 seeds, about a few minutes)
```
- `cartpole_pg.py` - the network, the training loop and the test games.
- `run_cartpole.py` - makes `figures/cartpole_training.png`, `cartpole_histogram.png` and `cartpole_seeds.png`.
- `sweep_cartpole.py` - compares learning rates; results are in `sweep_results.csv`.

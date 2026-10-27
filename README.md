# Advanced ML – Assignment 1

GitHub user: **ishan-monie**  |  Repository: **advanced-ml-hw1**

This repository holds all the code for Assignment 1. It is built one question at a time.

## What is here so far
| Folder | Assignment part | Status |
|---|---|---|
| `q1_shuttle/` | Value and Policy Iteration (the shuttle problem) | done |
| `q2_policy_gradient/` | Policy Gradient Part 1 (CartPole, Pong) | not started |
| `q3_baselines/` | Policy Gradient Part 2 (baselines) | not started |

## How to run the code

**1. Install what is needed** (Python 3.12 was used):
```
pip install numpy matplotlib
```
(Later parts also need `torch`, `gymnasium` and `ale-py`; these are added when those parts are done.)

**2. Shuttle problem** – open a terminal in the `q1_shuttle` folder:
```
python run_q1.py     # solves the problem 3 ways and saves the 3 plots into figures/
python test_q1.py    # runs the 13 edge-case / correctness checks (prints PASS or FAIL for each)
```

## What each shuttle file does
- `shuttle.py` – defines the problem and the three methods (enumeration, value iteration, policy iteration).
- `run_q1.py` – runs the methods and makes the plots: `figures/a_enumeration.png`, `b_value_iteration.png`, `c_policy_iteration.png`.
- `test_q1.py` – the checks. Each one prints what it tests and the evidence.

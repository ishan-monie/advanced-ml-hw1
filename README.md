# Advanced ML – Assignment 1

GitHub user: **ishan-monie**  |  Repository: **advanced-ml-hw1**

This repository holds all the code for Assignment 1. The report (`report/report.pdf`) is built by `report/make_report.py`.

## What is here
| Folder | Assignment part |
|---|---|
| `q1_shuttle/` | Question 2: the shuttle problem (enumeration, value iteration, policy iteration) |
| `q3_policy_gradient/` | Question 3, Part 1: policy gradient on CartPole-v1 and Pong-v5, no baseline |
| `q4_baselines/` | Question 3, Part 2: the same with baselines (CartPole: constant and learned; Pong: learned; plus an extra convolutional Pong network) |
| `report/` | the report |

## How to run the code

**1. Install** (Python 3.12 was used):
```
pip install numpy matplotlib scipy torch "gymnasium[classic-control]" ale-py
```

**2. Shuttle problem** – open a terminal in `q1_shuttle`:
```
python run_q1.py     # solves the problem 3 ways and saves the 3 plots into figures/
python test_q1.py    # the 32 edge-case / correctness checks (prints PASS or FAIL for each)
```
Or open `shuttle_walkthrough.ipynb` to see the graphs one by one with a short description of each part.

**3. CartPole, no baseline** – open a terminal in `q3_policy_gradient`:
```
python run_cartpole.py      # trains the base model (1500 games), plays 500 test games, saves graphs in figures/
python test_cartpole.py     # the 24 checks (about 1 minute; uses the saved model cartpole_model_seed0.pt)
python sweep_cartpole.py    # optional: the learning-rate comparison (results in sweep_results.csv)
```

**4. Pong, no baseline** – in `q3_policy_gradient`. Pong is slow (hours), so the finished runs are included in `results/`:
```
python pong_pg.py --lr 0.001 --seed 0 --batch 4 --max-games 10000 --time-budget 21600 --out results/pong_lr0.001_seed0_10k
python pong_pg.py --lr 0.001 --seed 1 --batch 4 --max-games 10000 --time-budget 21600 --out results/pong_lr0.001_seed1_10k
python pong_pg.py --lr 0.01  --seed 0 --batch 4 --time-budget 14400 --out results/pong_lr0.01_seed0
python plot_pong.py results/pong_lr0.001_seed0_10k results/pong_lr0.001_seed1_10k results/pong_lr0.01_seed0   # the graphs
python test_pong.py         # the 29 checks (about 3 minutes; uses the saved models in results/)
```
The two 0.001 runs were made on one GPU card of the course server (`CUDA_VISIBLE_DEVICES=1`, `--device cuda`); the 0.01 run on a laptop CPU.

**5. Baselines** – in `q4_baselines`:
```
python run_cartpole_baselines.py      # CartPole: no / constant / learned baseline, 10 seeds each (about 20-30 minutes)
python test_cartpole_baselines.py     # the 22 checks
python pong_baseline.py --baseline learned --lr 0.001 --seed 0 --max-games 6500 --out results/pong_learned_seed0
python pong_baseline.py --network conv --baseline learned --lr 0.001 --seed 0 --max-games 6500 --out results/pong_conv_learned_seed0   # extra experiment
python plot_pong_baselines.py         # the Pong baseline graphs and summary
python test_pong_baseline.py          # the checks for the Pong baseline and the convolutional network
```
**Repeating results exactly:** runs use fixed random seeds, but the same seed can give a different run if the number of computer threads differs
(tiny rounding differences grow over thousands of games). The CartPole baseline runs and their tests use 1 thread per run.

## What the files do
- `q1_shuttle/shuttle.py` – the problem and the three methods; `run_q1.py` makes the plots; `test_q1.py` has the checks.
- `q3_policy_gradient/cartpole_pg.py`, `pong_pg.py` – the network, training loop and test games for each game (basic policy gradient).
- `q3_policy_gradient/run_cartpole.py`, `plot_pong.py` – make the graphs. `test_cartpole.py`, `test_pong.py` – the checks.
- `q4_baselines/cartpole_baseline.py`, `pong_baseline.py` – the same methods with a baseline (`--network conv` for the convolutional extra).
- `q4_baselines/run_cartpole_baselines.py`, `plot_pong_baselines.py` – experiments and graphs; `test_*.py` – the checks.
- Every `test_*.py` prints what it tests and the evidence for each check (PASS or FAIL).

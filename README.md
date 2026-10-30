# Advanced ML – Assignment 1

GitHub user: **ishan-monie**  |  Repository: **advanced-ml-hw1**

Code and report for Assignment 1: a shuttle-dispatch problem solved three ways (enumeration, value iteration, policy iteration), and policy gradient (REINFORCE) on CartPole-v1 and Pong-v5, first without and then with a baseline.
The report is `report/report.pdf` (built by `report/make_report.py`). It contains the graphs, the edge cases we considered, the tests, and the discussion.

## What is in this repository
| Folder | Assignment part |
|---|---|
| `q1_shuttle/` | Question 2: the shuttle problem (enumeration, value iteration, policy iteration) |
| `q3_policy_gradient/` | Question 3, Part 1: policy gradient on CartPole-v1 and Pong-v5, no baseline |
| `q4_baselines/` | Question 3, Part 2: the same with baselines (CartPole: constant and learned; Pong: learned; plus an extra convolutional Pong network) |
| `report/` | the PDF report and the script that builds it |

Every folder has `run_*` / `plot_*` scripts that make the graphs (saved in its `figures/` folder) and `test_*.py` files with the edge-case and correctness checks. Each check prints PASS or FAIL and the evidence.

## Requirements
Python 3.12. Install the packages with:
```
pip install -r requirements.txt
```
(`requirements.txt` lists numpy, matplotlib, scipy, torch, gymnasium, ale-py, and, only for the report and notebook, reportlab, pillow and jupyter.)
A GPU is not needed. The long Pong trainings simply take many hours on a CPU, so the finished Pong runs are included in the repository.

## How to run
**1. Shuttle problem** – in `q1_shuttle/`:
```
python run_q1.py     # solves the problem 3 ways and saves the 3 plots into figures/
python test_q1.py    # the 32 edge-case / correctness checks
```
Or open `shuttle_walkthrough.ipynb` to see the graphs one by one with a short description of each part.

**2. CartPole, no baseline** – in `q3_policy_gradient/`:
```
python run_cartpole.py      # trains the base model (1500 games), plays 500 test games, saves graphs in figures/
python test_cartpole.py     # the 24 checks (about 1 minute; uses the saved model cartpole_model_seed0.pt)
python sweep_cartpole.py    # optional: the learning-rate comparison (results in sweep_results.csv)
```

**3. Pong, no baseline** – in `q3_policy_gradient/`. Pong is slow (hours), so the finished runs are included in `results/`:
```
python pong_pg.py --lr 0.001 --seed 0 --batch 4 --max-games 10000 --time-budget 21600 --out results/pong_lr0.001_seed0_10k
python pong_pg.py --lr 0.001 --seed 1 --batch 4 --max-games 10000 --time-budget 21600 --out results/pong_lr0.001_seed1_10k
python pong_pg.py --lr 0.01  --seed 0 --batch 4 --time-budget 14400 --out results/pong_lr0.01_seed0
python plot_pong.py results/pong_lr0.001_seed0_10k results/pong_lr0.001_seed1_10k results/pong_lr0.01_seed0   # the graphs
python test_pong.py         # the 29 checks (about 3 minutes; uses the saved models in results/)
```
The two 0.001 runs were made on one GPU card of the course server (`CUDA_VISIBLE_DEVICES=1`, `--device cuda`); the 0.01 run on a laptop CPU.

**4. Baselines** – in `q4_baselines/`:
```
python run_cartpole_baselines.py      # CartPole: no / constant / learned baseline, 10 seeds each (about 20-30 minutes)
python test_cartpole_baselines.py     # the 22 checks
python pong_baseline.py --baseline learned --lr 0.001 --seed 0 --max-games 6500 --out results/pong_learned_lr0.001_seed0
python pong_baseline.py --baseline learned --lr 0.001 --seed 1 --max-games 6500 --out results/pong_learned_lr0.001_seed1
python pong_baseline.py --network conv --baseline learned --lr 0.001 --seed 0 --max-games 6500 --out results/pong_conv_learned_lr0.001_seed0   # extra experiment
python plot_pong_baselines.py         # the Pong baseline graphs and summary
python test_pong_baseline.py          # the 26 checks (about 7 minutes)
```
(The Pong baseline runs used `--time-budget 32400 --device cuda` on one GPU card, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.)

**5. Rebuild the report** – in `report/`: `python make_report.py` (writes `report.pdf`; reads the saved results and figures).

## Results at a glance
| Experiment | Result (500 test games unless noted) |
|---|---|
| Shuttle | enumeration, value iteration and policy iteration agree (see the report for the policy and graphs) |
| CartPole, no baseline (seed 0, main) | 155.4 ± 12.8 (seeds 0, 1, 2: 155.4, 500, 277.5; maximum is 500) |
| Pong, no baseline, learning rate 0.001 | seed 0 (main): +10.08 ± 4.70, seed 1: +10.48 ± 4.98 (score range −21 to +21) |
| Pong, no baseline, learning rate 0.01 (the assignment's value) | −21.00 ± 0.00: it never learned |
| CartPole baselines, average over 10 seeds | no baseline 154.9, learned baseline 330.9, constant baseline 482.1 |
| Pong, learned baseline | seed 0 (main): **+13.97 ± 3.54**, seed 1: +5.57 ± 5.18; learning starts about 2.5 times sooner than without a baseline |
| Pong, convolutional network + learned baseline (extra, untuned, 1 seed) | +0.57 ± 5.46 |

## Notes on repeating results
- Runs use fixed random seeds, but the same seed can give a different run if the number of computer threads differs (tiny rounding differences grow over thousands of games). The CartPole baseline runs and their tests use 1 thread per run.
- Pong runs are limited by time, not by number of games, so the number of games differs a little between runs. The report compares runs on an equal number of games.
- Trained models saved on a GPU are loaded onto the CPU by the tests (`map_location="cpu"`).

## What the main files do
- `q1_shuttle/shuttle.py` – the problem and the three methods; `run_q1.py` makes the plots; `test_q1.py` has the checks.
- `q3_policy_gradient/cartpole_pg.py`, `pong_pg.py` – the network, training loop and test games for each game (basic policy gradient).
- `q3_policy_gradient/run_cartpole.py`, `plot_pong.py` – make the graphs; `test_cartpole.py`, `test_pong.py` – the checks.
- `q4_baselines/cartpole_baseline.py`, `pong_baseline.py` – the same methods with a baseline (`--network conv` for the convolutional extra).
- `q4_baselines/run_cartpole_baselines.py`, `plot_pong_baselines.py` – experiments and graphs; `test_*.py` – the checks.

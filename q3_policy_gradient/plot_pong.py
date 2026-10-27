"""Question 3(b): makes the required Pong graphs from saved runs (the runs themselves come from pong_pg.py).
Run: python plot_pong.py MAIN_DIR OTHER_SEED_DIR [LR0.01_DIR]
  MAIN_DIR       the main model (learning rate 0.001, seed 0), chosen in advance by the rule in the report
  OTHER_SEED_DIR the same settings with another seed, to show how much luck matters
  LR0.01_DIR     optional: the run with the assignment's learning rate 0.01
Prints the numbers used in the report and saves three figures in figures/."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pong_pg import moving_average

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "figures")
os.makedirs(out, exist_ok=True)
if len(sys.argv) < 3:
    raise SystemExit(__doc__)
main_dir, other_dir = sys.argv[1], sys.argv[2]
lr01_dir = sys.argv[3] if len(sys.argv) > 3 else None


def load(folder):
    return np.load(os.path.join(folder, "train_rewards.npy")), np.load(os.path.join(folder, "eval_rewards.npy"))


runs = {"learning rate 0.001, seed 0 (main)": load(main_dir), "learning rate 0.001, seed 1": load(other_dir)}
if lr01_dir:
    runs["learning rate 0.01, seed 0"] = load(lr01_dir)
for name, (train_r, eval_r) in runs.items():
    print(f"{name}: {len(train_r)} training games, last-100 average {train_r[-100:].mean():.2f} | "
          f"500 test games: mean {eval_r.mean():.2f}, std {eval_r.std():.2f}, range {eval_r.min():.0f} to {eval_r.max():.0f}")

train_r, eval_r = runs["learning rate 0.001, seed 0 (main)"]

# Graph 1: score of every training game, with the 100-game moving average on top
plt.figure(figsize=(7, 4))
plt.plot(train_r, color="0.7", lw=0.6, label="score of each game")
plt.plot(moving_average(train_r), color="C3", lw=2, label="moving average of the last 100 games")
plt.xlabel("training game"); plt.ylabel("score in the game (-21 to +21)")
plt.title("Pong-v5: score during training"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(f"{out}/pong_training.png", dpi=150); plt.close()

# Graph 2: histogram of the 500 test games
plt.figure(figsize=(6, 4))
plt.hist(eval_r, bins=np.arange(-21.5, 22.5, 1) if eval_r.max() - eval_r.min() > 15 else 12, color="C0", edgecolor="w")
plt.axvline(eval_r.mean(), color="C3", ls="--", label=f"mean {eval_r.mean():.2f}, std {eval_r.std():.2f}")
plt.xlabel("score in the game"); plt.ylabel("number of games (out of 500)")
plt.title("Pong-v5: 500 games with the trained policy"); plt.legend(); plt.tight_layout()
plt.savefig(f"{out}/pong_histogram.png", dpi=150); plt.close()

# Graph 3: all runs together (learning rates and seeds)
plt.figure(figsize=(7, 4))
for (name, (r, _)), color in zip(runs.items(), ["C3", "C0", "C2"]):
    plt.plot(moving_average(r), color=color, lw=2 if "main" in name else 1.4, label=name)
plt.xlabel("training game"); plt.ylabel("moving average of the last 100 games")
plt.title("Pong-v5: learning rates and random seeds"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(f"{out}/pong_runs.png", dpi=150); plt.close()

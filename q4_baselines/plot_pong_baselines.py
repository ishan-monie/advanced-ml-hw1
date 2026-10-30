"""Question 3 Part 2, Pong-v5: makes the baseline graphs and a summary from the saved runs.
Run: python plot_pong_baselines.py
Expects (folders with train_rewards.npy and, once finished, eval_rewards.npy):
  ../q3_policy_gradient/results/pong_lr0.001_seed{0,1}_10k      no baseline (Part 1)
  results/pong_learned_lr0.001_seed{0,1}                         learned baseline
  results/pong_conv_learned_lr0.001_seed0                        convolutional network, learned baseline (extra experiment)
  results/pong_conv_none_lr0.001_seed0                           convolutional network, no baseline (extra experiment, optional)
Main model: learned baseline, seed 0 (chosen in advance). Runs are compared at an equal number of games (the first COMMON games)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
part1 = os.path.join(here, "..", "q3_policy_gradient", "results")
res = os.path.join(here, "results")
out = os.path.join(here, "figures")
os.makedirs(out, exist_ok=True)
COMMON = 6500                         # every run is compared on its first 6,500 games


def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)
    return np.array([values[max(0, i - window + 1): i + 1].mean() for i in range(len(values))])


RUNS = {
    "no baseline, seed 0": os.path.join(part1, "pong_lr0.001_seed0_10k"),
    "no baseline, seed 1": os.path.join(part1, "pong_lr0.001_seed1_10k"),
    "learned baseline, seed 0 (main)": os.path.join(res, "pong_learned_lr0.001_seed0"),
    "learned baseline, seed 1": os.path.join(res, "pong_learned_lr0.001_seed1"),
    "convolutional + learned baseline, seed 0": os.path.join(res, "pong_conv_learned_lr0.001_seed0"),
    "convolutional, no baseline, seed 0": os.path.join(res, "pong_conv_none_lr0.001_seed0"),
}
COLORS = {"no baseline": "C7", "learned baseline": "C3", "convolutional + learned": "C2", "convolutional, no": "C0"}

loaded = {}
for name, folder in RUNS.items():
    f = os.path.join(folder, "train_rewards.npy")
    if os.path.exists(f):
        e = os.path.join(folder, "eval_rewards.npy")
        loaded[name] = (np.load(f), np.load(e) if os.path.exists(e) else None)

summary = {}
for name, (t, e) in loaded.items():
    ma = moving_average(t)
    reached = np.where(ma >= -10)[0]
    window = t[:COMMON][-500:]
    summary[name] = {"games": len(t), "last500_common": float(window.mean()) if len(t) >= 500 else None,
                     "first_game_average_reaches_minus10": int(reached[0]) + 1 if len(reached) else None,
                     "finished": e is not None,
                     "mean": float(e.mean()) if e is not None else None, "std": float(e.std()) if e is not None else None,
                     "min": float(e.min()) if e is not None else None, "max": float(e.max()) if e is not None else None}
    print(f"{name:42s}: {len(t):5d} games | average of games {max(1, min(len(t), COMMON) - 499)}-{min(len(t), COMMON)}: "
          f"{'n/a' if summary[name]['last500_common'] is None else format(summary[name]['last500_common'], '.2f')} | average reaches -10 at game {summary[name]['first_game_average_reaches_minus10']} | "
          + (f"500 test games: mean {e.mean():.2f}, std {e.std():.2f}, range {e.min():.0f} to {e.max():.0f}" if e is not None else "not finished (no test result yet)"))
json.dump(summary, open(os.path.join(out, "pong_baseline_summary.json"), "w"), indent=1)

main_name = "learned baseline, seed 0 (main)"
if main_name in loaded:
    train_r, eval_r = loaded[main_name]

    # Graph 1: score of every training game with the 100-game moving average (main model)
    plt.figure(figsize=(7, 4))
    plt.plot(train_r, color="0.7", lw=0.6, label="score of each game")
    plt.plot(moving_average(train_r), color="C3", lw=2, label="moving average of the last 100 games")
    plt.xlabel("training game"); plt.ylabel("score in the game (-21 to +21)")
    plt.title("Pong-v5 with the learned baseline: score during training"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/pong_baseline_training.png", dpi=150); plt.close()

    # Graph 2: histogram of the 500 test games (main model)
    if eval_r is not None:
        plt.figure(figsize=(6, 4))
        plt.hist(eval_r, bins=np.arange(-21.5, 22.5, 1), color="C0", edgecolor="w")
        plt.axvline(eval_r.mean(), color="C3", ls="--", label=f"mean {eval_r.mean():.2f}, std {eval_r.std():.2f}")
        plt.xlabel("score in the game"); plt.ylabel("number of games (out of 500)")
        plt.title("Pong-v5 with the learned baseline: 500 test games"); plt.legend(); plt.tight_layout()
        plt.savefig(f"{out}/pong_baseline_histogram.png", dpi=150); plt.close()

# Graph 3: no baseline against learned baseline (two seeds each), then the convolutional extra runs
def comparison(names, filename, title):
    plt.figure(figsize=(7, 4))
    for name in names:
        if name not in loaded:
            continue
        key = next(k for k in COLORS if name.startswith(k))
        seed1 = name.endswith("seed 1")
        plt.plot(moving_average(loaded[name][0][:COMMON]), color=COLORS[key], lw=1.4 if seed1 else 2.2, ls="--" if seed1 else "-", label=name)
    plt.xlabel("training game"); plt.ylabel("moving average of the last 100 games")
    plt.title(title); plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{out}/{filename}", dpi=150); plt.close()


comparison(["no baseline, seed 0", "no baseline, seed 1", "learned baseline, seed 0 (main)", "learned baseline, seed 1"],
           "pong_baseline_comparison.png", "Pong-v5: no baseline against learned baseline")
comparison(["no baseline, seed 0", "learned baseline, seed 0 (main)", "convolutional + learned baseline, seed 0", "convolutional, no baseline, seed 0"],
           "pong_conv_comparison.png", "Pong-v5: the plain network against the convolutional network")

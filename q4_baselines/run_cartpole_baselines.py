"""Question 4, CartPole-v1: trains no baseline / constant baseline / learned baseline on 10 seeds each and makes the graphs.
Run: python run_cartpole_baselines.py          (about 15-25 minutes; the runs go in parallel)
Everything is identical to Part 1 (network, discount, learning rate 0.01, 1500 games) except the baseline.
Main model: learned baseline, seed 0 (chosen in advance). The other seeds show how much luck matters:
with only 3 seeds the first version of this experiment was too noisy to compare the methods."""
import json
import os
from multiprocessing import Pool
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cartpole_baseline import train, evaluate, moving_average

EPISODES = 1500                      # same as Part 1
SEEDS = list(range(10))
BASELINES = ("none", "constant", "learned")
NAMES = {"none": "no baseline", "constant": "constant baseline", "learned": "learned baseline"}
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "figures")
res = os.path.join(here, "results")


def one_run(job):
    baseline, seed = job
    torch.set_num_threads(1)
    policy, rewards = train(EPISODES, seed=seed, baseline=baseline)
    scores = evaluate(policy, episodes=500)
    np.save(os.path.join(res, f"cartpole_{baseline}_train_seed{seed}.npy"), np.array(rewards))
    np.save(os.path.join(res, f"cartpole_{baseline}_eval_seed{seed}.npy"), scores)
    if baseline == "learned" and seed == 0:
        torch.save(policy.state_dict(), os.path.join(res, "cartpole_learned_model_seed0.pt"))
    return job, np.array(rewards), scores


if __name__ == "__main__":
    os.makedirs(out, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    jobs = [(b, s) for s in SEEDS for b in BASELINES]
    with Pool(8) as pool:
        done = pool.map(one_run, jobs)
    runs = {b: {} for b in BASELINES}
    for (b, s), r, e in done:
        runs[b][s] = (r, e)

    summary = {}
    for b, by_seed in runs.items():
        for s, (r, e) in sorted(by_seed.items()):
            summary[f"{b}_seed{s}"] = {"last100": float(r[-100:].mean()), "mean": float(e.mean()), "std": float(e.std()),
                                       "min": float(e.min()), "max": float(e.max())}
            print(f"{NAMES[b]:18s} seed {s}: last-100 training average {r[-100:].mean():6.1f} | "
                  f"500 test games: mean {e.mean():6.1f}, std {e.std():5.1f}, range {e.min():.0f} to {e.max():.0f}")
    json.dump(summary, open(os.path.join(out, "cartpole_baseline_summary.json"), "w"), indent=1)
    for b in BASELINES:
        means = [runs[b][s][1].mean() for s in SEEDS]
        print(f"{NAMES[b]:18s}: average test score over {len(SEEDS)} seeds {np.mean(means):6.1f} (seed-to-seed std {np.std(means):5.1f}, "
              f"{sum(m >= 499 for m in means)} of {len(SEEDS)} seeds at the 500 limit)")

    train_r, eval_r = runs["learned"][0]

    # Graph 1: reward of every training game with the 100-game moving average (main model)
    plt.figure(figsize=(7, 4))
    plt.plot(train_r, color="0.7", lw=0.8, label="reward of each game")
    plt.plot(moving_average(train_r), color="C3", lw=2, label="moving average of the last 100 games")
    plt.xlabel("training game"); plt.ylabel("total reward in the game (maximum 500)")
    plt.title("CartPole-v1 with the learned baseline: reward during training"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/cartpole_baseline_training.png", dpi=150); plt.close()

    # Graph 2: histogram of the 500 test games (main model)
    plt.figure(figsize=(6, 4))
    plt.hist(eval_r, bins=np.arange(eval_r.min() - 0.5, eval_r.max() + 1.5, 1), color="C0", edgecolor="w")   # one bin per whole score
    plt.axvline(eval_r.mean(), color="C3", ls="--", label=f"mean {eval_r.mean():.1f}, std {eval_r.std():.1f}")
    plt.xlabel("total reward in the game"); plt.ylabel("number of games (out of 500)")
    plt.title("CartPole-v1 with the learned baseline: 500 test games"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/cartpole_baseline_histogram.png", dpi=150); plt.close()

    # Graph 3: the three methods; every seed (faint) and the average over seeds (bold)
    plt.figure(figsize=(7, 4))
    for color, b in zip(["C7", "C0", "C3"], BASELINES):
        curves = np.array([moving_average(runs[b][s][0]) for s in SEEDS])
        for c in curves:
            plt.plot(c, color=color, lw=0.7, alpha=0.3)
        plt.plot(curves.mean(0), color=color, lw=2.4, label=f"{NAMES[b]} (average of {len(SEEDS)} seeds)")
    plt.xlabel("training game"); plt.ylabel("moving average of the last 100 games")
    plt.title("CartPole-v1: no baseline, constant baseline, learned baseline"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/cartpole_baseline_comparison.png", dpi=150); plt.close()

    # Graph 4: the final test score of every seed, per method
    plt.figure(figsize=(6, 4))
    rng = np.random.default_rng(0)
    for i, b in enumerate(BASELINES):
        means = np.array([runs[b][s][1].mean() for s in SEEDS])
        plt.scatter(i + rng.uniform(-0.12, 0.12, len(means)), means, color=["C7", "C0", "C3"][i], s=40, zorder=3)
        plt.hlines(means.mean(), i - 0.25, i + 0.25, color="k", lw=2, zorder=4)
    plt.xticks(range(3), [NAMES[b] for b in BASELINES]); plt.ylabel("mean score on 500 test games")
    plt.title("CartPole-v1: one dot per seed, black line = average"); plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(f"{out}/cartpole_baseline_seeds.png", dpi=150); plt.close()

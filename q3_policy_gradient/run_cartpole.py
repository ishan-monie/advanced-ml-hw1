"""Question 3(a): trains the base model on CartPole-v1 and makes the required graphs.
Run: python run_cartpole.py
Main model: seed 0. Two more seeds are trained only to show how much the results depend on luck."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cartpole_pg import *

EPISODES = 1500
MAIN_SEED, OTHER_SEEDS = 0, [1, 2]
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "figures")
os.makedirs(out, exist_ok=True)

runs = {}
for seed in [MAIN_SEED] + OTHER_SEEDS:
    policy, rewards = train(EPISODES, seed=seed)
    scores = evaluate(policy, episodes=500)
    runs[seed] = (np.array(rewards), scores)
    print(f"seed {seed}: last-100 training average {np.mean(rewards[-100:]):6.1f} | "
          f"500 evaluation games: mean {scores.mean():6.1f}, std {scores.std():5.1f}")
    np.save(os.path.join(here, f"cartpole_train_seed{seed}.npy"), np.array(rewards))
    np.save(os.path.join(here, f"cartpole_eval_seed{seed}.npy"), scores)

# Graph 1: reward of every training game, with the 100-game moving average on top
train_rewards, eval_scores = runs[MAIN_SEED]
plt.figure(figsize=(7, 4))
plt.plot(train_rewards, color="0.7", lw=0.8, label="reward of each game")
plt.plot(moving_average(train_rewards), color="C3", lw=2, label="moving average of the last 100 games")
plt.xlabel("training game"); plt.ylabel("total reward in the game (maximum 500)")
plt.title("CartPole-v1: reward during training"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(f"{out}/cartpole_training.png", dpi=150); plt.close()

# Graph 2: histogram of the 500 evaluation games
plt.figure(figsize=(6, 4))
plt.hist(eval_scores, bins=30, color="C0", edgecolor="w")
plt.axvline(eval_scores.mean(), color="C3", ls="--", label=f"mean {eval_scores.mean():.1f}, std {eval_scores.std():.1f}")
plt.xlabel("total reward in the game"); plt.ylabel("number of games (out of 500)")
plt.title("CartPole-v1: 500 games with the trained policy"); plt.legend(); plt.tight_layout()
plt.savefig(f"{out}/cartpole_histogram.png", dpi=150); plt.close()

# Graph 3: all three seeds (how much does luck matter?)
plt.figure(figsize=(7, 4))
for seed, (r, _) in runs.items():
    plt.plot(moving_average(r), lw=2 if seed == MAIN_SEED else 1.2, label=f"seed {seed}" + (" (main)" if seed == MAIN_SEED else ""))
plt.xlabel("training game"); plt.ylabel("moving average of the last 100 games")
plt.title("CartPole-v1: three different random seeds"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(f"{out}/cartpole_seeds.png", dpi=150); plt.close()

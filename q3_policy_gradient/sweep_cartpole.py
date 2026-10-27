"""Compare a few learning rates on CartPole (3 seeds each) so the choice for the main run is not a guess.
Run: python sweep_cartpole.py       Prints a table and saves sweep_results.csv.
The choice is made on the average reward of the LAST 100 TRAINING games, not on the 500 evaluation games."""
import csv
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

EPISODES = 1500
LEARNING_RATES = [0.01, 0.003, 0.001]
SEEDS = [0, 1, 2]


def one_run(args):
    learning_rate, seed = args
    import torch
    torch.set_num_threads(1)
    from cartpole_pg import train
    _, rewards = train(EPISODES, seed=seed, learning_rate=learning_rate)
    rewards = np.array(rewards)
    return learning_rate, seed, rewards[-100:].mean(), rewards.max(), rewards[:100].mean()


if __name__ == "__main__":
    jobs = [(lr, s) for lr in LEARNING_RATES for s in SEEDS]
    with ProcessPoolExecutor(max_workers=9) as pool:
        results = list(pool.map(one_run, jobs))
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "sweep_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["learning_rate", "seed", "average_of_last_100_training_games", "best_single_game", "average_of_first_100_games"])
        writer.writerows([[lr, s, f"{last:.1f}", f"{best:.0f}", f"{first:.1f}"] for lr, s, last, best, first in results])
    print(f"{EPISODES} training games per run\n")
    print(f"{'learning rate':>13} | {'seed 0':>7} {'seed 1':>7} {'seed 2':>7} | {'average':>7}   (average reward of the last 100 training games)")
    for lr in LEARNING_RATES:
        lasts = [last for (l, s, last, _, _) in results if l == lr]
        print(f"{lr:>13} | " + " ".join(f"{x:7.1f}" for x in lasts) + f" | {np.mean(lasts):7.1f}")

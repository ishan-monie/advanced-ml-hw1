"""Question 4 (policy gradient, part 2), Pong-v5: the same method as Part 1 plus a learned baseline.

How it works, in plain terms (see pong_pg.py in q3_policy_gradient for the rest):
  - Part 1 credited every push with the discounted reward that followed it, until the next point.
  - Here a second small network looks at the same screen-difference the policy sees and guesses how much reward to expect.
    Each push is credited with  (reward that followed) - (that guess).  A push counts as good only if things went better than expected.
  - The guessing network is trained separately, to guess better. Its guesses carry no learning information into the policy.
  Everything else (network, discount, learning rate, 4 games per update, credit restarting at each point) is as in Part 1.

Run (example):  python pong_baseline.py --lr 0.001 --seed 0 --max-games 5000 --out results/pong_learned_seed0
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q3_policy_gradient"))
from pong_pg import (Policy, discounted_returns, make_env, play_game, evaluate, moving_average, GAMMA, HIDDEN)   # noqa: E402,F401

BASELINES = ("none", "learned")


class ValueNet(nn.Module):
    """The learned baseline: 6400 numbers in (the same input as the policy) -> 1 number out (the reward expected from here until the point ends)."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(6400, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def update(policy, optimizer, games, gamma=GAMMA, device="cpu", reset_at_points=True, value_net=None, value_optimizer=None):
    """One learning step from a list of games. Returns True if the network changed, False if skipped
    (nothing to learn from, or numbers that are NaN/infinite). Nothing is changed when skipping."""
    games = [g for g in games if len(g[2]) > 0]
    if not games:
        return False
    x = torch.from_numpy(np.concatenate([g[0] for g in games])).to(device)
    a = torch.from_numpy(np.concatenate([g[1] for g in games])).to(device)
    returns = torch.from_numpy(np.concatenate([discounted_returns(g[2], gamma, reset_at_points) for g in games])).to(device)
    if not torch.isfinite(x).all() or not torch.isfinite(returns).all():      # checked before the networks are used, so nothing can crash or change
        return False
    if value_net is not None:
        with torch.no_grad():                           # the baseline is only a reference here, it gets no policy gradient
            reference = value_net(x)
    else:
        reference = torch.zeros_like(returns)
    loss = -(policy(x).log_prob(a) * (returns - reference)).sum() / len(games)     # per game, as in Part 1
    if not torch.isfinite(loss):
        return False
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if value_net is not None:                           # teach the baseline network to guess the rewards better
        value_loss = ((value_net(x) - returns) ** 2).mean()
        if torch.isfinite(value_loss):
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
    return True


def train(args):
    if args.batch < 1 or args.lr <= 0 or not (0 <= args.gamma < 1) or args.baseline not in BASELINES:
        raise ValueError(f"batch must be at least 1, lr above 0, gamma at least 0 and below 1, and baseline one of {BASELINES}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    env = make_env(args.seed)
    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    value_net = value_optimizer = None
    if args.baseline == "learned":                      # made after the policy, so the policy starts exactly as in Part 1
        value_net = ValueNet().to(device)
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)
    scores, batch, skipped, start = [], [], 0, time.time()

    def save():
        np.save(os.path.join(args.out, "train_rewards.npy"), np.array(scores))
        torch.save(policy.state_dict(), os.path.join(args.out, "model.pt"))

    print(f"device: {device}, baseline: {args.baseline}, learning rate {args.lr}, {args.batch} games per update", flush=True)
    while len(scores) < args.max_games and time.time() - start < args.time_budget:
        game = play_game(env, policy, device)
        scores.append(float(game[2].sum()))
        batch.append(game)
        if len(batch) == args.batch:
            skipped += not update(policy, optimizer, batch, args.gamma, device, True, value_net, value_optimizer)
            batch = []
        if len(scores) % args.log_every == 0:
            print(f"game {len(scores):5d} | last {scores[-1]:5.0f} | average of last 100: {np.mean(scores[-100:]):6.2f} | "
                  f"{time.time() - start:6.0f}s", flush=True)
            save()
    if batch:                                           # games left over when the limit ended mid-batch are still used
        skipped += not update(policy, optimizer, batch, args.gamma, device, True, value_net, value_optimizer)
    save()
    json.dump({**vars(args), "games_trained": len(scores), "device_used": device, "skipped_updates": int(skipped),
               "train_seconds": time.time() - start}, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    eval_scores = evaluate(policy, args.eval_games, device=device)
    np.save(os.path.join(args.out, "eval_rewards.npy"), eval_scores)
    print(f"EVALUATION over {args.eval_games} games: mean {eval_scores.mean():.2f}, std {eval_scores.std():.2f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="learned", choices=BASELINES)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch", type=int, default=4, help="games per learning step")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget", type=float, default=21600, help="training time limit in seconds")
    p.add_argument("--max-games", type=int, default=100000)
    p.add_argument("--eval-games", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto, cpu or cuda (one GPU card)")
    p.add_argument("--out", required=True)
    train(p.parse_args())

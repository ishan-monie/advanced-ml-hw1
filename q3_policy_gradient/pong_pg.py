"""Question 3(b): basic policy gradient (REINFORCE) on Pong (ALE/Pong-v5), with no baseline.

How it works, in plain terms:
  - Each frame is cleaned up with the assignment's preprocess() into an 80x80 grid of 0s and 1s
    (paddles and ball = 1). The network is shown the DIFFERENCE between this frame and the previous one,
    so it can see which way things are moving.
  - The network outputs the chance of pushing the paddle UP (action 2, "RIGHT"); otherwise DOWN (action 3, "LEFT").
  - We play a few whole games, choosing every push at random from those chances.
  - Every push is credited with the discounted reward that followed it, until the next point is scored
    (each point is treated as its own small game). Pushes that led to winning points become more likely.

Run (example):  python pong_pg.py --lr 0.001 --batch 4 --time-budget 14400 --out results/pong_lr0.001_seed0
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

GAMMA = 0.99            # given in the assignment for Pong
HIDDEN = 256
RIGHT, LEFT = 2, 3      # the only two of Pong's 6 actions that we use


def preprocess(image):
    """prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array  (the function given in the assignment)"""
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float32).ravel(), [80, 80])


class Policy(nn.Module):
    """6400 numbers in -> 1 score out; the score becomes the chance of pushing UP.
    Probabilities are built from the raw score (logit), so log(0) can never happen."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(6400, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.distributions.Bernoulli(logits=self.layers(x).squeeze(-1))


def discounted_returns(rewards, gamma=GAMMA, reset_at_points=True):
    """For each step: its reward plus the discounted rewards after it. Pong scores +1 or -1 when a point ends;
    with reset_at_points=True the running total restarts there, so a push is only credited for its own point."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        if reset_at_points and rewards[t] != 0:
            running = 0.0
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def make_env(seed):
    import ale_py
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5")
    env.reset(seed=seed)
    return env


def play_game(env, policy, device="cpu", greedy=False):
    """Play one whole game (until someone reaches 21). Returns (inputs shown to the network, actions taken (1=up), rewards)."""
    obs, _ = env.reset()
    previous = None
    inputs, actions, rewards = [], [], []
    done = False
    while not done:
        frame = preprocess(obs.copy()).ravel()                 # copy: preprocess changes the array it is given
        x = frame - previous if previous is not None else np.zeros_like(frame)   # the first frame has nothing before it
        previous = frame
        with torch.no_grad():
            dist = policy(torch.from_numpy(x).to(device))
            up = (dist.probs > 0.5).float() if greedy else dist.sample()
        obs, reward, terminated, truncated, _ = env.step(RIGHT if int(up) == 1 else LEFT)
        inputs.append(x)
        actions.append(float(up))
        rewards.append(reward)
        done = terminated or truncated
    return np.stack(inputs), np.array(actions, dtype=np.float32), np.array(rewards, dtype=np.float32)


def update(policy, optimizer, games, gamma=GAMMA, device="cpu", reset_at_points=True):
    """One learning step from a list of games. Returns True if the network changed,
    False if skipped (nothing to learn from, or a loss that is NaN/infinite)."""
    games = [g for g in games if len(g[2]) > 0]
    if not games:
        return False
    x = torch.from_numpy(np.concatenate([g[0] for g in games])).to(device)
    a = torch.from_numpy(np.concatenate([g[1] for g in games])).to(device)
    returns = torch.from_numpy(np.concatenate([discounted_returns(g[2], gamma, reset_at_points) for g in games])).to(device)
    loss = -(policy(x).log_prob(a) * returns).sum() / len(games)   # per game, so the learning rate does not depend on the batch size
    if not torch.isfinite(loss):
        return False
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return True


def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)
    return np.array([values[max(0, i - window + 1): i + 1].mean() for i in range(len(values))])


def evaluate(policy, games=500, seed=12345, device="cpu", greedy=False):
    env = make_env(seed)
    torch.manual_seed(seed)
    scores = [float(play_game(env, policy, device, greedy)[2].sum()) for _ in range(games)]
    env.close()
    return np.array(scores)


def train(args):
    if args.batch < 1 or args.lr <= 0 or not (0 <= args.gamma < 1):
        raise ValueError("batch must be at least 1, lr above 0, gamma at least 0 and below 1")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    env = make_env(args.seed)
    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scores, batch, skipped, start = [], [], 0, time.time()

    def save():
        np.save(os.path.join(args.out, "train_rewards.npy"), np.array(scores))
        torch.save(policy.state_dict(), os.path.join(args.out, "model.pt"))

    print(f"device: {device}, learning rate {args.lr}, {args.batch} games per update", flush=True)
    while len(scores) < args.max_games and time.time() - start < args.time_budget:
        game = play_game(env, policy, device)
        scores.append(float(game[2].sum()))
        batch.append(game)
        if len(batch) == args.batch:
            skipped += not update(policy, optimizer, batch, args.gamma, device)
            batch = []
        if len(scores) % args.log_every == 0:
            print(f"game {len(scores):5d} | last {scores[-1]:5.0f} | average of last 100: {np.mean(scores[-100:]):6.2f} | "
                  f"{time.time() - start:6.0f}s", flush=True)
            save()
    if batch:                                    # games left over when the time ran out mid-batch are still used
        skipped += not update(policy, optimizer, batch, args.gamma, device)
    save()
    json.dump({**vars(args), "games_trained": len(scores), "device_used": device, "skipped_updates": int(skipped),
               "train_seconds": time.time() - start}, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    eval_scores = evaluate(policy, args.eval_games, device=device)
    np.save(os.path.join(args.out, "eval_rewards.npy"), eval_scores)
    print(f"EVALUATION over {args.eval_games} games: mean {eval_scores.mean():.2f}, std {eval_scores.std():.2f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch", type=int, default=4, help="games per learning step")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget", type=float, default=14400, help="training time limit in seconds")
    p.add_argument("--max-games", type=int, default=100000)
    p.add_argument("--eval-games", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto, cpu or cuda (one GPU card)")
    p.add_argument("--out", required=True)
    train(p.parse_args())

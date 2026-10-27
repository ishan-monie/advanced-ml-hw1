"""Question 3(b) checks for the Pong base model (edge cases + "does it behave as predicted?").
Run: python test_pong.py        (takes about 2-3 minutes: it plays a few dozen real games)
Each check prints PASS/FAIL and the evidence. The checks are grouped as in the report."""
import json
import os
import tempfile
from argparse import Namespace
from math import sqrt
import numpy as np
import torch
from pong_pg import *

HERE = os.path.dirname(os.path.abspath(__file__))
results = []


def check(name, ok, evidence=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {evidence}" if evidence else ""))


def raises(error, fn):
    try:
        fn()
    except error:
        return True
    except Exception:
        return False
    return False


def weights(policy):
    return torch.cat([p.detach().flatten() for p in policy.parameters()]).clone()


def short_run(seed, games, batch=2, lr=0.001, gamma=GAMMA):
    """A tiny training run in a temporary folder. Returns (scores, path of the folder's contents loaded)."""
    out = tempfile.mkdtemp()
    args = Namespace(lr=lr, batch=batch, gamma=gamma, seed=seed, time_budget=1e9, max_games=games, eval_games=1,
                     log_every=1000, threads=2, device="cpu", out=out)
    train(args)
    return np.load(os.path.join(out, "train_rewards.npy")), out


def welch_t(a, b):
    return (a.mean() - b.mean()) / sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


# =====================================================================================
print("=== A. SCORING AND GAME RULES ===")
r = np.array([0, 0, 1, 0, -1], dtype=np.float32)
reset = discounted_returns(r, 0.9, reset_at_points=True)
leak = discounted_returns(r, 0.9, reset_at_points=False)
expected_reset = [0.81, 0.9, 1.0, -0.9, -1.0]
check("A1. credits worked by hand: a point won after 3 pushes and one lost after 2 more give 0.81, 0.9, 1, -0.9, -1; "
      "the lost point does not spill back onto pushes from the earlier point",
      np.allclose(reset, expected_reset), f"{reset.round(3).tolist()}")
check("A1b. without the reset the earlier pushes would be wrongly blamed for the later loss (so the reset really does something)",
      not np.allclose(leak, expected_reset) and leak[0] < reset[0], f"{leak.round(3).tolist()}")

check("A2. bad settings are rejected with a message (batch 0, learning rate 0, discount 1.0 or -0.1)",
      raises(ValueError, lambda: short_run(0, 1, batch=0)) and raises(ValueError, lambda: short_run(0, 1, lr=0))
      and raises(ValueError, lambda: short_run(0, 1, gamma=1.0)) and raises(ValueError, lambda: short_run(0, 1, gamma=-0.1)))

env = make_env(0)
meanings = env.unwrapped.get_action_meanings()
check("A3. the right game and buttons: ALE/Pong-v5 with 6 actions, where button 2 is up (RIGHT) and button 3 is down (LEFT)",
      env.spec.id == "ALE/Pong-v5" and env.action_space.n == 6 and meanings[RIGHT] == "RIGHT" and meanings[LEFT] == "LEFT",
      f"buttons 2 and 3 are {meanings[2]} and {meanings[3]}")

torch.manual_seed(0)
untrained = Policy()
inputs, actions, rewards = play_game(env, untrained)
check("A4. a full game: ends on its own, rewards are only -1, 0 or +1, and the final score is between -21 and +21",
      set(np.unique(rewards)) <= {-1.0, 0.0, 1.0} and -21 <= rewards.sum() <= 21,
      f"{len(rewards)} steps, score {rewards.sum():.0f}, {int((rewards != 0).sum())} points played")
check("A5. a game ends when one side reaches 21 points",
      max((rewards == 1).sum(), (rewards == -1).sum()) == 21, f"points won {(rewards == 1).sum()}, lost {(rewards == -1).sum()}")

# =====================================================================================
print("\n=== B. WHAT THE NETWORK SEES ===")
check("B1. the first picture of a game has nothing before it, so the network is shown all zeros",
      not inputs[0].any(), f"largest value in first input {np.abs(inputs[0]).max():.0f}")
check("B2. later inputs show motion: only -1, 0 and +1 appear, and both the +1 (appeared) and -1 (disappeared) kinds occur",
      set(np.unique(inputs)) <= {-1.0, 0.0, 1.0} and (inputs == 1).any() and (inputs == -1).any() and inputs.shape == (len(rewards), 6400),
      f"input size {inputs.shape[1]} numbers; {int((inputs[1:] != 0).sum(1).mean())} changed pixels per step on average")

frame = np.zeros((210, 160, 3), dtype=np.uint8)
frame[101, 80] = 236                                       # something that is not background
frame[61, 40] = 144                                        # background colour 1
frame[71, 40] = 109                                        # background colour 2
original = frame.copy()
cleaned = preprocess(frame)
check("B3. cleaning a picture: result is 80x80 with only 0s and 1s, and both background colours are erased",
      cleaned.shape == (80, 80) and set(np.unique(cleaned)) <= {0.0, 1.0} and cleaned.sum() == 1, f"{int(cleaned.sum())} object pixel kept")

grabbed, _ = env.reset()
before_play = grabbed.copy()
preprocess(grabbed.copy())
check("B4. the game's own picture is not altered by the cleaning (the assignment's function changes what it is given, so we pass a copy)",
      np.array_equal(grabbed, before_play) and not np.array_equal(original, frame))

# =====================================================================================
print("\n=== C. THE NETWORK AND ITS PROBABILITIES ===")
torch.manual_seed(0)
fresh = Policy()
sample_inputs = torch.from_numpy(inputs[1:400:2])
p_up = fresh(sample_inputs).probs.mean().item()
check("C1. before any training, up and down are about equally likely", 0.4 < p_up < 0.6, f"average chance of pushing up {p_up:.3f}")

wild = Policy()
with torch.no_grad():
    for p in wild.parameters():
        p.mul_(1e4)
d = wild(torch.randn(50, 6400) * 1e6)
check("C2. an extreme network (weights x 10,000) and extreme inputs (millions) still give finite probabilities and log-probabilities",
      torch.isfinite(d.probs).all().item() and torch.isfinite(d.log_prob(d.sample())).all().item())

check("C3. one game fits easily in memory (about 1,000 steps, a few dozen MB), so four per learning step is safe",
      inputs.nbytes < 100e6, f"{inputs.nbytes / 1e6:.0f} MB for this game")

# =====================================================================================
print("\n=== D. THE TRAINING LOOP ===")
policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=0.01)
before = weights(policy)
bad = (inputs[:5], actions[:5], np.array([1, np.nan, 0, 0, 1], dtype=np.float32))
check("D1. a game with a NaN reward makes the update skip, and the network is left untouched",
      update(policy, opt, [bad]) is False and torch.equal(before, weights(policy)))
check("D2. an empty list of games, or only empty games, is skipped instead of crashing",
      update(policy, opt, []) is False
      and update(policy, opt, [(np.zeros((0, 6400), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))]) is False)

one_game_scores, one_game_out = short_run(seed=5, games=1, batch=4)
start_weights = (torch.manual_seed(5), weights(Policy()))[1]
final = torch.load(os.path.join(one_game_out, "model.pt"))
final_weights = torch.cat([v.flatten() for v in final.values()])
check("D3. when the time or game limit ends in the middle of a batch, the games played so far are still learned from",
      len(one_game_scores) == 1 and not torch.equal(start_weights, final_weights), "1 game played with a batch size of 4; the network still changed")

a1, _ = short_run(seed=3, games=2)
a2, _ = short_run(seed=3, games=2)
b1, _ = short_run(seed=4, games=2)
check("D4. the same seed gives an identical result; a different seed gives a different one", np.array_equal(a1, a2) and not np.array_equal(a1, b1),
      f"seed 3 twice: {a1.tolist()} and {a2.tolist()}; seed 4: {b1.tolist()}")

before = weights(policy)
evaluate(policy, games=1)
check("D5. test games never change the network", torch.equal(before, weights(policy)))

e1, e2 = evaluate(untrained, games=2, seed=99), evaluate(untrained, games=2, seed=99)
check("D6. test games with the same seed give identical scores (so the 500-game result can be re-created)", np.array_equal(e1, e2), f"{e1.tolist()}")

# Made-up problem with a known answer: one fixed input, pushing up always pays 1, down pays 0.
# update() with plain SGD (learning rate 1) moves every weight by minus its gradient, which exposes the learning signal.
STATE = torch.randn(6400) * 0.1


def learning_signal(reward_fn):
    """Direction match and relative size error between update()'s signal and the exact gradient of P(up)."""
    torch.manual_seed(0)
    toy = Policy()
    params = list(toy.parameters())
    exact = torch.autograd.grad(toy(STATE).probs, params)
    acts = toy(STATE.repeat(2000, 1)).sample()
    one_step_games = [(STATE.numpy()[None, :], np.array([float(a)], dtype=np.float32), np.array([reward_fn(float(a))], dtype=np.float32))
                      for a in acts]
    sgd = torch.optim.SGD(toy.parameters(), lr=1.0)
    before = weights(toy)
    update(toy, sgd, one_step_games)
    moved = weights(toy) - before
    x = torch.cat([g.flatten() for g in exact])
    return torch.nn.functional.cosine_similarity(moved, x, dim=0).item(), ((moved - x).norm() / x.norm()).item()


cosine, rel_error = learning_signal(lambda a: a)
check("D7. made-up problem with a known answer: our learning signal matches the exact gradient of the expected reward",
      rel_error < 0.15 and cosine > 0.98, f"direction match {cosine:.4f}, relative size error {rel_error:.3f}")
flipped = learning_signal(lambda a: -a)
ignores = learning_signal(lambda a: 1.0)
check("D7b. the D7 check can really fail: a learner with the sign flipped (learns the wrong way) or one that ignores what was earned is caught",
      not (flipped[1] < 0.15 and flipped[0] > 0.98) and not (ignores[1] < 0.15 and ignores[0] > 0.98),
      f"sign flipped: direction match {flipped[0]:+.2f}; ignoring the reward: {ignores[0]:+.2f}")

torch.manual_seed(0)
toy = Policy()
toy_opt = torch.optim.Adam(toy.parameters(), lr=0.01)
p_before = toy(STATE).probs.item()
for _ in range(150):
    acts = toy(STATE.repeat(4, 1)).sample()
    update(toy, toy_opt, [(STATE.numpy()[None, :], np.array([float(a)], dtype=np.float32), np.array([float(a)], dtype=np.float32)) for a in acts])
p_after = toy(STATE).probs.item()
check("D8. in the made-up problem, training makes the better action much more likely", p_after > 0.8 and p_after > p_before,
      f"chance of pushing up {p_before:.2f} before, {p_after:.2f} after")

# =====================================================================================
print("\n=== E. DOES IT BEHAVE AS PREDICTED? (plays real games with the saved models) ===")
main_dir = os.path.join(HERE, "results", "pong_lr0.001_seed0")
if os.path.exists(os.path.join(main_dir, "eval_rewards.npy")):
    saved_eval = np.load(os.path.join(main_dir, "eval_rewards.npy"))
    cfg = json.load(open(os.path.join(main_dir, "config.json")))
    check("E1. the saved test result is exactly 500 games, every score between -21 and +21",
          len(saved_eval) == 500 and saved_eval.min() >= -21 and saved_eval.max() <= 21,
          f"mean {saved_eval.mean():.2f}, std {saved_eval.std():.2f}, range {saved_eval.min():.0f} to {saved_eval.max():.0f}")
    check("E2. no learning step was skipped during the main run", cfg["skipped_updates"] == 0, f"{cfg['games_trained']} games trained")

    main_policy = Policy()
    main_policy.load_state_dict(torch.load(os.path.join(main_dir, "model.pt")))
    trained = evaluate(main_policy, games=30, seed=777)
    torch.manual_seed(7)
    blank = evaluate(Policy(), games=30, seed=777)
    t = welch_t(trained, blank)
    check("E3. the trained network is far better than an untrained one over 30 fresh games each (statistical test, t > 10)", t > 10,
          f"trained {trained.mean():.1f} vs untrained {blank.mean():.1f}, t = {t:.0f}")
    greedy = evaluate(main_policy, games=30, seed=777, greedy=True)
    check("E4. taking away the randomness (always the most likely push) does not do much worse than random choice",
          greedy.mean() >= trained.mean() - 3, f"most-likely push {greedy.mean():.1f} vs random choice {trained.mean():.1f}")
else:
    print("(saved main model not found, so checks E1-E4 were skipped)")

ma = moving_average(np.arange(1, 5), window=2)
games = np.arange(1, 301, dtype=float)
ma100 = moving_average(games, 100)
check("E5. moving average by hand: [1,2,3,4] with window 2 gives 1, 1.5, 2.5, 3.5; with window 100 the first value is the first game, "
      "game 100 is the mean of games 1-100, and game 101 drops game 1",
      np.allclose(ma, [1, 1.5, 2.5, 3.5]) and ma100[0] == 1 and np.isclose(ma100[99], games[:100].mean()) and np.isclose(ma100[100], games[1:101].mean()))

print(f"\n{sum(results)} of {len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)

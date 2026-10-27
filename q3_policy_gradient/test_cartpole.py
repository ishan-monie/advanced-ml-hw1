"""Question 3(a) checks for the CartPole base model (edge cases + "does it behave as predicted?").
Run: python test_cartpole.py        (takes about a minute: it uses the saved main model, cartpole_model_seed0.pt)
Each check prints PASS/FAIL and the evidence. The checks are grouped as in the report."""
import os
from math import sqrt
import numpy as np
import torch
import gymnasium as gym
from cartpole_pg import *

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


def welch_t(a, b):
    return (a.mean() - b.mean()) / sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


STATE = torch.tensor([0.1, -0.2, 0.05, 0.3])        # one fixed example state, used in several checks

# =====================================================================================
print("=== A. RETURNS, GAME RULES ===")
g = discounted_returns(np.array([1, 1, 1], dtype=np.float32), 0.5)
leak = discounted_returns(np.array([5, 0, 0], dtype=np.float32), 0.9)
check("A1. credits worked by hand: rewards 1, 1, 1 with discount 0.5 give 1.75, 1.5, 1.0; and a reward never credits earlier pushes",
      np.allclose(g, [1.75, 1.5, 1.0]) and np.allclose(leak, [5, 0, 0]), f"{g.tolist()} and {leak.tolist()}")

policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=0.01)
before = weights(policy)
dist = policy(STATE)
action = dist.sample()
applied = update(opt, [dist.log_prob(action)], np.array([1.0], dtype=np.float32))
check("A2. a one-step game (the pole falls straight away) still gives a valid update",
      applied and not torch.equal(before, weights(policy)) and torch.isfinite(weights(policy)).all())

check("A3. discount 0 gives credit = reward only; 0.99 works; 1.0 and -0.1 are rejected",
      np.allclose(discounted_returns(np.array([1, 2, 3], dtype=np.float32), 0.0), [1, 2, 3])
      and np.isfinite(discounted_returns(np.ones(500, dtype=np.float32), 0.99)).all()
      and raises(ValueError, lambda: train(5, gamma=1.0)) and raises(ValueError, lambda: train(5, gamma=-0.1)))

longest = discounted_returns(np.ones(500, dtype=np.float32), GAMMA)
check("A4. in the longest game (500 steps) no credit can exceed 1/(1-0.95) = 20",
      longest.max() <= 1 / (1 - GAMMA) + 1e-3, f"largest credit {longest.max():.3f}")

env = gym.make("CartPole-v1")
check("A5. the right game: CartPole-v1 with a 500-step limit (not v0, which stops at 200)",
      env.spec.id == "CartPole-v1" and env.spec.max_episode_steps == 500, f"{env.spec.id}, limit {env.spec.max_episode_steps}")

log_probs, rewards = play_episode(env, Policy(), keep_gradients=False)
check("A6. every step gives a reward of 1, so a game's score equals its length",
      np.all(rewards == 1) and rewards.sum() == len(rewards) == len(log_probs), f"game of {len(rewards)} steps")

# =====================================================================================
print("\n=== B. THE NETWORK AND ITS PROBABILITIES ===")
torch.manual_seed(0)
fresh = Policy()
states = []
state, _ = env.reset(seed=0)
while len(states) < 300:
    states.append(state)
    state, _, terminated, truncated, _ = env.step(env.action_space.sample())
    if terminated or truncated:
        state, _ = env.reset()
p_left = fresh(torch.as_tensor(np.array(states), dtype=torch.float32)).probs[:, 0].mean().item()
check("B1. before any training, both actions are about equally likely", 0.4 < p_left < 0.6, f"average chance of pushing left {p_left:.3f}")

many = torch.randn(1000, 4) * 3
probs = fresh(many).probs
check("B2. the two probabilities add up to 100% for every state, and none is NaN", torch.allclose(probs.sum(1), torch.ones(1000), atol=1e-5)
      and torch.isfinite(probs).all(), f"largest error {(probs.sum(1) - 1).abs().max():.1e}")

extreme = Policy()
with torch.no_grad():
    for p in extreme.parameters():
        p.mul_(1e4)
d = extreme(torch.randn(50, 4) * 1e6)
check("B3. an extreme network (weights x 10,000) and extreme inputs (millions) still give finite log-probabilities",
      torch.isfinite(d.log_prob(d.sample())).all().item() and torch.isfinite(d.probs).all().item())

# =====================================================================================
print("\n=== C. THE TRAINING LOOP ===")
policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=0.01)
before = weights(policy)
dist = policy(STATE)
lps = [dist.log_prob(dist.sample()) for _ in range(3)]
check("C1. a NaN reward makes the update skip, and the network is left untouched",
      update(opt, lps, np.array([1.0, np.nan, 1.0], dtype=np.float32)) is False and torch.equal(before, weights(policy)))
check("C2. an empty game is skipped instead of crashing", update(opt, [], np.array([], dtype=np.float32)) is False)

check("C3. bad settings are rejected with a message (0 games, learning rate 0 or negative)",
      raises(ValueError, lambda: train(0)) and raises(ValueError, lambda: train(5, learning_rate=0))
      and raises(ValueError, lambda: train(5, learning_rate=-1)))

same_a, same_b = np.array(train(30, seed=3)[1]), np.array(train(30, seed=3)[1])
other = np.array(train(30, seed=4)[1])
check("C4. the same seed gives an identical training curve; a different seed gives a different one",
      np.array_equal(same_a, same_b) and not np.array_equal(same_a, other), f"{len(same_a)} games compared")

before = weights(policy)
evaluate(policy, episodes=20)
lp_test, _ = play_episode(env, policy, keep_gradients=False)
check("C5. test games never change the network, and keep no learning information",
      torch.equal(before, weights(policy)) and not lp_test[0].requires_grad)

# Made-up problem with a known answer: one fixed state, action 1 always pays 1, action 0 pays 0.
# The true gradient of the expected reward is the gradient of P(action 1). Our loss, averaged over many
# one-step games (discount 0 so each game is independent), must point the same way and have the same size.
def gradient_match(loss_fn):
    """How well does the learning signal from loss_fn match the exact gradient? Returns (direction match, relative size error)."""
    torch.manual_seed(0)
    toy = Policy()
    params = list(toy.parameters())
    exact = torch.autograd.grad(toy(STATE).probs[1], params)
    N = 20000
    batch = toy(STATE.repeat(N, 1))
    acts = batch.sample()
    toy_rewards = (acts == 1).float().numpy()
    loss = loss_fn(list(batch.log_prob(acts)), toy_rewards, 0.0)
    estimate = [-g / N for g in torch.autograd.grad(loss, params)]
    e, x = torch.cat([g.flatten() for g in estimate]), torch.cat([g.flatten() for g in exact])
    return torch.nn.functional.cosine_similarity(e, x, dim=0).item(), ((e - x).norm() / x.norm()).item()


cosine, rel_error = gradient_match(policy_loss)
check("C6. made-up problem with a known answer: our learning signal matches the exact gradient of the expected reward",
      rel_error < 0.1 and cosine > 0.99, f"direction match {cosine:.4f}, relative size error {rel_error:.3f}")

flipped = gradient_match(lambda lp, r, g: -policy_loss(lp, r, g))
ignores = gradient_match(lambda lp, r, g: -torch.stack(lp).sum())
check("C6b. the C6 check can really fail: a learner with the sign flipped (learns the wrong way) or one that ignores the reward is caught",
      not (flipped[1] < 0.1 and flipped[0] > 0.99) and not (ignores[1] < 0.1 and ignores[0] > 0.99),
      f"sign flipped: direction match {flipped[0]:+.2f}; ignoring the reward: {ignores[0]:+.2f}")

torch.manual_seed(0)
toy = Policy()
toy_opt = torch.optim.Adam(toy.parameters(), lr=0.05)
p_before = toy(STATE).probs[1].item()
for _ in range(300):
    dist = toy(STATE)
    a = dist.sample()
    update(toy_opt, [dist.log_prob(a)], np.array([float(a == 1)], dtype=np.float32))
p_after = toy(STATE).probs[1].item()
check("C7. in the made-up problem, training makes the better action much more likely", p_after > 0.8 and p_after > p_before,
      f"chance of the better action {p_before:.2f} before, {p_after:.2f} after")

# =====================================================================================
print("\n=== D. DOES IT BEHAVE AS PREDICTED? (uses the saved main model) ===")
MODEL_FILE = os.path.join(HERE, "cartpole_model_seed0.pt")
saved_train = os.path.join(HERE, "cartpole_train_seed0.npy")
if os.path.exists(MODEL_FILE):                       # normal case: use the saved main model, no retraining
    main_policy = Policy()
    main_policy.load_state_dict(torch.load(MODEL_FILE))
    main_rewards = np.load(saved_train)
else:                                                # first run only: train it once (same seed, so identical) and save it
    main_policy, main_rewards = train(1500, seed=0)
    main_rewards = np.array(main_rewards)
    torch.save(main_policy.state_dict(), MODEL_FILE)
main_scores = evaluate(main_policy, episodes=500)

torch.manual_seed(7)
untrained_scores = evaluate(Policy(), episodes=500)
t = welch_t(main_scores, untrained_scores)
check("D1. the trained network is far better than an untrained one (statistical test, t > 10)", t > 10,
      f"trained {main_scores.mean():.1f} vs untrained {untrained_scores.mean():.1f}, t = {t:.0f}")

_, tiny_rewards = train(300, seed=0, learning_rate=1e-6)
check("D2. with a tiny learning rate (1e-6) it hardly learns and stays near random play (about 22)",
      np.mean(tiny_rewards[-100:]) < 40, f"average of the last 100 games {np.mean(tiny_rewards[-100:]):.1f}")

greedy_scores = evaluate(main_policy, episodes=500, greedy=True)
check("D3. taking away the randomness (always the most likely action) does about as well or better",
      greedy_scores.mean() >= 0.9 * main_scores.mean(), f"most-likely action {greedy_scores.mean():.1f} vs random choice {main_scores.mean():.1f}")

first, last = main_rewards[:100].mean(), main_rewards[-100:].mean()
check("D4. the last 100 training games score much higher than the first 100", last - first >= 50, f"{first:.1f} then {last:.1f}")

# =====================================================================================
print("\n=== E. THE 500 TEST GAMES AND THE GRAPHS ===")
check("E1. exactly 500 test games, every score between 1 and 500",
      len(main_scores) == 500 and main_scores.min() >= 1 and main_scores.max() <= 500,
      f"mean {main_scores.mean():.1f}, std {main_scores.std():.1f}, range {main_scores.min():.0f} to {main_scores.max():.0f}")

ma = moving_average(np.arange(1, 5), window=2)
games = np.arange(1, 301, dtype=float)
ma100 = moving_average(games, 100)
check("E2. moving average by hand: [1,2,3,4] with window 2 gives 1, 1.5, 2.5, 3.5; with window 100 the first value is the first game, "
      "game 100 is the mean of games 1-100, and game 101 drops game 1",
      np.allclose(ma, [1, 1.5, 2.5, 3.5]) and ma100[0] == 1 and np.isclose(ma100[99], games[:100].mean()) and np.isclose(ma100[100], games[1:101].mean()))

saved_eval = os.path.join(HERE, "cartpole_eval_seed0.npy")
same = os.path.exists(saved_eval) and np.array_equal(main_scores, np.load(saved_eval))
retrained_same = np.array_equal(np.array(train(200, seed=0)[1]), main_rewards[:200])
check("E3. the saved model reproduces exactly the 500 test scores behind the graphs in the report, and retraining from the same seed repeats the start of the saved training curve",
      same and retrained_same, "the 500 test scores are identical; the first 200 training games are identical")

print(f"\n{sum(results)} of {len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)

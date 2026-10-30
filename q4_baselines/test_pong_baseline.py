"""Question 4, Pong-v5 baseline checks (edge cases + "does it behave as predicted?").
Run: python test_pong_baseline.py        (takes a few minutes: it plays a few real games and does a few hundred tiny updates)
Each check prints PASS/FAIL and the evidence. The checks are grouped as in the report."""
import json
import os
import tempfile
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
from pong_baseline import *
import pong_pg

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


def weights(net):
    return torch.cat([p.detach().flatten() for p in net.parameters()]).clone()


def args_for(**kw):
    base = dict(baseline="learned", lr=0.001, batch=2, gamma=GAMMA, seed=0, time_budget=1e9, max_games=1, eval_games=1,
                log_every=1000, threads=2, device="cpu", out=tempfile.mkdtemp())
    base.update(kw)
    return Namespace(**base)


def made_up_game(steps=5, rewards=None, seed=0):
    """A short made-up game: sparse random screen-differences, random pushes, and the given rewards."""
    g = np.random.default_rng(seed)
    x = (g.random((steps, 6400)) < 0.01).astype(np.float32) - (g.random((steps, 6400)) < 0.01).astype(np.float32)
    a = (g.random(steps) < 0.5).astype(np.float32)
    r = np.array([0, 0, 1, 0, -1][:steps] if rewards is None else rewards, dtype=np.float32)
    return x, a, r


def fresh(seed=1):
    torch.manual_seed(seed)
    return Policy(), ValueNet()


# =====================================================================================
print("=== A. THE REFERENCE AND THE CREDITS ===")
games = [made_up_game(5, seed=1), made_up_game(5, [0, 1, 0, 0, -1], seed=2)]


def moved(with_value, pol_seed=1):
    pol, vn = fresh(pol_seed)
    sgd = torch.optim.SGD(pol.parameters(), lr=1.0)
    x = torch.from_numpy(np.concatenate([g[0] for g in games]))
    a = torch.from_numpy(np.concatenate([g[1] for g in games]))
    returns = torch.from_numpy(np.concatenate([discounted_returns(g[2], 0.9, True) for g in games]))
    reference = vn(x).detach() if with_value else torch.zeros_like(returns)
    hand = -(pol(x).log_prob(a) * (returns - reference)).sum() / len(games)
    expected = torch.cat([-g.flatten() for g in torch.autograd.grad(hand, list(pol.parameters()))])
    before = weights(pol)
    update(pol, sgd, games, 0.9, "cpu", True, vn if with_value else None, torch.optim.SGD(vn.parameters(), lr=0.0) if with_value else None)
    return torch.allclose(weights(pol) - before, expected, atol=1e-6)


check("A1. the learning step is exactly the hand-written one: credit = (reward that followed, restarting at each point) - reference, "
      "averaged over the games, with and without the learned baseline", moved(False) and moved(True))

pol_a, _ = fresh()
pol_b, _ = fresh()
opt_a, opt_b = torch.optim.Adam(pol_a.parameters(), lr=0.01), torch.optim.Adam(pol_b.parameters(), lr=0.01)
pong_pg.update(pol_a, opt_a, games, 0.9)
update(pol_b, opt_b, games, 0.9)
check("A2. 'no baseline' is exactly Part 1: one learning step gives identical weights", torch.equal(weights(pol_a), weights(pol_b)))

calls = []
pol, vn = fresh()
vn.register_forward_hook(lambda module, inputs, output: calls.append(output.requires_grad))
update(pol, torch.optim.Adam(pol.parameters()), games, 0.9, "cpu", True, vn, torch.optim.Adam(vn.parameters()))
check("A3. the learned baseline is only a reference for the credits: its numbers carry no learning information into the policy's step "
      "(it is trained separately, on its own)", calls[0] is False and calls[1] is True, "first use without learning information, second use (its own training) with")

check("A4. bad settings are rejected with a message (batch 0, learning rate 0, discount 1.0, an unknown baseline name)",
      raises(ValueError, lambda: train(args_for(batch=0))) and raises(ValueError, lambda: train(args_for(lr=0)))
      and raises(ValueError, lambda: train(args_for(gamma=1.0))) and raises(ValueError, lambda: train(args_for(baseline="constant"))))

# =====================================================================================
print("\n=== B. A BASELINE MUST NOT CHANGE WHERE LEARNING POINTS, ONLY HOW NOISY IT IS ===")
# Made-up problem with a known answer: one fixed screen-difference; pushing down pays 10, pushing up pays 12.
# The exact gradient of the expected reward is the gradient of 2 x P(up).
g0 = np.random.default_rng(0)
X = ((g0.random(6400) < 0.02).astype(np.float32) - (g0.random(6400) < 0.02).astype(np.float32))
XT = torch.from_numpy(X)


def toy_games(acts):
    return [(X[None, :], np.array([float(a)], dtype=np.float32), np.array([10.0 + 2.0 * float(a)], dtype=np.float32)) for a in acts]


def signal(value, n, seed):
    """update()'s learning signal from n one-step games (plain SGD, learning rate 1), and the exact gradient it should match."""
    torch.manual_seed(seed)
    pol = Policy()
    exact = torch.cat([g.flatten() for g in torch.autograd.grad(2.0 * pol(XT).probs, list(pol.parameters()))])
    acts = pol(XT.repeat(n, 1)).sample()
    before = weights(pol)
    update(pol, torch.optim.SGD(pol.parameters(), lr=1.0), toy_games(acts.tolist()), 0.9, "cpu", True, value, torch.optim.SGD(value.parameters(), lr=0.0) if value else None)
    return weights(pol) - before, exact


def match(value, n=4000):
    e, x = signal(value, n, 0)
    return torch.nn.functional.cosine_similarity(e, x, dim=0).item(), ((e - x).norm() / x.norm()).item()


def spread(value, n=20, repeats=60):
    est = torch.stack([signal(value, n, s)[0] for s in range(repeats)])
    return (est - est.mean(0)).norm(dim=1).pow(2).mean().sqrt().item()


class Fixed(nn.Module):
    """A pretend baseline network that returns whatever we choose (used to try good and bad baselines)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) + self.dummy.sum()          # the dummy is 0; it only lets the pretend network take part in training


torch.manual_seed(0)
learned = ValueNet()
learned_opt = torch.optim.Adam(learned.parameters(), lr=0.01)
idle = Policy()                                       # a policy that is never changed (learning rate 0), so only the baseline network learns
idle_opt = torch.optim.SGD(idle.parameters(), lr=0.0)
for _ in range(300):                                  # teach the toy baseline network what to expect (about 11), using update()'s own training of it
    acts = (torch.rand(8) < 0.5).float().tolist()
    update(idle, idle_opt, toy_games(acts), 0.9, "cpu", True, learned, learned_opt)
guess = learned(XT).item()
check("B1. the learned baseline network learns to expect about the right reward in the made-up problem (it should be near 11)",
      abs(guess - 11.0) < 0.6, f"its guess: {guess:.2f}")

res = {"none": match(None), "learned": match(learned)}
check("B2. made-up problem with a known answer: with no baseline and with a learned baseline, our learning signal points the same way as the exact gradient of the expected reward",
      res["none"][0] > 0.9 and res["learned"][0] > 0.98 and res["learned"][1] < 0.2,
      "; ".join(f"{k}: direction match {c:.4f}, size error {err:.3f}" for k, (c, err) in res.items()))

# a baseline that peeks at each push's own reward (the games are built so that the nth step's reward is known)
def match_with(returns_multiplier):
    torch.manual_seed(0)
    pol = Policy()
    exact = torch.cat([g.flatten() for g in torch.autograd.grad(2.0 * pol(XT).probs, list(pol.parameters()))])
    acts = pol(XT.repeat(4000, 1)).sample()
    rewards = 10.0 + 2.0 * acts
    cheat = Fixed(lambda x: returns_multiplier * rewards)
    before = weights(pol)
    update(pol, torch.optim.SGD(pol.parameters(), lr=1.0), toy_games(acts.tolist()), 0.9, "cpu", True, cheat, torch.optim.SGD(cheat.parameters(), lr=0.0))
    e = weights(pol) - before
    return torch.nn.functional.cosine_similarity(e, exact, dim=0).item() if e.norm() > 0 else 0.0, ((e - exact).norm() / exact.norm()).item()


bad = {"peeks at the reward": match_with(1.0), "subtracts twice the reward": match_with(2.0)}
check("B2b. the B2 check can really fail: a baseline that depends on the push itself (it peeks at the reward it is meant to judge, or subtracts twice as much) is caught",
      all(not (c > 0.98 and err < 0.2) for c, err in bad.values()),
      "; ".join(f"{k}: direction match {c:+.2f}, size error {err:.2f}" for k, (c, err) in bad.items()))

s_none, s_learned = spread(None), spread(learned)
check("B3. a baseline makes the learning signal much less noisy (repeated small batches of 20 pushes: typical spread, smaller is better)",
      s_learned < 0.5 * s_none, f"no baseline {s_none:.3f}, learned {s_learned:.3f}")

# =====================================================================================
print("\n=== C. THE TRAINING LOOP ===")
pol, vn = fresh()
pol_before, val_before = weights(pol), weights(vn)
nan_game = made_up_game(5, [0, 0, np.nan, 0, -1])
skipped = update(pol, torch.optim.Adam(pol.parameters()), [nan_game], 0.9, "cpu", True, vn, torch.optim.Adam(vn.parameters())) is False
check("C1. a NaN reward makes the update skip, and neither network is touched", skipped and torch.equal(pol_before, weights(pol)) and torch.equal(val_before, weights(vn)))

x, a, r = made_up_game(5)
x[1, 7] = np.nan
skipped = update(pol, torch.optim.Adam(pol.parameters()), [(x, a, r)], 0.9, "cpu", True, vn, torch.optim.Adam(vn.parameters())) is False
check("C1b. a NaN inside a game's screen input (so the learned baseline outputs NaN) makes the update skip, and neither network is touched",
      skipped and torch.equal(pol_before, weights(pol)) and torch.equal(val_before, weights(vn)))

empty = (np.zeros((0, 6400), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
check("C2. an empty list of games, or only empty games, is skipped instead of crashing (both settings)",
      all(update(*fresh()[:1], torch.optim.Adam(Policy().parameters()), g, 0.9, "cpu", True, v, None) is False for g in ([], [empty]) for v in (None, ValueNet())))

one = [made_up_game(1, [-1.0])]
pol, vn = fresh()
before = weights(pol)
ok = update(pol, torch.optim.Adam(pol.parameters()), one, 0.9, "cpu", True, vn, torch.optim.Adam(vn.parameters()))
check("C3. a one-step game still gives a valid update", ok and not torch.equal(before, weights(pol)) and torch.isfinite(weights(pol)).all() and torch.isfinite(weights(vn)).all())

# short real runs: reproducible, and a partly filled batch is still learned from
runs = {}
for name, seed in (("a", 3), ("b", 3), ("c", 4)):
    cfg = args_for(seed=seed, max_games=2)
    train(cfg)
    runs[name] = np.load(os.path.join(cfg.out, "train_rewards.npy"))
check("C4. the same seed gives an identical training curve and a different seed a different one (learned baseline, real games)",
      np.array_equal(runs["a"], runs["b"]) and not np.array_equal(runs["a"], runs["c"]), f"seed 3 twice: {runs['a'].tolist()}; seed 4: {runs['c'].tolist()}")

cfg = args_for(seed=5, max_games=1, batch=4)
train(cfg)
torch.manual_seed(5)
start_weights = weights(Policy())
final = torch.load(os.path.join(cfg.out, "model.pt"))
check("C5. when the game limit ends in the middle of a batch (1 game played, batch of 4), that game is still learned from",
      not torch.equal(start_weights, torch.cat([v.flatten() for v in final.values()])))

# =====================================================================================
print("\n=== D. THE CONVOLUTIONAL NETWORK (extra experiment) ===")
torch.manual_seed(0)
conv = ConvPolicy()
screens = torch.from_numpy(np.concatenate([made_up_game(5, seed=s)[0] for s in range(20)]))
d_one, d_many = conv(screens[0]), conv(screens)
check("D1. one screen gives one probability and a batch of screens one probability each, all between 0 and 1 (same as the plain network)",
      d_one.probs.shape == () and d_many.probs.shape == (100,) and bool(((d_many.probs >= 0) & (d_many.probs <= 1)).all()),
      f"shapes {tuple(d_one.probs.shape)} and {tuple(d_many.probs.shape)}")
check("D2. before any training, up and down are about equally likely", 0.4 < d_many.probs.mean().item() < 0.6, f"average chance of pushing up {d_many.probs.mean():.3f}")

wild = ConvPolicy()
with torch.no_grad():
    for p in wild.parameters():
        p.mul_(30)
dw = wild(torch.randn(20, 6400) * 1e5)
check("D3. an extreme network (weights x 30) and extreme inputs (hundreds of thousands) still give finite probabilities and log-probabilities",
      torch.isfinite(dw.probs).all().item() and torch.isfinite(dw.log_prob(dw.sample())).all().item())

torch.manual_seed(0)
cp = ConvPolicy()
exact_conv = torch.cat([g.flatten() for g in torch.autograd.grad(2.0 * cp(XT).probs, list(cp.parameters()))])
acts = cp(XT.repeat(4000, 1)).sample()
before = weights(cp)
update(cp, torch.optim.SGD(cp.parameters(), lr=1.0), toy_games(acts.tolist()), 0.9, "cpu", True, None, None)
moved_conv = weights(cp) - before
c_dir = torch.nn.functional.cosine_similarity(moved_conv, exact_conv, dim=0).item()
check("D4. made-up problem with a known answer, now with the convolutional network: our learning signal points the same way as the exact gradient of the expected reward",
      c_dir > 0.9, f"direction match {c_dir:.4f}")

conv_pol, conv_val = ConvPolicy(), ConvValueNet()
conv_before = weights(conv_pol)
ok = update(conv_pol, torch.optim.Adam(conv_pol.parameters(), lr=0.01), games, 0.9, "cpu", True, conv_val, torch.optim.Adam(conv_val.parameters(), lr=0.01))
conv_after = weights(conv_pol)
check("D5. the convolutional network with the learned baseline gives a valid learning step that changes the policy network",
      ok and torch.isfinite(conv_after).all().item() and not torch.equal(conv_before, conv_after))
nan_game = made_up_game(5, [0, 0, np.nan, 0, -1])
cp2, cv2 = ConvPolicy(), ConvValueNet()
cb = weights(cp2), weights(cv2)
skipped = update(cp2, torch.optim.Adam(cp2.parameters()), [nan_game], 0.9, "cpu", True, cv2, torch.optim.Adam(cv2.parameters())) is False
check("D6. a NaN reward makes the update skip, with both convolutional networks untouched", skipped and torch.equal(cb[0], weights(cp2)) and torch.equal(cb[1], weights(cv2)))

filt_before = conv_pol.layers[0].weight.detach().clone()
update(conv_pol, torch.optim.Adam(conv_pol.parameters(), lr=0.01), games, 0.9, "cpu", True, None, None)
check("D7. the first layer (the picture filters) itself receives learning, not just the layers after it", not torch.equal(filt_before, conv_pol.layers[0].weight.detach()))

cfg1, cfg2 = (args_for(seed=3, max_games=2, network="conv") for _ in range(2))
for c in (cfg1, cfg2):
    train(c)
r1 = np.load(os.path.join(cfg1.out, "train_rewards.npy"))
r2 = np.load(os.path.join(cfg2.out, "train_rewards.npy"))
check("D8. a short real run with the convolutional network works, repeats exactly with the same seed, and an unknown network name is rejected",
      len(r1) == 2 and np.array_equal(r1, r2) and raises(ValueError, lambda: train(args_for(network="transformer"))), f"two runs of seed 3: {r1.tolist()} and {r2.tolist()}")

# =====================================================================================
print("\n=== E. DOES IT BEHAVE AS PREDICTED? (uses the saved results of the experiments) ===")
summary_file = os.path.join(HERE, "figures", "pong_baseline_summary.json")
PART1 = os.path.join(HERE, "..", "q3_policy_gradient", "results")
if os.path.exists(summary_file):
    from scipy import stats
    S = json.load(open(summary_file))
    RUNS = {"learned baseline, seed 0 (main)": ("pong_learned_lr0.001_seed0", Policy),
            "learned baseline, seed 1": ("pong_learned_lr0.001_seed1", Policy),
            "convolutional + learned baseline, seed 0": ("pong_conv_learned_lr0.001_seed0", ConvPolicy)}
    bad_runs = []
    for name, (folder, _) in RUNS.items():
        scores = np.load(os.path.join(HERE, "results", folder, "eval_rewards.npy"))
        cfg = json.load(open(os.path.join(HERE, "results", folder, "config.json")))
        if not (len(scores) == 500 and scores.min() >= -21 and scores.max() <= 21 and cfg["skipped_updates"] == 0):
            bad_runs.append(name)
    check("E1. each saved run has exactly 500 test games, every score between -21 and +21, and no skipped learning step",
          not bad_runs, "; ".join(f"{n}: {S[n]['games']:,} games, mean {S[n]['mean']:.2f}, std {S[n]['std']:.2f}" for n in RUNS))

    sooner = all(S[f"learned baseline, seed {s}" + (" (main)" if s == 0 else "")]["first_game_average_reaches_minus10"]
                 < 0.6 * S[f"no baseline, seed {s}"]["first_game_average_reaches_minus10"] for s in (0, 1))
    check("E2. the learned baseline makes Pong take off much sooner: the 100-game average first reaches -10 in under 60% of the games needed without a baseline, for both seeds",
          sooner, "; ".join(f"seed {s}: game {S['learned baseline, seed ' + str(s) + (' (main)' if s == 0 else '')]['first_game_average_reaches_minus10']} "
                            f"against game {S['no baseline, seed ' + str(s)]['first_game_average_reaches_minus10']}" for s in (0, 1)))

    none0 = np.load(os.path.join(PART1, "pong_lr0.001_seed0_10k", "eval_rewards.npy"))
    main_scores = np.load(os.path.join(HERE, "results", "pong_learned_lr0.001_seed0", "eval_rewards.npy"))
    t = stats.ttest_ind(main_scores, none0, equal_var=False)
    check("E3. the main model (learned baseline, seed 0) scores higher on its 500 test games than the no-baseline main model, and the difference is unlikely to be luck (p below 0.001)",
          main_scores.mean() > none0.mean() and t.pvalue < 0.001, f"{main_scores.mean():.2f} against {none0.mean():.2f}, p = {t.pvalue:.1e}")

    torch.set_num_threads(2)
    torch.manual_seed(7)
    blank = evaluate(Policy(), games=20, seed=777)
    verdicts = []
    for name, (folder, cls) in RUNS.items():
        net = cls()
        net.load_state_dict(torch.load(os.path.join(HERE, "results", folder, "model.pt"), map_location="cpu"))
        trained = evaluate(net, games=20, seed=777)
        tt = (trained.mean() - blank.mean()) / np.sqrt(trained.var(ddof=1) / 20 + blank.var(ddof=1) / 20)
        verdicts.append((name, trained.mean(), tt))
    check("E4. each trained network (both plain ones and the convolutional one) is far better than an untrained one over 20 fresh games each (statistical test t above 10)",
          all(v[2] > 10 for v in verdicts), "; ".join(f"{n}: {m:.1f} (t = {tt:.0f})" for n, m, tt in verdicts) + f"; untrained {blank.mean():.1f}")
else:
    print("(no saved Pong baseline results yet, so this section was skipped; run the experiments first)")

print(f"\n{sum(results)} of {len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)

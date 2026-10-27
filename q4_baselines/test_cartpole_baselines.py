"""Question 4, CartPole-v1 baseline checks (edge cases + "does it behave as predicted?").
Run: python test_cartpole_baselines.py        (takes about 1 minute; it does not retrain the main models)
Each check prints PASS/FAIL and the evidence. The checks are grouped as in the report."""
import json
import os
import numpy as np
import torch
from cartpole_baseline import *
import cartpole_pg

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


STATE = torch.tensor([0.1, -0.2, 0.05, 0.3])        # one fixed example state, used in several checks


def one_game(policy, steps=4, reward=None):
    """A short made-up game on the fixed state: (states, log-probs, rewards)."""
    states = np.tile(STATE.numpy(), (steps, 1))
    dist = policy(torch.as_tensor(states))
    actions = dist.sample()
    rewards = np.ones(steps, dtype=np.float32) if reward is None else np.array(reward, dtype=np.float32)
    return states, list(dist.log_prob(actions)), rewards


# =====================================================================================
print("=== A. THE REFERENCE AND THE CREDITS ===")
lp = [torch.tensor(-0.5), torch.tensor(-1.0), torch.tensor(-2.0)]
loss = policy_loss(lp, torch.tensor([3.0, 2.0, 1.0]), torch.tensor([2.0, 2.0, 2.0]))
check("A1. worked by hand: rewards 3, 2, 1 against a reference of 2 give credits +1, 0, -1, so the loss is -(-0.5 x 1 + -1 x 0 + -2 x -1) = -1.5",
      torch.isclose(loss, torch.tensor(-1.5)), f"loss {loss.item():.2f}")

same_a = np.array(train(60, seed=3, baseline="none")[1])
same_b = np.array(cartpole_pg.train(60, seed=3)[1])
check("A2. 'no baseline' is exactly Part 1: the same training curve, game for game", np.array_equal(same_a, same_b), f"{len(same_a)} games compared")


def moved_by_update(baseline, constant=0.0, with_value=False):
    """Does update() move the policy by exactly minus the gradient of the hand-written loss? (plain SGD with learning rate 1)"""
    torch.manual_seed(1)
    pol = Policy()
    sgd = torch.optim.SGD(pol.parameters(), lr=1.0)
    value_net = ValueNet() if with_value else None
    states, lps, rewards = one_game(pol)
    returns = torch.as_tensor(discounted_returns(rewards, GAMMA))
    if baseline == "constant":
        reference = torch.full_like(returns, constant)
    elif baseline == "learned":
        reference = value_net(torch.as_tensor(states)).detach()
    else:
        reference = torch.zeros_like(returns)
    expected = [-g for g in torch.autograd.grad(policy_loss(lps, returns, reference), list(pol.parameters()), retain_graph=True)]
    before = weights(pol)
    update(pol, sgd, states, lps, rewards, GAMMA, baseline, constant, value_net, torch.optim.SGD(value_net.parameters(), lr=0.0) if with_value else None)
    moved = weights(pol) - before
    return torch.allclose(moved, torch.cat([g.flatten() for g in expected]), atol=1e-6)


check("A3. in all three settings the learning step is exactly the hand-written one (credit = reward - reference)",
      moved_by_update("none") and moved_by_update("constant", 1.7) and moved_by_update("learned", with_value=True))

def policy_after_one_update(baseline, constant):
    torch.manual_seed(1)
    pol = Policy()
    sgd = torch.optim.SGD(pol.parameters(), lr=1.0)
    states, lps, rewards = one_game(pol)
    update(pol, sgd, states, lps, rewards, GAMMA, baseline, constant)
    return weights(pol)


check("A3b. before the first game the constant baseline is 0, so the first update is identical to 'no baseline'",
      torch.equal(policy_after_one_update("none", 0.0), policy_after_one_update("constant", 0.0)))

calls = []
value_net = ValueNet()
value_net.register_forward_hook(lambda module, inputs, output: calls.append(output.requires_grad))
pol = Policy()
states, lps, rewards = one_game(pol)
update(pol, torch.optim.Adam(pol.parameters()), states, lps, rewards, GAMMA, "learned", 0.0, value_net, torch.optim.Adam(value_net.parameters()))
check("A4. the learned baseline is only a reference for the credits: its numbers carry no learning information into the policy's step "
      "(it is trained separately, on its own)", calls[0] is False and calls[1] is True, "first use without learning information, second use (its own training) with")

check("A5. bad settings are rejected with a message (0 games, learning rate 0, discount 1.0, an unknown baseline name)",
      raises(ValueError, lambda: train(0)) and raises(ValueError, lambda: train(5, learning_rate=0))
      and raises(ValueError, lambda: train(5, gamma=1.0)) and raises(ValueError, lambda: train(5, baseline="mean")))

# =====================================================================================
print("\n=== B. A BASELINE MUST NOT CHANGE WHERE LEARNING POINTS, ONLY HOW NOISY IT IS ===")
# Made-up problem with a known answer: one fixed state; pushing 0 pays 10, pushing 1 pays 12. The exact gradient of the
# expected reward is the gradient of 2 x P(push 1). A baseline that does not depend on the push should keep this direction.
EXACT_FACTOR = 2.0


def estimate(reference_fn, n, seed):
    """The learning signal from n one-step games, and the exact gradient it should match."""
    torch.manual_seed(seed)
    toy = Policy()
    params = list(toy.parameters())
    exact = torch.cat([g.flatten() for g in torch.autograd.grad(EXACT_FACTOR * toy(STATE).probs[1], params)])
    dist = toy(STATE.repeat(n, 1))
    acts = dist.sample()
    returns = 10.0 + 2.0 * acts.float()
    grads = torch.autograd.grad(policy_loss(list(dist.log_prob(acts)), returns, reference_fn(returns)), params)
    return torch.cat([-g.flatten() / n for g in grads]), exact


def match(reference_fn, n=20000):
    e, x = estimate(reference_fn, n, 0)
    return torch.nn.functional.cosine_similarity(e, x, dim=0).item(), ((e - x).norm() / x.norm()).item()


def spread(reference_fn, n=20, repeats=200):
    """How noisy the signal is from a small batch: typical distance between repeated estimates."""
    est = torch.stack([estimate(reference_fn, n, s)[0] for s in range(repeats)])
    return (est - est.mean(0)).norm(dim=1).pow(2).mean().sqrt().item()


learned = ValueNet()
learned_opt = torch.optim.Adam(learned.parameters(), lr=0.01)
for _ in range(400):                                 # teach the toy baseline network what to expect (about 11)
    pred = learned(STATE)
    target = torch.tensor(11.0)
    learned_opt.zero_grad(); ((pred - target) ** 2).backward(); learned_opt.step()
guess = learned(STATE).item()
none_ref, const_ref = (lambda r: torch.zeros_like(r)), (lambda r: torch.full_like(r, 11.0))
learned_ref = lambda r: learned(STATE).detach().expand_as(r)

results_match = {name: match(fn) for name, fn in [("none", none_ref), ("constant", const_ref), ("learned", learned_ref)]}
check("B1. made-up problem with a known answer: with no baseline, a constant baseline and a learned baseline, our learning signal "
      "points the same way as the exact gradient of the expected reward",
      all(c > 0.98 and err < 0.25 for c, err in results_match.values()),
      "; ".join(f"{k}: direction match {c:.4f}, size error {err:.3f}" for k, (c, err) in results_match.items()))

cheat = lambda r: r                                  # a baseline that peeks at the push's own reward: it must NOT be accepted
flipped = lambda r: 2 * r                            # subtracts too much, so the credit changes sign
bad = {"peeks at the reward": match(cheat), "subtracts twice the reward": match(flipped)}
check("B1b. the B1 check can really fail: a baseline that depends on the push itself (it peeks at the reward it is meant to judge, or subtracts twice as much) is caught",
      all(not (c > 0.98 and err < 0.25) for c, err in bad.values()),
      "; ".join(f"{k}: direction match {c:+.2f}, size error {err:.2f}" for k, (c, err) in bad.items()))

s_none, s_const, s_learned = spread(none_ref), spread(const_ref), spread(learned_ref)
check("B2. a baseline makes the learning signal much less noisy (repeated small batches of 20 pushes: typical spread, smaller is better)",
      s_const < 0.5 * s_none and s_learned < 0.5 * s_none, f"no baseline {s_none:.3f}, constant {s_const:.3f}, learned {s_learned:.3f}")
check("B3. the learned baseline network learns to expect about the right reward in the made-up problem (it should be near 11)",
      abs(guess - 11.0) < 0.5, f"its guess: {guess:.2f}")

# =====================================================================================
print("\n=== C. THE TRAINING LOOP ===")
for baseline in ("constant", "learned"):
    pol = Policy()
    vn = ValueNet()
    opt, vopt = torch.optim.Adam(pol.parameters(), lr=0.01), torch.optim.Adam(vn.parameters(), lr=0.01)
    pol_before, val_before = weights(pol), weights(vn)
    states, lps, rewards = one_game(pol, steps=3)
    skipped = update(pol, opt, states, lps, np.array([1.0, np.nan, 1.0], dtype=np.float32), GAMMA, baseline, 0.0, vn, vopt) is False
    untouched = torch.equal(pol_before, weights(pol)) and torch.equal(val_before, weights(vn))
    check(f"C1 ({baseline}). a NaN reward makes the update skip, and neither network is touched", skipped and untouched)

pol = Policy()
vn = ValueNet()
states, lps, rewards = one_game(pol)
states[1, 2] = np.nan                                  # a broken state reaching the baseline network
pol_before, val_before = weights(pol), weights(vn)
skipped = update(pol, torch.optim.Adam(pol.parameters()), states, lps, rewards, GAMMA, "learned", 0.0, vn, torch.optim.Adam(vn.parameters())) is False
check("C1b. a NaN inside a game's states (the learned baseline would output NaN) makes the update skip, and neither network is touched",
      skipped and torch.equal(pol_before, weights(pol)) and torch.equal(val_before, weights(vn)))

check("C2. an empty game is skipped instead of crashing (all three settings)",
      all(update(Policy(), torch.optim.Adam(Policy().parameters()), np.zeros((0, 4), dtype=np.float32), [], np.array([], dtype=np.float32), GAMMA, b,
                 0.0, ValueNet(), None) is False for b in BASELINES))

for baseline in ("constant", "learned"):
    pol = Policy()
    vn = ValueNet()
    opt, vopt = torch.optim.Adam(pol.parameters(), lr=0.01), torch.optim.Adam(vn.parameters(), lr=0.01)
    states, lps, rewards = one_game(pol, steps=1)
    before = weights(pol)
    ok = update(pol, opt, states, lps, rewards, GAMMA, baseline, 0.0, vn, vopt)
    check(f"C3 ({baseline}). a one-step game (the pole falls straight away) still gives a valid update",
          ok and not torch.equal(before, weights(pol)) and torch.isfinite(weights(pol)).all() and torch.isfinite(weights(vn)).all())

a1, a2 = np.array(train(40, seed=3, baseline="learned")[1]), np.array(train(40, seed=3, baseline="learned")[1])
b1 = np.array(train(40, seed=4, baseline="learned")[1])
c1, c2 = np.array(train(40, seed=3, baseline="constant")[1]), np.array(train(40, seed=3, baseline="constant")[1])
check("C4. the same seed gives an identical training curve and a different seed a different one (learned and constant baselines)",
      np.array_equal(a1, a2) and not np.array_equal(a1, b1) and np.array_equal(c1, c2))

# =====================================================================================
print("\n=== D. DOES IT BEHAVE AS PREDICTED? (uses the saved results of the experiments, 10 seeds per method) ===")
summary_file = os.path.join(HERE, "figures", "cartpole_baseline_summary.json")
if os.path.exists(summary_file):
    from scipy import stats
    S = json.load(open(summary_file))
    SEEDS = sorted({int(k.split("seed")[1]) for k in S})
    per_seed = lambda b, key: [S[f"{b}_seed{s}"][key] for s in SEEDS]
    check("D1. the 500 test scores of every run are valid (between 1 and 500)", all(1 <= v["min"] and v["max"] <= 500 for v in S.values()),
          f"{len(S)} runs; lowest {min(v['min'] for v in S.values()):.0f}, highest {max(v['max'] for v in S.values()):.0f}")
    for number, name in ((2, "learned"), (3, "constant")):
        t = stats.ttest_ind(per_seed(name, "mean"), per_seed("none", "mean"), equal_var=False)
        u = stats.mannwhitneyu(per_seed(name, "mean"), per_seed("none", "mean"), alternative="greater")
        check(f"D{number}. the {name} baseline does better than no baseline over {len(SEEDS)} seeds, and the difference is unlikely to be luck (two statistical tests, both p below 0.05)",
              np.mean(per_seed(name, "mean")) > np.mean(per_seed("none", "mean")) and t.pvalue < 0.05 and u.pvalue < 0.05,
              f"{name} {np.mean(per_seed(name, 'mean')):.1f} against none {np.mean(per_seed('none', 'mean')):.1f}; p = {t.pvalue:.3f} and {u.pvalue:.3f}")
    torch.set_num_threads(1)                           # the experiments used 1 thread per run; other settings can change a run completely
    main = Policy()
    main.load_state_dict(torch.load(os.path.join(HERE, "results", "cartpole_learned_model_seed0.pt")))
    saved = np.load(os.path.join(HERE, "results", "cartpole_learned_eval_seed0.npy"))
    check("D4. the saved main model (learned baseline, seed 0) reproduces exactly the 500 test scores behind the graphs",
          np.array_equal(evaluate(main, episodes=500), saved), f"{len(saved)} scores compared")
    saved_curve = np.load(os.path.join(HERE, "results", "cartpole_learned_train_seed0.npy"))
    check("D5. retraining the main model from the same seed (1 thread) repeats the start of the saved training curve",
          np.array_equal(np.array(train(200, seed=0, baseline="learned")[1]), saved_curve[:200]), "the first 200 training games were compared")
else:
    print("(saved results not found, so checks D1-D5 were skipped; run run_cartpole_baselines.py first)")

print(f"\n{sum(results)} of {len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)

"""Question 1 checks (edge cases + "does it behave as predicted?").   Run: python test_q1.py
Each check prints PASS/FAIL and the evidence. The checks are grouped as in the report."""
import contextlib
import numpy as np
import shuttle
from shuttle import *

rng = np.random.default_rng(0)
results = []


def check(name, ok, evidence=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {evidence}" if evidence else ""))


@contextlib.contextmanager
def change(**settings):
    """Temporarily change the problem (e.g. change(DISPATCH_COST=0)), then restore it."""
    old = {k: getattr(shuttle, k) for k in settings}
    for k, v in settings.items():
        setattr(shuttle, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(shuttle, k, v)


def solve():
    """Value iteration on the current (possibly changed) problem: returns (values, best actions)."""
    c, p = shuttle.build_tables()
    values, policy, _ = shuttle.value_iteration(c, p)
    return values, policy


def first_dispatch(policy):
    return int(np.argmax(policy == DISPATCH))


costs, probs = build_tables()
values, policy, _ = value_iteration(costs, probs)

# =====================================================================================
print("=== A. ORDINARY SITUATIONS ===")
check("A1. empty station (0 people): best action is to wait", policy[0] == WAIT,
      "dispatching would pay 100 to carry nobody")

cost, nxt = cost_and_next_state_probs(4, DISPATCH)
check("A2. dispatching with fewer than 15 people (4): takes everyone, costs only 100, next state is 1-5",
      cost == 100 and np.isclose(nxt[1:6].sum(), 1.0), f"cost = {cost}")

with change(MAX_PEOPLE=300):
    v300, p300 = solve()
check("A3. the 200-person limit does not matter: with a limit of 300 the best actions are the same (up to 150 people)",
      np.array_equal(p300[:151], policy[:151]), f"largest cost difference {np.abs(v300[:151] - values[:151]).max():.1e}")

sums_ok = np.allclose(probs.sum(axis=2), 1)
broken = probs.copy()
broken[WAIT, 50] *= 0.9                      # deliberately damage one row
check("A4. chances of the next state add up to 100% everywhere (and the check does catch a broken table)",
      sums_ok and not np.allclose(broken.sum(axis=2), 1), f"largest error {np.abs(probs.sum(axis=2) - 1).max():.1e}")

cutoffs = []
for k in (5, 15, 30):
    with change(CAPACITY=k):
        cutoffs.append(first_dispatch(solve()[1]))
check("A5. a bigger shuttle waits for more people before dispatching", cutoffs[0] < cutoffs[1] < cutoffs[2],
      f"shuttle size 5 / 15 / 30 -> dispatch at {cutoffs[0]} / {cutoffs[1]} / {cutoffs[2]} people")

# =====================================================================================
print("\n=== B. EDGES OF THE STATION ===")
c15, n15 = cost_and_next_state_probs(15, DISPATCH)
c16, n16 = cost_and_next_state_probs(16, DISPATCH)
check("B1. exactly 15 people empties the station (cost 100, next 1-5); 16 people leaves one behind (cost 102, next 2-6)",
      c15 == 100 and np.isclose(n15[1:6].sum(), 1) and c16 == 102 and np.isclose(n16[2:7].sum(), 1), f"costs {c15} and {c16}")

_, n200 = cost_and_next_state_probs(200, WAIT)
check("B2. full station (200): the best action is to dispatch, and waiting keeps it at 200",
      policy[200] == DISPATCH and np.isclose(n200[200], 1.0))

_, n198 = cost_and_next_state_probs(198, WAIT)
_, n199 = cost_and_next_state_probs(199, WAIT)
check("B3. nearly full: at 199 everyone arriving is turned away (stays 200); at 198 one arrival fits, the rest do not",
      np.isclose(n199[200], 1.0) and np.isclose(n198[199], 0.2) and np.isclose(n198[200], 0.8),
      f"from 198: 20% -> 199, 80% -> 200")

check("B4. no off-by-one: 201 situations (0 to 200 people) everywhere, and the plots have 201 points",
      costs.shape == (2, 201) and probs.shape == (2, 201, 201) and len(policy) == 201 == len(values),
      f"tables are {probs.shape}")

# =====================================================================================
print("\n=== C. ARRIVALS ===")
# Hand calculation: nobody ever arrives, 10 people waiting.
#   wait forever: 2*10 = 20 per period -> 20/(1-0.95) = 400.   dispatch once: 100, then nothing.
with change(ARRIVALS=[0], MAX_PEOPLE=30):
    v_none, _ = solve()
check("C1. nobody ever arrives, 10 waiting: cheapest is 100 (dispatch once) rather than 400 (wait forever)",
      np.isclose(v_none[10], 100.0), f"program says {v_none[10]:.3f}")

with change(ARRIVALS=[16, 17, 18, 19, 20]):
    v_heavy, p_heavy = solve()
# Prediction: the queue can never empty, so each person left waiting costs 2/(1-0.95) = 40 forever.
# A dispatch (100) removes up to 15 people, so it pays off from 100/40 = 2.5 -> 3 people.
# (My first guess here was "15 people"; the reasoning above shows that was wrong.)
last_dispatch = int(np.where(p_heavy == DISPATCH)[0].max())
with change(ARRIVALS=[16, 17, 18, 19, 20], MAX_PEOPLE=300):
    last_dispatch_300 = int(np.where(solve()[1] == DISPATCH)[0].max())
check("C2. more arrivals than the shuttle can carry (16-20 per period): each person left costs 2/(1-0.95) = 40 forever, "
      "so dispatching (100) pays off from 100/40 -> 3 people, and the plan dispatches for every number from 3 up to 152",
      first_dispatch(p_heavy) == 3 and np.all(p_heavy[3:last_dispatch + 1] == DISPATCH),
      f"dispatch from 3 to {last_dispatch}")
check("C2b. ...the plan only waits again for the last ~48 places below the station limit, and that point moves with the limit "
      "(limit 200 -> 152, limit 300 -> 252): it is an effect of the cap, not of the arrivals",
      last_dispatch == 152 and last_dispatch_300 == 252, f"last dispatch at {last_dispatch} (limit 200) and {last_dispatch_300} (limit 300)")

# =====================================================================================
print("\n=== D. EXTREME PRICES ===")
with change(DISPATCH_COST=0):
    _, p_free = solve()
check("D1a. free shuttle: dispatch whenever somebody is waiting", np.all(p_free[1:] == DISPATCH),
      f"dispatches in {int(p_free[1:].sum())} of 200 states")
with change(WAIT_COST=0):
    v_nowait, p_nowait = solve()
check("D1b. free waiting: never dispatch (total cost 0)", np.all(p_nowait == WAIT) and np.allclose(v_nowait, 0))

with change(DISPATCH_COST=1_000_000):
    _, p_huge = solve()
check("D2. enormous shuttle cost (1,000,000): never dispatch, even with a full station", p_huge.sum() == 0)

# =====================================================================================
print("\n=== E. TIME AND DISCOUNTING ===")
enum = enumeration(costs, probs, periods=500)
one = enumeration(costs, probs, periods=1)[0]
two = enumeration(costs, probs, periods=2)[0]
s = np.arange(0, 101)
check("E1. very short horizon: 1 period left costs 2 x people (never dispatch), 2 periods left costs 3.9 x people + 5.7 (hand-worked)",
      np.allclose(one[s], 2 * s) and np.allclose(two[s], 3.9 * s + 5.7), "matches the hand calculation")

gaps = {}
for T in (100, 200, 500):
    gaps[T] = np.max(np.abs(enumeration(costs, probs, periods=T)[0] - values))
check("E2. 500 periods is long enough: the answer stops changing as the horizon grows (100 -> 200 -> 500)",
      gaps[100] > gaps[200] > gaps[500] and gaps[500] < 1e-6,
      f"difference from the no-end-date answer: {gaps[100]:.1e} / {gaps[200]:.1e} / {gaps[500]:.1e}")

# =====================================================================================
print("\n=== F. CHECKS ON EACH METHOD ===")
check("F1. enumeration base case: nothing is left to pay after the last period, and more periods left never costs less",
      np.all(enum[500] == 0) and np.all(enum[:-1] >= enum[1:] - 1e-9), "value after period 500 is 0 for every state")

starts = {"all 0": np.zeros(201), "all 1000": np.full(201, 1000.0), "random": rng.uniform(0, 5000, 201)}
diffs = {k: np.abs(value_iteration(costs, probs, start_values=v)[0] - values).max() for k, v in starts.items()}
check("F2. value iteration reaches the same answer from different starting guesses (all 0, all 1000, random)",
      max(diffs.values()) < 1e-6, "largest difference " + f"{max(diffs.values()):.1e}")

hist = []
policy_iteration(costs, probs, history=hist)
improves = all(np.all(hist[i + 1] <= hist[i] + 1e-9) for i in range(len(hist) - 1))
with change(DISPATCH_COST=0, WAIT_COST=0):
    c0, p0 = shuttle.build_tables()
    _, _, rounds_ties = shuttle.policy_iteration(c0, p0)       # every action ties
check("F3. policy iteration: every round improves (or keeps) the plan, and exact ties do not make it loop",
      improves and rounds_ties == 1, f"{len(hist)} rounds, costs never went up; all-ties problem stopped after {rounds_ties} round")

# =====================================================================================
print("\n=== G. OVERALL SANITY CHECKS ===")
pi_v, pi_p, _ = policy_iteration(costs, probs)
check("G1. enumeration, value iteration and policy iteration all give the same answer",
      np.allclose(enum[0], values, atol=1e-6) and np.allclose(pi_v, values, atol=1e-6) and np.array_equal(pi_p, policy),
      f"biggest gap {max(np.abs(enum[0] - values).max(), np.abs(pi_v - values).max()):.1e}")

_, p_alt, _ = policy_iteration(costs, probs, start_policy=np.ones(201, dtype=int))
check("G2. policy iteration starting from 'always dispatch' still finds the same best plan", np.array_equal(p_alt, policy))


def simulate(start, periods=400):
    people, total, discount = start, 0.0, 1.0
    for _ in range(periods):
        if policy[people] == DISPATCH:
            left, paid = max(people - CAPACITY, 0), DISPATCH_COST
        else:
            left, paid = people, 0
        total += discount * (paid + WAIT_COST * left)
        people = min(left + rng.choice(ARRIVALS), MAX_PEOPLE)
        discount *= GAMMA
    return total


runs = np.array([simulate(0) for _ in range(3000)])
margin = 3 * runs.std() / np.sqrt(len(runs))
check("G3. simulating the station 3000 times gives the same average cost the maths predicted",
      abs(runs.mean() - values[0]) < margin, f"simulated {runs.mean():.1f} (+/- {margin:.1f}) vs predicted {values[0]:.1f}")

# =====================================================================================
print("\n=== H. PROTECTIONS (bad inputs are stopped with a clear message) ===")


def raises(error, fn):
    """True if fn() stops with exactly this kind of error."""
    try:
        fn()
    except error:
        return True
    except Exception:
        return False
    return False


bad_problems = {"no arrivals listed": dict(ARRIVALS=[]), "negative arrivals": dict(ARRIVALS=[-1, 2]),
                "negative shuttle size": dict(CAPACITY=-1), "negative shuttle cost": dict(DISPATCH_COST=-5),
                "negative waiting cost": dict(WAIT_COST=-2), "station limit 0": dict(MAX_PEOPLE=0)}
rejected = []
for name, setting in bad_problems.items():
    with change(**setting):
        rejected.append(raises(ValueError, shuttle.build_tables))
check("H1. nonsense problem numbers are rejected with a clear message (no arrivals, negative arrivals/size/costs, limit 0)",
      all(rejected), f"{sum(rejected)} of {len(rejected)} rejected")

short_costs, short_probs = costs[:, :100], probs[:, :100, :100]
check("H1b. tables that do not match the problem, or whose chances do not add to 100%, are rejected",
      raises(ValueError, lambda: value_iteration(short_costs, short_probs)) and
      raises(ValueError, lambda: value_iteration(costs, broken)), "wrong size and damaged table both stopped")

rejected = []
for g in (1.0, 1.5, -0.1):
    with change(GAMMA=g):
        rejected.append(raises(ValueError, shuttle.build_tables))
with change(GAMMA=0.0):
    accepted_zero = not raises(ValueError, shuttle.build_tables)
check("H2. a discount of 1 or more (or below 0) is rejected, because 'forever' would have no finite answer; a discount of 0 is allowed",
      all(rejected) and accepted_zero, f"1.0, 1.5 and -0.1 rejected; 0.0 accepted")

check("H3. value iteration cannot loop forever: with a tolerance it can never meet, it stops with an error after max_rounds",
      raises(RuntimeError, lambda: value_iteration(costs, probs, tolerance=0, max_rounds=50)))

check("H4. policy iteration cannot loop forever: if it needs more than max_rounds (it needs 6, we allow 1), it stops with an error",
      raises(RuntimeError, lambda: policy_iteration(costs, probs, max_rounds=1)))

check("H5. enumeration needs a whole number of periods, at least 1 (0, -3 and 2.5 are rejected)",
      all(raises(ValueError, lambda T=T: enumeration(costs, probs, periods=T)) for T in (0, -3, 2.5)))

nan_costs = costs.copy(); nan_costs[WAIT, 5] = np.nan
inf_costs = costs.copy(); inf_costs[WAIT, 7] = np.inf     # (an infinite cost on an action nobody uses is not an error)
check("H6. a NaN or infinite number stops the calculation instead of quietly spreading (all three methods)",
      raises(FloatingPointError, lambda: enumeration(nan_costs, probs, 5)) and
      raises(FloatingPointError, lambda: value_iteration(nan_costs, probs)) and
      raises(FloatingPointError, lambda: policy_iteration(inf_costs, probs)), "NaN and infinity both caught")

cost_w, next_w = cost_and_next_state_probs(10, WAIT)
check("H7. waiting leaves everyone behind: with 10 waiting we pay 20 and next period have 11 to 15 people",
      cost_w == 20 and np.allclose(next_w[11:16], 0.2) and np.isclose(next_w.sum(), 1), f"cost = {cost_w}")

slow = np.zeros(201)                       # evaluate the best plan the slow way: repeat 'cost + discounted future' 3000 times
P_plan, c_plan = probs[policy, np.arange(201)], costs[policy, np.arange(201)]
for _ in range(3000):
    slow = c_plan + GAMMA * P_plan @ slow
check("H8. the exact equation solving used by policy iteration agrees with a slow repeat-until-settled calculation",
      np.allclose(evaluate_policy(costs, probs, policy), slow, atol=1e-6),
      f"largest gap {np.abs(evaluate_policy(costs, probs, policy) - slow).max():.1e}")

# =====================================================================================
print("\n=== I. HARDER CHECKS (other problems, to see the methods really are right) ===")
import itertools


def solve_all(table_args):
    """(value iteration, policy iteration) on the current problem."""
    c, p = shuttle.build_tables()
    v1, a1, r1 = shuttle.value_iteration(c, p, max_rounds=1_000_000)
    v2, a2, r2 = shuttle.policy_iteration(c, p)
    return c, p, v1, a1, r1, v2, a2, r2


# I1. brute force: on a tiny problem try every possible plan, the best one must match the methods
with change(CAPACITY=3, MAX_PEOPLE=8, ARRIVALS=[1, 2, 3], GAMMA=0.9):
    c, p, v1, a1, r1, v2, a2, r2 = solve_all(None)
    best = None
    for bits in itertools.product([0, 1], repeat=9):
        v = shuttle.evaluate_policy(c, p, np.array(bits))
        best = v if best is None else np.minimum(best, v)
    gap_vi, gap_pi = np.abs(best - v1).max(), np.abs(best - v2).max()
check("I1. brute force on a tiny problem: the cheapest of all 512 possible plans costs the same as the plan the methods find",
      gap_vi < 1e-8 and gap_pi < 1e-8, f"largest gap {gap_vi:.1e} (value iteration), {gap_pi:.1e} (policy iteration)")

# I2. many random problems
rrng = np.random.default_rng(0)
worst_vp, worst_enum, disagree, clear_total = 0.0, 0.0, 0, 0
for _ in range(300):
    cap, maxp = int(rrng.integers(1, 12)), int(rrng.integers(12, 45))
    arr = sorted(rrng.choice(range(0, 9), size=int(rrng.integers(1, 5)), replace=False).tolist())
    g, cf, ch = float(rrng.uniform(0.3, 0.97)), float(rrng.uniform(0, 300)), float(rrng.uniform(0, 8))
    with change(CAPACITY=cap, MAX_PEOPLE=maxp, ARRIVALS=arr, GAMMA=g, DISPATCH_COST=cf, WAIT_COST=ch):
        c, p, v1, a1, r1, v2, a2, r2 = solve_all(None)
        enum0 = shuttle.enumeration(c, p, 400)[0]
    scale = max(1.0, np.abs(v1).max())
    worst_vp = max(worst_vp, np.abs(v1 - v2).max() / scale)
    worst_enum = max(worst_enum, np.abs(enum0 - v1).max() / scale)
    qw, qd = c[0] + g * p[0] @ v1, c[1] + g * p[1] @ v1
    clear = np.abs(qw - qd) > 1e-6                       # ignore situations where both actions cost (almost) the same
    disagree += int((a1[clear] != a2[clear]).sum()); clear_total += int(clear.sum())
check("I2. 300 random problems (random shuttle size, station size, arrivals, discount, prices): all three methods agree",
      worst_vp < 1e-8 and worst_enum < 1e-4 and disagree == 0,
      f"value vs policy iteration within {worst_vp:.0e}; enumeration (400 periods) within {worst_enum:.0e}; "
      f"{disagree} action disagreements in {clear_total} clear-cut situations")

# I3. a discount close to 1
rows = []
for g in (0.9, 0.99, 0.999):
    with change(GAMMA=g):
        c, p, v1, a1, r1, v2, a2, r2 = solve_all(None)
    rows.append((g, r1, r2, np.array_equal(a1, a2), np.abs(v1 - v2).max()))
check("I3. a discount close to 1 (0.9, 0.99, 0.999): value iteration needs far more rounds, policy iteration hardly more, and both find the same plan",
      all(r[3] for r in rows) and rows[0][1] < rows[1][1] < rows[2][1] and rows[2][2] <= 10 and all(r[4] < 1e-6 for r in rows),
      "; ".join(f"{g}: {r1} rounds vs {r2}" for g, r1, r2, _, _ in rows))

# I4. value iteration stops with an error no bigger than theory allows
bound = 1e-10 * GAMMA / (1 - GAMMA)
real_gap = np.abs(values - evaluate_policy(costs, probs, policy)).max()
check("I4. the leftover error of value iteration is within the bound the stopping rule guarantees (tolerance x discount / (1 - discount))",
      real_gap <= bound, f"real error {real_gap:.2e}, bound {bound:.2e}")

# I5. a shuttle bigger than the whole station
with change(CAPACITY=250):
    c, p, v1, a1, r1, v2, a2, r2 = solve_all(None)
    empties = np.allclose(c[1], DISPATCH_COST) and np.all(np.argmax(p[1], axis=1) <= max(ARRIVALS))
check("I5. a shuttle bigger than the station (250 against 200): every dispatch empties the station, and the methods still agree",
      empties and np.array_equal(a1, a2) and np.abs(v1 - v2).max() < 1e-8,
      f"dispatch from {first_dispatch(a1)} people; every dispatch costs exactly {DISPATCH_COST}")

print(f"\n{sum(results)} of {len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)

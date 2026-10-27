"""Question 1 checks (edge cases + "does it behave as predicted?").   Run: python test_q1.py
Each check prints PASS/FAIL and the evidence."""
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
    c, p = shuttle.build_tables()
    values, policy, _ = shuttle.value_iteration(c, p)
    return values, policy


costs, probs = build_tables()
values, policy, _ = value_iteration(costs, probs)

print("--- EDGE CASES ---")
# 1. Every state/action must lead somewhere: the chances of the next state add up to 1.
check("1. next-state chances add up to 100% in every situation (including a full station)",
      np.allclose(probs.sum(axis=2), 1), f"largest error {np.abs(probs.sum(axis=2) - 1).max():.1e}")

# 2. A full station can't grow: extra arrivals are turned away.
check("2. waiting with a full station (200) keeps it at 200", np.isclose(probs[WAIT, 200, 200], 1.0))

# 3. Sending a shuttle with fewer than 15 people just empties the station (no negative numbers).
cost, nxt = cost_and_next_state_probs(4, DISPATCH)
check("3. dispatching with only 4 people costs just 100 (nobody left to pay for) and is never negative",
      cost == 100 and np.isclose(nxt[1:6].sum(), 1.0), f"cost = {cost}")

# 4. Nobody waiting: dispatching would waste 100.
check("4. with 0 people waiting, the best action is to wait", policy[0] == WAIT)

# 5. A full station: it is best to dispatch.
check("5. with a full station (200), the best action is to dispatch", policy[200] == DISPATCH)

# 6. The 200 limit is an assumption of ours. If the limit were 300, would the answer change?
with change(MAX_PEOPLE=300):
    v300, p300 = solve()
check("6. raising the station limit from 200 to 300 does not change the best actions (up to 150 people)",
      np.array_equal(p300[:151], policy[:151]), f"largest cost difference {np.abs(v300[:151] - values[:151]).max():.1e}")

print("\n--- DOES IT BEHAVE AS PREDICTED? ---")
# 7. Hand calculation: no one ever arrives, 10 people waiting.
#    Option 1: wait forever -> pay 2*10 = 20 every period = 20/(1-0.95) = 400.
#    Option 2: dispatch once -> pay 100 and then nothing. Cheapest = 100.
with change(ARRIVALS=[0], MAX_PEOPLE=30):
    v_none, _ = solve()
check("7. hand calculation (nobody arrives, 10 waiting): cheapest is 100 (dispatch once) vs 400 (wait forever)",
      np.isclose(v_none[10], 100.0), f"program says {v_none[10]:.3f}")

# 8. Make dispatching free -> should always dispatch. Make waiting free -> should never dispatch.
with change(DISPATCH_COST=0):
    _, p_free = solve()
check("8a. if the shuttle were free, it should dispatch whenever someone is waiting", np.all(p_free[1:] == DISPATCH),
      f"dispatches in {int(p_free[1:].sum())} of 200 states")
with change(WAIT_COST=0):
    v_nowait, p_nowait = solve()
check("8b. if waiting cost nothing, it should never dispatch (total cost 0)", np.all(p_nowait == WAIT) and np.allclose(v_nowait, 0))

# 9. A bigger shuttle should make us wait longer before sending it.
cutoffs = []
for k in (5, 15, 30):
    with change(CAPACITY=k):
        cutoffs.append(int(np.argmax(solve()[1] == DISPATCH)))
check("9. a bigger shuttle means waiting for more people before dispatching", cutoffs[0] < cutoffs[1] < cutoffs[2],
      f"shuttle size 5 / 15 / 30 -> dispatch at {cutoffs[0]} / {cutoffs[1]} / {cutoffs[2]} people")

# 10. Independent check: actually simulate the station with the best rule.
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
check("10. simulating the station 3000 times gives the same average cost the maths predicted",
      abs(runs.mean() - values[0]) < margin, f"simulated {runs.mean():.1f} (+/- {margin:.1f}) vs predicted {values[0]:.1f}")

# 11. The three methods must agree.
enum_v = enumeration(costs, probs, 500)[0]
pi_v, pi_p, _ = policy_iteration(costs, probs)
check("11. enumeration, value iteration and policy iteration all give the same answer",
      np.allclose(enum_v, values, atol=1e-6) and np.allclose(pi_v, values, atol=1e-6) and np.array_equal(pi_p, policy),
      f"biggest gap {max(np.abs(enum_v - values).max(), np.abs(pi_v - values).max()):.1e}")

# 12. Policy iteration should reach the same answer even from a bad starting guess.
_, p_alt, _ = policy_iteration(costs, probs, start_policy=np.ones(201, dtype=int))
check("12. policy iteration starting from 'always dispatch' still finds the same best rule", np.array_equal(p_alt, policy))

print(f"\n{sum(results)} of {len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)

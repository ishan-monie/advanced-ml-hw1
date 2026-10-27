"""Question 1: the shuttle problem, solved three ways.

The setup (all numbers come from the assignment):
  - People wait at a station. Each period we choose: WAIT (0) or DISPATCH a shuttle (1).
  - A shuttle costs 100 and takes up to 15 people.
  - Every person still waiting after the decision costs 2.
  - Then 1-5 new people arrive (equally likely).
  - The station holds at most 200 people.
  - Future costs are discounted by 0.95 per period.

We track "cost", so LOWER is better. The "value" of a state = expected total discounted cost
from that state onward, if we act as cheaply as possible.
"""
import numpy as np

CAPACITY = 15         # K
DISPATCH_COST = 100   # cf
WAIT_COST = 2         # ch, per person left waiting
MAX_PEOPLE = 200
GAMMA = 0.95
ARRIVALS = [1, 2, 3, 4, 5]     # each has probability 1/5

WAIT, DISPATCH = 0, 1


# ---------------------------------------------------------------- protections (inputs, safety stops)
def check_inputs(costs=None, probs=None):
    """N1/N2: stop with a clear message if the problem's numbers make no sense.
    If the tables are given, also check they match the problem and that every row of chances adds to 100%."""
    if len(ARRIVALS) == 0 or min(ARRIVALS) < 0:
        raise ValueError("ARRIVALS must be a non-empty list of numbers that are 0 or more")
    if CAPACITY < 0 or DISPATCH_COST < 0 or WAIT_COST < 0:
        raise ValueError("CAPACITY, DISPATCH_COST and WAIT_COST must not be negative")
    if MAX_PEOPLE < 1:
        raise ValueError("MAX_PEOPLE must be at least 1")
    if not (0 <= GAMMA < 1):
        raise ValueError("GAMMA (the discount) must be at least 0 and below 1; with 1 the 'forever' problem has no finite answer")
    if costs is not None and probs is not None:
        n = MAX_PEOPLE + 1
        if costs.shape != (2, n) or probs.shape != (2, n, n):
            raise ValueError(f"tables do not match MAX_PEOPLE: expected 2 x {n} costs, got {costs.shape}")
        if not np.allclose(probs.sum(axis=2), 1):
            raise ValueError("the chances of the next situation do not add up to 100% in every situation")


def check_finite(values, where):
    """N6: stop if a value has turned into NaN or infinity, so bad numbers cannot spread quietly."""
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"{where}: a value became NaN or infinite")


def cost_and_next_state_probs(people, action):
    """For a given number of people and action, return:
         cost            - what we pay this period
         next_probs      - array: next_probs[n] = chance of having n people next period
    """
    if action == DISPATCH:
        left = max(people - CAPACITY, 0)      # the shuttle takes up to 15 people
        cost = DISPATCH_COST + WAIT_COST * left
    else:
        left = people
        cost = WAIT_COST * left
    next_probs = np.zeros(MAX_PEOPLE + 1)
    for a in ARRIVALS:
        # if the station would go over 200, the extra people are simply turned away
        next_probs[min(left + a, MAX_PEOPLE)] += 1 / len(ARRIVALS)
    return cost, next_probs


def build_tables():
    """Precompute costs[action][people] and probs[action][people][next_people]."""
    check_inputs()
    n = MAX_PEOPLE + 1
    costs = np.zeros((2, n))
    probs = np.zeros((2, n, n))
    for action in (WAIT, DISPATCH):
        for people in range(n):
            costs[action, people], probs[action, people] = cost_and_next_state_probs(people, action)
    check_inputs(costs, probs)
    return costs, probs


# ---------------------------------------------------------------- (a) enumeration, 500 periods
def enumeration(costs, probs, periods=500):
    """Work backward from the last period. At the very end nothing is left to pay (value 0).
    One step earlier, for every number of people we try both actions and keep the cheaper one,
    and so on back to period 0. Returns the value table: values[t][people]."""
    check_inputs(costs, probs)
    if int(periods) != periods or periods < 1:             # N5
        raise ValueError("periods must be a whole number, at least 1")
    values = np.zeros((periods + 1, MAX_PEOPLE + 1))
    for t in range(periods - 1, -1, -1):
        cost_wait = costs[WAIT] + GAMMA * probs[WAIT] @ values[t + 1]
        cost_dispatch = costs[DISPATCH] + GAMMA * probs[DISPATCH] @ values[t + 1]
        values[t] = np.minimum(cost_wait, cost_dispatch)
    check_finite(values, "enumeration")
    return values


# ---------------------------------------------------------------- (b) value iteration
def value_iteration(costs, probs, tolerance=1e-10, start_values=None, max_rounds=100_000):
    """Same idea with no end date: keep re-applying the 'try both, keep the cheaper' step
    until the values stop changing. Returns (values, best_action_for_each_state, rounds).
    start_values is only for testing: the starting guess (default: every state worth 0).
    max_rounds (N3) stops an endless loop: if the values have not settled by then, we raise an error."""
    check_inputs(costs, probs)
    values = np.zeros(MAX_PEOPLE + 1) if start_values is None else np.array(start_values, dtype=float)
    rounds = 0
    while True:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(f"value iteration did not settle within {max_rounds} rounds")
        q_wait = costs[WAIT] + GAMMA * probs[WAIT] @ values
        q_dispatch = costs[DISPATCH] + GAMMA * probs[DISPATCH] @ values
        new_values = np.minimum(q_wait, q_dispatch)
        check_finite(new_values, "value iteration")
        done = np.max(np.abs(new_values - values)) < tolerance
        values = new_values
        if done:
            break
    best_action = (q_dispatch < q_wait).astype(int)      # 1 = dispatch is strictly cheaper
    return values, best_action, rounds


# ---------------------------------------------------------------- (c) policy iteration
def evaluate_policy(costs, probs, policy):
    """Exact cost of following a fixed policy forever (solves a system of linear equations)."""
    n = MAX_PEOPLE + 1
    states = np.arange(n)
    P = probs[policy, states]          # row i = next-state chances when following the policy
    c = costs[policy, states]
    values = np.linalg.solve(np.eye(n) - GAMMA * P, c)
    check_finite(values, "policy evaluation")
    return values


def policy_iteration(costs, probs, start_policy=None, history=None, max_rounds=1000):
    """Start with a guess, measure how good it is, improve it, repeat until nothing changes.
    If a list is passed as `history`, the plan's values after each round are appended to it (for testing).
    max_rounds (N4) stops an endless loop: if the plan has not settled by then, we raise an error."""
    check_inputs(costs, probs)
    n = MAX_PEOPLE + 1
    policy = np.zeros(n, dtype=int) if start_policy is None else start_policy.copy()   # default: never dispatch
    rounds = 0
    while True:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(f"policy iteration did not settle within {max_rounds} rounds")
        values = evaluate_policy(costs, probs, policy)
        if history is not None:
            history.append(values)
        q_wait = costs[WAIT] + GAMMA * probs[WAIT] @ values
        q_dispatch = costs[DISPATCH] + GAMMA * probs[DISPATCH] @ values
        better = np.where(q_dispatch < q_wait - 1e-9, DISPATCH,
                          np.where(q_wait < q_dispatch - 1e-9, WAIT, policy))   # on a tie keep what we had
        if np.array_equal(better, policy):
            return values, policy, rounds
        policy = better

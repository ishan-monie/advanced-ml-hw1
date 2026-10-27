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
    n = MAX_PEOPLE + 1
    costs = np.zeros((2, n))
    probs = np.zeros((2, n, n))
    for action in (WAIT, DISPATCH):
        for people in range(n):
            costs[action, people], probs[action, people] = cost_and_next_state_probs(people, action)
    return costs, probs


# ---------------------------------------------------------------- (a) enumeration, 500 periods
def enumeration(costs, probs, periods=500):
    """Work backward from the last period. At the very end nothing is left to pay (value 0).
    One step earlier, for every number of people we try both actions and keep the cheaper one,
    and so on back to period 0. Returns the value table: values[t][people]."""
    values = np.zeros((periods + 1, MAX_PEOPLE + 1))
    for t in range(periods - 1, -1, -1):
        cost_wait = costs[WAIT] + GAMMA * probs[WAIT] @ values[t + 1]
        cost_dispatch = costs[DISPATCH] + GAMMA * probs[DISPATCH] @ values[t + 1]
        values[t] = np.minimum(cost_wait, cost_dispatch)
    return values


# ---------------------------------------------------------------- (b) value iteration
def value_iteration(costs, probs, tolerance=1e-10):
    """Same idea with no end date: keep re-applying the 'try both, keep the cheaper' step
    until the values stop changing. Returns (values, best_action_for_each_state, rounds)."""
    values = np.zeros(MAX_PEOPLE + 1)
    rounds = 0
    while True:
        rounds += 1
        q_wait = costs[WAIT] + GAMMA * probs[WAIT] @ values
        q_dispatch = costs[DISPATCH] + GAMMA * probs[DISPATCH] @ values
        new_values = np.minimum(q_wait, q_dispatch)
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
    return np.linalg.solve(np.eye(n) - GAMMA * P, c)


def policy_iteration(costs, probs, start_policy=None):
    """Start with a guess, measure how good it is, improve it, repeat until nothing changes."""
    n = MAX_PEOPLE + 1
    policy = np.zeros(n, dtype=int) if start_policy is None else start_policy.copy()   # default: never dispatch
    rounds = 0
    while True:
        rounds += 1
        values = evaluate_policy(costs, probs, policy)
        q_wait = costs[WAIT] + GAMMA * probs[WAIT] @ values
        q_dispatch = costs[DISPATCH] + GAMMA * probs[DISPATCH] @ values
        better = np.where(q_dispatch < q_wait - 1e-9, DISPATCH,
                          np.where(q_wait < q_dispatch - 1e-9, WAIT, policy))   # on a tie keep what we had
        if np.array_equal(better, policy):
            return values, policy, rounds
        policy = better

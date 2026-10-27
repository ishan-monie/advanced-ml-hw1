"""Makes the three Question 1 plots and prints the key numbers.   Run: python run_q1.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shuttle import *

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out, exist_ok=True)
costs, probs = build_tables()
people = np.arange(MAX_PEOPLE + 1)

# (a) enumeration over 500 periods
enum_values = enumeration(costs, probs, periods=500)
plt.figure(figsize=(6, 4))
plt.plot(people, enum_values[0])
plt.xlabel("people waiting at the station"); plt.ylabel("expected total cost")
plt.title("(a) Enumeration, 500 periods: best value at time 0")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{out}/a_enumeration.png", dpi=150); plt.close()

# (b) value iteration
vi_values, vi_policy, vi_rounds = value_iteration(costs, probs)
plt.figure(figsize=(6, 4))
plt.plot(people, vi_values, color="C1")
plt.xlabel("people waiting at the station"); plt.ylabel("expected total cost")
plt.title("(b) Value iteration (no end date): best value")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{out}/b_value_iteration.png", dpi=150); plt.close()

# (c) policy iteration
pi_values, pi_policy, pi_rounds = policy_iteration(costs, probs)
plt.figure(figsize=(6, 4))
plt.step(people, pi_policy, where="post", color="C2")
plt.yticks([0, 1], ["wait", "dispatch"])
plt.xlabel("people waiting at the station"); plt.ylabel("best action")
plt.title("(c) Policy iteration: best action")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{out}/c_policy_iteration.png", dpi=150); plt.close()

cutoff = int(np.argmax(pi_policy == DISPATCH))
print(f"Value iteration took {vi_rounds} rounds; policy iteration took {pi_rounds} rounds")
print(f"Best rule: dispatch when {cutoff} or more people are waiting (checked: everything above also dispatches = "
      f"{bool(np.all(pi_policy[cutoff:] == DISPATCH))})")
print(f"Largest difference between enumeration and value iteration:  {np.max(np.abs(enum_values[0] - vi_values)):.1e}")
print(f"Largest difference between policy iteration and value iteration: {np.max(np.abs(pi_values - vi_values)):.1e}")
print(f"Same best action from all methods: {np.array_equal(vi_policy, pi_policy)}")
print(f"Expected cost starting with 0 / 15 / 200 people waiting: {vi_values[0]:.1f} / {vi_values[15]:.1f} / {vi_values[200]:.1f}")

# -*- coding: utf-8 -*-
"""
ADPO replication on a small pairwise-preference benchmark.

Reference: Ji, He, Gu (2024) "RLHF with Active Queries" (arXiv:2402.09401)
Paper code: https://github.com/jkx19/ActiveQuery  (scripts/trainer.py, lines 1065-1071)

Paper's ADPO rule (ported verbatim):
    if |chosen_reward - rejected_reward| > gamma:
        # confident -> skip the oracle query; use pseudo-label = sign(margin)
    else:
        # uncertain -> query the oracle for the true preference label

Benchmark: synthetic Bradley-Terry preferences over linear rewards -- a
different benchmark from the paper's LLM experiments (ARC/TruthfulQA/HellaSwag).
We make the oracle noisy (reward scale < 1) so the paper's claim is testable:
ADPO's confident pseudo-labels can beat noisy oracle labels near the boundary,
and ADPO gets many more total updates per oracle query than DPO, so ADPO
should reach a higher plateau with far fewer queries.

Both methods are compared at equal oracle-query budgets k in [0, 60], matching
the x-axis range in the paper's Figure 2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ====== Settings ======
D = 16                   # feature dim
N_SEEDS = 30
N_TEST = 3000
LR = 0.1
GAMMA = 1.3              # paper default
REWARD_SCALE = 0.5       # shrinks |r1 - r2| so the BT oracle is noisier
MAX_STEPS = 4000         # hard cap on per-run training steps
QUERY_BUDGETS = np.arange(0, 62, 2)   # x-axis: oracle queries k in {0,2,...,60}


# ====== Environment ======
def sample_x(rng, n=None):
  if n is None:
    return rng.standard_normal(D)
  return rng.standard_normal((n, D))


def bt_label(r1, r2, rng):
  p = 1.0 / (1.0 + np.exp(-(r1 - r2)))
  return 1 if rng.random() < p else -1


def make_test_set(theta_star, rng):
  x1 = sample_x(rng, N_TEST)
  x2 = sample_x(rng, N_TEST)
  r1 = x1 @ theta_star
  r2 = x2 @ theta_star
  y = np.where(r1 > r2, 1, -1)  # noiseless oracle-optimal label for evaluation
  return (torch.tensor(x1, dtype=torch.float32),
          torch.tensor(x2, dtype=torch.float32),
          torch.tensor(y, dtype=torch.float32))


# ====== Model ======
class LinearReward(nn.Module):
  def __init__(self, d):
    super().__init__()
    self.w = nn.Linear(d, 1, bias=False)

  def forward(self, x):
    return self.w(x).squeeze(-1)


@torch.no_grad()
def test_accuracy(model, test_set):
  x1, x2, y = test_set
  pred = torch.where(model(x1) > model(x2), 1.0, -1.0)
  return (pred == y).float().mean().item()


# ====== Single run: returns accuracy[k] for every k in QUERY_BUDGETS ======
def run(mode, seed):
  """
  mode in {'dpo', 'adpo', 'adpo_no_pl'}.
  For each k in QUERY_BUDGETS we record the test accuracy reached by the
  method after it has used exactly k oracle queries.
  """
  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)

  theta_star = rng.standard_normal(D) * REWARD_SCALE
  test_set = make_test_set(theta_star, rng)

  model = LinearReward(D)
  opt = optim.SGD(model.parameters(), lr=LR)

  k_target_idx = 0
  out = np.full(len(QUERY_BUDGETS), np.nan)

  # Record acc at k=0 if it is in the grid.
  if QUERY_BUDGETS[0] == 0:
    out[0] = test_accuracy(model, test_set)
    k_target_idx = 1

  q = 0
  for _ in range(MAX_STEPS):
    if k_target_idx >= len(QUERY_BUDGETS):
      break

    x1 = sample_x(rng)
    x2 = sample_x(rng)
    r1 = float(theta_star @ x1)
    r2 = float(theta_star @ x2)

    x1_t = torch.tensor(x1, dtype=torch.float32)
    x2_t = torch.tensor(x2, dtype=torch.float32)
    s1 = model(x1_t)
    s2 = model(x2_t)
    margin = (s1 - s2).item()

    if mode == 'dpo':
      label = bt_label(r1, r2, rng)
      q += 1
    elif mode == 'adpo':
      if abs(margin) > GAMMA:
        label = 1 if margin > 0 else -1
      else:
        label = bt_label(r1, r2, rng)
        q += 1
    elif mode == 'adpo_no_pl':
      if abs(margin) > GAMMA:
        continue
      label = bt_label(r1, r2, rng)
      q += 1
    else:
      raise ValueError(mode)

    y = torch.tensor(label, dtype=torch.float32)
    loss = -torch.nn.functional.logsigmoid(y * (s1 - s2))
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Snapshot every time q hits the next target budget.
    while k_target_idx < len(QUERY_BUDGETS) and q >= QUERY_BUDGETS[k_target_idx]:
      out[k_target_idx] = test_accuracy(model, test_set)
      k_target_idx += 1

  # If the method ran out of queries (e.g. ADPO never hits high k), carry
  # forward the last recorded accuracy — meaning "no more queries available,
  # model is done training".
  last = out[~np.isnan(out)][-1] if np.any(~np.isnan(out)) else 0.5
  out[np.isnan(out)] = last
  return out


def aggregate(mode):
  curves = np.stack([run(mode, seed) for seed in range(N_SEEDS)], axis=0)
  return curves.mean(axis=0), curves.std(axis=0) / np.sqrt(N_SEEDS)


def main():
  modes = [('dpo',        'DPO',                              'black', 'v',  '-'),
           ('adpo',       'ADPO (pseudo-label, γ=%.1f)' % GAMMA, 'red',   'o',  '-'),
           ('adpo_no_pl', 'Active only (no pseudo-label)',    'gray',  's',  '--')]

  plt.figure(figsize=(7, 4.5))
  for mode, label, color, marker, ls in modes:
    mean, se = aggregate(mode)
    plt.plot(QUERY_BUDGETS, mean * 100, marker=marker, ms=4, color=color,
             linestyle=ls, label=label, linewidth=1.8)
    plt.fill_between(QUERY_BUDGETS, (mean - se) * 100, (mean + se) * 100,
                     color=color, alpha=0.12)

  plt.xlabel('Number of Queries (k)')
  plt.ylabel('Accuracy (%)')
  plt.title('DPO vs ADPO on synthetic BT benchmark\n'
            '(d=%d, noisy oracle, %d seeds, γ=%.1f)' % (D, N_SEEDS, GAMMA))
  plt.legend(loc='lower right')
  plt.grid(alpha=0.3)
  plt.tight_layout()
  out = 'adpo_vs_dpo_accuracy_vs_queries.png'
  plt.savefig(out, dpi=150)
  print('Saved', out)
  plt.show()


if __name__ == '__main__':
  main()

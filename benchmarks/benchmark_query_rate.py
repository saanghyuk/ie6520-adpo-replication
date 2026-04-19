# -*- coding: utf-8 -*-
"""
Benchmark #4 -- query-rate over training time.

Records the fraction of pairs that ADPO actually queries the oracle for, in
a sliding window of training steps. Validates the burn-in -> steady-state
mechanism predicted by the theory: at the start the design matrix is small
so |phi^1 - phi^2|_{Sigma^{-1}} > Gamma almost always, and the algorithm
queries on essentially every pair. As Sigma grows the bonus shrinks, the
threshold check fails on most pairs, and the query rate drops.

Output: bench_query_rate.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

OUT = os.path.join(os.path.dirname(__file__), 'bench_query_rate.png')
import matplotlib.pyplot as plt

D = 16
N_SEEDS = 20
LR = 0.1
GAMMA = 1.3
REWARD_SCALE = 0.5
N_STEPS = 3000
WINDOW = 100


def sample_x(rng):
  return rng.standard_normal(D)


def bt_label(r1, r2, rng):
  p = 1.0 / (1.0 + np.exp(-(r1 - r2)))
  return 1 if rng.random() < p else -1


class LinearReward(nn.Module):
  def __init__(self, d):
    super().__init__()
    self.w = nn.Linear(d, 1, bias=False)

  def forward(self, x):
    return self.w(x).squeeze(-1)


def run_adpo(seed):
  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)
  theta_star = rng.standard_normal(D) * REWARD_SCALE
  model = LinearReward(D)
  opt = optim.SGD(model.parameters(), lr=LR)

  queried = np.zeros(N_STEPS, dtype=np.int8)
  for t in range(N_STEPS):
    x1, x2 = sample_x(rng), sample_x(rng)
    r1 = float(theta_star @ x1)
    r2 = float(theta_star @ x2)
    x1_t = torch.tensor(x1, dtype=torch.float32)
    x2_t = torch.tensor(x2, dtype=torch.float32)
    s1, s2 = model(x1_t), model(x2_t)
    margin = (s1 - s2).item()

    if abs(margin) > GAMMA:
      label = 1 if margin > 0 else -1
    else:
      label = bt_label(r1, r2, rng)
      queried[t] = 1

    y = torch.tensor(label, dtype=torch.float32)
    loss = -torch.nn.functional.logsigmoid(y * (s1 - s2))
    opt.zero_grad(); loss.backward(); opt.step()

  return queried


def rolling_mean(x, w):
  c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
  return (c[w:] - c[:-w]) / w


def main():
  print(f'Running {N_SEEDS} seeds, {N_STEPS} steps each...')
  rates = np.stack([rolling_mean(run_adpo(s), WINDOW) for s in range(N_SEEDS)])
  mean = rates.mean(0) * 100
  se = rates.std(0) / np.sqrt(N_SEEDS) * 100
  x = np.arange(WINDOW, N_STEPS + 1)

  plt.figure(figsize=(7, 4.5))
  plt.plot(x, mean, color='red', linewidth=2, label='ADPO ($\\gamma=1.3$)')
  plt.fill_between(x, mean - se, mean + se, color='red', alpha=0.18)
  plt.axhline(100, color='black', linestyle='--', linewidth=1.2,
              label='DPO (queries every step)')
  plt.xlabel('Training step $t$')
  plt.ylabel(f'Query rate (%) over rolling window of {WINDOW} steps')
  plt.title(f'ADPO query rate over training\n'
            f'($d={D}$, BT noise scale {REWARD_SCALE}, {N_SEEDS} seeds)')
  plt.ylim(-5, 110)
  plt.legend(loc='center right')
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(OUT, dpi=150)
  print('Saved bench_query_rate.png')
  cum_q = rates.mean(0).cumsum() / np.arange(1, len(mean) + 1)
  print(f'Cumulative query fraction at end: {cum_q[-1] * 100:.1f}%')


if __name__ == '__main__':
  main()

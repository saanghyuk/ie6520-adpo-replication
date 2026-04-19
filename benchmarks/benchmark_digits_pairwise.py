# -*- coding: utf-8 -*-
"""
Benchmark #2 — real-data pairwise preference on sklearn load_digits (8x8 digits).

Setting (ARC-like, simpler): given two digit images, the "preferred" one is
the one whose digit value is larger (0..9). Features are 64-d flattened
pixel vectors. The oracle is corrupted by label-flip noise (p=20%) so
there's a real noise ceiling for DPO to hit.

This tests whether the paper's mechanism generalizes beyond synthetic
Gaussian features to a real, structured feature distribution.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

OUT = os.path.join(os.path.dirname(__file__), 'bench_digits_pairwise.png')

N_SEEDS = 20
LR = 0.05
GAMMA = 1.3
FLIP_NOISE = 0.2
MAX_STEPS = 30000
QUERY_BUDGETS = np.arange(0, 520, 20)

# Load once — 1797 samples, 64 features, labels 0..9.
_DIGITS = load_digits()
_X_ALL = _DIGITS.data.astype(np.float32)
_X_ALL = (_X_ALL - _X_ALL.mean(0)) / (_X_ALL.std(0) + 1e-6)
_Y_ALL = _DIGITS.target.astype(np.int64)
D = _X_ALL.shape[1]


def make_split(rng):
  n = _X_ALL.shape[0]
  idx = rng.permutation(n)
  split = int(0.7 * n)
  return idx[:split], idx[split:]


def sample_pair(rng, pool_idx):
  i, j = rng.choice(pool_idx, size=2, replace=False)
  return _X_ALL[i], _X_ALL[j], _Y_ALL[i], _Y_ALL[j]


def oracle_label(yi, yj, rng):
  """+1 if image i is preferred (larger digit), with flip noise."""
  true = 1 if yi > yj else (-1 if yi < yj else 0)
  if true == 0:
    return 1 if rng.random() < 0.5 else -1  # tie -> random
  if rng.random() < FLIP_NOISE:
    true = -true
  return true


@torch.no_grad()
def test_accuracy(model, test_pairs):
  x1, x2, y = test_pairs
  pred = torch.where(model(x1) > model(x2), 1.0, -1.0)
  return (pred == y).float().mean().item()


def make_test_set(test_idx, rng, n_pairs=3000):
  i = rng.choice(test_idx, n_pairs)
  j = rng.choice(test_idx, n_pairs)
  x1 = torch.tensor(_X_ALL[i], dtype=torch.float32)
  x2 = torch.tensor(_X_ALL[j], dtype=torch.float32)
  yi, yj = _Y_ALL[i], _Y_ALL[j]
  y = np.where(yi > yj, 1, np.where(yi < yj, -1, 0)).astype(np.float32)
  return x1, x2, torch.tensor(y)


class LinearReward(nn.Module):
  def __init__(self, d):
    super().__init__()
    self.w = nn.Linear(d, 1, bias=False)

  def forward(self, x):
    return self.w(x).squeeze(-1)


def run(mode, seed):
  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)
  train_idx, test_idx = make_split(rng)
  test_pairs = make_test_set(test_idx, rng)
  model = LinearReward(D)
  opt = optim.SGD(model.parameters(), lr=LR)

  k_target_idx = 0
  out = np.full(len(QUERY_BUDGETS), np.nan)
  if QUERY_BUDGETS[0] == 0:
    out[0] = test_accuracy(model, test_pairs); k_target_idx = 1

  q = 0
  for _ in range(MAX_STEPS):
    if k_target_idx >= len(QUERY_BUDGETS):
      break
    xi, xj, yi, yj = sample_pair(rng, train_idx)
    xi_t = torch.tensor(xi, dtype=torch.float32)
    xj_t = torch.tensor(xj, dtype=torch.float32)
    s1, s2 = model(xi_t), model(xj_t)
    margin = (s1 - s2).item()

    if mode == 'dpo':
      label = oracle_label(yi, yj, rng); q += 1
    elif mode == 'adpo':
      if abs(margin) > GAMMA:
        label = 1 if margin > 0 else -1
      else:
        label = oracle_label(yi, yj, rng); q += 1
    else:  # adpo_no_pl
      if abs(margin) > GAMMA:
        continue
      label = oracle_label(yi, yj, rng); q += 1

    y = torch.tensor(label, dtype=torch.float32)
    loss = -torch.nn.functional.logsigmoid(y * (s1 - s2))
    opt.zero_grad(); loss.backward(); opt.step()

    while k_target_idx < len(QUERY_BUDGETS) and q >= QUERY_BUDGETS[k_target_idx]:
      out[k_target_idx] = test_accuracy(model, test_pairs); k_target_idx += 1

  last = out[~np.isnan(out)][-1] if np.any(~np.isnan(out)) else 0.5
  out[np.isnan(out)] = last
  return out


def aggregate(mode):
  curves = np.stack([run(mode, s) for s in range(N_SEEDS)], axis=0)
  return curves.mean(0), curves.std(0) / np.sqrt(N_SEEDS)


def main():
  modes = [('dpo', 'DPO', 'black', 'v', '-'),
           ('adpo', 'ADPO (γ=%.1f)' % GAMMA, 'red', 'o', '-'),
           ('adpo_no_pl', 'Active only (no PL)', 'gray', 's', '--')]
  plt.figure(figsize=(7, 4.5))
  for mode, label, color, marker, ls in modes:
    mean, se = aggregate(mode)
    plt.plot(QUERY_BUDGETS, mean * 100, marker=marker, ms=4, color=color,
             linestyle=ls, linewidth=1.8, label=label)
    plt.fill_between(QUERY_BUDGETS, (mean - se) * 100, (mean + se) * 100,
                     color=color, alpha=0.12)
  plt.xlabel('Number of Queries (k)')
  plt.ylabel('Test accuracy (%)')
  plt.title('sklearn digits pairwise (which digit is larger)\n'
            'flip-noise=%.0f%%, %d seeds, γ=%.1f' % (FLIP_NOISE * 100, N_SEEDS, GAMMA))
  plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
  plt.savefig(OUT, dpi=150)
  print('Saved bench_digits_pairwise.png')


if __name__ == '__main__':
  main()

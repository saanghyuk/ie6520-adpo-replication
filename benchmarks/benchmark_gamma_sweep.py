# -*- coding: utf-8 -*-
"""
Benchmark #1 — γ sensitivity sweep on the synthetic BT toy.

Goal: show that ADPO is NOT a free win. The paper picks γ=1.3 and reports
big gains. We sweep γ and check whether the gain survives bad γ choices.
If γ is too small, pseudo-labels dominate the loss before the model is
reliable, and ADPO ends up *below* DPO -- a contradicting regime.

For each γ we run the same BT setup as the main replication (d=16, noisy
oracle, 20 seeds), measure test accuracy at query budget k = 300, and
compare to DPO (which does not depend on γ).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'bench_gamma_sweep.png')

D = 16
N_SEEDS = 20
N_TEST = 3000
LR = 0.1
REWARD_SCALE = 0.5
MAX_STEPS = 20000
QUERY_BUDGET = 300
GAMMAS = [0.1, 0.3, 0.6, 1.0, 1.3, 2.0, 3.0]


def sample_x(rng, n=None):
  return rng.standard_normal(D) if n is None else rng.standard_normal((n, D))


def bt_label(r1, r2, rng):
  p = 1.0 / (1.0 + np.exp(-(r1 - r2)))
  return 1 if rng.random() < p else -1


def make_test_set(theta_star, rng):
  x1 = sample_x(rng, N_TEST)
  x2 = sample_x(rng, N_TEST)
  y = np.where(x1 @ theta_star > x2 @ theta_star, 1, -1)
  return (torch.tensor(x1, dtype=torch.float32),
          torch.tensor(x2, dtype=torch.float32),
          torch.tensor(y, dtype=torch.float32))


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


def run(mode, seed, gamma):
  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)
  theta_star = rng.standard_normal(D) * REWARD_SCALE
  test_set = make_test_set(theta_star, rng)
  model = LinearReward(D)
  opt = optim.SGD(model.parameters(), lr=LR)

  q = 0
  for _ in range(MAX_STEPS):
    if q >= QUERY_BUDGET:
      break
    x1, x2 = sample_x(rng), sample_x(rng)
    r1, r2 = float(theta_star @ x1), float(theta_star @ x2)
    x1_t = torch.tensor(x1, dtype=torch.float32)
    x2_t = torch.tensor(x2, dtype=torch.float32)
    s1, s2 = model(x1_t), model(x2_t)
    margin = (s1 - s2).item()

    if mode == 'dpo':
      label = bt_label(r1, r2, rng)
      q += 1
    else:  # adpo
      if abs(margin) > gamma:
        label = 1 if margin > 0 else -1
      else:
        label = bt_label(r1, r2, rng)
        q += 1

    y = torch.tensor(label, dtype=torch.float32)
    loss = -torch.nn.functional.logsigmoid(y * (s1 - s2))
    opt.zero_grad(); loss.backward(); opt.step()

  return test_accuracy(model, test_set)


def mean_acc(mode, gamma):
  return np.mean([run(mode, s, gamma) for s in range(N_SEEDS)])


def main():
  dpo_acc = mean_acc('dpo', gamma=0.0)  # γ unused for DPO
  adpo_by_gamma = {g: mean_acc('adpo', g) for g in GAMMAS}

  plt.figure(figsize=(7, 4.5))
  gs = list(adpo_by_gamma.keys())
  accs = [adpo_by_gamma[g] * 100 for g in gs]
  plt.plot(gs, accs, marker='o', color='red', label='ADPO', linewidth=2)
  plt.axhline(dpo_acc * 100, color='black', linestyle='--',
              label='DPO baseline (%.1f%%)' % (dpo_acc * 100))
  plt.xlabel('γ (ADPO confidence threshold)')
  plt.ylabel('Test accuracy at k=%d (%%)' % QUERY_BUDGET)
  plt.title('γ sensitivity — ADPO can lose to DPO when γ is mis-tuned\n'
            '(synthetic BT, d=%d, %d seeds)' % (D, N_SEEDS))
  plt.legend()
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(OUT, dpi=150)
  print('Saved bench_gamma_sweep.png')
  print('DPO:', dpo_acc)
  for g, a in adpo_by_gamma.items():
    print('ADPO γ=%.2f → %.4f' % (g, a))


if __name__ == '__main__':
  main()

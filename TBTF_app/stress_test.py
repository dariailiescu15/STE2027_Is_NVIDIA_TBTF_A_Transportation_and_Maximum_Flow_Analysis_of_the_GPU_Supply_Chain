# -*- coding: utf-8 -*-
"""Stress test for the GPU supply chain (Algorithm 1 in the paper).

For each supplier k we draw N random disruption scenarios:
    theta   ~ U(0.2, 1.0)   share of k's capacity that is lost
    sigma_i ~ U(0.0, 0.2)   reserve capacity each supplier can activate in a crisis
    eps_j   ~ U(-0.1, 0.1)  demand variation of market j
The deficit of a scenario is total demand minus the maximum flow of the capacity model.
With uncapacitated supplier-market routes the maximum flow equals min(total supply, total demand);
with CUDA lock-in (a share lam of every market's demand that only NVIDIA can serve) NVIDIA
serves the locked demand first and the rest of the capacity serves the flexible demand.
The deficit caused by the failure is measured against the same scenario without the failure.

Run `python3 stress_test.py` to reproduce the numbers of the paper (seed 2026, N = 20,000).
"""

import numpy as np

SUPPLIERS = ["NVIDIA", "AMD", "Intel Arc"]
A0 = np.array([500.0, 300.0, 200.0])  # capacities, scenario S1
B0 = np.array([300.0, 150.0, 200.0, 250.0, 100.0])  # demand: USA, Germany, Japan, China, Romania


def deficit(a, demand, lam=0.0):
    """Unmet demand = total demand - maximum flow (NVIDIA is supplier 0)."""
    locked, flexible = lam * demand, (1 - lam) * demand
    served_locked = np.minimum(locked, a[..., 0])
    served = served_locked + np.minimum(flexible, a[..., 1:].sum(-1) + a[..., 0] - served_locked)
    return demand - served


def simulate(k, n=20000, seed=2026, lockin=False, a0=A0, b0=B0):
    """Per-scenario results for a failure of supplier k (a0, b0: baseline capacities and demand)."""
    a0, b0 = np.asarray(a0, float), np.asarray(b0, float)
    rng = np.random.default_rng(seed + k)
    theta = rng.uniform(0.2, 1.0, n)
    sigma = rng.uniform(0.0, 0.2, (n, len(a0)))
    eps = rng.uniform(-0.1, 0.1, (n, len(b0)))
    lam = rng.uniform(0.1, 0.5, n) if lockin else np.zeros(n)
    demand = (b0 * (1 + eps)).sum(1)
    a_ref = a0 * (1 + sigma)
    a_fail = a_ref.copy()
    a_fail[:, k] = a0[k] * (1 - theta)
    d_lost = deficit(a_fail, demand, lam) - deficit(a_ref, demand, lam)
    capacity_lost = a_ref[:, k] - a_fail[:, k]
    with np.errstate(divide="ignore", invalid="ignore"):
        absorbed = np.where(capacity_lost > 0, 1 - d_lost / capacity_lost, np.nan)
    return {
        "share": d_lost / demand,
        "absorbed": absorbed,
        "d_lost": d_lost,
        "demand": demand,
        "theta": theta,
    }


def stress_test(k, n=20000, seed=2026, lockin=False, a0=A0, b0=B0):
    """Summary indicators for a failure of supplier k."""
    r = simulate(k, n, seed, lockin, a0, b0)
    share = r["share"]
    return {
        "SI": r["d_lost"].mean() / r["demand"].mean(),
        "mean": share.mean(),
        "p5": np.percentile(share, 5),
        "q1": np.percentile(share, 25),
        "median": np.median(share),
        "q3": np.percentile(share, 75),
        "p95": np.percentile(share, 95),
        "absorbed": float(np.nanmean(r["absorbed"])) if np.isfinite(r["absorbed"]).any() else np.nan,
        "P(share>20%)": (share > 0.2).mean(),
    }


if __name__ == "__main__":
    for lockin in (False, True):
        print("CUDA lock-in 10-50%" if lockin else "No lock-in")
        for k, name in enumerate(SUPPLIERS):
            r = stress_test(k, lockin=lockin)
            print(
                f"  {name:9s} SI {100*r['SI']:5.1f}%  median {100*r['median']:5.1f}%  "
                f"p95 {100*r['p95']:5.1f}%  absorbed {100*r['absorbed']:5.1f}%  "
                f"P(>20%) {100*r['P(share>20%)']:5.1f}%"
            )
    a_cal = np.array([880.0, 120.0, 0.0]) * np.array([0.2, 1.0, 1.0])
    print(
        f"Calibrated S2: supply {a_cal.sum():.0f}, unmet {1000 - a_cal.sum():.0f} "
        f"({100 * (1 - a_cal.sum() / 1000):.1f}%)"
    )

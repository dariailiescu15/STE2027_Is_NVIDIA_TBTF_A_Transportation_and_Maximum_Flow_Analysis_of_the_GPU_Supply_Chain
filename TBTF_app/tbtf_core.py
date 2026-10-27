# -*- coding: utf-8 -*-
"""Models behind the "Too Big to Fail" application (no Streamlit code here).

- demand: synthetic 2018-2024 series and a Random Forest forecast of 2026 demand
- transportation problem: balancing with a fictitious supplier, North-West Corner,
  MODI / stepping-stone iterations, HiGHS check
- capacity model: Edmonds-Karp (Ford-Fulkerson with BFS), minimum cut, augmenting paths
- recovery agent: period-by-period re-optimisation with shipment lots and market priorities
"""

from collections import deque

import numpy as np
from scipy.optimize import LinearConstraint, linprog, milp
from sklearn.ensemble import RandomForestRegressor

SUPPLIERS = ["NVIDIA", "AMD", "Intel Arc"]
MARKETS_EN = ["USA", "Germany", "Japan", "China", "Romania"]

# unit costs (m.u. per product unit); rows = suppliers, columns = markets (slide 13 / paper, Table 1)
COST = np.array([[2, 5, 4, 6, 7], [3, 4, 6, 5, 6], [5, 6, 7, 4, 5]], dtype=float)

INSTANCES = {
    "paper": np.array([500.0, 300.0, 200.0]),  # stylised instance used in the paper
    "calibrated": np.array([880.0, 120.0, 0.0]),  # observed add-in-board shares, JPR Q1 2024
}
DEMAND_TARGET = np.array([300.0, 150.0, 200.0, 250.0, 100.0])  # case-study demand, sum = 1,000

# Omdia estimates of Hopper GPUs bought in 2024 (DatacenterDynamics, 18 Dec 2024)
HOPPER_2024 = [
    ("Microsoft", 485_000, "Maia"),
    ("ByteDance", 230_000, "-"),
    ("Tencent", 230_000, "-"),
    ("Meta", 224_000, "MTIA"),
    ("xAI / Tesla", 200_000, "-"),
    ("Amazon", 196_000, "Trainium"),
    ("Google", 169_000, "TPU"),
]


# ----------------------------------------------------------------------------------- demand
def synthetic_history(seed=42):
    """Synthetic annual demand 2018-2024 that grows towards the case-study demand."""
    rng = np.random.default_rng(seed)
    years = np.arange(2018, 2025)
    growth = rng.uniform(0.155, 0.175, len(DEMAND_TARGET))
    data = {}
    for k, target in enumerate(DEMAND_TARGET):
        trend = target * np.exp(-growth[k] * (2026 - years))
        data[k] = trend * (1 + rng.normal(0, 0.02, len(years)))
    return years, data


def rf_forecast(years, data, seed=42):
    """Random Forest per region on log-growth (lag 1 -> next year); returns 2025, 2026 forecasts."""
    out = {}
    for k, series in data.items():
        g = np.diff(np.log(series))
        model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=seed)
        model.fit(g[:-1].reshape(-1, 1), g[1:])
        g25 = float(model.predict([[g[-1]]])[0])
        g26 = float(model.predict([[g25]])[0])
        d25 = series[-1] * np.exp(g25)
        d26 = d25 * np.exp(g26)
        out[k] = (d25, d26)
    return out


def demand_vector(forecast, step=50):
    """Stylised instance: the 2026 forecast rounded to multiples of `step` product units."""
    return np.array([step * round(forecast[k][1] / step) for k in sorted(forecast)], dtype=float)


# ---------------------------------------------------------------------- transportation model
def balance(C, A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    sa, sb = A.sum(), B.sum()
    if abs(sa - sb) < 1e-9:
        return C.copy(), A.copy(), B.copy(), "balanced", 0.0
    if sa < sb:
        return (
            np.vstack([C, np.zeros(C.shape[1])]),
            np.append(A, sb - sa),
            B.copy(),
            "dummy_supplier",
            sb - sa,
        )
    return np.hstack([C, np.zeros((C.shape[0], 1))]), A.copy(), np.append(B, sa - sb), "dummy_market", sa - sb


def north_west(A, B):
    """North-West Corner rule; returns the plan and a basis of m+n-1 cells (degeneracy handled)."""
    a, b = list(map(float, A)), list(map(float, B))
    m, n = len(a), len(b)
    X = np.zeros((m, n))
    basis = []
    i = j = 0
    while i < m and j < n:
        q = min(a[i], b[j])
        X[i, j] = q
        basis.append((i, j))
        a[i] -= q
        b[j] -= q
        if a[i] <= 1e-9 and b[j] <= 1e-9:
            if i == m - 1 and j == n - 1:
                break
            if j < n - 1 and i < m - 1:
                basis.append((i + 1, j))  # zero basic cell keeps the basis connected
                i += 1
                j += 1
            elif j < n - 1:
                j += 1
            else:
                i += 1
        elif a[i] <= 1e-9:
            i += 1
        else:
            j += 1
    return X, basis


def potentials(C, basis):
    m, n = C.shape
    u, v = [None] * m, [None] * n
    u[0] = 0.0
    changed = True
    while changed:
        changed = False
        for i, j in basis:
            if u[i] is not None and v[j] is None:
                v[j] = C[i, j] - u[i]
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = C[i, j] - v[j]
                changed = True
    return u, v


def _cycle(basis, enter):
    adj = {}
    for i, j in basis:
        adj.setdefault(("r", i), []).append(("c", j))
        adj.setdefault(("c", j), []).append(("r", i))
    start, goal = ("c", enter[1]), ("r", enter[0])
    prev = {start: None}
    dq = deque([start])
    while dq:
        x = dq.popleft()
        if x == goal:
            break
        for y in adj.get(x, []):
            if y not in prev:
                prev[y] = x
                dq.append(y)
    path, x = [], goal
    while x is not None:
        path.append(x)
        x = prev[x]
    cells = [enter]
    for p, q in zip(path, path[1:]):
        r = p[1] if p[0] == "r" else q[1]
        c = p[1] if p[0] == "c" else q[1]
        cells.append((r, c))
    return cells  # + - + - ... starting with the entering cell


def modi(C, X, basis, max_iter=100):
    """MODI / stepping-stone iterations from a basic plan. Returns the optimum and a log."""
    X, basis, log = X.copy(), list(basis), []
    m, n = C.shape
    for _ in range(max_iter):
        u, v = potentials(C, basis)
        red = {(i, j): C[i, j] - u[i] - v[j] for i in range(m) for j in range(n) if (i, j) not in basis}
        if not red:
            break
        enter, dmin = min(red.items(), key=lambda kv: (kv[1], kv[0]))
        if dmin >= -1e-9:
            return X, basis, u, v, red, log
        cyc = _cycle(basis, enter)
        minus = cyc[1::2]
        theta = min(X[c] for c in minus)
        leave = next(c for c in minus if X[c] == theta)
        before = float((X * C).sum())
        for k, c in enumerate(cyc):
            X[c] += theta if k % 2 == 0 else -theta
        basis.remove(leave)
        basis.append(enter)
        log.append(
            {
                "enter": enter,
                "delta": float(dmin),
                "theta": float(theta),
                "leave": leave,
                "cost_before": before,
                "cost_after": float((X * C).sum()),
            }
        )
    u, v = potentials(C, basis)
    red = {(i, j): C[i, j] - u[i] - v[j] for i in range(m) for j in range(n) if (i, j) not in basis}
    return X, basis, u, v, red, log


def lp_cost(C, A, B):
    m, n = C.shape
    Aeq, beq = [], []
    for i in range(m):
        r = np.zeros(m * n)
        r[i * n : (i + 1) * n] = 1
        Aeq.append(r)
        beq.append(A[i])
    for j in range(n):
        r = np.zeros(m * n)
        r[j::n] = 1
        Aeq.append(r)
        beq.append(B[j])
    res = linprog(C.ravel(), A_eq=np.array(Aeq), b_eq=beq, bounds=(0, None), method="highs")
    return float(res.fun)


def solve_transport(A, B, C=COST):
    """Full pipeline used by the app for one scenario."""
    Cb, Ab, Bb, kind, gap = balance(C, A, B)
    X0, basis0 = north_west(Ab, Bb)
    X, basis, u, v, red, log = modi(Cb, X0, basis0)
    real = X[: len(A), : len(B)]
    delivered = real.sum(axis=0)
    unmet = np.maximum(0, np.asarray(B, float) - delivered)
    return {
        "C": Cb,
        "A": Ab,
        "B": Bb,
        "kind": kind,
        "gap": gap,
        "X0": X0,
        "basis0": basis0,
        "z0": float((X0 * Cb).sum()),
        "X": X,
        "basis": basis,
        "z": float((X * Cb).sum()),
        "u": u,
        "v": v,
        "reduced": red,
        "log": log,
        "z_lp": lp_cost(Cb, Ab, Bb),
        "alt_optima": sorted(c for c, d in red.items() if abs(d) < 1e-9),
        "delivered": delivered,
        "unmet": unmet,
    }


# ---------------------------------------------------------------------------- capacity model
def max_flow(A, B, big=10**6):
    """Edmonds-Karp on source -> suppliers -> markets -> sink. Node 0 = s, last node = t."""
    m, n = len(A), len(B)
    N = m + n + 2
    s, t = 0, N - 1
    cap = [[0.0] * N for _ in range(N)]
    for i, a in enumerate(A):
        cap[s][1 + i] = float(a)
        for j in range(n):
            cap[1 + i][1 + m + j] = float(big)
    for j, b in enumerate(B):
        cap[1 + m + j][t] = float(b)
    R = [row[:] for row in cap]
    flow, paths = 0.0, []
    while True:
        par = [-1] * N
        par[s] = s
        dq = deque([s])
        while dq and par[t] == -1:
            x = dq.popleft()
            for y in range(N):
                if par[y] == -1 and R[x][y] > 1e-9:
                    par[y] = x
                    dq.append(y)
        if par[t] == -1:
            break
        alpha, y = float("inf"), t
        while y != s:
            alpha = min(alpha, R[par[y]][y])
            y = par[y]
        y, path = t, [t]
        while y != s:
            R[par[y]][y] -= alpha
            R[y][par[y]] += alpha
            y = par[y]
            path.append(y)
        flow += alpha
        paths.append((path[::-1], alpha))
    side = {s}
    dq = deque([s])
    while dq:
        x = dq.popleft()
        for y in range(N):
            if y not in side and R[x][y] > 1e-9:
                side.add(y)
                dq.append(y)
    cut = [(x, y, cap[x][y]) for x in side for y in range(N) if y not in side and cap[x][y] > 0]
    arc_flow = [[max(0.0, cap[x][y] - R[x][y]) if cap[x][y] > 0 else 0.0 for y in range(N)] for x in range(N)]
    return {
        "value": flow,
        "cut": cut,
        "source_side": side,
        "paths": paths,
        "flow": arc_flow,
        "cap": cap,
        "m": m,
        "n": n,
    }


def sweep(base_caps, B, losses, C=COST):
    """Cost and deficit as NVIDIA loses a growing share of its capacity."""
    rows = []
    for loss in losses:
        A = np.array(base_caps, float)
        A[0] *= 1 - loss
        res = solve_transport(A, B, C)
        rows.append(
            {
                "loss": loss,
                "nvidia": A[0],
                "cost": res["z"],
                "deficit": res["unmet"].sum(),
                "unmet": res["unmet"],
            }
        )
    return rows


# --------------------------------------------------------------------------- recovery agent
def recovery_plan(A_start, A_target, B, periods=6, lot=1, weights=None, C=COST, penalty=100.0):
    """At every period the agent re-optimises shipments for the capacity it has.

    NVIDIA's capacity recovers linearly from A_start[0] to A_target[0]. Shipments are made in
    lots of `lot` units (integer multiples). Unmet demand is penalised with penalty * weight_j,
    so the agent first serves as much as possible, then protects high-priority markets and only
    then minimises transport cost."""
    m, n = C.shape
    B = np.asarray(B, float)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    plan = []
    for p in range(periods + 1):
        A = np.array(A_start, float)
        A[0] = A_start[0] + (A_target[0] - A_start[0]) * p / periods
        nx = m * n
        c = np.concatenate([lot * C.ravel(), penalty * w])
        rows, lb, ub = [], [], []
        for i in range(m):  # supply: sum_j lot*k_ij <= a_i
            r = np.zeros(nx + n)
            r[i * n : (i + 1) * n] = lot
            rows.append(r)
            lb.append(-np.inf)
            ub.append(A[i])
        for j in range(n):  # demand: sum_i lot*k_ij + u_j = b_j
            r = np.zeros(nx + n)
            r[j:nx:n] = lot
            r[nx + j] = 1
            rows.append(r)
            lb.append(B[j])
            ub.append(B[j])
        integrality = np.concatenate([np.full(nx, 1 if lot > 1 else 0), np.zeros(n)])
        res = milp(
            c,
            constraints=LinearConstraint(np.array(rows), lb, ub),
            integrality=integrality,
            bounds=(0, np.inf),
        )
        k = res.x[:nx].reshape(m, n)
        X = lot * np.round(k) if lot > 1 else k
        unmet = np.maximum(0, B - X.sum(axis=0))
        plan.append(
            {
                "period": p,
                "capacity": A,
                "X": X,
                "unmet": unmet,
                "cost": float((X * C).sum()),
                "idle": float(A.sum() - X.sum()),
            }
        )
    return plan

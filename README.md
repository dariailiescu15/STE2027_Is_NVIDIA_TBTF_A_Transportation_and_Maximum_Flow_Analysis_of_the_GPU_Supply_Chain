# The "Too Big to Fail" Paradigm — GPU Supply Chain

> Mathematical modelling of strategic dependence on a dominant player in a GPU supply network.
> Application: Machine Learning for demand estimation + the Transportation Problem for cost optimisation + the Ford-Fulkerson algorithm for the physical capacity of the network + a Monte Carlo stress test.

---

## Contents

1. [Motivation and general problem](#1-motivation-and-general-problem)
2. [Demand estimation with Random Forest](#2-demand-estimation-with-random-forest)
3. [The Transportation Problem (TP)](#3-the-transportation-problem-tp)
   - 3.1 [General formulation](#31-general-formulation)
   - 3.2 [Balancing the problem](#32-balancing-the-problem)
   - 3.3 [Initial solution: the North-West Corner method](#33-initial-solution-the-north-west-corner-method)
   - 3.4 [Optimisation: the MODI algorithm](#34-optimisation-the-modi-algorithm)
   - 3.5 [The ε-perturbation technique](#35-the-ε-perturbation-technique)
4. [Network flow and Ford-Fulkerson](#4-network-flow-and-ford-fulkerson)
   - 4.1 [Network graph and flow](#41-network-graph-and-flow)
   - 4.2 [Cuts and the Ford-Fulkerson theorem](#42-cuts-and-the-ford-fulkerson-theorem)
   - 4.3 [The labelling algorithm](#43-the-labelling-algorithm)
5. [Combining the two models](#5-combining-the-two-models)
6. [Case study: the GPU ecosystem](#6-case-study-the-gpu-ecosystem)
   - 6.1 [Scenario 1 — Normal market](#61-scenario-1--normal-market)
   - 6.2 [Scenario 2 — NVIDIA collapse](#62-scenario-2--nvidia-collapse)
7. [The apparent-cost paradox](#7-the-apparent-cost-paradox)
8. [Implementation](#8-implementation)
9. [Extensions and a scalable architecture](#9-extensions-and-a-scalable-architecture)

---

## 1. Motivation and general problem

The concept of **"Too Big to Fail" (TBTF)** describes a situation in which an economic player becomes so important to a system that its failure would cause systemic damage. The term comes from the 2007–2008 financial crisis, when certain banks were considered too important to be allowed to fail.

In today's technology ecosystem, the dependence of global AI infrastructure on NVIDIA shows similar features. The question we address is:

> **How can we quantify mathematically the vulnerability of a supply network when a dominant player becomes partly or fully unavailable?**

To answer it, we split the problem into three steps:

1. **Estimating future demand** by region — with a Random Forest on historical series
2. **Optimising the minimum-cost allocation** — with the Transportation Problem
3. **Checking physical feasibility** — with the Ford-Fulkerson algorithm

Each model answers a different question, and together they expose the TBTF phenomenon in a measurable way. A Monte Carlo stress test then compares how much demand is lost when each supplier fails.

---

## 2. Demand estimation with Random Forest

### Motivation

The demand vector **B = (b₁, b₂, ..., bₙ)** that feeds the optimisation model is not set arbitrarily. It is derived from historical data by regression.

### The model

For each region r ∈ {USA, Germany, Japan, China, Romania}, the 2018–2024 data are modelled as:

$$D_r(t) = \beta_r \cdot e^{\alpha_r (t - 2018)} \cdot (1 + \epsilon), \quad \epsilon \sim \mathcal{N}(0, 0.02^2)$$

where:
- $\beta_r$ = base volume in 2018
- $\alpha_r$ = annual growth rate (between 15.5% and 17.5%, chosen so that the series reaches the case-study demand)
- $\epsilon$ = Gaussian noise reflecting market fluctuations

> The historical series are **synthetic**: the model illustrates the estimation mechanism, it does not estimate the real market.

### Random Forest Regressor

A **Random Forest** model with the following hyperparameters:
- 200 trees (`n_estimators=200`)
- maximum depth 4 (`max_depth=4`)
- fixed seed for reproducibility (`random_state=42`)

is trained, for each region, on the year-on-year log growth of the 2018–2024 series (growth in year t → growth in year t+1) and applied twice, for 2025 and 2026. The 2026 value is rounded to multiples of 50 units.

### Methodological justification

| RF advantage | Detail |
|---|---|
| Robust to outliers | Averaging the predictions of many trees reduces the impact of extreme values |
| Captures non-linearities | No specific model form (linear, exponential) is assumed |
| Uncertainty estimate | The spread across trees gives a confidence band |
| Guards against overfitting | `max_depth=4` on 7 historical points keeps the model simple |

### Alternatives considered

- **ARIMA**: assumes stationarity, which is problematic for exponentially growing series
- **LSTM**: far too complex for a 7-point series (over 100 parameters vs 7 observations)
- **Polynomial regression**: tends to overfit at degree ≥ 3

### Output

For 2026 the model gives the demand vector **B = (300, 150, 200, 250, 100)**, with Σ bⱼ = 1,000 units, which becomes the input of the next step.

---

## 3. The Transportation Problem (TP)

### 3.1 General formulation

We consider:
- **m** production centres (sources) **Aᵢ**, i = 1..m, with availabilities **aᵢ ≥ 0**
- **n** destinations (markets) **Bⱼ**, j = 1..n, with requirements **bⱼ ≥ 0**
- the unit transport cost **cᵢⱼ ≥ 0** from Aᵢ to Bⱼ

The decision variable xᵢⱼ is the quantity shipped from source i to destination j.

**General form:**

$$\min \, f(x) = \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} \cdot x_{ij}$$

subject to:

$$\sum_{j=1}^{n} x_{ij} \leq a_i, \quad i = \overline{1, m} \quad \text{(availability)}$$

$$\sum_{i=1}^{m} x_{ij} \geq b_j, \quad j = \overline{1, n} \quad \text{(demand)}$$

$$x_{ij} \geq 0, \quad \forall i, j \quad \text{(non-negativity)}$$

### 3.2 Balancing the problem

The problem is **balanced** if:

$$\sum_{i=1}^{m} a_i = \sum_{j=1}^{n} b_j$$

In this case all constraints become equalities and an optimal solution is guaranteed to exist (fundamental theorem of the TP).

If the problem is **unbalanced**, it is turned into a balanced one by adding a fictitious player:

- **Supply shortfall** (Σa < Σb): a **fictitious supplier** Aₘ₊₁ is added with availability aₘ₊₁ = Σbⱼ − Σaᵢ and costs cₘ₊₁,ⱼ = 0
- **Supply surplus** (Σa > Σb): a **fictitious destination** Bₙ₊₁ is added with requirement bₙ₊₁ = Σaᵢ − Σbⱼ and costs cᵢ,ₙ₊₁ = 0

> **Conceptual warning:** the fictitious supplier or destination is a mathematical artefact. The units allocated to it do not exist physically — they are only variables that satisfy the balance constraint at zero cost. **This fact is central to the apparent-cost paradox.**

**Standard form:**

$$\min \, f(x) = \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} \cdot x_{ij}$$

subject to:

$$\sum_{j=1}^{n} x_{ij} = a_i, \quad i = \overline{1, m}$$

$$\sum_{i=1}^{m} x_{ij} = b_j, \quad j = \overline{1, n}$$

$$x_{ij} \geq 0, \quad \sum_{i} a_i = \sum_{j} b_j$$

### Fundamental theorems

| Theorem | Statement |
|---|---|
| **T1** | Every balanced TP has at least one feasible solution |
| **T2** | Every balanced TP has at least one optimal solution |
| **T3** | The rank of the system matrix of the standard form is `m + n − 1` |

A **basic solution** has at most m + n − 1 non-zero variables. If it has exactly m + n − 1 non-zero components it is **non-degenerate**; otherwise it is **degenerate**.

### 3.3 Initial solution: the North-West Corner method

To start the optimisation algorithm we need a feasible initial solution. The **North-West Corner method** is the simplest one:

```
1. Start in cell (1, 1) (the north-west corner)
2. Allocate x₁₁ = min(a₁, b₁)
3. Subtract x₁₁ from the availability aᵢ and the requirement bⱼ
4. If aᵢ is exhausted → move down to the next row (i+1, j)
   If bⱼ is exhausted → move right to the next column (i, j+1)
5. Repeat until all aᵢ and bⱼ are zero
```

**Advantage:** simple, and guaranteed to produce a basic solution.
**Drawback:** it ignores the costs cᵢⱼ, so the initial solution is usually far from optimal. This is why the next step is needed.

### 3.4 Optimisation: the MODI algorithm

**MODI** (Modified Distribution Method) checks whether a solution is optimal and, if it is not, gives the direction of improvement.

#### The dual problem

For the standard form, the dual problem is:

$$\max \, g(u, v) = \sum_{i=1}^{m} a_i u_i + \sum_{j=1}^{n} b_j v_j$$

subject to:

$$u_i + v_j \leq c_{ij}, \quad \forall i, j$$

$$u_i, v_j \in \mathbb{R}$$

#### Complementary slackness theorem

The optimal solutions of the primal and the dual satisfy:

$$x_{ij} \cdot (c_{ij} - u_i - v_j) = 0, \quad \forall i, j$$

In other words, for every basic variable we have **uᵢ + vⱼ = cᵢⱼ**.

#### The algorithm

**Step 1.** Start from the initial solution X given by the North-West Corner method. Let J be the set of basic cells (m + n − 1 cells).

**Step 2.** Compute the **potentials** by solving the system:

$$u_i + v_j = c_{ij}, \quad (i, j) \in J$$

The system has m + n unknowns and m + n − 1 equations, so one unknown is free. The usual convention is **u₁ = 0**; the others follow by propagation.

**Step 3.** Compute the **modified costs** of the non-basic cells:

$$\tilde{c}_{ij} = u_i + v_j, \quad (i, j) \notin J$$

**Step 4.** Compute the **differences**:

$$\delta_{ij} = c_{ij} - \tilde{c}_{ij} = c_{ij} - (u_i + v_j)$$

**Step 5. Optimality criterion:**

- If **all δᵢⱼ ≥ 0** → the solution X is **optimal**. STOP.
- If **some (i,j) ∉ J has δᵢⱼ < 0** → the solution is not optimal; go to step 6.

**Step 6.** Choose the entering cell:

$$\delta_{p,q} = \min_{(i,j) \notin J} \{\delta_{ij} < 0\}$$

**Step 7.** Build the **cycle** that starts at (p,q), passes only through basic cells and returns to (p,q). This cycle is unique.

**Step 8.** Mark the cells of the cycle with alternating signs: (+θ), (−θ), (+θ), (−θ)...

**Step 9.** Compute:

$$\theta = \min \{x_{ij} \mid (i,j) \text{ is a } (-\theta) \text{ cell of the cycle}\}$$

**Step 10.** Shift θ around the cycle:
- cells marked (+θ): xᵢⱼ + θ
- cells marked (−θ): xᵢⱼ − θ

The cell that drops to zero leaves the basis and (p,q) enters it.

**Step 11.** The new solution X' has a lower cost:

$$f'_{min} = f_{min} + \delta_{p,q} \cdot \theta$$

(δ\_{p,q} is negative, so the cost falls)

**Step 12.** Go back to step 2.

#### Convergence

The algorithm converges in a finite number of iterations because:
- there is a finite number of basic solutions (combinations of m+n−1 cells out of mn)
- each iteration gives a solution with a strictly lower cost (for non-degenerate problems)
- so the algorithm cannot loop forever

### 3.5 The ε-perturbation technique

**The degeneracy problem:** if the number of non-zero components is below m + n − 1, **cycling** can occur — the algorithm may return to an earlier solution without making progress.

**Solution:** the perturbation technique adds a small ε > 0 to the values aᵢ and bₙ:

$$a_i' = a_i + \epsilon, \quad i = \overline{1, m}, \qquad b_n' = b_n + m\epsilon, \qquad 0 \leq \epsilon \ll 1$$

This guarantees that partial sums of the aᵢ over subsets I₁ ⊂ I and of the bⱼ over subsets J₁ ⊂ J never coincide, which removes degeneracy.

MODI is applied to the perturbed problem, and **ε = 0** is set in the final solution. The application uses an equivalent approach: a zero-valued cell is kept in the basis so that it always has m + n − 1 cells.

---

## 4. Network flow and Ford-Fulkerson

The Transportation Problem assumes that the network can actually carry any quantity required. **This assumption may be false in practice** — there are physical capacity limits. To check feasibility we use the network-flow model.

### 4.1 Network graph and flow

**Definition 1 (finite connected graph).** The graph G = (X, U) is finite and connected if X is a finite set and for any x, y ∈ X, x ≠ y, there is a chain (a sequence of adjacent arcs) joining them.

**Definition 2 (network graph).** A finite connected graph G = (X, U) without loops is a **network graph** if there are:
- a vertex **xₛ ∈ X**, the **source**, with Γ⁻¹(xₛ) = ∅ (no arc enters xₛ)
- a vertex **xₜ ∈ X**, the **sink**, with Γ(xₜ) = ∅ (no arc leaves xₜ)

**Definition 3 (capacity).** For the set of arcs U we define the function:

$$c: U \rightarrow \mathbb{R}_+$$

where c(u) is the **capacity of arc** u.

**Definition 4 (flow).** A function φ: U → ℝ₊ is a **flow** in the network graph G if it satisfies three conditions:

**(i) Capacity:**

$$0 \leq \varphi(u) \leq c(u), \quad \forall u \in U$$

**(ii) Conservation** (at intermediate nodes):

$$\sum_{x_i \in \Gamma_{x_k}} \varphi(x_k, x_i) = \sum_{x_j \in \Gamma^{-1}_{x_k}} \varphi(x_j, x_k), \quad \forall k \neq s, t$$

(the flow leaving a node equals the flow entering it)

**(iii) Flow value:**

$$\sum_{x_i \in \Gamma_{x_s}} \varphi(x_s, x_i) = \sum_{x_j \in \Gamma^{-1}_{x_t}} \varphi(x_j, x_t) = v(\varphi)$$

**Definition 5 (saturated arc).** An arc u ∈ U is:
- **saturated** if φ(u) = c(u)
- **unsaturated** if φ(u) < c(u)

### 4.2 Cuts and the Ford-Fulkerson theorem

**Definition 6 (cut).** A subset A ⊂ X, A ≠ ∅, is a **cut** in the network graph G if xₛ ∉ A and xₜ ∈ A.

The cut splits the vertices into two disjoint subsets: S = X \ A (containing the source) and T = A (containing the sink).

**Definition 7 (arcs of the cut).**

$$U_A^- = \{u = (x_i, x_j) \mid x_i \notin A, \, x_j \in A\}$$

**Definition 8 (capacity of the cut).**

$$c(U_A^-) = \sum_{u \in U_A^-} c(u)$$

### The Ford-Fulkerson theorem

> **In a network graph G, the maximum flow value equals the minimum capacity of its cuts:**
>
> $$\max_{\varphi \in \mathcal{F}} v(\varphi) = \min_{A \subset \mathcal{T}} c(U_A^-)$$

**Proof (sketch).** For any flow φ and any cut A:
- all flow from xₛ to xₜ must cross the cut A
- the flow on each arc cannot exceed its capacity
- hence **v(φ) ≤ c(U_A⁻)** for every pair (φ, A)

Equality holds when the cut A is saturated by the flow φ (every arc of the cut has φ(u) = c(u)). ∎

**Practical consequence:** to find the maximum flow it is enough to identify a cut of minimum capacity. The Ford-Fulkerson algorithm does this constructively.

### 4.3 The labelling algorithm

**Step 1. Initial flow.** Choose an initial flow φ₀ ∈ F by picking a few paths from xₛ to xₜ and saturating them. The simplest choice: φ₀(u) = 0 for all arcs.

**Step 2. Labelling procedure** (optimality test):

- Label the source **xₛ with [+]**
- For each arc (xᵢ, xⱼ):
  - if xᵢ is labelled and the arc is unsaturated, label xⱼ with **[+xᵢ]**
  - if xⱼ is labelled and the arc carries positive flow, label xᵢ with **[−xⱼ]**
  - a saturated forward arc is not used for labelling
- Propagate the labels through the graph

**Step 3. Decision:**

- **If the sink xₜ cannot be labelled** → the flow φ is **optimal**; STOP. The labelled nodes give the minimum cut.
- **If xₜ has been labelled** → the flow is not optimal; go to step 4.

**Step 4. Rebuild the path.** Following the labels backwards, rebuild the chain μ from xₛ to xₜ.

**Step 5. Improvement amount.** Along the chain μ compute:

$$\alpha = \min \{c(u) - \varphi(u) \mid u \in \mu\}$$

(on backward arcs of μ, the available amount is φ(u))

**Step 6. Update the flow.** The improved flow is:

$$\varphi'(u) = \begin{cases} \varphi(u) + \alpha & \text{on forward arcs of } \mu \\ \varphi(u) - \alpha & \text{on backward arcs of } \mu \\ \varphi(u) & \text{otherwise} \end{cases}$$

with value **v(φ') = v(φ) + α**.

**Step 7.** Go back to step 2 with the new flow φ'.

#### Convergence

The algorithm converges because:
- each iteration increases v(φ) by α > 0
- v(φ) is bounded by the minimum cut capacity
- with integer capacities α is an integer, so the number of iterations is finite

#### Important remark

The order in which paths are chosen **does not change the final value** v(φ\*), only the intermediate values. The application uses the **Edmonds–Karp** variant, which always picks the shortest augmenting path (breadth-first search) and therefore needs a polynomial number of iterations.

---

## 5. Combining the two models

The Transportation Problem and Ford-Fulkerson answer **different questions** about the same network:

| Aspect | Transportation Problem | Ford-Fulkerson (FF) |
|---|---|---|
| **Question** | "What is the cheapest allocation?" | "How much can actually be shipped?" |
| **Output** | X\* (optimal allocation), f\* (minimum cost) | v(φ\*) (maximum flow), minimum cut |
| **Assumes** | The network can carry all the demand | Only the physical capacities of the arcs |
| **Limitation** | Accepts fictitious solutions when unbalanced | Cannot be "fooled" — the flow is real |

**Together they give a complete view** of the distribution network:
- ✓ Optimal cost
- ✓ Physical feasibility check
- ✓ Size of the deficit (when there is one)
- ✓ Location of the bottleneck (through the minimum cut)

---

## 6. Case study: the GPU ecosystem

### Problem data

- **Suppliers:** NVIDIA, AMD, Intel Arc (m = 3)
- **Markets:** USA, Germany, Japan, China, Romania (n = 5)
- **Cost matrix** (m.u. per unit):

$$
C = \begin{pmatrix}
2 & 5 & 4 & 6 & 7 \\
3 & 4 & 6 & 5 & 6 \\
5 & 6 & 7 & 4 & 5
\end{pmatrix}
$$

- **Demand vector** (from the ML step, for 2026): **B = (300, 150, 200, 250, 100)**, Σ bⱼ = 1,000 units

### 6.1 Scenario 1 — Normal market

**Capacities:** NVIDIA = 500, AMD = 300, Intel Arc = 200 → Σ aᵢ = 1,000

**Balance:** Σ aᵢ = Σ bⱼ = 1,000 → the problem is balanced, no fictitious supplier is needed.

**Initial solution** (North-West Corner): cost **4,100 m.u.**

**Optimal solution** (one MODI iteration: AMD → Germany enters with δ = −3, θ = 150, AMD → Japan leaves):

| X* | USA | Germany | Japan | China | Romania | Σ aᵢ |
|---|---|---|---|---|---|---|
| **NVIDIA** | 300 | 0 | 200 | 0 | 0 | 500 |
| **AMD** | 0 | 150 | 0 | 150 | 0 | 300 |
| **Intel Arc** | 0 | 0 | 0 | 100 | 100 | 200 |
| **Σ bⱼ** | 300 | 150 | 200 | 250 | 100 | 1,000 |

The cells NVIDIA → China, NVIDIA → Romania and AMD → Romania have δ = 0, so there are **alternative optima**: other plans with the same cost.

**Results:**

| Indicator | Value |
|---|---|
| Optimal TP cost (f\*) | **3,650 m.u.** (checked with HiGHS) |
| Ford-Fulkerson maximum flow (v(φ)) | **1,000 units** |
| Demand coverage | **100% ✓** |
| Physical deficit | 0 |
| Minimum cut capacity | 1,000 = v(φ) ✓ |

### 6.2 Scenario 2 — NVIDIA collapse

**Capacities:** NVIDIA = 100 (an 80% cut), AMD = 300, Intel Arc = 200 → real Σ aᵢ = 600

**Balance:** Σ aᵢ = 600 < Σ bⱼ = 1,000 → the problem is **unbalanced** and needs a **fictitious supplier A\* with 400 units at zero cost**.

**Initial solution** (North-West Corner): cost **2,550 m.u.**

**Optimal solution** (two MODI iterations: Intel Arc → China enters with δ = −3, θ = 150 → 2,100 m.u.; then A\* → Germany with δ = −2, θ = 50 → 2,000 m.u.):

| X* | USA | Germany | Japan | China | Romania | Σ aᵢ |
|---|---|---|---|---|---|---|
| **NVIDIA** | 100 | 0 | 0 | 0 | 0 | 100 |
| **AMD** | 200 | 100 | 0 | 0 | 0 | 300 |
| **Intel Arc** | 0 | 0 | 0 | 200 | 0 | 200 |
| **A\* (fictitious)** | 0 | 50 | 200 | 50 | 100 | 400 |
| **Σ bⱼ** | 300 | 150 | 200 | 250 | 100 | 1,000 |

The deficit is concentrated in Japan (100%) and Romania (100%); Germany loses 33%, China 20%, and the USA is fully served.

**Results:**

| Indicator | Value |
|---|---|
| Optimal TP cost (f\*) | **2,000 m.u.** (apparently 45% lower!) |
| Ford-Fulkerson maximum flow (v(φ)) | **600 units** (the physical truth) |
| Demand coverage | **60%** |
| Physical deficit | **400 units** (40% of the market unsupplied) |
| Minimum cut capacity | 600 = v(φ) ✓ |
| Bottleneck | All 3 source → supplier arcs are saturated |

---

## 7. The apparent-cost paradox

### The mathematical fact

Optimal TP cost:
- S1: f\* = **3,650 m.u.**
- S2: f\* = **2,000 m.u.** (paradoxically **lower**, by 45%)

### The interpretation trap

Judged by the objective function f\* alone, the crisis seems to **improve** the situation. The Transportation Problem is mathematically correct — it minimises exactly what it is asked to minimise. A conclusion based on cost alone is nevertheless **wrong**.

### Why the paradox appears

The **400 units** missing from physical production are absorbed by the fictitious supplier at zero cost. The model does not "know" that these units **do not exist** — for the algorithm they are just variables that satisfy the balance constraint at minimum cost. The apparent cost falls precisely because **40% of the allocation is free and fictitious**.

### The role of Ford-Fulkerson

Ford-Fulkerson does not allow this illusion. It works only on the physical network, with no fictitious suppliers. Its answer, **v(φ) = 600 units**, is the observable truth: total real supply is 600, which means a deficit of 400 units — **exactly the volume hidden by the Transportation Problem**.

### The key identity

$$\boxed{\underbrace{v(\varphi^\ast)}_{\text{real flow}} + \underbrace{\sum_j x^\ast_{m+1,j}}_{\text{fictitious volume}} = \underbrace{\sum_j b_j}_{\text{total demand}}}$$

For S2: **600 + 400 = 1,000** ✓

Or equivalently:

$$\boxed{\text{Physical deficit} = \sum_j b_j - v(\varphi^\ast) = \text{Fictitious supplier volume}}$$

### The stress test

For each supplier, 20,000 random failures are simulated (seed 2026): the share of capacity lost θ ~ U(0.2, 1), the other suppliers' reserves σᵢ ~ U(0, 0.2) and the demand variation εⱼ ~ U(−0.1, 0.1). The deficit caused by the failure is measured against the same scenario without the failure.

| Supplier | Systemic importance SI | Median | 95th percentile | Absorbed by the others | P(loss > 20%) |
|---|---|---|---|---|---|
| NVIDIA | **25.0%** | 25.1% | 43.5% | 31.8% | 62.7% |
| AMD | 11.2% | 11.0% | 23.6% | 51.6% | 14.2% |
| Intel Arc | 5.0% | 4.1% | 14.0% | 68.8% | 0.1% |

NVIDIA has 1.67 times AMD's capacity, but its failure costs 2.23 times as much; against Intel Arc the ratios are 2.5 and 4.99. The code is in `stress_test.py` (`python3 stress_test.py` reproduces the table).

### Link to the TBTF paradigm

Concentrating capacity in a single dominant player (NVIDIA = 50% of supply in S1) creates a vulnerability that **does not show up in the nominal costs** of the network. Traditional financial models underestimate systemic risk for exactly this reason.

> **Theoretical conclusion:** TBTF vulnerability is **mathematically measurable** as the gap between total demand and the maximum physical flow or, equivalently, as the volume of the fictitious supplier added for balancing.

---

## 8. Implementation

### Technology stack

- **Python 3.10+**
- **Streamlit** — interactive web interface
- **NumPy / Pandas** — matrix computation and data handling
- **SciPy** (`linprog`, `milp`, HiGHS method) — check of the optimal cost and the recovery agent
- **scikit-learn** (`RandomForestRegressor`) — demand forecast
- **Plotly** — interactive charts (lines, bars, heatmap, Sankey, box plots)
- **Graphviz (DOT, rendered directly by Streamlit)** — flow network and flowcharts

### Project files

```
ML-AFF.py               Streamlit interface
tbtf_core.py            models: Random Forest, North-West Corner, MODI, Edmonds–Karp, recovery agent
stress_test.py          the stress test from the paper
.streamlit/config.toml  visual theme of the application
```

### Application tabs

The sidebar selects the data set (the paper instance 500/300/200, the real market shares 880/120/0 or custom data), the scenario S1/S2 and the share of NVIDIA capacity lost.

| Tab | Content |
|---|---|
| **Overview** | Supplier → market flow diagram, demand coverage by market, the cost paradox |
| **Demand (ML)** | Random Forest demand forecast by region |
| **Transport (cost)** | North-West Corner, MODI iterations, potentials uᵢ, vⱼ, differences δᵢⱼ, HiGHS check |
| **Maximum flow** | Capacity network with the minimum cut and the Edmonds–Karp augmenting paths |
| **Stress test** | Algorithm 1 (Monte Carlo) and sensitivity analysis |
| **Recovery agent** | Prototype of multi-period rerouting, with shipment lots and priority markets |
| **Who depends on NVIDIA** | Exposure of the largest GPU buyers (Omdia estimates, 2024) |
| **Method and theory** | Flowcharts and theory guide |

### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run
streamlit run ML-AFF.py
```

### Pseudocode of the key algorithms

**Random Forest for demand:**
```python
for each region r:
    g = diff(log(historical_demand[r]))        # annual log growth
    model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(g[:-1], g[1:])                   # growth in year t -> growth in year t+1
    B[r] = 50 * round(demand_2026[r] / 50)     # after two applications (2025, 2026)
```

**Transportation Problem (North-West Corner + MODI, checked with HiGHS):**
```python
C, A, B = balance(C, A, B)                     # fictitious supplier or market, cost 0
X, basis = north_west_corner(A, B)             # m+n-1 basic cells
while True:
    u, v = potentials(C, basis)                # uᵢ + vⱼ = cᵢⱼ on basic cells
    delta = C - (u + v.T)                      # on non-basic cells
    if all(delta >= 0): break                  # optimal plan
    (p, q) = argmin(delta)                     # entering cell
    cycle = cycle_through_basis(basis, (p, q))
    theta = min(X on the (−) positions of the cycle)
    X = X ± theta along the cycle; update the basis
assert cost(X) == linprog(..., method="highs").fun
```

**Ford-Fulkerson (BFS for augmenting paths):**
```python
while bfs_path(residual_graph, source, sink) exists:
    alpha = min(residual_graph[u][v] for (u, v) in path)
    update the residual graph with ±alpha
    max_flow += alpha
return max_flow, min_cut_S, min_cut_T
```

---

## 9. Extensions and a scalable architecture

The current demo (3 suppliers × 5 markets × 1 product) is the minimal version of an extensible decision platform.

### Proposed architecture for scaling

```
[ SOURCE DATA ]  →  [ INGEST ]  →  [ ML LAYER ]  →  [ OPTIM LAYER ]  →  [ DECISION ]
   N companies       ETL /          RF per           TP + FF               Aggregated
   M markets         streaming      product          (in parallel)         dashboard
   K products                       × region                               + alerts
```

### Components

- **Source data:** multiple databases with time series, capacities, prices, contracts
- **Ingest:** ETL (Airflow/dbt), normalisation, aggregation, validation
- **ML layer:** Random Forest per (product, region), producing the demand vectors B
- **Optimisation layer:** TP + FF in parallel, one instance per product
- **Decision:** aggregated dashboard with alerts at the critical threshold (the capacities below which a deficit appears)

### Applications beyond GPUs

The mathematical structure is **generic**. The same architecture applies to:

| Field | Dominant player | Product |
|---|---|---|
| Semiconductors | TSMC | 3 nm / 5 nm chips |
| Energy | Gazprom (pre-2022) | Natural gas |
| Cloud | AWS | Compute capacity |
| Pharma | Pfizer | Specific vaccines |
| Cobalt | DR Congo (mining) | Battery raw materials |

In every case, dependence on a dominant player creates a systemic vulnerability that our model can quantify.

---

## Copyright

© 2026 National University of Science and Technology POLITEHNICA Bucharest (UNSTPB)
Faculty of Applied Sciences

All rights reserved. This material is the property of the authors and of the coordinating institution. Full or partial reproduction or distribution without the written consent of the authors is prohibited.

## Authors

- Dedu Anișoara-Nicoleta
- Dumitrescu Andreea-Mihaela
- Iliescu Daria
- Lungu Diana-Ionela

**National University of Science and Technology POLITEHNICA Bucharest**
Faculty of Applied Sciences · Year III · Group 1333a
Supervisor: **Lect. Dr. Simona Mihaela BIBIC**

---

> *"The 'Too Big to Fail' paradigm is not only an economic intuition — it is a mathematical phenomenon, measurable as the gap between total demand Σbⱼ and the maximum flow v(φ\*)."*

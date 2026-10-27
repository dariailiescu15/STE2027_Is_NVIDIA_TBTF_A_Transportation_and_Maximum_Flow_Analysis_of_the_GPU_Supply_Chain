# -*- coding: utf-8 -*-
"""Is NVIDIA "Too Big to Fail"? Decision-support application for the GPU supply chain.

Layers: Random Forest demand forecast -> transportation problem (North-West Corner + MODI,
checked with HiGHS) -> maximum flow and minimum cut (Edmonds-Karp) -> Monte Carlo stress test
-> recovery agent (prototype) -> downstream exposure of the large GPU buyers.

The models live in tbtf_core.py and stress_test.py; this file only builds the interface.
Run locally with:  streamlit run ML-AFF.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import stress_test as stx
import tbtf_core as core

st.set_page_config(
    page_title="TBTF · GPU supply chain",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="auto",
)

# ------------------------------------------------------------------------------------ palette
INK, INK2, MUTED = "#14171F", "#4B5363", "#7A8394"
GRID, LINE = "#E6E9EE", "#C9CED6"
NAVY = "#0E1B33"
SUP_COL = ["#2a78d6", "#1baf7a", "#4a3aa7"]  # NVIDIA, AMD, Intel Arc (validated all-pairs with orange)
DEFICIT = "#eb6834"  # unmet demand / fictitious supplier, used for nothing else
DEFICIT_TXT = "#c24f1f"
IDLE = "#9AA1AD"
MARKET_COL = "#5B6475"
HATCH = dict(shape="/", fgcolor="rgba(255,255,255,.55)", size=7, solidity=0.22, fillmode="overlay")
ORANGE_RAMP = [[0, "#fff4ee"], [0.25, "#fbd3bf"], [0.5, "#f5a582"], [0.75, "#eb6834"], [1, "#a8400f"]]


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


CSS = """
<style>
.block-container {padding: 1.6rem 2.6rem 3rem; max-width: 1380px;}
.hero {position: relative; overflow: hidden; border-radius: 18px; padding: 30px 34px 26px;
  color: #fff; margin-bottom: 18px;
  background:
    linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px) 0 0 / 30px 30px,
    linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px) 0 0 / 30px 30px,
    radial-gradient(circle at 88% 12%, rgba(57,135,229,.55), transparent 46%),
    linear-gradient(135deg, #0E1B33 0%, #13284b 55%, #1c5cab 100%);}
.hero .kicker {font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; color: #9ec5f4;
  font-weight: 600;}
.hero h1 {color: #fff; font-size: 2.15rem; line-height: 1.15; margin: 6px 0 8px; padding: 0;
  font-weight: 700; letter-spacing: -.01em;}
.hero .sub {color: #cde2fb; font-size: .98rem; margin: 0 0 14px; max-width: 900px;}
.hero .pills {display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;}
.hero .pill {font-size: .8rem; font-weight: 600; padding: 4px 11px; border-radius: 999px;
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); color: #fff;}
.hero .pill.warn {background: rgba(235,104,52,.25); border-color: rgba(235,104,52,.75);}
.hero .pill.ok {background: rgba(27,175,122,.22); border-color: rgba(27,175,122,.7);}
.hero .meta {border-top: 1px solid rgba(255,255,255,.16); padding-top: 12px; font-size: .82rem;
  color: #b7d3f6; line-height: 1.6;}
.hero .meta b {color: #fff; font-weight: 600;}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 12px;
  margin: 0 0 8px;}
.kpi {background: #fff; border: 1px solid #E3E6EB; border-radius: 14px; padding: 14px 16px 12px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04);}
.kpi .lab {font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; color: #7A8394;
  font-weight: 600;}
.kpi .val {font-size: 1.75rem; font-weight: 700; color: #14171F; line-height: 1.2; margin-top: 4px;
  font-variant-numeric: tabular-nums;}
.kpi .val .u {font-size: .9rem; font-weight: 500; color: #7A8394; margin-left: 3px;}
.kpi .sub {font-size: .8rem; color: #4B5363; margin-top: 2px;}
.kpi.alert {border-color: #f5c3ad; background: linear-gradient(180deg, #fff7f3 0%, #fff 70%);}
.kpi.alert .val {color: #c24f1f;}
.kpi.good .val {color: #0f7a54;}
.sec {font-size: 1.06rem; font-weight: 600; color: #14171F; margin: 6px 0 0;}
.sec-sub {font-size: .85rem; color: #7A8394; margin: 0 0 6px;}
.call {border-radius: 12px; padding: 13px 16px; margin: 6px 0 12px; font-size: .92rem;
  line-height: 1.5; color: #14171F; border: 1px solid;}
.call b {font-weight: 600;}
.call.info {background: #F1F6FD; border-color: #cde2fb;}
.call.warn {background: #FFF5F0; border-color: #f8cdb9;}
.call.ok {background: #EFF8F3; border-color: #bfe5d3;}
.call.neutral {background: #F6F7F9; border-color: #E3E6EB;}
.call .ttl {font-size: .76rem; letter-spacing: .07em; text-transform: uppercase; font-weight: 700;
  margin-bottom: 3px;}
.call.info .ttl {color: #1c5cab;} .call.warn .ttl {color: #c24f1f;}
.call.ok .ttl {color: #0f7a54;} .call.neutral .ttl {color: #4B5363;}
.tbl-wrap {overflow-x: auto; margin: 4px 0 6px;}
.tbl {border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 14px;}
.tbl th {font-weight: 600; color: #4B5363; font-size: 12px; padding: 6px 3px; text-align: center;
  white-space: nowrap; background: transparent; border: none;}
.tbl th.rh {text-align: left; padding-left: 2px;}
.tbl th.dm {color: #c24f1f;}
.tbl td {border: 1px solid #E3E6EB; text-align: center; height: 48px; min-width: 50px;
  position: relative; background: #fff; padding: 0;}
.tbl td .c {position: absolute; top: 0; right: 0; font-size: 11px; color: #7A8394; padding: 1px 5px;
  border-left: 1px solid #E3E6EB; border-bottom: 1px solid #E3E6EB; background: #FAFBFC;}
.tbl td .x {font-size: 16px; font-weight: 600; color: #14171F;}
.tbl td .d {font-size: 11.5px; color: #9AA1AD;}
.tbl td.bas {background: #EAF2FC;}
.tbl td.bas .x {color: #1c5cab;}
.tbl td.basdm {background: #FDEEE7;}
.tbl td.basdm .x {color: #c24f1f;}
.tbl td.alt {box-shadow: inset 0 0 0 2px #86b6ef;}
.tbl td.alt .d {color: #1c5cab; font-weight: 600;}
.tbl td.mg {background: #F4F6F9; color: #4B5363; font-weight: 500; height: 34px;}
.tbl td.pt {background: #F5F3FD; color: #4a3aa7; font-weight: 600; height: 34px;}
.tbl th.pt {color: #4a3aa7;}
.legend-row {display: flex; flex-wrap: wrap; gap: 14px; font-size: .8rem; color: #4B5363;
  margin: 2px 0 8px;}
.legend-row span {display: inline-flex; align-items: center; gap: 6px;}
.sw {width: 12px; height: 12px; border-radius: 3px; display: inline-block;}
.foot {text-align: center; color: #7A8394; font-size: .8rem; margin-top: 28px; padding-top: 14px;
  border-top: 1px solid #E3E6EB;}
section[data-testid="stSidebar"] .side-title {font-weight: 700; font-size: 1.05rem; color: #14171F;
  margin-bottom: 0;}
section[data-testid="stSidebar"] .side-sub {font-size: .8rem; color: #7A8394; margin-top: 0;}
section[data-testid="stSidebar"] .side-lab {font-size: .72rem; letter-spacing: .08em;
  text-transform: uppercase; color: #7A8394; font-weight: 700; margin: 10px 0 2px;}
.stTabs [data-baseweb="tab-list"] {gap: 2px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {padding: 8px 12px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------- formatting
def fmt(x, dec=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    x = float(x)
    if abs(x) < 0.5 * 10 ** (-dec):
        x = 0.0
    return ("−" if x < 0 else "") + f"{abs(x):,.{dec}f}"


def pct(x, dec=0):
    return "–" if x is None or np.isnan(x) else fmt(100 * x, dec) + "%"


def signed(x, dec=0):
    return ("+" if x > 1e-9 else "") + fmt(x, dec)


U = "units"
CU = "m.u."
MK = core.MARKETS_EN
SUP = core.SUPPLIERS

# ---------------------------------------------------------------------------- sidebar (inputs)
with st.sidebar:
    st.markdown(
        f'<p class="side-title">TBTF · GPU</p><p class="side-sub">' f"Scenario control panel</p>",
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="side-lab">Input data</p>', unsafe_allow_html=True)
    preset = st.radio(
        "Data set",
        ["paper", "calibrated", "custom"],
        key="preset",
        label_visibility="collapsed",
        format_func=lambda k: {
            "paper": "Paper instance · 500 / 300 / 200",
            "calibrated": "Real market shares · 880 / 120 / 0",
            "custom": "Custom (edit the data)",
        }[k],
    )
    if preset == "calibrated":
        st.caption(
            "Add-in-board shares in Q1 2024 (Jon Peddie Research): NVIDIA 88%, AMD 12%, "
            "Intel below the reported threshold. Same total demand of 1,000 units."
        )

    st.markdown(f'<p class="side-lab">Scenario</p>', unsafe_allow_html=True)
    scen = st.radio(
        "Scenario",
        ["S1", "S2"],
        index=1,
        key="scen",
        label_visibility="collapsed",
        format_func=lambda s: {"S1": "S1 · normal market", "S2": "S2 · NVIDIA blackout"}[s],
    )
    S2 = scen == "S2"
    loss_pct = 0
    if S2:
        loss_pct = st.slider(
            "NVIDIA capacity lost (%)",
            10,
            100,
            80,
            10,
            key="loss",
            help="Export controls, sanctions, a crisis at the TSMC foundry.",
        )
    LOSS = loss_pct / 100


# --------------------------------------------------------------------------------- input data
@st.cache_data(show_spinner=False)
def forecast_data():
    years, data = core.synthetic_history()
    fc = core.rf_forecast(years, data)
    return years, data, fc


YEARS, HIST, FC = forecast_data()
B_RF = core.demand_vector(FC)

top = st.container()  # hero and KPI cards are filled in after the inputs are known

if preset == "custom":
    with st.expander("Edit the input data", expanded=True, icon=":material/edit_square:"):
        st.caption(
            "Unit costs cᵢⱼ (m.u. per unit) and capacities aᵢ by row; demand bⱼ below. "
            "Scenario S2 applies the capacity loss on top of NVIDIA's aᵢ entered here."
        )
        init = pd.DataFrame(core.COST, index=SUP, columns=MK)
        init["a_i"] = core.INSTANCES["paper"]
        ed = st.data_editor(
            init,
            key="ed_cost",
            width="stretch",
            column_config={c: st.column_config.NumberColumn(c, min_value=0, step=1) for c in MK + ["a_i"]},
        )
        dem = st.data_editor(
            pd.DataFrame([core.DEMAND_TARGET], index=["b_j"], columns=MK),
            key="ed_dem",
            width="stretch",
            column_config={c: st.column_config.NumberColumn(c, min_value=0, step=10) for c in MK},
        )
    C = np.nan_to_num(ed[MK].to_numpy(float))
    BASE_A = np.nan_to_num(ed["a_i"].to_numpy(float))
    B = np.nan_to_num(dem[MK].to_numpy(float)[0])
    if B.sum() <= 0 or BASE_A.sum() <= 0:
        st.error("Total demand and total capacity must be positive.")
        st.stop()
else:
    C = core.COST.copy()
    BASE_A = core.INSTANCES[preset].copy()
    B = B_RF.copy()

A = BASE_A.copy()
A[0] = BASE_A[0] * (1 - LOSS)

# --------------------------------------------------------------------------------- models
tup = lambda x: tuple(float(v) for v in np.ravel(x))


@st.cache_data(show_spinner=False)
def transport(a, b, c):
    return core.solve_transport(np.array(a), np.array(b), np.array(c).reshape(3, 5))


@st.cache_data(show_spinner=False)
def flow(a, b):
    return core.max_flow(list(a), list(b))


@st.cache_data(show_spinner=False)
def stress(a0, b0, n, seed, lockin):
    out = []
    for k in range(3):
        if a0[k] <= 0:
            out.append(None)
            continue
        out.append(stx.stress_test(k, n=n, seed=seed, lockin=lockin, a0=np.array(a0), b0=np.array(b0)))
    return out


@st.cache_data(show_spinner=False)
def sweep(a0, b, c):
    return core.sweep(np.array(a0), np.array(b), [i / 10 for i in range(11)], np.array(c).reshape(3, 5))


@st.cache_data(show_spinner=False)
def agent(a_start, a_target, b, c, periods, lot, weights):
    return core.recovery_plan(
        np.array(a_start),
        np.array(a_target),
        np.array(b),
        periods=periods,
        lot=lot,
        weights=np.array(weights),
        C=np.array(c).reshape(3, 5),
    )


R = transport(tup(A), tup(B), tup(C))
REF = transport(tup(BASE_A), tup(B), tup(C))
MF = flow(tup(A), tup(B))

DEM = B.sum()
UNMET = R["unmet"]
DELIV = DEM - UNMET.sum()
ROWS = SUP + (["A* (fictitious)"] if R["kind"] == "dummy_supplier" else [])
COLS = MK + (["B* (fictitious)"] if R["kind"] == "dummy_market" else [])

# --------------------------------------------------------------------------------- helpers
PLOT_CFG = {"displayModeBar": False, "responsive": True}


def style(fig, height=320, legend=True, top=34):
    fig.update_layout(
        template="none",
        height=height,
        margin=dict(l=6, r=10, t=top, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, Source Sans Pro, sans-serif", size=13, color=INK2),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(color=INK, size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            x=0,
            title=None,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
            traceorder="normal",
        ),
        showlegend=legend,
        separators=".,",
        bargap=0.32,
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=LINE,
        showline=True,
        ticks="outside",
        tickcolor=LINE,
        ticklen=4,
        automargin=True,
        tickfont=dict(size=12, color=INK2),
        title_font=dict(size=12, color=MUTED),
        title_standoff=8,
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=LINE,
        automargin=True,
        tickfont=dict(size=12, color=INK2),
        title_font=dict(size=12, color=MUTED),
        title_standoff=8,
    )
    return fig


def show(fig):
    st.plotly_chart(fig, theme=None, config=PLOT_CFG)


def sec(title, sub=None):
    st.markdown(
        f'<p class="sec">{title}</p>' + (f'<p class="sec-sub">{sub}</p>' if sub else ""),
        unsafe_allow_html=True,
    )


def call(kind, title, body):
    st.markdown(
        f'<div class="call {kind}"><div class="ttl">{title}</div>{body}</div>', unsafe_allow_html=True
    )


def legend_row(items):
    st.markdown(
        '<div class="legend-row">'
        + "".join(
            f'<span><i class="sw" style="background:{bg}; box-shadow: inset 0 0 0 1.5px {bd}"></i>{lab}</span>'
            for bg, bd, lab in items
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def tableau(X, Cm, rlab, clab, a, b, basis=None, red=None, u=None, v=None):
    m, n = X.shape
    bas = set(basis) if basis is not None else {(i, j) for i in range(m) for j in range(n) if X[i, j] > 1e-9}
    dr = R["kind"] == "dummy_supplier"
    dc = R["kind"] == "dummy_market"
    h = ['<div class="tbl-wrap"><table class="tbl"><thead><tr><th></th>']
    for j in range(n):
        h.append(f'<th class="{"dm" if dc and j == n - 1 else ""}">{clab[j]}</th>')
    h.append("<th>a<sub>i</sub></th>" + ('<th class="pt">u<sub>i</sub></th>' if u is not None else ""))
    h.append("</tr></thead><tbody>")
    for i in range(m):
        h.append(f'<tr><th class="rh{" dm" if dr and i == m - 1 else ""}">{rlab[i]}</th>')
        for j in range(n):
            fict = (dr and i == m - 1) or (dc and j == n - 1)
            inner = f'<span class="c">{fmt(Cm[i, j])}</span>'
            cls = ""
            if (i, j) in bas:
                cls = "basdm" if fict else "bas"
                inner += f'<span class="x">{fmt(X[i, j])}</span>'
            elif red is not None and (i, j) in red:
                d = red[(i, j)]
                cls = "alt" if abs(d) < 1e-9 else ""
                inner += f'<span class="d">δ {signed(d)}</span>'
            h.append(f'<td class="{cls}">{inner}</td>')
        h.append(
            f'<td class="mg">{fmt(a[i])}</td>' + (f'<td class="pt">{fmt(u[i])}</td>' if u is not None else "")
        )
        h.append("</tr>")
    h.append(
        '<tr><th class="rh">b<sub>j</sub></th>'
        + "".join(f'<td class="mg">{fmt(x)}</td>' for x in b)
        + f'<td class="mg"><b>{fmt(sum(b))}</b></td>'
        + ("<td></td>" if u is not None else "")
        + "</tr>"
    )
    if v is not None:
        h.append(
            '<tr><th class="rh pt">v<sub>j</sub></th>'
            + "".join(f'<td class="pt">{fmt(x)}</td>' for x in v)
            + "<td></td><td></td></tr>"
        )
    h.append("</tbody></table></div>")
    return "".join(h)


def cell_name(c):
    i, j = c
    return f"{ROWS[i]} → {COLS[j]}"


# ------------------------------------------------------------------------------ hero + KPIs
with top:
    scen_pill = (
        f'<span class="pill warn">S2 · NVIDIA −{loss_pct}%</span>'
        if S2
        else f'<span class="pill ok">S1 · normal market</span>'
    )
    preset_name = {
        "paper": "Paper instance",
        "calibrated": "Real shares 88 / 12 / 0",
        "custom": "Custom data",
    }[preset]
    st.markdown(
        f"""<div class="hero">
<div class="kicker">Systemic risk in the GPU supply chain</div>
<h1>Is NVIDIA “too big to fail”?</h1>
<p class="sub">Demand forecast with a Random Forest, minimum-cost allocation (North-West Corner + MODI), maximum flow with the minimum cut and a Monte Carlo stress test — recomputed every time the slider moves.</p>
<div class="pills">{scen_pill}<span class="pill">{preset_name}</span><span class="pill">NVIDIA {fmt(A[0])} · AMD {fmt(A[1])} · Intel Arc {fmt(A[2])} {U}</span></div>
<div class="meta"><b>National University of Science and Technology POLITEHNICA Bucharest</b> · Faculty of Applied Sciences · Year III · Group 1333<br>
Authors: <b>Anișoara-Nicoleta Dedu</b> · <b>Andreea-Mihaela Dumitrescu</b> · <b>Daria-Gabriela Iliescu</b> · <b>Diana-Ionela Lungu</b> &nbsp;·&nbsp; Supervisor: <b>Lect. Dr. Simona Mihaela Bibic</b></div>
</div>""",
        unsafe_allow_html=True,
    )

    cov = DELIV / DEM
    dz = (R["z"] - REF["z"]) / REF["z"] if REF["z"] else 0
    cards = [
        (
            "",
            "Total supply Σaᵢ",
            fmt(A.sum()),
            U,
            f"vs {fmt(BASE_A.sum())} in S1" if S2 else "all suppliers at capacity",
        ),
        (
            "",
            "Demand Σbⱼ",
            fmt(DEM),
            U,
            "entered manually" if preset == "custom" else "RF forecast for 2026, rounded to 50",
        ),
        (
            "good" if UNMET.sum() == 0 else "",
            "Delivered (max flow)",
            fmt(MF["value"]),
            U,
            f"{pct(cov)} of demand",
        ),
        (
            "alert" if UNMET.sum() > 0 else "good",
            "Unmet demand D",
            fmt(UNMET.sum()),
            U,
            f"{pct(1 - cov)} of demand" if UNMET.sum() > 0 else "all demand is served",
        ),
        (
            "",
            "Optimal cost z*",
            fmt(R["z"]),
            CU,
            f"S1: {fmt(REF['z'])} ({signed(100 * dz)}%)" if S2 else f"North-West Corner: {fmt(R['z0'])}",
        ),
    ]
    st.markdown(
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi {k}"><div class="lab">{lab}</div><div class="val">{val}<span class="u">{u}</span></div>'
            f'<div class="sub">{sub}</div></div>'
            for k, lab, val, u, sub in cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------- tabs
tabs = st.tabs(
    [
        ":material/dashboard: Overview",
        ":material/trending_up: Demand (ML)",
        ":material/local_shipping: Transport (cost)",
        ":material/account_tree: Maximum flow",
        ":material/bolt: Stress test",
        ":material/smart_toy: Recovery agent",
        ":material/domain: Who depends on NVIDIA",
        ":material/menu_book: Method and theory",
    ]
)

STRESS_DEFAULT = stress(tup(BASE_A), tup(B), 20000, 2026, False)

# ===================================================================================== 1. overview
with tabs[0]:
    m_real, n_real = 3, 5
    X = R["X"]
    sec(
        "Who supplies which market",
        (
            "The minimum-cost plan. The orange flow comes from the fictitious supplier A*: it is demand that cannot be served."
            if R["kind"] == "dummy_supplier"
            else "The minimum-cost plan: the width of each band is the quantity shipped."
        ),
    )
    node_lab, node_col = [], []
    for i, name in enumerate(ROWS):
        node_lab.append(f"{name} · {fmt(X[i].sum())}")
        node_col.append(SUP_COL[i] if i < 3 else DEFICIT)
    for j, name in enumerate(COLS):
        node_lab.append(f"{name} · {fmt(R['B'][j])}")
        node_col.append(MARKET_COL if j < 5 else IDLE)
    src, tgt, val, lcol = [], [], [], []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] > 1e-9:
                src.append(i)
                tgt.append(len(ROWS) + j)
                val.append(X[i, j])
                lcol.append(rgba(SUP_COL[i], 0.42) if i < 3 else rgba(DEFICIT, 0.55))
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            valueformat=",.0f",
            valuesuffix=f" {U}",
            node=dict(
                label=node_lab,
                color=node_col,
                pad=16,
                thickness=14,
                line=dict(width=0),
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=src,
                target=tgt,
                value=val,
                color=lcol,
                hovertemplate="%{source.label} → %{target.label}<br><b>%{value}</b><extra></extra>",
            ),
        )
    )
    style(fig, height=360, legend=False, top=10)
    fig.update_layout(font=dict(size=13, color=INK))
    show(fig)

    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        sec("Demand coverage by market", "Deliveries by supplier and the demand left unmet.")
        fig = go.Figure()
        ylab = MK[::-1]
        for i in range(3):
            fig.add_bar(
                y=ylab,
                x=X[i, :5][::-1],
                name=SUP[i],
                orientation="h",
                marker_color=SUP_COL[i],
                marker_line=dict(color="white", width=2),
                hovertemplate=f"{SUP[i]} → %{{y}}: <b>%{{x:,.0f}}</b> {U}<extra></extra>",
            )
        fig.add_bar(
            y=ylab,
            x=UNMET[::-1],
            name="unmet",
            orientation="h",
            marker=dict(color=DEFICIT, pattern=HATCH, line=dict(color="white", width=2)),
            hovertemplate="unmet in" + f" %{{y}}: <b>%{{x:,.0f}}</b> {U}<extra></extra>",
        )
        for j in range(5):
            share = UNMET[j] / B[j] if B[j] else 0
            fig.add_annotation(
                x=B[j],
                y=MK[j],
                xanchor="left",
                xshift=6,
                showarrow=False,
                text=(f"<b>{pct(share)}</b> " + "unmet") if share > 0 else "✓",
                font=dict(size=12, color=DEFICIT_TXT if share > 0 else "#0f7a54"),
            )
        style(fig, height=300)
        fig.update_layout(barmode="stack", barcornerradius=4)
        fig.update_xaxes(title=U, range=[0, max(B) * 1.32])
        fig.update_yaxes(showgrid=False)
        show(fig)
    with c2:
        sec("What the numbers say")
        if R["kind"] == "dummy_supplier":
            call(
                "warn",
                "The cost paradox",
                f"The optimal cost falls from <b>{fmt(REF['z'])}</b> to <b>{fmt(R['z'])} {CU}</b> ({signed(100 * dz)}%) "
                f"although {pct(1 - cov)} of demand goes unserved. Units “shipped” by A* cost nothing, "
                "so a lower cost here means a deeper crisis, not a leaner network.",
            )
            share_m = sorted(
                [(UNMET[j] / B[j], MK[j]) for j in range(5) if B[j] > 0 and UNMET[j] > 0], reverse=True
            )
            worst = ", ".join(f"{name} ({pct(s)})" for s, name in share_m[:3])
            call(
                "neutral",
                "Where the shortage lands",
                f"Most exposed markets: <b>{worst}</b>. The optimiser serves the cheap routes first, "
                "so the markets that are expensive to reach are the first to go without.",
            )
        else:
            call(
                "ok",
                "Balanced network",
                f"All demand is served at a minimum cost of <b>{fmt(R['z'])} {CU}</b>. "
                "Switch to S2 to see what happens when NVIDIA loses capacity.",
            )
        s = STRESS_DEFAULT
        if s[0] is not None:
            parts = []
            for k in (1, 2):
                if s[k] is not None and s[k]["SI"] > 1e-6:
                    parts.append(f"{fmt(s[0]['SI'] / s[k]['SI'], 1)}× as much as {SUP[k]}")
            rel = (" — " + " and ".join(parts)) if parts else ""
            call(
                "info",
                "Stress-test verdict",
                f"A random failure at NVIDIA leaves on average <b>{pct(s[0]['SI'])}</b> of demand unserved{rel}. "
                "Details in the “Stress test” tab.",
            )

# ===================================================================================== 2. demand
with tabs[1]:
    sec(
        "Demand forecast by region",
        "Synthetic 2018–2024 series (calibrated to the case-study demand) and the Random Forest forecast for 2025–2026.",
    )
    fig = make_subplots(rows=1, cols=5, shared_yaxes=True, subplot_titles=MK, horizontal_spacing=0.025)
    for k in range(5):
        d25, d26 = FC[k]
        sl = k == 0
        fig.add_scatter(
            x=YEARS,
            y=HIST[k],
            mode="lines+markers",
            line=dict(color=SUP_COL[0], width=2),
            marker=dict(size=6),
            name="history (synthetic)",
            legendgroup="h",
            showlegend=sl,
            row=1,
            col=k + 1,
            hovertemplate="%{x}: <b>%{y:,.1f}</b>" + f" {U}<extra></extra>",
        )
        fig.add_scatter(
            x=[YEARS[-1], 2025, 2026],
            y=[HIST[k][-1], d25, d26],
            mode="lines+markers",
            line=dict(color=SUP_COL[0], width=2, dash="dot"),
            marker=dict(size=7, symbol="circle-open"),
            name="RF forecast",
            legendgroup="f",
            showlegend=sl,
            row=1,
            col=k + 1,
            hovertemplate="%{x}: <b>%{y:,.1f}</b>" + f" {U}<extra></extra>",
        )
        fig.add_scatter(
            x=[2026],
            y=[B_RF[k]],
            mode="markers",
            marker=dict(size=11, symbol="diamond", color=INK, line=dict(color="white", width=2)),
            name="bⱼ used in the model",
            legendgroup="b",
            showlegend=sl,
            row=1,
            col=k + 1,
            hovertemplate="bⱼ 2026: <b>%{y:,.0f}</b>" + f" {U}<extra></extra>",
        )
    style(fig, height=330, top=60)
    fig.update_xaxes(tickvals=[2018, 2022, 2026], range=[2017.4, 2026.6], showgrid=False)
    fig.update_yaxes(title=None)
    fig.update_yaxes(title=U, row=1, col=1)
    fig.update_annotations(font=dict(size=13, color=INK))
    fig.update_layout(legend=dict(y=1.2))
    show(fig)

    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        df = pd.DataFrame(
            {
                "Region": MK,
                "2024": [HIST[k][-1] for k in range(5)],
                "RF 2025": [FC[k][0] for k in range(5)],
                "RF 2026": [FC[k][1] for k in range(5)],
                "bⱼ (rounded to 50)": B_RF,
            }
        )
        if preset == "custom":
            df["bⱼ used (manual)"] = B
        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.1f") for c in df.columns[1:4]},
        )
    with c2:
        call(
            "info",
            "How the demand vector is built",
            "For each region a Random Forest (200 trees, depth 4) learns the year-on-year log growth and applies "
            "it twice, for 2025 and 2026. The 2026 value is rounded to multiples of 50 units and becomes the demand "
            "bⱼ of the transportation problem. The history is synthetic: the model shows the mechanism, it does "
            "not estimate the real market.",
        )
        if preset == "custom":
            call(
                "neutral",
                "Custom mode",
                "The models use the demand entered manually; the forecast is kept for comparison.",
            )

# ===================================================================================== 3. transport
with tabs[2]:
    sa, sb = A.sum(), B.sum()
    if R["kind"] == "dummy_supplier":
        call(
            "warn",
            "Unbalanced problem → fictitious supplier A*",
            f"Σaᵢ = {fmt(sa)} < Σbⱼ = {fmt(sb)}. A fictitious supplier A* is added with a* = <b>{fmt(R['gap'])} units</b> "
            "and zero cost on every route. Whatever A* “ships” is unmet demand.",
        )
    elif R["kind"] == "dummy_market":
        call(
            "info",
            "Unbalanced problem → fictitious market B*",
            f"Σaᵢ = {fmt(sa)} > Σbⱼ = {fmt(sb)}. A fictitious market B* is added with b* = <b>{fmt(R['gap'])} units</b> "
            "and zero cost: it is the capacity left idle.",
        )
    else:
        call("ok", "Balanced problem", f"Σaᵢ = Σbⱼ = {fmt(sa)} units, so the method applies directly.")

    c1, c2 = st.columns([1, 1.14], gap="medium")
    with c1:
        sec("1 · Initial plan — North-West Corner", f"Cost z₀ = <b>{fmt(R['z0'])} {CU}</b>")
        st.markdown(
            tableau(R["X0"], R["C"], ROWS, COLS, R["A"], R["B"], basis=R["basis0"]), unsafe_allow_html=True
        )
    with c2:
        sec(
            "2 · Optimal plan — MODI",
            f"Cost z* = <b>{fmt(R['z'])} {CU}</b> · potentials uᵢ, vⱼ and reduced costs δᵢⱼ",
        )
        st.markdown(
            tableau(
                R["X"],
                R["C"],
                ROWS,
                COLS,
                R["A"],
                R["B"],
                basis=R["basis"],
                red=R["reduced"],
                u=R["u"],
                v=R["v"],
            ),
            unsafe_allow_html=True,
        )
    legend_row(
        [
            ("#EAF2FC", "#86b6ef", "basic cell (xᵢⱼ)"),
            ("#FDEEE7", "#f5a582", "fictitious shipment = unmet demand"),
            ("#F5F3FD", "#9085e9", "potentials uᵢ, vⱼ"),
            ("#FFFFFF", "#86b6ef", "δᵢⱼ = 0: alternative optimum"),
        ]
    )
    st.caption(
        "Each cell's corner shows the unit cost cᵢⱼ. Non-basic cells show δᵢⱼ = cᵢⱼ − (uᵢ + vⱼ); "
        "the plan is optimal when every δᵢⱼ ≥ 0."
    )

    c1, c2 = st.columns([1.9, 1], gap="large")
    with c1:
        sec(
            "3 · MODI iterations",
            "Entering cell, most negative δ, quantity θ moved around the cycle and leaving cell.",
        )
        if R["log"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Iteration": k + 1,
                            "Enters": cell_name(e["enter"]),
                            "δ": e["delta"],
                            "θ": e["theta"],
                            "Leaves": cell_name(e["leave"]),
                            "Cost before": e["cost_before"],
                            "Cost after": e["cost_after"],
                        }
                        for k, e in enumerate(R["log"])
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "δ": st.column_config.NumberColumn(format="%.0f"),
                    "θ": st.column_config.NumberColumn(format="%.0f"),
                },
            )
        else:
            st.info("The North-West Corner plan is already optimal: no iterations.", icon=":material/check:")
    with c2:
        sec("4 · Check")
        ok = abs(R["z"] - R["z_lp"]) < 1e-6
        call(
            "ok" if ok else "warn",
            "MODI vs HiGHS",
            f"z*(MODI) = <b>{fmt(R['z'])}</b>, z*(HiGHS, SciPy) = <b>{fmt(R['z_lp'])}</b> "
            + ("— the same optimal value." if ok else "— the values differ!"),
        )
        if R["alt_optima"]:
            cells = ", ".join(cell_name(c) for c in R["alt_optima"])
            call(
                "info",
                "Alternative optima",
                f"δ = 0 in {cells}: other plans have the same cost z*, so the route allocation is not unique — the cost is.",
            )

    with st.expander("Problem data", icon=":material/table:"):
        df = pd.DataFrame(C, index=SUP, columns=MK)
        df["Capacity aᵢ"] = A
        df.loc["Demand bⱼ"] = list(B) + [B.sum()]
        st.dataframe(
            df,
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.0f") for c in df.columns},
        )
        st.caption("Unit costs cᵢⱼ in m.u. per unit; capacities follow the selected scenario.")

# ===================================================================================== 4. max flow
with tabs[3]:
    m, n = 3, 5
    s_node, t_node = 0, m + n + 1
    cut_arcs = {(x, y) for x, y, _ in MF["cut"]}
    Xr = R["X"][:3, :5]
    xmax = max(Xr.max(), 1)

    def node_id(k):
        if k == s_node:
            return "s"
        if k == t_node:
            return "t"
        return f"S{k - 1}" if k <= m else f"M{k - m - 1}"

    lines = [
        "digraph G {",
        'rankdir=LR; bgcolor="transparent"; nodesep=0.28; ranksep=1.35; pad=0.15;',
        'node [fontname="Helvetica", fontsize=13, shape=box, style="rounded,filled", color="#C9CED6", '
        'fillcolor="white", fontcolor="#14171F", penwidth=1.2, margin="0.14,0.05"];',
        'edge [fontname="Helvetica", fontsize=10.5, color="#B8BEC8", fontcolor="#4B5363", arrowsize=0.6];',
        f's [shape=circle, label="s", style=filled, fillcolor="{NAVY}", fontcolor="white", color="{NAVY}", width=0.5, fixedsize=true];',
        f't [shape=circle, label="t", style=filled, fillcolor="{NAVY}", fontcolor="white", color="{NAVY}", width=0.5, fixedsize=true];',
    ]
    for i in range(m):
        lines.append(
            f'S{i} [label="{SUP[i]}\\na = {fmt(A[i])}", color="{SUP_COL[i]}", fillcolor="{SUP_COL[i]}1c", penwidth=1.8];'
        )
    for j in range(n):
        lab = f"{MK[j]}\\nb = {fmt(B[j])}"
        extra = f', color="{DEFICIT}", fillcolor="#FFF5F0"' if UNMET[j] > 1e-9 else ""
        lines.append(f'M{j} [label="{lab}"{extra}];')
    lines.append("{rank=same; S0 -> S1 -> S2 [style=invis];}")
    lines.append("{rank=same; M0 -> M1 -> M2 -> M3 -> M4 [style=invis];}")
    for i in range(m):
        f_, c_ = Xr[i].sum(), A[i]
        if (s_node, 1 + i) in cut_arcs:
            lines.append(
                f's -> S{i} [label="{fmt(f_)} / {fmt(c_)}", color="{DEFICIT}", fontcolor="{DEFICIT_TXT}", penwidth=3.2];'
            )
        else:
            lines.append(f's -> S{i} [label="{fmt(f_)} / {fmt(c_)}"];')
        for j in range(n):
            if Xr[i, j] > 1e-9:
                w = 1 + 3.2 * Xr[i, j] / xmax
                lines.append(
                    f'S{i} -> M{j} [label="{fmt(Xr[i, j])}", color="{SUP_COL[i]}", penwidth={w:.2f}];'
                )
    for j in range(n):
        f_ = Xr[:, j].sum()
        if (1 + m + j, t_node) in cut_arcs:
            lines.append(
                f'M{j} -> t [label="{fmt(f_)} / {fmt(B[j])}", color="{DEFICIT}", fontcolor="{DEFICIT_TXT}", penwidth=3.2];'
            )
        else:
            lines.append(f'M{j} -> t [label="{fmt(f_)} / {fmt(B[j])}"];')
    lines.append("}")

    c1, c2 = st.columns([2.1, 1], gap="large")
    with c1:
        sec(
            "The capacity network",
            "s → suppliers (capacity aᵢ) → markets (uncapacitated routes) → t (demand bⱼ). The orange arcs form the minimum cut.",
        )
        st.graphviz_chart("\n".join(lines), width="stretch")
        st.caption(
            "Labels: flow / capacity. Supplier → market arcs show the minimum-cost plan, which is also a "
            "maximum flow (a minimum-cost maximum flow)."
        )
    with c2:
        cut_cap = sum(c for _, _, c in MF["cut"])
        sec("Max-flow min-cut theorem")
        st.markdown(
            '<div class="kpis" style="grid-template-columns:1fr 1fr">'
            + "".join(
                f'<div class="kpi {k}"><div class="lab">{lab}</div><div class="val">{val}<span class="u">{U}</span></div></div>'
                for k, lab, val in [
                    ("", "Max flow v", fmt(MF["value"])),
                    ("", "Minimum cut", fmt(cut_cap)),
                    ("alert" if UNMET.sum() > 0 else "good", "D = Σbⱼ − v", fmt(DEM - MF["value"])),
                    ("", "Shipped by A*", fmt(R["X"][3].sum()) if R["kind"] == "dummy_supplier" else fmt(0)),
                ]
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        same = abs((DEM - MF["value"]) - UNMET.sum()) < 1e-6
        call(
            "ok" if same else "warn",
            "The two models agree",
            f"The network deficit, Σbⱼ − v = {fmt(DEM - MF['value'])}, equals exactly the quantity shipped by the "
            "fictitious supplier in the transportation problem.",
        )
        src_cut = all(x == s_node for x, _ in cut_arcs)
        call(
            "info",
            "Where the bottleneck is",
            (
                "The minimum cut runs through the source arcs: the limit is production capacity, not the routes. "
                "No rerouting can deliver more than is produced."
                if src_cut
                else "The minimum cut runs through the arcs into the sink: demand limits the flow and part of the "
                "capacity stays idle."
            ),
        )

    names = ["s"] + SUP + MK + ["t"]
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        sec(
            "Augmenting paths (Edmonds–Karp)",
            "Each path is a shortest s → t path in the residual graph; α is its smallest residual capacity.",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"#": k + 1, "Path": " → ".join(names[x] for x in p), "α": a}
                    for k, (p, a) in enumerate(MF["paths"])
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={"α": st.column_config.NumberColumn(format="%.0f")},
        )
    with c2:
        sec("Minimum-cut arcs")
        st.dataframe(
            pd.DataFrame([{"Arc": f"{names[x]} → {names[y]}", "Capacity": c} for x, y, c in MF["cut"]]),
            hide_index=True,
            width="stretch",
            column_config={"Capacity": st.column_config.NumberColumn(format="%.0f")},
        )

# ===================================================================================== 5. stress test
with tabs[4]:
    sec(
        "How systemically important is each supplier?",
        "For each supplier k, N random failures are simulated (Algorithm 1 in the paper)."
        + (
            " With the paper instance, the default values reproduce the published results."
            if preset == "paper"
            else " The baseline is the S1 capacity of the selected data set."
        ),
    )
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 1, 1.3], vertical_alignment="center")
        n_sim = c1.select_slider(
            "Simulated scenarios N",
            [2000, 5000, 10000, 20000, 50000],
            20000,
            key="n_sim",
            format_func=lambda v: fmt(v),
        )
        seed = int(c2.number_input("Seed", 0, 10**6, 2026, 1, key="seed"))
        lock = c3.toggle(
            "CUDA lock-in (λ ~ U(0.1, 0.5))",
            False,
            key="lock",
            help="A share λ of every market's demand can only be served by NVIDIA.",
        )
    ST = stress(tup(BASE_A), tup(B), n_sim, seed, lock)
    live = [k for k in range(3) if ST[k] is not None]

    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        sec(
            "Distribution of demand lost",
            "Box: quartiles 1–3 and the median; whiskers: 5th and 95th percentiles; diamond: the mean (SIₖ).",
        )
        fig = go.Figure()
        for k in live[::-1]:
            r = ST[k]
            fig.add_trace(
                go.Box(
                    y=[SUP[k]],
                    q1=[100 * r["q1"]],
                    median=[100 * r["median"]],
                    q3=[100 * r["q3"]],
                    lowerfence=[100 * r["p5"]],
                    upperfence=[100 * r["p95"]],
                    orientation="h",
                    name=SUP[k],
                    fillcolor=rgba(SUP_COL[k], 0.22),
                    line=dict(color=SUP_COL[k], width=2),
                    width=0.5,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_scatter(
                x=[100 * r["SI"]],
                y=[SUP[k]],
                mode="markers",
                showlegend=False,
                marker=dict(symbol="diamond", size=12, color=INK, line=dict(color="white", width=2)),
                hovertemplate=(
                    f"<b>{SUP[k]}</b><br>SI = {pct(r['SI'], 1)}<br>"
                    f"median {pct(r['median'], 1)} · p95 {pct(r['p95'], 1)}"
                    "<extra></extra>"
                ),
            )
            fig.add_annotation(
                x=100 * r["p95"],
                y=SUP[k],
                xanchor="left",
                xshift=8,
                showarrow=False,
                text=f"<b>{pct(r['SI'])}</b>",
                font=dict(size=13, color=INK),
            )
        style(fig, height=280, legend=False, top=10)
        fig.update_xaxes(title="demand lost (%)", ticksuffix="%", rangemode="tozero")
        fig.update_yaxes(showgrid=False)
        show(fig)
    with c2:
        sec("Capacity share vs expected loss", "If risk were proportional, the bars would be equally tall.")
        cap_share = BASE_A / BASE_A.sum()
        fig = go.Figure()
        fig.add_bar(
            x=[SUP[k] for k in live],
            y=[100 * cap_share[k] for k in live],
            name="capacity share",
            marker_color=IDLE,
            hovertemplate="%{x}: <b>%{y:.1f}%</b><extra></extra>",
        )
        fig.add_bar(
            x=[SUP[k] for k in live],
            y=[100 * ST[k]["SI"] for k in live],
            name="SI (demand lost)",
            marker_color=DEFICIT,
            hovertemplate="%{x}: <b>%{y:.1f}%</b><extra></extra>",
        )
        style(fig, height=280, top=30)
        fig.update_layout(barmode="group", barcornerradius=4, bargroupgap=0.12)
        fig.update_yaxes(ticksuffix="%")
        show(fig)

    rows = []
    for k in live:
        r = ST[k]
        rows.append(
            {
                "Supplier": SUP[k],
                "Capacity": BASE_A[k],
                "SI": 100 * r["SI"],
                "Median": 100 * r["median"],
                "p95": 100 * r["p95"],
                "Absorbed by the others": 100 * r["absorbed"],
                "P(> 20%)": 100 * r["P(share>20%)"],
            }
        )
    pc = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "SI": pc,
            "Median": pc,
            "p95": pc,
            "Absorbed by the others": pc,
            "P(> 20%)": pc,
            "Capacity": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    if len(live) < 3:
        st.caption("Suppliers with zero capacity in this data set have nothing to lose and are left out.")
    if len(live) == 3 and ST[1]["SI"] > 0 and ST[2]["SI"] > 0:
        call(
            "info",
            "Risk grows faster than size",
            f"NVIDIA has {fmt(BASE_A[0] / BASE_A[1], 2)}× AMD's capacity, but its failure costs "
            f"{fmt(ST[0]['SI'] / ST[1]['SI'], 2)}× as much; against Intel Arc the ratio is {fmt(BASE_A[0] / BASE_A[2], 2)}× "
            f"in capacity and {fmt(ST[0]['SI'] / ST[2]['SI'], 2)}× in loss. The others' reserves absorb on average "
            f"only {pct(ST[0]['absorbed'])} of the capacity NVIDIA loses.",
        )

    st.divider()
    sec(
        "Deterministic sensitivity analysis",
        "NVIDIA loses 0–100% of its capacity step by step; AMD and Intel Arc stay at their base level.",
    )
    SW = sweep(tup(BASE_A), tup(B), tup(C))
    xs = [100 * r["loss"] for r in SW]
    c1, c2 = st.columns(2, gap="large")
    for col, key, title, color in [
        (c1, "deficit", "Unmet demand", DEFICIT),
        (c2, "cost", "Optimal cost z*", SUP_COL[0]),
    ]:
        with col:
            ys = [r[key] for r in SW]
            fig = go.Figure()
            fig.add_scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=8, line=dict(color="white", width=2)),
                name=title,
                hovertemplate="loss" + " %{x:.0f}%: <b>%{y:,.0f}</b><extra></extra>",
            )
            if S2:
                fig.add_vline(x=loss_pct, line=dict(color=INK2, width=1, dash="dash"))
                fig.add_annotation(
                    x=loss_pct,
                    y=1,
                    yref="paper",
                    text="current scenario",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=11, color=INK2),
                )
            style(fig, height=270, legend=False, top=36)
            fig.update_layout(
                title=dict(
                    text=f"{title} ({U if key == 'deficit' else CU})",
                    x=0,
                    y=0.98,
                    font=dict(size=13, color=INK),
                )
            )
            fig.update_xaxes(title="NVIDIA capacity lost (%)", ticksuffix="%", dtick=20)
            fig.update_yaxes(rangemode="tozero")
            show(fig)
    st.caption(
        "The two measures have different units, so they are drawn separately. The larger the loss, the "
        "higher the deficit and the lower the cost — the fictitious-supplier paradox."
    )

    with st.expander("Algorithm 1 — the stress test", icon=":material/code:"):
        st.code(
            "for each supplier k:\n"
            "    repeat N times:\n"
            "        θ ~ U(0.2, 1)          # share of k's capacity that is lost\n"
            "        σᵢ ~ U(0, 0.2)         # reserve activated by each supplier\n"
            "        εⱼ ~ U(−0.1, 0.1)      # demand variation in market j\n"
            "        D_ref  = Σb' − maxflow(a·(1+σ), b')\n"
            "        D_fail = Σb' − maxflow(a·(1+σ) with aₖ ← aₖ(1−θ), b')\n"
            "        ΔD = D_fail − D_ref\n"
            "    SIₖ = E[ΔD] / E[Σb'];   absorbed = 1 − ΔD / capacity lost",
            language=None,
        )
        st.caption(
            "With uncapacitated routes the maximum flow is min(Σa, Σb), so the simulation uses the closed form; "
            "it matches the linear-programming solution. Code: stress_test.py."
        )

# ===================================================================================== 6. agent
with tabs[5]:
    call(
        "neutral",
        "Prototype",
        "The agent is not part of the paper's results; it is a first step towards the multi-period extension "
        "listed there as future work. In each period NVIDIA's capacity recovers linearly towards its S1 level and "
        "the agent re-optimises shipments in whole lots, protecting the priority markets first.",
    )
    if not S2:
        st.info("There is nothing to recover in S1. Choose S2 in the sidebar.", icon=":material/info:")
    else:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1, 1, 1.6, 1], vertical_alignment="center")
            periods = c1.slider("Recovery periods", 2, 12, 6, key="periods")
            lot = c2.select_slider("Shipment lot (units)", [1, 10, 25, 50, 100], 50, key="lot")
            prio = c3.multiselect("Priority markets", MK, default=[MK[4]], key="prio")
            wgt = c4.slider("Priority weight", 1, 5, 3, key="wgt")
        weights = tuple(float(wgt) if MK[j] in prio else 1.0 for j in range(5))
        PLAN = agent(tup(A), tup(BASE_A), tup(B), tup(C), periods, lot, weights)
        P = [f"P{p['period']}" for p in PLAN]

        c1, c2 = st.columns([1.15, 1], gap="large")
        with c1:
            sec("Shipments by period", "What each supplier ships and what stays unmet.")
            fig = go.Figure()
            for i in range(3):
                fig.add_bar(
                    x=P,
                    y=[p["X"][i].sum() for p in PLAN],
                    name=SUP[i],
                    marker_color=SUP_COL[i],
                    marker_line=dict(color="white", width=2),
                    hovertemplate=f"{SUP[i]} · %{{x}}: <b>%{{y:,.0f}}</b> {U}<extra></extra>",
                )
            fig.add_bar(
                x=P,
                y=[p["unmet"].sum() for p in PLAN],
                name="unmet",
                marker=dict(color=DEFICIT, pattern=HATCH, line=dict(color="white", width=2)),
                hovertemplate="unmet" + f" · %{{x}}: <b>%{{y:,.0f}}</b> {U}<extra></extra>",
            )
            style(fig, height=320)
            fig.update_layout(barmode="stack", barcornerradius=4)
            fig.update_yaxes(title=U)
            show(fig)
        with c2:
            sec("Unmet demand by market", "Priority markets are served first.")
            Z = np.array([p["unmet"] for p in PLAN]).T
            fig = go.Figure(
                go.Heatmap(
                    z=Z,
                    x=P,
                    y=MK,
                    colorscale=ORANGE_RAMP,
                    zmin=0,
                    zmax=max(B.max(), 1),
                    xgap=2,
                    ygap=2,
                    text=[[fmt(v) if v > 0 else "" for v in row] for row in Z],
                    texttemplate="%{text}",
                    textfont=dict(size=12),
                    colorbar=dict(thickness=10, outlinewidth=0, title=dict(text=U, font=dict(size=11))),
                    hovertemplate="%{y} · %{x}: <b>%{z:,.0f}</b>" + f" {U}<extra></extra>",
                )
            )
            style(fig, height=320, legend=False, top=10)
            fig.update_yaxes(
                autorange="reversed",
                showgrid=False,
                ticktext=[f"<b>{m_}</b> ★" if m_ in prio else m_ for m_ in MK],
                tickvals=MK,
            )
            fig.update_xaxes(showgrid=False, showline=False)
            show(fig)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Period": f"P{p['period']}",
                        "NVIDIA capacity": p["capacity"][0],
                        "Delivered": p["X"].sum(),
                        "Unmet": p["unmet"].sum(),
                        "Transport cost": p["cost"],
                        "Idle capacity (lot remainder)": p["idle"],
                    }
                    for p in PLAN
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%.0f")
                for c in [
                    "NVIDIA capacity",
                    "Delivered",
                    "Unmet",
                    "Transport cost",
                    "Idle capacity (lot remainder)",
                ]
            },
        )
        st.caption(
            "Formulation: in each period, min Σ lot·cᵢⱼ·kᵢⱼ + M·Σ wⱼ·uⱼ with integer kᵢⱼ (number of lots), "
            "Σⱼ lot·kᵢⱼ ≤ aᵢ and Σᵢ lot·kᵢⱼ + uⱼ = bⱼ. The large penalty M makes the agent serve as much as possible "
            "first, then the high-weight markets, then cut cost. Solved with HiGHS (scipy.optimize.milp)."
        )

# ===================================================================================== 7. dependents
with tabs[6]:
    sec(
        "Who would feel an NVIDIA failure first",
        "Omdia estimates of Hopper GPUs bought in 2024, with the loss shared pro rata under the current scenario.",
    )
    data = sorted(core.HOPPER_2024, key=lambda r: r[1])
    names_ = [r[0] for r in data]
    units = np.array([r[1] for r in data], float)
    lost = units * LOSS
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        fig = go.Figure()
        fig.add_bar(
            y=names_,
            x=(units - lost) / 1000,
            orientation="h",
            name="kept",
            marker_color=SUP_COL[0],
            marker_line=dict(color="white", width=2),
            hovertemplate="%{y}: <b>%{x:,.0f}k</b> " + "kept" + "<extra></extra>",
        )
        fig.add_bar(
            y=names_,
            x=lost / 1000,
            orientation="h",
            name="lost in the scenario",
            marker=dict(color=DEFICIT, pattern=HATCH, line=dict(color="white", width=2)),
            hovertemplate="%{y}: <b>%{x:,.0f}k</b> " + "lost" + "<extra></extra>",
        )
        for nm, u_, l_ in zip(names_, units, lost):
            fig.add_annotation(
                x=u_ / 1000,
                y=nm,
                xanchor="left",
                xshift=6,
                showarrow=False,
                text=(f"−{fmt(l_ / 1000)}k" if l_ > 0 else f"{fmt(u_ / 1000)}k"),
                font=dict(size=12, color=DEFICIT_TXT if l_ > 0 else INK2),
            )
        style(fig, height=330)
        fig.update_layout(barmode="stack", barcornerradius=4)
        fig.update_xaxes(title="thousand Hopper GPUs (2024)", range=[0, 560])
        fig.update_yaxes(showgrid=False)
        show(fig)
    with c2:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Company": r[0],
                        "Hopper 2024": r[1],
                        "Lost": int(round(r[1] * LOSS)),
                        "In-house chip": r[2] if r[2] != "-" else "no",
                    }
                    for r in core.HOPPER_2024
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Hopper 2024": st.column_config.NumberColumn(format="localized"),
                "Lost": st.column_config.NumberColumn(format="localized"),
            },
        )
        if S2:
            call(
                "warn",
                "Domino effect",
                f"With a {loss_pct}% loss, Microsoft alone would be short of about <b>{fmt(485 * LOSS)} thousand</b> "
                "GPUs. The large cloud providers have their own accelerators (Maia, MTIA, Trainium, TPU), but the smaller "
                "firms that rent compute from them do not, so the shock would travel further down the chain.",
            )
        else:
            call(
                "neutral",
                "Scenario S1",
                "With no capacity loss nobody loses anything. Choose S2 to see the exposure.",
            )
    st.caption(
        "Source: Omdia, as reported by DatacenterDynamics (18 December 2024). The pro-rata assumption is a "
        "simplification: in practice allocation would depend on contracts and on NVIDIA's priorities."
    )

# ===================================================================================== 8. method
with tabs[7]:
    BOX = 'shape=box, style="rounded,filled", fillcolor="white", color="#C9CED6", fontcolor="#14171F", penwidth=1.3'
    DEC = (
        'shape=diamond, style=filled, fillcolor="#EAF2FC", color="#2a78d6", fontcolor="#14171F", penwidth=1.3'
    )
    FIC = f'shape=box, style="rounded,filled", fillcolor="#FFF5F0", color="{DEFICIT}", fontcolor="#14171F", penwidth=1.3'
    RES = 'shape=box, style="rounded,filled", fillcolor="#EFF8F3", color="#1baf7a", fontcolor="#14171F", penwidth=1.3'
    TER = f'shape=ellipse, style=filled, fillcolor="{NAVY}", color="{NAVY}", fontcolor="white"'
    HEAD = (
        'digraph G {{ rankdir={rd}; bgcolor="transparent"; nodesep=0.35; ranksep=0.45; pad=0.15; '
        'node [fontname="Helvetica", fontsize=11.5, margin="0.16,0.07"]; '
        'edge [fontname="Helvetica", fontsize=10, color="#9AA1AD", fontcolor="#4B5363", arrowsize=0.65]; '
    )

    def dot(rd, body):
        return HEAD.format(rd=rd) + body + " }"

    yes, no = "Yes", "No"
    charts = {
        "arch": (
            "Application architecture",
            dot(
                "LR",
                f"""
            newrank=true; ranksep=0.55;
            data [label="Synthetic series\\n2018–2024", {BOX}];
            rf [label="Random Forest\\nper region", {BOX}];
            c [label="Costs cᵢⱼ", {RES}];
            b [label="Demand bⱼ\\n2026", {RES}];
            a [label="Capacities aᵢ\\n(scenario slider)", {RES}];
            tp [label="Transportation problem\\nNW Corner + MODI", {BOX}];
            mf [label="Maximum flow\\nEdmonds–Karp", {BOX}];
            stt [label="Stress test\\nMonte Carlo", {BOX}];
            ag [label="Recovery agent\\n(prototype)", {BOX}, style="rounded,dashed"];
            ind [label="Indicators\\nz*, D, v, SIₖ", {TER}, shape=box, style="rounded,filled"];
            {{rank=same; c -> b -> a [style=invis];}}
            {{rank=same; tp -> mf -> stt [style=invis];}}
            data -> rf -> b;
            c -> tp; b -> tp; a -> tp; b -> mf; a -> mf; b -> stt; a -> stt;
            tp -> ind; mf -> ind; stt -> ind;
            tp -> ag [style=dashed]; ag -> ind [style=dashed];""",
            ),
        ),
        "bal": (
            "Balancing the problem",
            dot(
                "TB",
                f"""
            s [label="Start", {TER}];
            calc [label="S_A = Σaᵢ,  S_B = Σbⱼ", {BOX}];
            q1 [label="S_A = S_B ?", {DEC}];
            q2 [label="S_A < S_B ?", {DEC}];
            fs [label="Fictitious supplier A*\\na* = S_B − S_A, c = 0\\n(unmet demand)", {FIC}];
            fm [label="Fictitious market B*\\nb* = S_A − S_B, c = 0\\n(idle capacity)", {FIC}];
            ok [label="Balanced problem", {RES}];
            s -> calc -> q1; q1 -> ok [label="{yes}"]; q1 -> q2 [label="{no}"];
            q2 -> fs [label="{yes}"]; q2 -> fm [label="{no}"]; fs -> ok; fm -> ok;""",
            ),
        ),
        "modi": (
            "North-West Corner + MODI",
            dot(
                "TB",
                f"""
            s [label="Start", {TER}];
            nw [label="Initial plan\\nNorth-West Corner (m+n−1 cells)", {BOX}];
            pot [label="Potentials: uᵢ + vⱼ = cᵢⱼ\\non basic cells", {BOX}];
            dl [label="δᵢⱼ = cᵢⱼ − (uᵢ + vⱼ)", {BOX}];
            q [label="all δᵢⱼ ≥ 0 ?", {DEC}];
            opt [label="Optimal plan X*, z*", {RES}];
            en [label="Enter (p,q) = argmin δᵢⱼ", {BOX}];
            cy [label="Cycle through the basis\\nθ = min x on the (−)", {BOX}];
            up [label="Shift θ around the cycle\\nthe cell that hits 0 leaves", {BOX}];
            s -> nw -> pot -> dl -> q; q -> opt [label="{yes}"]; q -> en [label="{no}"]; en -> cy -> up;
            up -> pot [label="next iteration", color="#2a78d6", fontcolor="#1c5cab"];""",
            ),
        ),
        "ff": (
            "Maximum flow (Ford–Fulkerson / Edmonds–Karp)",
            dot(
                "TB",
                f"""
            s [label="Start", {TER}];
            g [label="Network s → suppliers → markets → t\\nφ = 0", {BOX}];
            bfs [label="BFS in the residual graph", {BOX}];
            q [label="is there a path s → t ?", {DEC}];
            al [label="α = min (c − φ) on the path", {BOX}];
            up [label="φ ← φ + α along the path", {BOX}];
            cut [label="Nodes reachable from s → minimum cut\\nv(φ) = c(cut)", {RES}];
            s -> g -> bfs -> q; q -> al [label="{yes}"]; al -> up;
            up -> bfs [label="repeat", color="#2a78d6", fontcolor="#1c5cab"]; q -> cut [label="{no}"];""",
            ),
        ),
        "stress": (
            "Stress test (Algorithm 1)",
            dot(
                "TB",
                f"""
            s [label="Start", {TER}];
            k [label="Next supplier k", {BOX}];
            draw [label="Draw θ ~ U(0.2, 1), σᵢ ~ U(0, 0.2), εⱼ ~ U(−0.1, 0.1)", {BOX}];
            ref [label="D_ref = Σb′ − maxflow(a(1+σ), b′)", {BOX}];
            fail [label="aₖ ← aₖ(1−θ)\\nD_fail = Σb′ − maxflow(…)", {FIC}];
            dd [label="ΔD = D_fail − D_ref", {BOX}];
            q [label="N scenarios done?", {DEC}];
            si [label="SIₖ = E[ΔD] / E[Σb′]\\nquantiles, absorption, P(>20%)", {RES}];
            q2 [label="all suppliers done?", {DEC}];
            e [label="Stop", {TER}];
            s -> k -> draw -> ref -> fail -> dd -> q; q -> draw [label="{no}", color="#2a78d6", fontcolor="#1c5cab"];
            q -> si [label="{yes}"]; si -> q2; q2 -> k [label="{no}", color="#2a78d6", fontcolor="#1c5cab"];
            q2 -> e [label="{yes}"];""",
            ),
        ),
    }
    pick = st.segmented_control(
        "Flowchart",
        list(charts),
        default="arch",
        required=True,
        format_func=lambda k: charts[k][0],
        key="chart",
    )
    st.graphviz_chart(charts[pick][1], width="stretch" if pick == "arch" else "content")

    st.divider()
    sec("Theory guide")
    g1, g2, g3, g4, g5 = st.tabs(
        ["Transportation problem", "Balancing", "MODI", "Max flow and cut", "Stress test"]
    )
    with g1:
        st.markdown(
            "A product available at **m** suppliers Aᵢ (capacity aᵢ) must reach **n** markets Bⱼ (demand bⱼ). "
            "The unit cost on route Aᵢ → Bⱼ is cᵢⱼ and xᵢⱼ is the quantity shipped."
        )
        st.latex(
            r"\min z=\sum_{i=1}^{m}\sum_{j=1}^{n}c_{ij}x_{ij}\quad\text{s.t.}\quad"
            r"\sum_{j}x_{ij}\le a_i,\ \ \sum_{i}x_{ij}\ge b_j,\ \ x_{ij}\ge 0"
        )
    with g2:
        st.markdown("The problem is balanced when total supply equals total demand:")
        st.latex(r"\sum_{i=1}^{m}a_i=\sum_{j=1}^{n}b_j")
        st.markdown(
            "Otherwise a fictitious supplier A* (if supply is short) or a fictitious market B* (if there is a surplus) "
            "is added at zero cost. In our model, whatever A* ships is exactly the unmet demand:"
        )
        st.latex(r"D=\sum_{j}x^{*}_{A^{*}j}=\sum_j b_j-\sum_i a_i")
        st.markdown(
            "A balanced problem always has feasible solutions and at least one optimal solution; "
            "the rank of the system is m + n − 1."
        )
    with g3:
        st.markdown(
            "The **North-West Corner** rule starts at cell (1,1), allocates min(aᵢ, bⱼ) and moves right or down until it "
            "has m + n − 1 basic cells. **MODI** (the method of potentials) checks optimality through the dual problem:"
        )
        st.latex(r"\max\ \sum_i a_i u_i+\sum_j b_j v_j\quad\text{s.t.}\quad u_i+v_j\le c_{ij}")
        st.markdown(
            "On basic cells uᵢ + vⱼ = cᵢⱼ (complementary slackness). For the other cells one computes"
        )
        st.latex(r"\delta_{ij}=c_{ij}-(u_i+v_j)")
        st.markdown(
            "If every δᵢⱼ ≥ 0 the plan is optimal. Otherwise the cell with the smallest δ enters the basis, the cycle "
            "through the basic cells is built, θ = the minimum on the negative positions is shifted and the cell that "
            "reaches zero leaves. For degenerate plans a zero-valued cell is kept in the basis (equivalent to the ε technique)."
        )
    with g4:
        st.markdown(
            "The network G = (X, U) has a source s and a sink t, with capacities c(u) ≥ 0. A flow φ respects every arc's "
            "capacity and conservation at every intermediate node:"
        )
        st.latex(
            r"0\le\varphi(u)\le c(u),\qquad \sum_{\text{in}}\varphi=\sum_{\text{out}}\varphi\ \ (k\ne s,t)"
        )
        st.markdown(
            "A cut separates s from t; its capacity is the sum of the capacities of the arcs that cross it. "
            "**Ford–Fulkerson theorem:**"
        )
        st.latex(r"\max_{\varphi}\ v(\varphi)=\min_{(S,\bar S)}\ c(S,\bar S)")
        st.markdown(
            "Edmonds–Karp is the Ford–Fulkerson variant that always picks the shortest augmenting path (BFS), which "
            "guarantees a polynomial number of iterations. When no path is left, the nodes reachable from s give the minimum cut."
        )
    with g5:
        st.markdown(
            "The systemic importance of supplier k is the share of demand lost, on average, when it fails:"
        )
        st.latex(
            r"SI_k=\frac{\mathbb{E}[\Delta D_k]}{\mathbb{E}[\sum_j b'_j]},\qquad "
            r"\text{absorbed}_k=1-\frac{\Delta D_k}{\text{capacity lost}}"
        )
        st.markdown(
            "ΔDₖ is the extra deficit compared with the same scenario without the failure, so demand noise and reserves "
            "are not blamed on the supplier. The idea comes from bank stress tests: an institution is “too big to fail” "
            "when its loss cannot be absorbed by the rest of the system."
        )

    with st.expander("Sources", icon=":material/link:"):
        st.markdown(
            "- Jon Peddie Research, *Shipments of graphics add-in boards decline in Q1 of '24…*, 6 June 2024.\n"
            "- NVIDIA Corporation, *Annual Report on Form 10-K for the fiscal year ended January 25, 2026*, SEC.\n"
            "- C. Trueman, [*Microsoft bought twice as many Nvidia Hopper GPUs as other Big Tech companies*]"
            "(https://www.datacenterdynamics.com/en/news/microsoft-bought-twice-as-many-nvidia-hopper-gpus-as-other-big-tech-companies-report), "
            "DatacenterDynamics, 18 December 2024.\n"
            "- M. J. Roe, [*The costs of “too big to fail”*](https://corpgov.law.harvard.edu/2013/06/26/the-costs-of-too-big-to-fail/), "
            "Harvard Law School Forum on Corporate Governance, 2013.\n"
            "- J. Edmonds, R. M. Karp, *Theoretical improvements in algorithmic efficiency for network flow problems*, "
            "J. ACM 19(2), 1972.\n"
            "- L. Breiman, *Random forests*, Machine Learning 45(1), 2001."
        )

st.markdown(
    f'<div class="foot">The “Too Big to Fail” paradigm in the GPU supply chain · mathematical modelling application'
    f" · NumPy · SciPy (HiGHS) · scikit-learn · Plotly · Streamlit</div>",
    unsafe_allow_html=True,
)

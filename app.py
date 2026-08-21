import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Marco's Portfolio Optimizer", layout="wide", page_icon="📈")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2a3b 55%, #1e3a5f 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 0.2rem 0; letter-spacing: -0.5px; }
.hero .sub { font-size: 0.88rem; opacity: 0.65; margin: 0 0 1.4rem 0; }
.hero-stats { display: flex; gap: 0.9rem; flex-wrap: wrap; }
.hero-stat {
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    min-width: 130px;
}
.hero-stat .val { font-size: 1.4rem; font-weight: 700; line-height: 1.2; }
.hero-stat .lbl { font-size: 0.7rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }
.hero-stat.a { border-left: 3px solid #60a5fa; }
.hero-stat.b { border-left: 3px solid #fb923c; }
.hero-stat.t { border-left: 3px solid #34d399; }

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 0.9rem 1rem !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
[data-testid="stMetricValue"] { font-size: 1.45rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.4px; }

.insight {
    background: #f0f5ff;
    border-left: 4px solid #4f6ef7;
    padding: 0.75rem 1rem;
    border-radius: 0 10px 10px 0;
    margin: 0.6rem 0;
    font-size: 0.87rem;
    color: #1a2040;
    line-height: 1.55;
}
.insight.orange { background: #fff5ed; border-left-color: #f97316; }
.insight.green  { background: #f0fdf4; border-left-color: #16a34a; color: #14532d; }
.insight.red    { background: #fef2f2; border-left-color: #dc2626; color: #7f1d1d; }

.sec-hd {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #6b7280;
    margin: 1.1rem 0 0.45rem 0;
    padding-bottom: 0.35rem; border-bottom: 1px solid #e5e7eb;
}
.data-badge {
    display: inline-block; background: #f0fdf4; border: 1px solid #bbf7d0;
    color: #15803d; border-radius: 6px; padding: 0.2rem 0.65rem;
    font-size: 0.75rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Marco's Parameters")
    st.markdown('<div class="sec-hd">Capital</div>', unsafe_allow_html=True)
    total_capital = st.number_input("Total Capital ($)", 10_000, 500_000, 80_000, step=5_000)
    goal_a_amt    = st.number_input("Goal A Amount ($)", 5_000, 200_000, 30_000, step=5_000)
    goal_b_amt    = total_capital - goal_a_amt
    if goal_b_amt < 0:
        st.error("Goal A exceeds total capital!")
        st.stop()
    st.metric("Goal B Capital", f"${goal_b_amt:,}")

    st.markdown('<div class="sec-hd">Time Horizons</div>', unsafe_allow_html=True)
    years_a = st.slider("Goal A: Years to target", 2, 7, 4)
    years_b = st.slider("Goal B: Years to retirement", 15, 25, 20)

    st.markdown('<div class="sec-hd">Rebalancing</div>', unsafe_allow_html=True)
    drift_threshold = st.slider("Alert threshold (%)", 2, 15, 5)

    st.caption(
        f"Optimizer uses **{max(years_a, 2)}y** of history for Goal A, "
        f"**{min(years_b, 20)}y** for Goal B."
    )

# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>📈 Marco's Portfolio Optimizer</h1>
  <p class="sub">43-year-old corporate manager &nbsp;·&nbsp; Two goals, one disciplined plan</p>
  <div class="hero-stats">
    <div class="hero-stat t">
      <div class="val">${total_capital:,}</div>
      <div class="lbl">Total Capital</div>
    </div>
    <div class="hero-stat a">
      <div class="val">${goal_a_amt:,}</div>
      <div class="lbl">Goal A · {years_a}yr horizon</div>
    </div>
    <div class="hero-stat b">
      <div class="val">${goal_b_amt:,}</div>
      <div class="lbl">Goal B · {years_b}yr retirement</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Assets ─────────────────────────────────────────────────────────────────────
TICKERS_A = ["TIP", "BND", "SCHD", "VNQ"]
TICKERS_B = ["QQQ", "ARKK", "IEMG", "SPY"]

NAMES_A = {"TIP": "Inflation Bonds", "BND": "Aggregate Bonds",
           "SCHD": "Dividend Stocks", "VNQ": "Real Estate"}
NAMES_B = {"QQQ": "Tech Growth", "ARKK": "Disruptive Innovation",
           "IEMG": "Emerging Markets", "SPY": "US Market"}

C_SHARPE = "#f59e0b"
C_PARITY = "#06b6d4"
C_EQUAL  = "#ef4444"
C_A      = "#3b82f6"
C_B      = "#f97316"

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_prices(tickers: list, years: int) -> pd.DataFrame:
    end   = datetime.today()
    # FIX: was max(years, 5) — changing years_a from 2→4 produced identical
    # data and identical optimizer results. Now uses actual horizon (min 2y).
    start = end - timedelta(days=max(years, 2) * 365)
    raw   = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    elif "Close" in raw.columns:
        prices = raw[["Close"]]
        prices.columns = tickers[:1]
    else:
        prices = raw
    available = [t for t in tickers if t in prices.columns]
    return prices[available].dropna()

with st.spinner("Fetching live prices from Yahoo Finance…"):
    try:
        prices_a = load_prices(TICKERS_A, years_a)
        prices_b = load_prices(TICKERS_B, min(years_b, 20))
        if prices_a.empty or prices_b.empty:
            raise ValueError("yfinance returned no data — market may be closed or tickers unavailable.")
        rets_a = prices_a.pct_change().dropna()
        rets_b = prices_b.pct_change().dropna()
        if rets_a.empty or rets_b.empty:
            raise ValueError("Not enough data to compute returns.")
        data_ok = True
    except Exception as e:
        st.error(f"Data load failed: {e}")
        data_ok = False

if data_ok:
    st.markdown(
        f'<span class="data-badge">✓ Yahoo Finance — {datetime.today().strftime("%b %d, %Y")}'
        f" &nbsp;|&nbsp; Goal A: {prices_a.index[0].strftime('%b %Y')} → {prices_a.index[-1].strftime('%b %Y')}"
        f" ({len(prices_a)} trading days)"
        f" &nbsp;|&nbsp; Goal B: {prices_b.index[0].strftime('%b %Y')} → {prices_b.index[-1].strftime('%b %Y')}"
        f" ({len(prices_b)} trading days)</span>",
        unsafe_allow_html=True,
    )
    st.write("")

# ── Portfolio Math ─────────────────────────────────────────────────────────────
def port_stats(weights, rets):
    w  = np.array(weights)
    r  = float(np.dot(rets.mean(), w) * 252)
    v  = float(np.sqrt(w @ (rets.cov() * 252).values @ w))
    sh = r / v if v > 0 else 0.0
    return r, v, sh

def max_sharpe(rets):
    n    = rets.shape[1]
    init = np.ones(n) / n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bnds = [(0.05, 0.70)] * n
    res  = minimize(lambda w: -port_stats(w, rets)[2], init,
                    method="SLSQP", bounds=bnds, constraints=cons)
    return res.x if res.success else init

def risk_parity(rets):
    n    = rets.shape[1]
    cov  = rets.cov().values * 252
    init = np.ones(n) / n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bnds = [(0.01, 0.99)] * n
    def objective(w):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc  = w * mrc
        return np.sum((rc - port_var / n) ** 2)
    res = minimize(objective, init, method="SLSQP", bounds=bnds, constraints=cons)
    return res.x if res.success else init

def simulate_frontier(rets, n=600):
    n_assets = rets.shape[1]
    rows = []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        r, v, s = port_stats(w, rets)
        rows.append((v, r, s))
    return rows

# ── TABS ───────────────────────────────────────────────────────────────────────
if data_ok:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Efficient Frontier", "🌩 Scenario Analysis", "⚖️ Rebalancing Check", "🔍 Verification"]
    )

    # ── TAB 1: Efficient Frontier ──────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)

        for col, rets, tickers, names, label, capital, horizon, accent, cscale, gkey in [
            (col_a, rets_a, TICKERS_A, NAMES_A, "Goal A — Conservative", goal_a_amt, years_a, C_A, "Blues", "A"),
            (col_b, rets_b, TICKERS_B, NAMES_B, "Goal B — Growth",       goal_b_amt, years_b, C_B, "Oranges", "B"),
        ]:
            with col:
                st.markdown(f'<div class="sec-hd">{label}</div>', unsafe_allow_html=True)

                frontier = simulate_frontier(rets)
                vols     = [x[0] for x in frontier]
                ret_vals = [x[1] for x in frontier]
                sharpes  = [x[2] for x in frontier]

                opt_w = max_sharpe(rets)
                rp_w  = risk_parity(rets)
                eq_w  = np.ones(len(tickers)) / len(tickers)

                opt_r, opt_v, opt_s = port_stats(opt_w, rets)
                rp_r,  rp_v,  rp_s  = port_stats(rp_w,  rets)
                eq_r,  eq_v,  eq_s  = port_stats(eq_w,  rets)

                # Efficient frontier scatter
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=vols, y=ret_vals, mode="markers",
                    marker=dict(color=sharpes, colorscale=cscale, size=4, opacity=0.45,
                               showscale=True,
                               colorbar=dict(title="Sharpe", thickness=10, len=0.55, x=1.01)),
                    name="Simulated Portfolios",
                    hovertemplate="Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"))
                fig.add_trace(go.Scatter(
                    x=[opt_v], y=[opt_r], mode="markers+text",
                    marker=dict(color=C_SHARPE, size=18, symbol="star",
                               line=dict(color="white", width=1.5)),
                    text=[f"  Max Sharpe ({opt_s:.2f})"], textposition="middle right",
                    name=f"Max Sharpe",
                    hovertemplate=f"Return: {opt_r:.1%}<br>Risk: {opt_v:.1%}<br>Sharpe: {opt_s:.2f}<extra>Max Sharpe</extra>"))
                fig.add_trace(go.Scatter(
                    x=[rp_v], y=[rp_r], mode="markers+text",
                    marker=dict(color=C_PARITY, size=15, symbol="diamond",
                               line=dict(color="white", width=1.5)),
                    text=[f"  Risk Parity ({rp_s:.2f})"], textposition="middle right",
                    name=f"Risk Parity",
                    hovertemplate=f"Return: {rp_r:.1%}<br>Risk: {rp_v:.1%}<br>Sharpe: {rp_s:.2f}<extra>Risk Parity</extra>"))
                fig.add_trace(go.Scatter(
                    x=[eq_v], y=[eq_r], mode="markers+text",
                    marker=dict(color=C_EQUAL, size=14, symbol="x",
                               line=dict(color="white", width=2)),
                    text=[f"  Equal-Wt ({eq_s:.2f})"], textposition="middle right",
                    name=f"Equal-Weight",
                    hovertemplate=f"Return: {eq_r:.1%}<br>Risk: {eq_v:.1%}<br>Sharpe: {eq_s:.2f}<extra>Equal-Weight</extra>"))
                fig.update_layout(
                    xaxis_title="Annual Risk (Volatility)",
                    yaxis_title="Annual Return",
                    xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                    height=390, margin=dict(t=10, b=10, l=10, r=50),
                    plot_bgcolor="#f8faff" if gkey == "A" else "#fff8f2",
                    paper_bgcolor="white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)))
                fig.update_xaxes(gridcolor="#e5e7eb", zeroline=False)
                fig.update_yaxes(gridcolor="#e5e7eb", zeroline=False)
                st.plotly_chart(fig, use_container_width=True)

                # Key metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Annual Return",  f"{opt_r:.1%}")
                m2.metric("Annual Risk",    f"{opt_v:.1%}")
                m3.metric("Sharpe Ratio",   f"{opt_s:.2f}")

                projected = capital * (1 + opt_r) ** horizon
                gain      = projected - capital
                mood      = "green" if opt_r > 0.06 else "orange"
                st.markdown(
                    f'<div class="insight {mood}">'
                    f"💰 <strong>${capital:,}</strong> grows to <strong>${projected:,.0f}</strong> "
                    f"in {horizon} years &nbsp;·&nbsp; +${gain:,.0f} gain ({opt_r:.1%}/yr)"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Year-by-year projection (expandable)
                with st.expander("📅 Year-by-year growth projection"):
                    yrs = list(range(1, horizon + 1))
                    proj_df = pd.DataFrame({
                        "Year":            yrs,
                        "Projected Value": [f"${capital * (1 + opt_r) ** y:,.0f}" for y in yrs],
                        "Total Gain":      [f"+${capital * (1 + opt_r) ** y - capital:,.0f}" for y in yrs],
                        "Cumulative %":    [f"{((1 + opt_r) ** y - 1):.1%}" for y in yrs],
                    })
                    st.dataframe(proj_df, hide_index=True, use_container_width=True)

                # Allocation breakdown
                asset_labels = [f"{t} — {names[t]}" for t in tickers]
                fig_bar = go.Figure()
                for w_arr, name, color in [
                    (opt_w, "Max Sharpe",   C_SHARPE),
                    (rp_w,  "Risk Parity",  C_PARITY),
                    (eq_w,  "Equal-Weight", C_EQUAL),
                ]:
                    fig_bar.add_trace(go.Bar(
                        name=name, x=asset_labels,
                        y=[w * 100 for w in w_arr], marker_color=color,
                        text=[f"{w*100:.0f}%" for w in w_arr], textposition="outside"))
                fig_bar.update_layout(
                    barmode="group", yaxis_title="Allocation (%)", yaxis_range=[0, 85],
                    height=280, margin=dict(t=5, b=5, l=10, r=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)))
                fig_bar.update_yaxes(gridcolor="#f0f0f0", zeroline=False)
                st.caption("Asset allocation behind each strategy:")
                st.plotly_chart(fig_bar, use_container_width=True)

    # ── TAB 2: Scenario Analysis ───────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="sec-hd">Market Crisis Stress Test</div>', unsafe_allow_html=True)
        st.caption("How would Marco's portfolio have survived a major market event?")

        scenario = st.selectbox("Select scenario", [
            "2020 COVID Crash (Feb–Mar 2020)",
            "2022 Rate Hike Year",
            "2008 Financial Crisis",
        ])
        scenario_dates = {
            "2020 COVID Crash (Feb–Mar 2020)": ("2020-01-01", "2020-12-31"),
            "2022 Rate Hike Year":             ("2022-01-01", "2022-12-31"),
            "2008 Financial Crisis":           ("2007-10-01", "2009-06-30"),
        }
        s_start, s_end = scenario_dates[scenario]

        @st.cache_data(ttl=3600)
        def load_scenario(tickers, start, end):
            raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw
            # FIX: original dropna() (how='any') silently killed the 2008 scenario because
            # ARKK (launched 2014) had an all-NaN column, emptying the entire DataFrame.
            # Drop columns with zero coverage first, then forward-fill intra-column gaps.
            prices = prices.dropna(axis=1, how="all")
            prices = prices.ffill().dropna()
            return prices

        with st.spinner("Loading scenario data…"):
            sp = load_scenario(tuple(TICKERS_A + TICKERS_B), s_start, s_end)

        if sp.empty:
            st.warning("No data available for this scenario period.")
        else:
            opt_w_a = max_sharpe(rets_a)
            opt_w_b = max_sharpe(rets_b)
            eq_w_a  = np.ones(len(TICKERS_A)) / len(TICKERS_A)
            eq_w_b  = np.ones(len(TICKERS_B)) / len(TICKERS_B)

            def bucket_perf(prices, tickers, weights):
                # FIX: original used weights[:len(cols)] — positional slicing.
                # If ARKK (index 1) was missing, IEMG inherited ARKK's weight.
                # Now we map by ticker name so absent tickers are simply skipped.
                weight_map = dict(zip(tickers, weights))
                cols = [t for t in tickers if t in prices.columns]
                if not cols:
                    return pd.Series(dtype=float)
                p = prices[cols]
                w = np.array([weight_map[t] for t in cols])
                w /= w.sum()
                return (p / p.iloc[0]).dot(w)

            missing_b = [t for t in TICKERS_B if t not in sp.columns]
            if missing_b:
                st.info(
                    f"ℹ️ {', '.join(missing_b)} had no coverage during this period "
                    f"and were excluded — remaining weights re-normalized."
                )

            series = {
                "Goal A — Optimal":      bucket_perf(sp, TICKERS_A, opt_w_a),
                "Goal A — Equal-Weight": bucket_perf(sp, TICKERS_A, eq_w_a),
                "Goal B — Optimal":      bucket_perf(sp, TICKERS_B, opt_w_b),
                "Goal B — Equal-Weight": bucket_perf(sp, TICKERS_B, eq_w_b),
            }

            line_style = {
                "Goal A — Optimal":      (C_A, "solid",  2.5),
                "Goal A — Equal-Weight": (C_A, "dash",   1.5),
                "Goal B — Optimal":      (C_B, "solid",  2.5),
                "Goal B — Equal-Weight": (C_B, "dash",   1.5),
            }

            fig2 = go.Figure()
            for name, s in series.items():
                if not s.empty:
                    color, dash, width = line_style[name]
                    fig2.add_trace(go.Scatter(
                        x=s.index, y=s, name=name,
                        line=dict(color=color, dash=dash, width=width),
                        hovertemplate="%{y:.3f}x<extra>" + name + "</extra>"))
            fig2.add_hline(
                y=1.0, line_dash="dot", line_color="#9ca3af",
                annotation_text="Starting value (1.0x)", annotation_font_size=11)
            fig2.update_layout(
                yaxis_title="Portfolio Value (1.0 = starting point)",
                xaxis_title="Date", height=420,
                plot_bgcolor="#fafafa", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
                margin=dict(t=10, b=10))
            fig2.update_xaxes(gridcolor="#e5e7eb", zeroline=False)
            fig2.update_yaxes(gridcolor="#e5e7eb", zeroline=False)
            st.plotly_chart(fig2, use_container_width=True)

            def max_dd(s):
                return ((s - s.cummax()) / s.cummax()).min()

            cols4 = st.columns(4)
            labels   = ["Goal A Optimal", "Goal A Equal-Wt", "Goal B Optimal", "Goal B Equal-Wt"]
            ser_list = [
                series["Goal A — Optimal"], series["Goal A — Equal-Weight"],
                series["Goal B — Optimal"], series["Goal B — Equal-Weight"],
            ]
            for c, lbl, s in zip(cols4, labels, ser_list):
                if not s.empty:
                    dd = max_dd(s)
                    c.metric(lbl, f"{dd:.1%}", delta="lower risk" if abs(dd) < 0.15 else "high drawdown",
                             delta_color="normal" if abs(dd) < 0.15 else "inverse")

            st.markdown("""
            <div class="insight">
              <strong>How to read this:</strong> Goal A (blue) is Marco's conservative bucket — built to hold up
              in crises. Goal B (orange) accepts deeper drawdowns for long-run upside. The 20-year
              retirement horizon gives Goal B time to recover. Solid line = Max Sharpe · Dashed = Equal-Weight.
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 3: Rebalancing Check ───────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="sec-hd">Portfolio Drift Monitor</div>', unsafe_allow_html=True)
        st.caption("Enter current holdings to check how far Marco has drifted from the optimal allocation.")

        opt_w_a = max_sharpe(rets_a)
        opt_w_b = max_sharpe(rets_b)

        for tickers, opt_w, names, bucket_label in [
            (TICKERS_A, opt_w_a, NAMES_A, "Goal A — Conservative"),
            (TICKERS_B, opt_w_b, NAMES_B, "Goal B — Growth"),
        ]:
            st.markdown(f"**{bucket_label}**")
            input_cols = st.columns(len(tickers))
            current = []
            for t, ic in zip(tickers, input_cols):
                val = ic.number_input(
                    f"{t}", 0.0, 100.0,
                    float(round(opt_w[tickers.index(t)] * 100, 1)),
                    step=0.5, key=f"rb_{bucket_label}_{t}", help=names[t])
                current.append(val / 100)

            total = sum(current)
            if abs(total - 1.0) > 0.015:
                st.warning(f"Weights sum to {total:.0%} — adjust to reach 100%")
            else:
                drifts       = [(c - t) * 100 for c, t in zip(current, opt_w)]
                bar_colors   = [
                    "#dc2626" if abs(d) > drift_threshold else
                    "#f59e0b" if abs(d) > drift_threshold * 0.6 else
                    "#16a34a"
                    for d in drifts
                ]
                asset_labels = [f"{t}\n{names[t]}" for t in tickers]

                # Visual drift bars
                fig_drift = go.Figure()
                fig_drift.add_trace(go.Bar(
                    x=asset_labels, y=drifts,
                    marker_color=bar_colors,
                    text=[f"{d:+.1f}%" for d in drifts],
                    textposition="outside",
                    hovertemplate="Drift: %{y:+.1f}%<extra></extra>"))
                fig_drift.add_hline(
                    y= drift_threshold, line_dash="dash", line_color="#f59e0b",
                    annotation_text=f"+{drift_threshold}% alert", annotation_font_size=10)
                fig_drift.add_hline(
                    y=-drift_threshold, line_dash="dash", line_color="#f59e0b",
                    annotation_text=f"−{drift_threshold}% alert", annotation_font_size=10)
                fig_drift.add_hline(y=0, line_color="#d1d5db", line_width=1)
                fig_drift.update_layout(
                    yaxis_title="Drift from Target (%)",
                    height=250, margin=dict(t=10, b=10, l=10, r=10),
                    plot_bgcolor="white", paper_bgcolor="white")
                fig_drift.update_yaxes(gridcolor="#f3f4f6", zeroline=False)
                st.plotly_chart(fig_drift, use_container_width=True)

                drift_df = pd.DataFrame({
                    "Asset":   [f"{t} — {names[t]}" for t in tickers],
                    "Current": [f"{w*100:.1f}%" for w in current],
                    "Target":  [f"{w*100:.1f}%" for w in opt_w],
                    "Drift":   [f"{d:+.1f}%" for d in drifts],
                    "Action":  [
                        "🔴 REBALANCE" if abs(d) > drift_threshold else
                        "🟡 WATCH"     if abs(d) > drift_threshold * 0.6 else
                        "✅ OK"
                        for d in drifts
                    ],
                })
                st.dataframe(drift_df, hide_index=True, use_container_width=True)

            st.divider()

    # ── TAB 4: Verification ────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="sec-hd">Independent Verification — Sharpe Ratio</div>', unsafe_allow_html=True)
        st.caption(
            "Recomputing Sharpe ratio manually (daily weighted return series) to confirm "
            "the optimizer's pipeline output. Both methods must agree."
        )

        all_match = True
        for rets, tickers, label in [
            (rets_a, TICKERS_A, "Goal A — Conservative"),
            (rets_b, TICKERS_B, "Goal B — Growth"),
        ]:
            opt_w = max_sharpe(rets)
            r, v, s = port_stats(opt_w, rets)

            daily_port = (rets * opt_w).sum(axis=1)
            manual_r   = daily_port.mean() * 252
            manual_v   = daily_port.std() * np.sqrt(252)
            manual_s   = manual_r / manual_v

            diff    = abs(s - manual_s)
            matched = diff < 0.001
            if not matched:
                all_match = False

            st.markdown(f"**{label}**")
            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("Pipeline Sharpe", f"{s:.4f}")
            vc2.metric("Manual Sharpe",   f"{manual_s:.4f}")
            vc3.metric("Difference",      f"{diff:.6f}",
                       delta="✅ Match" if matched else "⚠️ Mismatch",
                       delta_color="normal" if matched else "inverse")
            st.divider()

        if all_match:
            st.markdown(
                '<div class="insight green">✅ All calculations verified — pipeline and manual methods '
                "agree to within 0.001.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="insight red">⚠️ Mismatch detected — review the optimization pipeline.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("**Known Limitations**")
        st.markdown(
            "- Historical returns do not guarantee future performance\n"
            "- ARKK has a shorter history (since 2014); pre-2014 estimates for Goal B exclude it\n"
            "- No transaction costs, taxes, or rebalancing friction are modeled\n"
            "- Conclusions become less reliable if correlations shift structurally "
            "(e.g. prolonged stagflation where bonds and equities fall together)"
        )

# ── AI Disclosure ──────────────────────────────────────────────────────────────
with st.expander("AI-Use Disclosure"):
    st.markdown(
        "**Tools used:** Claude Code (Anthropic) assisted in designing, building, "
        "and debugging this application.  \n"
        "**Verification:** All financial calculations (return, volatility, Sharpe ratio, "
        "portfolio optimization) were independently reviewed against textbook formulas and "
        "cross-checked via the Verification tab.  \n"
        "**Known limitations:** The optimizer uses historical data; AI-generated code was "
        "reviewed line-by-line by team members who can explain every function."
    )

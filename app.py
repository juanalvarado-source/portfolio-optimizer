import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Marco's Portfolio Optimizer", layout="wide", page_icon="📈")

# ── Header ──────────────────────────────────────────────────────────────────
st.title("📈 Marco's Portfolio Optimizer")
st.caption("43-year-old corporate manager | $80,000 to invest | Two goals, one plan")

c1, c2, c3 = st.columns(3)
c1.metric("Total Capital", "$80,000")
c2.metric("Goal A — Short-Term", "$30,000 in 4 years")
c3.metric("Goal B — Retirement", "$50,000 in ~20 years")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Adjust Marco's Plan")
    total_capital = st.number_input("Total Capital ($)", 10_000, 500_000, 80_000, step=5_000)
    goal_a_amt    = st.number_input("Goal A Amount ($)", 5_000, 200_000, 30_000, step=5_000)
    goal_b_amt    = total_capital - goal_a_amt
    st.metric("Goal B Capital", f"${goal_b_amt:,}")
    years_a = st.slider("Goal A: Years to target", 2, 7, 4)
    years_b = st.slider("Goal B: Years to retirement", 15, 25, 20)
    drift_threshold = st.slider("Rebalancing alert threshold (%)", 2, 15, 5)

# ── Assets ───────────────────────────────────────────────────────────────────
TICKERS_A = ["TIP", "BND", "SCHD", "VNQ"]          # Conservative / inflation-beating
TICKERS_B = ["QQQ", "ARKK", "IEMG", "SPY"]         # Growth / retirement upside

NAMES_A = {"TIP": "Inflation Bonds", "BND": "Aggregate Bonds",
           "SCHD": "Dividend Stocks", "VNQ": "Real Estate"}
NAMES_B = {"QQQ": "Tech Growth", "ARKK": "Disruptive Innovation",
           "IEMG": "Emerging Markets", "SPY": "US Market"}

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_prices(tickers: list, years: int) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=max(years, 5) * 365)
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

with st.spinner("Fetching live data from Yahoo Finance…"):
    try:
        prices_a = load_prices(TICKERS_A, years_a + 1)
        prices_b = load_prices(TICKERS_B, min(years_b, 20))
        if prices_a.empty or prices_b.empty:
            raise ValueError("yfinance returned no data — market may be closed or tickers unavailable.")
        rets_a   = prices_a.pct_change().dropna()
        rets_b   = prices_b.pct_change().dropna()
        if rets_a.empty or rets_b.empty:
            raise ValueError("Not enough data to compute returns.")
        data_ok  = True
    except Exception as e:
        st.error(f"Data load failed: {e}")
        data_ok = False

if data_ok:
    st.success(
        f"Data loaded — Source: Yahoo Finance (yfinance) | "
        f"Retrieval date: {datetime.today().date()} | "
        f"Range: {prices_a.index[0].date()} → {prices_a.index[-1].date()}"
    )

# ── Portfolio Math ────────────────────────────────────────────────────────────
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

# ── TABS ──────────────────────────────────────────────────────────────────────
if data_ok:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Efficient Frontier", "🌩 Scenario Analysis", "⚖️ Rebalancing Check", "🔍 Verification"]
    )

    # ── TAB 1: Efficient Frontier ─────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)

        for col, rets, tickers, names, label, capital, horizon in [
            (col_a, rets_a, TICKERS_A, NAMES_A, "Goal A — Conservative", goal_a_amt, years_a),
            (col_b, rets_b, TICKERS_B, NAMES_B, "Goal B — Growth",       goal_b_amt, years_b),
        ]:
            with col:
                st.subheader(label)

                frontier  = simulate_frontier(rets)
                vols      = [x[0] for x in frontier]
                ret_vals  = [x[1] for x in frontier]
                sharpes   = [x[2] for x in frontier]

                opt_w  = max_sharpe(rets)
                rp_w   = risk_parity(rets)
                eq_w   = np.ones(len(tickers)) / len(tickers)

                opt_r, opt_v, opt_s = port_stats(opt_w, rets)
                rp_r,  rp_v,  rp_s  = port_stats(rp_w,  rets)
                eq_r,  eq_v,  eq_s  = port_stats(eq_w,  rets)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=vols, y=ret_vals, mode="markers",
                    marker=dict(color=sharpes, colorscale="Viridis", size=4, opacity=0.4),
                    name="Simulated", hovertemplate="Risk: %{x:.1%}<br>Return: %{y:.1%}"))
                fig.add_trace(go.Scatter(
                    x=[opt_v], y=[opt_r], mode="markers+text",
                    marker=dict(color="gold", size=16, symbol="star"),
                    text=[f"Optimal<br>Sharpe {opt_s:.2f}"], textposition="top center",
                    name="Max Sharpe"))
                fig.add_trace(go.Scatter(
                    x=[rp_v], y=[rp_r], mode="markers+text",
                    marker=dict(color="cyan", size=14, symbol="diamond"),
                    text=[f"Risk Parity<br>Sharpe {rp_s:.2f}"], textposition="top center",
                    name="Risk Parity"))
                fig.add_trace(go.Scatter(
                    x=[eq_v], y=[eq_r], mode="markers+text",
                    marker=dict(color="red", size=14, symbol="x"),
                    text=[f"Equal-Weight<br>Sharpe {eq_s:.2f}"], textposition="top center",
                    name="Equal-Weight"))
                fig.update_layout(
                    xaxis_title="Annual Risk (Volatility)",
                    yaxis_title="Annual Return",
                    xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                    height=380, margin=dict(t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig, use_container_width=True)

                # Weights breakdown
                w_df = pd.DataFrame({
                    "Asset":         [f"{t} — {names[t]}" for t in tickers],
                    "Max Sharpe %":  [f"{w*100:.1f}%" for w in opt_w],
                    "Risk Parity %": [f"{w*100:.1f}%" for w in rp_w],
                    "Equal-Wt %":    [f"{w*100:.1f}%" for w in eq_w],
                })
                st.dataframe(w_df, hide_index=True, use_container_width=True)

                projected = capital * (1 + opt_r) ** horizon
                st.info(
                    f"**Projected value in {horizon}y:** ${projected:,.0f} &nbsp;|&nbsp; "
                    f"Annual return: {opt_r:.1%} &nbsp;|&nbsp; Risk: {opt_v:.1%}"
                )

    # ── TAB 2: Scenario Analysis ──────────────────────────────────────────────
    with tab2:
        st.subheader("How would Marco's portfolio have survived a market crisis?")

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
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                return raw["Close"].dropna()
            return raw.dropna()

        sp = load_scenario(TICKERS_A + TICKERS_B, s_start, s_end)

        if sp.empty:
            st.warning("No data available for this scenario period.")
        else:
            opt_w_a = max_sharpe(rets_a)
            opt_w_b = max_sharpe(rets_b)
            eq_w_a  = np.ones(len(TICKERS_A)) / len(TICKERS_A)
            eq_w_b  = np.ones(len(TICKERS_B)) / len(TICKERS_B)

            def bucket_perf(prices, tickers, weights):
                cols = [t for t in tickers if t in prices.columns]
                if not cols:
                    return pd.Series(dtype=float)
                p = prices[cols]
                w = np.array(weights[:len(cols)])
                w /= w.sum()
                return (p / p.iloc[0] * w).sum(axis=1)

            series = {
                "Goal A — Optimal":      bucket_perf(sp, TICKERS_A, opt_w_a),
                "Goal A — Equal-Weight": bucket_perf(sp, TICKERS_A, eq_w_a),
                "Goal B — Optimal":      bucket_perf(sp, TICKERS_B, opt_w_b),
                "Goal B — Equal-Weight": bucket_perf(sp, TICKERS_B, eq_w_b),
            }
            colors = {"Goal A — Optimal": "blue", "Goal A — Equal-Weight": "blue",
                      "Goal B — Optimal": "orange", "Goal B — Equal-Weight": "orange"}
            dashes = {"Goal A — Optimal": "solid", "Goal A — Equal-Weight": "dash",
                      "Goal B — Optimal": "solid", "Goal B — Equal-Weight": "dash"}

            fig2 = go.Figure()
            for name, s in series.items():
                if not s.empty:
                    fig2.add_trace(go.Scatter(
                        x=s.index, y=s,
                        name=name,
                        line=dict(color=colors[name], dash=dashes[name])))
            fig2.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Starting value")
            fig2.update_layout(
                yaxis_title="Normalized Value (1.0 = start)",
                xaxis_title="Date", height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig2, use_container_width=True)

            def max_dd(s):
                roll_max = s.cummax()
                return ((s - roll_max) / roll_max).min()

            cols4 = st.columns(4)
            labels = ["Goal A Optimal", "Goal A Equal-Wt", "Goal B Optimal", "Goal B Equal-Wt"]
            ser_list = [series["Goal A — Optimal"], series["Goal A — Equal-Weight"],
                        series["Goal B — Optimal"], series["Goal B — Equal-Weight"]]
            for c, lbl, s in zip(cols4, labels, ser_list):
                if not s.empty:
                    c.metric(f"{lbl} Max Drawdown", f"{max_dd(s):.1%}")

            st.caption(
                "Takeaway: Marco's conservative bucket (Goal A) is designed to hold up in crises. "
                "The growth bucket (Goal B) accepts higher drawdowns for long-term upside — "
                "his 20-year horizon gives it time to recover."
            )

    # ── TAB 3: Rebalancing Check ──────────────────────────────────────────────
    with tab3:
        st.subheader("Is Marco's portfolio still on target?")
        st.caption("Enter current holdings to check drift from the optimal allocation.")

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
                    step=0.5, key=f"rb_{bucket_label}_{t}",
                    help=names[t])
                current.append(val / 100)

            total = sum(current)
            if abs(total - 1.0) > 0.015:
                st.warning(f"Weights sum to {total:.0%} — adjust to reach 100%")
            else:
                drift_df = pd.DataFrame({
                    "Asset":    [f"{t} ({names[t]})" for t in tickers],
                    "Current":  [f"{w*100:.1f}%" for w in current],
                    "Target":   [f"{w*100:.1f}%" for w in opt_w],
                    "Drift":    [f"{(c-t)*100:+.1f}%" for c, t in zip(current, opt_w)],
                    "Status":   [
                        "🔴 REBALANCE" if abs(c - t) * 100 > drift_threshold else "✅ OK"
                        for c, t in zip(current, opt_w)
                    ],
                })
                st.dataframe(drift_df, hide_index=True, use_container_width=True)
            st.divider()

    # ── TAB 4: Verification ───────────────────────────────────────────────────
    with tab4:
        st.subheader("Independent Verification — Sharpe Ratio")
        st.caption(
            "We recompute the Sharpe ratio manually from weights × daily returns "
            "to confirm the optimizer's output. Both methods must match."
        )

        for rets, tickers, label in [
            (rets_a, TICKERS_A, "Goal A — Conservative"),
            (rets_b, TICKERS_B, "Goal B — Growth"),
        ]:
            opt_w = max_sharpe(rets)
            r, v, s = port_stats(opt_w, rets)

            # Manual path: daily weighted return series → annualize
            daily_port  = (rets * opt_w).sum(axis=1)
            manual_r    = daily_port.mean() * 252
            manual_v    = daily_port.std() * np.sqrt(252)
            manual_s    = manual_r / manual_v

            st.markdown(f"**{label}**")
            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("Pipeline Sharpe",   f"{s:.4f}")
            vc2.metric("Manual Sharpe",     f"{manual_s:.4f}")
            vc3.metric("Difference",        f"{abs(s - manual_s):.6f}",
                       delta="✅ Match" if abs(s - manual_s) < 0.001 else "⚠️ Mismatch")
            st.divider()

        st.markdown("**Known Limitations**")
        st.markdown(
            "- Historical returns are not a guarantee of future performance\n"
            "- ARKK has a shorter history (since 2014); estimates are less reliable\n"
            "- No transaction costs, taxes, or rebalancing friction are modeled\n"
            "- Conclusion becomes unreliable if correlations shift structurally "
            "(e.g. a prolonged stagflation regime where bonds and stocks fall together)"
        )

# ── AI Disclosure ─────────────────────────────────────────────────────────────
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

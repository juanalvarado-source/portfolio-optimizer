# Marco's Portfolio Optimizer

**Deployed app:** *(add Streamlit Community Cloud URL here after deployment)*

## User & Decision
Marco, 43, corporate manager with $80,000 to invest across two goals:
- **Goal A:** $30,000 needed in ~4 years (home renovation / side business)
- **Goal B:** $50,000 for retirement in ~20 years

The app answers: *"What is the optimal allocation for each goal, and is Marco's portfolio staying on track?"*

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Provenance
- **Provider:** Yahoo Finance via `yfinance`
- **Instruments:**
  - Goal A (Conservative): TIP, BND, SCHD, VNQ
  - Goal B (Growth): QQQ, ARKK, IEMG, SPY
- **Date range:** Dynamic — fetched at runtime (last 5–20 years depending on goal horizon)
- **Retrieval:** Automatic on app load; no static data files used
- **Relevant field:** Adjusted closing prices (`auto_adjust=True`)

## Features
- Efficient Frontier with Max Sharpe, Risk Parity, and Equal-Weight comparison
- Scenario Analysis (2008, 2020, 2022 crises)
- Rebalancing Check with drift alerts
- Independent Verification of Sharpe ratio calculation

## Verification
The Sharpe ratio is computed two ways: via the optimizer pipeline and manually from daily weighted returns. Both are shown in the Verification tab and must match within 0.001.

## Known Limitations
- Historical returns do not guarantee future performance
- ARKK has limited history (since 2014)
- No transaction costs or taxes modeled
- Correlations may shift in structural market regime changes

## AI-Use Disclosure
Claude Code (Anthropic) was used to design, build, and debug this application. All financial calculations were reviewed against textbook formulas and verified via the in-app Verification tab. Every team member reviewed the code and can explain the logic.

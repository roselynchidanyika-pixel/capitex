# Capital Projects Finance AI Agent

AI-powered decision support for infrastructure and capital project finance in
Zimbabwe — built for commercial banks, development finance institutions (DFIs),
government ministries, corporate finance teams, and construction companies.

The agent predicts project cost overruns, computes Zimbabwe-adjusted capital
budgeting metrics, quantifies uncertainty with Monte Carlo simulation, and
optimizes project portfolios under budget and risk constraints — all grounded
in **live macroeconomic data** (World Bank, FRED, Open Exchange Rates).

---

## Features

| Feature | Description |
|---|---|
| **Overrun Risk Prediction** | ML model (Gradient Boosting / XGBoost) predicts P(cost overrun ≥ 20%). Returns risk category (LOW / MEDIUM / HIGH) and top 5 risk drivers per project. |
| **AI-Enhanced Capital Budgeting** | Zimbabwe-adjusted NPV, IRR, MIRR, payback, discounted payback, Profitability Index (PI), and DSCR — with inflation cost escalation, currency-depreciation-adjusted discount rates, and overrun-adjusted capex. |
| **Monte Carlo Simulation** | 10,000+ simulations produce NPV distributions, P(NPV>0), Value-at-Risk (95%), CVaR, percentiles, and tornado sensitivity charts. |
| **Portfolio Optimization** | Mixed-Integer Linear Programming (PuLP + CBC) selects the optimal project portfolio under budget, risk-exposure, sector-quota, and strategic-priority constraints. Includes budget sensitivity and efficient frontier. |
| **Live Macroeconomic Data** | Fetches exchange rates, inflation (Zimbabwe + US), policy interest rates, gold price (ZiG backing), foreign reserves, and GDP growth in real time from public APIs, with scenario-based FX forecasting. |
| **Automated Reporting** | Generates 1-page Project Risk Briefs, draft Credit Memos, and Portfolio Dashboards — downloadable as text. |

---

## Getting Started

### 1. Install dependencies

Requires **Python 3.10+**.

```bash
cd capital_projects_agent
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501`.

### 3. Typical workflow

1. **Dashboard** — review the live macroeconomic snapshot (ZiG/USD, inflation,
   policy rate, gold, reserves) and exchange-rate scenario forecasts.
2. **Overrun Risk** — upload your projects CSV (see template below), load sample
   projects, or enter a project manually; run the prediction to get risk scores.
3. **Capital Budgeting** — select a project, tune assumptions (debt ratio,
   tenor, revenue), and compute the full risk-adjusted valuation.
4. **Monte Carlo** — quantify NPV uncertainty and identify key drivers.
5. **Portfolio Optimization** — build a portfolio from the predictions and solve
   the MILP to see what to fund, defer, or reject.
6. **Reports** — export project briefs, credit memos, and portfolio dashboards.

---

## Data

### Project data template

Upload CSV files with the following columns (see
`data/sample_projects_template.csv` for a filled example):

| Column | Description | Example |
|---|---|---|
| `project_id` | Unique identifier | `ZIM-1001` |
| `sector` | roads, buildings, energy, water, telecoms, mining, agriculture | `roads` |
| `baseline_capex` | Baseline capital cost (USD) | `8500000` |
| `baseline_duration_days` | Planned duration in days | `730` |
| `complexity_score` | 1 (simple) to 5 (highly complex) | `4` |
| `design_completeness` | Design maturity at award, 0–1 | `0.55` |
| `procurement_delay_days` | Procurement slippage in days | `120` |
| `num_change_orders` | Number of variation orders | `5` |
| `contractor_type` | local_small, local_large, international | `local_large` |
| `funding_source` | govt, donor, PPP, mixed | `govt` |
| `procurement_method` | open_competitive, restricted, direct | `open_competitive` |
| `province` | Zimbabwe province | `Harare` |

### Macroeconomic data sources

Data is fetched live from public APIs with caching. When a source is
unavailable (rate limits, gaps in Zimbabwe-specific series), the agent
transparently falls back to cached or estimated values — each indicator
reports its source and status in the UI.

| Indicator | Primary source |
|---|---|
| ZWL/ZiG exchange rate | Open Exchange Rates, FRED (`DZXZWLUSDM`) |
| Zimbabwe inflation | World Bank (`FP.CPI.TOTL.ZG`) |
| US inflation | FRED (`CPIAUCSL`) |
| Zimbabwe policy rate | FRED (`INTDSRZWM193N`) |
| US Fed funds rate | FRED (`DFEDTARU`) |
| Gold price (ZiG backing) | metals.dev |
| Foreign reserves | World Bank (`FI.RES.TOTL.CD`) |
| GDP growth | World Bank (`NY.GDP.MKTP.KD.ZG`) |

#### Using real/paid API keys

The free FRED, Open Exchange Rates, and metals.dev keys have rate limits.
For production use, replace the demo keys in `utils/macro_data.py`:

```python
# utils/macro_data.py
params={"api_key": "YOUR_FRED_KEY", ...}      # FRED demo -> real key
params={"app_id": "YOUR_OER_APP_ID", ...}     # Open Exchange Rates
params={"api_key": "YOUR_METALS_DEV_KEY", ...}  # metals.dev
```

RBZ official series (monetary policy statements, statistical bulletins,
ZIMSTAT CPI) can be plugged into `fetch_inflation_zimbabwe()` and
`fetch_interest_rate_zim()` in the same file.

---

## Model details

- **Algorithm:** GradientBoostingClassifier (XGBoost if installed).
- **Training:** 5-fold cross-validation on historical project data
  (ROC-AUC ≈ 0.85 on generated Zimbabwean infrastructure data).
- **Auto-training:** the model trains on synthetic historical data on first
  run, then persists to `models/overrun_model.pkl`. Replace the synthetic
  dataset with your real 50–100 project history for production use — call
  `train_model(your_dataframe)` from `models/overrun_model.py`.
- **Features:** capex, duration, complexity, design completeness, procurement
  delay, change orders, contractor/funding/procurement categories, province,
  plus live macro features (inflation, FX, interest rates, reserves).

**Explainability:** each prediction reports its top 5 risk-driver features so
decisions are auditable, not a black box.

---

## Project structure

```
capital_projects_agent/
├── app.py                     # Streamlit web application (7 pages)
├── requirements.txt
├── data/
│   └── sample_projects_template.csv   # Upload template / sample dataset
├── models/
│   ├── capital_budgeting.py   # NPV, IRR, MIRR, DSCR, scenarios
│   ├── monte_carlo.py         # Monte Carlo + tornado sensitivity
│   ├── overrun_model.py       # ML overrun prediction
│   └── portfolio_optimizer.py # MILP portfolio selection + sensitivity
└── utils/
    ├── macro_data.py          # Live macro data fetchers + FX forecasting
    └── reporting.py           # Project briefs, credit memos, dashboards
```

---

## Ethics & usage notes

This agent is a **decision-support tool**, not a decision replacement. Credit
committees, Cabinet, and investment committees should review all
recommendations, weighing qualitative factors, stakeholder input, and
political context. Every output includes this disclaimer.

- **Data privacy:** anonymize sensitive data (project/company names) before
  upload. Comply with Zimbabwe's Data Protection Act (2021).
- **Explainability:** predictions expose their top risk drivers.
- **Bias mitigation:** review model behavior across regions, sectors, and
  contractor types; manual overrides are supported at the decision stage.
- **Continuous improvement:** retrain quarterly as project outcomes accrue and
  add new data sources as they become available.

---

## Roadmap

- [ ] RBZ / ZIMSTAT official API integration
- [ ] Real options valuation (defer / expand / abandon)
- [ ] Streamlit → FastAPI production API
- [ ] LSTM-based exchange-rate forecasting
- [ ] Satellite/drone imagery for construction progress monitoring
- [ ] PostgreSQL persistence and multi-user role-based access
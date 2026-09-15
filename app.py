"""Capital Projects Finance AI Agent - Streamlit Web Application (Zimbabwe)."""
import sys
import os
import importlib.util

# ---------------------------------------------------------------------------
# Layout-agnostic bootstrap.
# Preferred layout (repo root):
#     app.py, requirements.txt, utils/, models/, README.md
# Fallback layout (flat deploy zip): all modules next to app.py.
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
_SUB_DIR = os.path.join(APP_DIR, "capital_projects_agent")
if os.path.isdir(os.path.join(_SUB_DIR, "utils")):
    sys.path.insert(0, _SUB_DIR)


def _find_dir_with(both_files):
    """Return the first directory under APP_DIR containing every file in
    `both_files`, ignoring caches/virtualenvs."""
    skip = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".idea"}
    for root, dirs, files in os.walk(APP_DIR):
        dirs[:] = [d for d in dirs if d not in skip]
        if os.path.dirname(root) == os.path.dirname(APP_DIR) and root != APP_DIR:
            continue
        if all(f in files for f in both_files):
            return root
    return None


def _import_modules():
    """Import our modules either as a package (utils/models) or flat."""
    if importlib.util.find_spec("utils") is not None:
        from utils.macro_data import (
            fetch_all_macro, fetch_exchange_rate_history, fetch_exchange_rate_zig,
            forecast_exchange_rate, generate_cost_escalation_curve,
        )
        from utils.reporting import (
            generate_project_brief, generate_credit_memo, generate_portfolio_summary,
        )
        from models.overrun_model import (
            generate_synthetic_training_data, train_model, predict_overrun, get_model_info,
        )
        from models.capital_budgeting import full_project_valuation, risk_adjusted_discount_rate
        from models.monte_carlo import monte_carlo_npv, sensitivity_tornado
        from models.portfolio_optimizer import (
            optimize_portfolio, sensitivity_budget, efficient_frontier,
        )
    else:
        utils_dir = _find_dir_with(["macro_data.py", "reporting.py"])
        models_dir = _find_dir_with(["overrun_model.py", "capital_budgeting.py"])
        missing = []
        for label, path in (("utils/macro_data.py", utils_dir),
                            ("models/overrun_model.py", models_dir)):
            if not path:
                missing.append(label)
        if missing:
            raise ImportError(
                "Cannot find required project modules: " + ", ".join(missing) + ".\n"
                "Your GitHub repo must contain (next to app.py): utils/, models/, "
                "requirements.txt, OR the contents of the capitex_deploy/ folder "
                "uploaded flat to the repo root."
            )
        if utils_dir:
            sys.path.insert(0, utils_dir)
        if models_dir:
            sys.path.insert(0, models_dir)
        from macro_data import (
            fetch_all_macro, fetch_exchange_rate_history, fetch_exchange_rate_zig,
            forecast_exchange_rate, generate_cost_escalation_curve,
        )
        from reporting import (
            generate_project_brief, generate_credit_memo, generate_portfolio_summary,
        )
        from overrun_model import (
            generate_synthetic_training_data, train_model, predict_overrun, get_model_info,
        )
        from capital_budgeting import full_project_valuation, risk_adjusted_discount_rate
        from monte_carlo import monte_carlo_npv, sensitivity_tornado
        from portfolio_optimizer import (
            optimize_portfolio, sensitivity_budget, efficient_frontier,
        )

    return {
        "fetch_all_macro": fetch_all_macro,
        "fetch_exchange_rate_history": fetch_exchange_rate_history,
        "fetch_exchange_rate_zig": fetch_exchange_rate_zig,
        "forecast_exchange_rate": forecast_exchange_rate,
        "generate_cost_escalation_curve": generate_cost_escalation_curve,
        "generate_project_brief": generate_project_brief,
        "generate_credit_memo": generate_credit_memo,
        "generate_portfolio_summary": generate_portfolio_summary,
        "generate_synthetic_training_data": generate_synthetic_training_data,
        "train_model": train_model,
        "predict_overrun": predict_overrun,
        "get_model_info": get_model_info,
        "full_project_valuation": full_project_valuation,
        "risk_adjusted_discount_rate": risk_adjusted_discount_rate,
        "monte_carlo_npv": monte_carlo_npv,
        "sensitivity_tornado": sensitivity_tornado,
        "optimize_portfolio": optimize_portfolio,
        "sensitivity_budget": sensitivity_budget,
        "efficient_frontier": efficient_frontier,
    }


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

M = _import_modules()
globals().update(M)

st.set_page_config(
    page_title="Capital Projects Finance AI Agent - Zimbabwe",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "projects_df" not in st.session_state:
    st.session_state.projects_df = None
if "macro_data" not in st.session_state:
    st.session_state.macro_data = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "valuation" not in st.session_state:
    st.session_state.valuation = None
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = None
if "opt_result" not in st.session_state:
    st.session_state.opt_result = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_macro():
    if st.session_state.macro_data is None:
        with st.spinner("Fetching live macroeconomic data (World Bank, FRED, OER)..."):
            st.session_state.macro_data = fetch_all_macro()


def project_to_dict(row):
    """Convert a project row to full valuation input dict."""
    duration_days = int(row.get("baseline_duration_days", 365))
    duration_years = max(int(round(duration_days / 365)), 1)
    base_capex = float(row.get("baseline_capex", 0))
    revenue_factor = float(row.get("revenue_factor", 0.55))
    opex_factor = float(row.get("opex_factor", 0.15))
    overrun_prob = float(row.get("overrun_probability", 0.4))
    return {
        "project_id": str(row.get("project_id", "")),
        "sector": str(row.get("sector", "roads")),
        "baseline_capex": base_capex,
        "baseline_duration_days": duration_days,
        "complexity_score": int(row.get("complexity_score", 3)),
        "design_completeness": float(row.get("design_completeness", 0.7)),
        "procurement_delay_days": int(row.get("procurement_delay_days", 0)),
        "num_change_orders": int(row.get("num_change_orders", 0)),
        "contractor_type": str(row.get("contractor_type", "local_large")),
        "funding_source": str(row.get("funding_source", "govt")),
        "procurement_method": str(row.get("procurement_method", "open_competitive")),
        "province": str(row.get("province", "Harare")),
        "duration_years": duration_years,
        "annual_revenues": [base_capex * revenue_factor * (1 + i * 0.08)
                            for i in range(duration_years)],
        "annual_opex": [base_capex * opex_factor for _ in range(duration_years)],
        "debt_ratio": 0.6,
        "loan_tenor_years": min(duration_years + 2, 15),
        "interest_rate": 0.18,
        "overrun_probability": overrun_prob,
    }


PROJECT_SAMPLE_COLUMNS = [
    "project_id", "sector", "baseline_capex", "baseline_duration_days",
    "complexity_score", "design_completeness", "procurement_delay_days",
    "num_change_orders", "contractor_type", "funding_source",
    "procurement_method", "province",
]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Capital Projects Finance")
st.sidebar.caption("AI Decision Support for Zimbabwe")

nav = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Overrun Risk", "Capital Budgeting", "Monte Carlo",
     "Portfolio Optimization", "Macroeconomic Data", "Reports"],
)

# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------
if nav == "Dashboard":
    st.title("Capital Projects Finance AI Agent")
    st.caption("Infrastructure & Capital Project Finance Intelligence for Zimbabwe")

    ensure_macro()
    md = st.session_state.macro_data or {}

    if "exchange_rate" in md:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ZiG/USD", f"{md['exchange_rate'].get('zig_per_usd', 'N/A')}",
                    md["exchange_rate"].get("source", ""))
        col2.metric("Inflation (Zim)", f"{md['inflation'].get('zimbabwe_pct', 'N/A')}%",
                    "YoY CPI")
        col3.metric("Policy Rate", f"{md['interest_rates'].get('zim_policy_pct', 'N/A')}%")
        col4.metric("Gold (ZiG backing)", f"${md['gold'].get('price_usd_per_oz', 0):,.0f}/oz")
        col5.metric("Foreign Reserves", f"${md['reserves'].get('total_usd_bn', 0):.2f}B")

        st.markdown("---")
        st.subheader("Data Source Status")
        for label, name, status in [
            ("USD/ZiG Exchange Rate", md["exchange_rate"].get("source"), md["exchange_rate"].get("status")),
            ("Zimbabwe Inflation", md["inflation"].get("source_zim"), md["inflation"].get("status")),
            ("US Inflation", md["inflation"].get("source_us"), md["inflation"].get("status")),
            ("Policy Rate", md["interest_rates"].get("source_zim"), md["interest_rates"].get("status")),
            ("Gold Price", md["gold"].get("source"), md["gold"].get("status")),
            ("Foreign Reserves", md["reserves"].get("source"), md["reserves"].get("status")),
            ("GDP Growth", md["gdp"].get("source"), md["gdp"].get("status")),
        ]:
            st.write(f"- **{label}**: {name} `({status})`")

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Exchange Rate History")
            try:
                hist = fetch_exchange_rate_history(365)
                if hist is not None and not hist.empty:
                    fig = px.line(hist, x="date", y="rate",
                                  title="ZWL/ZiG per USD (12 months)")
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load exchange rate history: {e}")

        with col_r:
            st.subheader("Exchange Rate Forecast (Scenarios)")
            try:
                fc = forecast_exchange_rate(years=5)
                if fc is not None and not fc.empty:
                    fig2 = px.line(fc, x="year", y="zig_per_usd", color="scenario",
                                   title="ZiG/USD Forecast")
                    fig2.update_layout(height=350)
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load forecast: {e}")
    else:
        st.error("Macro data could not be fetched. Check internet connection.")

    st.info("**Purposes:** predict cost overruns, compute Zimbabwe-adjusted capital "
            "budgeting metrics, run Monte Carlo simulations, and optimize project "
            "portfolios. **This is a decision-support tool** - final decisions should "
            "consider qualitative factors, stakeholder input, and political context.")

# ---------------------------------------------------------------------------
# OVERRUN RISK
# ---------------------------------------------------------------------------
elif nav == "Overrun Risk":
    st.title("Project Overrun Risk Prediction")
    st.caption("Predicts P(cost overrun >= 20%) using Gradient Boosting / XGBoost")

    tab_data, tab_model = st.tabs(["Data & Predictions", "Model Info"])

    with tab_data:
        upload_option = st.radio(
            "Data source:",
            ["Sample projects (30)", "Upload CSV/Excel", "Enter single project"],
        )

        df = None
        if upload_option.startswith("Sample"):
            if st.button("Generate Sample Projects", type="primary"):
                synth = generate_synthetic_training_data(30)
                df = synth.drop(columns=["overrun_20pct", "actual_overrun_pct"], errors="ignore")
                st.session_state.projects_df = df
                st.success("Sample projects generated.")

        elif upload_option.startswith("Upload"):
            uploaded = st.file_uploader("Upload projects file", type=["csv", "xlsx"])
            if uploaded is not None:
                try:
                    df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
                    st.session_state.projects_df = df
                    st.success(f"Loaded {len(df)} projects")
                    st.dataframe(df.head(20), use_container_width=True)
                except Exception as ex:
                    st.error(f"Error loading file: {ex}")

        else:
            st.subheader("Manual Project Entry")
            col1, col2, col3 = st.columns(3)
            with col1:
                sector = st.selectbox("Sector", ["roads", "buildings", "energy", "water",
                                                 "telecoms", "mining", "agriculture"])
                capex = st.number_input("Baseline Capex (USD)", min_value=100000.0,
                                        value=10_000_000.0, step=500000.0)
                complexity = st.slider("Complexity (1-5)", 1, 5, 3)
                design = st.slider("Design Completeness (0-1)", 0.0, 1.0, 0.7)
            with col2:
                duration = st.number_input("Duration (days)", min_value=30.0, value=365.0)
                proc_delay = st.number_input("Procurement Delay (days)", min_value=0.0, value=30.0)
                change_orders = st.number_input("Number of Change Orders", min_value=0, value=2)
                contractor = st.selectbox("Contractor Category",
                                          ["local_small", "local_large", "international"])
            with col3:
                funding = st.selectbox("Funding Source", ["govt", "donor", "PPP", "mixed"])
                proc_method = st.selectbox("Procurement Method",
                                           ["open_competitive", "restricted", "direct"])
                province = st.selectbox("Province", [
                    "Harare", "Bulawayo", "Manicaland", "Mashonaland Central",
                    "Mashonaland East", "Mashonaland West", "Masvingo",
                    "Matabeleland North", "Matabeleland South", "Midlands"])
                project_id = st.text_input("Project ID", "NEW-001")

            if st.button("Add & Predict", type="primary"):
                df = pd.DataFrame([{
                    "project_id": project_id, "sector": sector,
                    "baseline_capex": capex, "baseline_duration_days": duration,
                    "complexity_score": complexity, "design_completeness": design,
                    "procurement_delay_days": proc_delay, "num_change_orders": change_orders,
                    "contractor_type": contractor, "funding_source": funding,
                    "procurement_method": proc_method, "province": province,
                }])
                st.session_state.projects_df = df

        if st.session_state.projects_df is not None:
            st.markdown("---")
            if st.button("Run Overrun Prediction", type="primary"):
                ensure_macro()
                with st.spinner("Predicting cost overrun probabilities..."):
                    preds = predict_overrun(st.session_state.projects_df,
                                            st.session_state.macro_data)
                    st.session_state.predictions = preds
                    st.session_state.projects_df = preds
                st.success(f"Predictions complete for {len(preds)} projects")

        if st.session_state.predictions is not None:
            preds = st.session_state.predictions
            st.subheader("Overrun Risk Results")
            risk_counts = preds["risk_category"].value_counts()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("LOW Risk", risk_counts.get("LOW", 0))
            col2.metric("MEDIUM Risk", risk_counts.get("MEDIUM", 0))
            col3.metric("HIGH Risk", risk_counts.get("HIGH", 0))
            col4.metric("Avg P(overrun)", f"{preds['overrun_probability'].mean() * 100:.1f}%")

            display = preds[[
                "project_id", "sector", "baseline_capex", "baseline_duration_days",
                "overrun_probability", "risk_category",
            ]].copy()
            display["baseline_capex"] = display["baseline_capex"].apply(lambda x: f"${x:,.0f}")
            display["overrun_probability"] = display["overrun_probability"].apply(
                lambda x: f"{x * 100:.1f}%")
            st.dataframe(display, use_container_width=True)

            fig = px.bar(
                preds.sort_values("overrun_probability"),
                x="project_id", y="overrun_probability", color="risk_category",
                title="Cost Overrun Probability by Project",
                color_discrete_map={"LOW": "green", "MEDIUM": "orange", "HIGH": "red"},
            )
            fig.update_layout(yaxis_tickformat=".0%", height=450)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View risk drivers (top 5 per project)"):
                import json as _json
                for _, row in preds.iterrows():
                    st.markdown(f"**{row['project_id']}** - {row['risk_category']}")
                    try:
                        drivers = _json.loads(row["top_risk_drivers"])
                        drivers = sorted(drivers, key=lambda x: abs(x[1]), reverse=True)[:5]
                        for feat, contrib in drivers:
                            st.write(f"  - {feat}: {contrib:+.4f}")
                    except Exception:
                        st.write("  (driver data unavailable)")

    with tab_model:
        info = get_model_info()
        if info.get("status") == "trained":
            st.json(info)
        else:
            st.warning("Model is not yet trained. Run a prediction in the other tab; "
                       "it will auto-train on synthetic historical data on first run.")

# ---------------------------------------------------------------------------
# CAPITAL BUDGETING
# ---------------------------------------------------------------------------
elif nav == "Capital Budgeting":
    st.title("AI-Enhanced Capital Budgeting")
    st.caption("Zimbabwe-adjusted NPV, IRR, MIRR, DSCR with inflation & currency risk")

    if st.session_state.predictions is None:
        st.warning("Run overrun predictions in the 'Overrun Risk' tab first.")
    else:
        ensure_macro()
        preds = st.session_state.predictions
        st.subheader("Select Project")
        project_selector = st.selectbox(
            "Project ID", preds["project_id"].astype(str).tolist(),
            format_func=lambda pid: f"{pid} - {preds[preds['project_id'].astype(str) == pid].iloc[0].get('sector', '')}",
        )
        row = preds[preds["project_id"].astype(str) == project_selector].iloc[0]

        st.markdown("---")
        st.subheader("Adjust Assumptions")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            debt_ratio = st.slider("Debt Ratio", 0.0, 0.9, 0.6, 0.05)
        with col2:
            loan_tenor = st.slider("Loan Tenor (years)", 1, 15, 5)
        with col3:
            loan_rate = st.slider("Loan Interest Rate", 5.0, 40.0, 18.0, 0.5) / 100
        with col4:
            revenue_mult = st.slider("Revenue Multiple of Capex", 0.1, 1.0, 0.55, 0.05)

        if st.button("Run Valuation", type="primary"):
            project_dict = project_to_dict(row)
            project_dict["debt_ratio"] = debt_ratio
            project_dict["loan_tenor_years"] = loan_tenor
            project_dict["interest_rate"] = loan_rate
            d_years = project_dict["duration_years"]
            base_rev = float(row.get("baseline_capex", 0)) * revenue_mult
            project_dict["annual_revenues"] = [base_rev * (1 + i * 0.08)
                                               for i in range(d_years)]
            with st.spinner("Computing risk-adjusted capital budgeting metrics..."):
                st.session_state.valuation = full_project_valuation(project_dict,
                                                                    st.session_state.macro_data)

        if st.session_state.valuation is not None:
            val = st.session_state.valuation
            st.markdown("---")
            st.subheader("Valuation Results")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("NPV (USD)", f"${val['npv_usd']:,.0f}",
                        "Positive" if val["npv_usd"] > 0 else "Negative")
            col2.metric("IRR", f"{val['irr_pct']}%" if val["irr_pct"] is not None else "N/A")
            col3.metric("MIRR", f"{val['mirr_pct']}%" if val["mirr_pct"] is not None else "N/A")
            col4.metric("Profitability Index", f"{val['profitability_index']:.2f}")
            col5.metric("Disc. Payback", f"{val['discounted_payback_years']} yrs")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Adjusted Capex", f"${val['adjusted_capex_usd']:,.0f}")
            col2.metric("Risk-Adj. Discount Rate", f"{val['risk_adjusted_discount_rate']}%")
            col3.metric("Expected NPV", f"${val['expected_npv_usd']:,.0f}")
            col4.metric("Recommendation", val["recommendation"])

            st.subheader("Scenario Analysis (Base / Downside / Upside)")
            sa = val["scenario_analysis"]
            st.dataframe(sa, use_container_width=True)

            fig_sc = px.bar(sa, x="scenario", y="npv_usd", color="scenario",
                            title="NPV by Scenario (USD)",
                            text=[f"${v:,.0f}" for v in sa["npv_usd"]])
            fig_sc.update_layout(yaxis_title="NPV (USD)", height=400,
                                 showlegend=False)
            st.plotly_chart(fig_sc, use_container_width=True)

            st.subheader("DSCR Profile")
            dscr_df = val["dscr_profile"]
            st.dataframe(dscr_df, use_container_width=True)
            fig_d = px.bar(dscr_df, x="period", y="dscr", color="dscr_adequate",
                           title="Debt Service Coverage Ratio by Period",
                           labels={"period": "Period (year)", "dscr": "DSCR"})
            fig_d.add_hline(y=1.2, line_dash="dash", line_color="green",
                            annotation_text="Covenant (1.2x)")
            fig_d.update_layout(height=350)
            st.plotly_chart(fig_d, use_container_width=True)

# ---------------------------------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------------------------------
elif nav == "Monte Carlo":
    st.title("Monte Carlo Simulation")
    st.caption("Quantify NPV uncertainty with 10,000+ simulations")

    ensure_macro()
    md = st.session_state.macro_data or {}
    default_inf = md.get("inflation", {}).get("zimbabwe_pct", 55) or 55
    default_ex = md.get("exchange_rate", {}).get("zig_per_usd", 13.5) or 13.5

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mc_capex = st.number_input("Baseline Capex (USD)", min_value=100000.0,
                                   value=25_000_000.0, step=1000000.0)
    with col2:
        mc_duration = st.number_input("Duration (years)", min_value=1, max_value=20, value=3)
    with col3:
        mc_rev = st.number_input("Annual Revenue (USD)", min_value=0.0,
                                 value=12_000_000.0, step=500000.0)
    with col4:
        mc_opex = st.number_input("Annual Opex (USD)", min_value=0.0,
                                  value=3_000_000.0, step=100000.0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mc_inf = st.slider("Inflation (%)", 5.0, 200.0, float(default_inf), 5.0) / 100
    with col2:
        mc_rate = st.slider("Discount Rate (%)", 5.0, 50.0, 25.0, 1.0) / 100
    with col3:
        mc_ex = st.number_input("Exchange Rate (ZiG/USD)", min_value=1.0,
                                value=float(default_ex), step=0.5)
    with col4:
        mc_n = st.number_input("Simulations", min_value=1000, max_value=100000,
                               value=10000, step=1000)

    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {mc_n:,} simulations..."):
            res = monte_carlo_npv(
                base_capex=mc_capex, duration_years=int(mc_duration),
                annual_revenues_base=mc_rev, annual_opex_base=mc_opex,
                discount_rate=mc_rate, inflation_rate=mc_inf,
                base_exchange_rate=mc_ex, n_simulations=int(mc_n),
            )
        st.session_state.mc_result = res

    if "mc_result" in st.session_state and st.session_state.mc_result:
        res = st.session_state.mc_result
        st.subheader("NPV Distribution")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Expected NPV", f"${res['expected_npv']:,.0f}")
        k2.metric("P(NPV > 0)", f"{res['prob_npv_positive']:.1f}%")
        k3.metric("Median NPV", f"${res['median_npv']:,.0f}")
        k4.metric("VaR (95%)", f"${res['var_95']:,.0f}")
        k5.metric("CVaR (95%)", f"${res['cvar_95']:,.0f}")
        k6.metric("Std Dev", f"${res['npv_std']:,.0f}")

        hist_counts = res["histogram"]["counts"]
        hist_edges = res["histogram"]["edges"]
        centers = [(hist_edges[i] + hist_edges[i + 1]) / 2 for i in range(len(hist_edges) - 1)]
        fig_h = go.Figure(go.Bar(x=centers, y=hist_counts,
                                 marker_color="steelblue", name="NPV"))
        fig_h.add_vline(x=res["expected_npv"], line_color="green", line_dash="dash",
                        annotation_text="E[NPV]")
        fig_h.add_vline(x=res["var_95"], line_color="red", line_dash="dot",
                        annotation_text="VaR 95%")
        fig_h.update_layout(title=f"NPV Distribution ({res['n_simulations']:,} simulations)",
                            xaxis_title="NPV (USD)", yaxis_title="Frequency", height=450)
        st.plotly_chart(fig_h, use_container_width=True)

        st.subheader("Tornado Sensitivity (what drives NPV)")
        with st.spinner("Computing sensitivity..."):
            tornado = sensitivity_tornado(
                base_capex=mc_capex, duration_years=int(mc_duration),
                annual_revenues=mc_rev, annual_opex=mc_opex,
                discount_rate=mc_rate, inflation_rate=mc_inf,
                base_exchange_rate=mc_ex,
            )
        if tornado is not None and not tornado.empty:
            st.dataframe(tornado, use_container_width=True)
            fig_t = px.bar(tornado, x="impact_range", y="variable", orientation="h",
                           title="Sensitivity to +/- 20% Changes",
                           labels={"impact_range": "NPV Impact Range (USD)",
                                   "variable": ""})
            fig_t.update_layout(height=400)
            st.plotly_chart(fig_t, use_container_width=True)

        st.info(f"Percentiles: P5=${res['percentiles']['p5']:,.0f}  "
                f"P25=${res['percentiles']['p25']:,.0f}  "
                f"P50=${res['percentiles']['p50']:,.0f}  "
                f"P75=${res['percentiles']['p75']:,.0f}  "
                f"P95=${res['percentiles']['p95']:,.0f}")

# ---------------------------------------------------------------------------
# PORTFOLIO OPTIMIZATION
# ---------------------------------------------------------------------------
elif nav == "Portfolio Optimization":
    st.title("Portfolio Optimization")
    st.caption("Mixed-Integer Linear Programming capital rationing (MILP)")

    tab_data, tab_opt = st.tabs(["Project Data", "Optimization"])

    with tab_data:
        if st.session_state.predictions is not None:
            if st.button("Build portfolio from current predictions", type="primary"):
                preds = st.session_state.predictions.copy()
                capex = preds["baseline_capex"].astype(float)
                preds["expected_npv"] = preds["expected_npv"] if "expected_npv" in preds.columns \
                    else capex * (1 - preds["overrun_probability"]) * 0.25
                preds["risk_score"] = preds["overrun_probability"]
                preds["strategic_score"] = np.where(preds["sector"].isin(["energy", "roads", "water"]), 1.5, 1.0)
                st.session_state.portfolio_df = preds
                st.success("Portfolio built from {} predictions".format(len(preds)))

        upload_option = st.radio("Or upload your own", ["Keep current", "Upload CSV"])
        if upload_option == "Upload CSV":
            uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"],
                                        key="port_csv")
            if uploaded is not None:
                try:
                    st.session_state.portfolio_df = pd.read_csv(uploaded)
                    st.success("Uploaded.")
                except Exception as ex:
                    st.error(f"Error: {ex}")

        if st.session_state.portfolio_df is not None:
            st.subheader("Portfolio Projects")
            st.dataframe(st.session_state.portfolio_df.head(50), use_container_width=True)
        else:
            st.info("Run the 'Overrun Risk' prediction first, or upload a CSV.")

    with tab_opt:
        if st.session_state.portfolio_df is None:
            st.warning("No project data. Build portfolio from predictions or upload CSV first.")
        else:
            pdf = st.session_state.portfolio_df
            capex_col = "baseline_capex" if "baseline_capex" in pdf.columns else "capex"
            if capex_col not in pdf.columns:
                st.error("CSV must contain a 'baseline_capex' or 'capex' column.")
            else:
                total_requested = float(pdf[capex_col].sum())
                col1, col2, col3 = st.columns(3)
                with col1:
                    budget = st.number_input("Total Budget (USD)", min_value=0.0,
                                             value=total_requested, step=1000000.0)
                with col2:
                    max_risk = st.number_input("Max Risk Exposure (weighted USD)",
                                               min_value=0.0,
                                               value=total_requested * 0.4,
                                               step=500000.0)
                with col3:
                    min_strat = st.slider("Min Total Strategic Score", 0.0,
                                          float(len(pdf) * 1.5), 0.0, 0.5)

                sector_choices = pdf["sector"].unique().tolist() if "sector" in pdf.columns else []
                quotas = {}
                idx = (st.multiselect("Sector quotas (min % of budget)",
                                      options=[str(s) for s in sector_choices])) if sector_choices else []
                for s in idx:
                    q = st.slider(f"Min % budget to {s}", 0, 100, 10, key=f"q_{s}")
                    quotas[s] = q / 100

                if st.button("Optimize Portfolio", type="primary"):
                    pdf2 = pdf.copy()
                    if "expected_npv" not in pdf2.columns:
                        pdf2["expected_npv"] = pdf2[capex_col] * 0.25
                    if "risk_score" not in pdf2.columns:
                        pdf2["risk_score"] = 0.5
                    if "strategic_score" not in pdf2.columns:
                        pdf2["strategic_score"] = 1.0
                    with st.spinner("Solving MILP portfolio optimization..."):
                        st.session_state.opt_result = optimize_portfolio(
                            pdf2, budget=budget, max_risk_exposure=max_risk,
                            min_strategic_score=min_strat, sector_quotas=quotas or None,
                        )

        if st.session_state.opt_result is not None:
            opt = st.session_state.opt_result
            st.markdown("---")
            st.subheader("Optimal Portfolio")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Funded", opt["n_funded"])
            col2.metric("Deferred", opt["n_deferred"])
            col3.metric("Capex Funded", f"${opt['total_capex_funded']:,.0f}")
            col4.metric("Budget Utilization", f"{opt['budget_utilization']}%")
            col5.metric("Total NPV", f"${opt['total_expected_npv']:,.0f}")

            if opt["selected_projects"]:
                st.subheader("Projects to Fund")
                funded_df = pd.DataFrame(opt["selected_projects"])
                st.dataframe(funded_df, use_container_width=True)
            if opt["deferred_projects"]:
                st.subheader("Projects to Defer / Reject")
                deferred_df = pd.DataFrame(opt["deferred_projects"])
                st.dataframe(deferred_df, use_container_width=True)

            st.subheader("Budget Sensitivity")
            if st.button("Run Budget Sensitivity"):
                with st.spinner("Running optimization across budget scenarios..."):
                    budget_range = [budget * f for f in [0.6, 0.8, 1.0, 1.2, 1.4]]
                    sens = sensitivity_budget(pdf2, budget_range, max_risk)
                if isinstance(sens, pd.DataFrame) and not sens.empty:
                    fig_s = px.line(sens, x="budget", y="total_npv",
                                    title="Portfolio NPV vs Budget",
                                    labels={"budget": "Budget (USD)", "total_npv": "Total NPV (USD)"})
                    fig_s.update_layout(height=350)
                    st.plotly_chart(fig_s, use_container_width=True)

            st.info("Shadow prices and sensitivity analyses are available for decision "
                    "support. Final allocation requires committee review.")

# ---------------------------------------------------------------------------
# MACROECONOMIC DATA
# ---------------------------------------------------------------------------
elif nav == "Macroeconomic Data":
    st.title("Macroeconomic Data")
    st.caption("Live economic indicators for project finance decisions")

    ensure_macro()
    md = st.session_state.macro_data or {}

    if "exchange_rate" in md:
        st.subheader("Current Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ZiG/USD", f"{md['exchange_rate'].get('zig_per_usd', 'N/A')}")
        col2.metric("Zimbabwe Inflation", f"{md['inflation'].get('zimbabwe_pct', 'N/A')}%")
        col3.metric("US Inflation", f"{md['inflation'].get('us_pct', 'N/A')}%")
        col4.metric("Inflation Differential", f"{md['inflation'].get('differential', 'N/A')}pp")
        col5.metric("Interest Rate Spread", f"{md['interest_rates'].get('spread_pct', 'N/A')}pp")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Zim Policy Rate", f"{md['interest_rates'].get('zim_policy_pct', 'N/A')}%")
        col2.metric("US Fed Rate", f"{md['interest_rates'].get('us_fed_pct', 'N/A')}%")
        col3.metric("Gold Price", f"${md['gold'].get('price_usd_per_oz', 0):,.0f}/oz")
        col4.metric("GDP Growth", f"{md['gdp'].get('growth_pct', 'N/A')}%")

        st.markdown("---")
        st.subheader("Exchange Rate Scenario Forecast")
        try:
            fc = forecast_exchange_rate(years=5)
            st.dataframe(fc, use_container_width=True)
            fig_fc = px.line(fc, x="year", y="zig_per_usd", color="scenario",
                             title="ZiG/USD Scenario Forecast Paths",
                             labels={"year": "Year", "zig_per_usd": "ZiG per USD"})
            fig_fc.update_layout(height=400)
            st.plotly_chart(fig_fc, use_container_width=True)
        except Exception as e:
            st.warning(f"Forecast error: {e}")

        st.subheader("Cost Escalation Calculator")
        col1, col2, col3 = st.columns(3)
        with col1:
            esc_base = st.number_input("Base Cost (USD)", min_value=10000.0,
                                       value=1_000_000.0, step=100000.0)
        with col2:
            esc_inf = st.slider("Inflation (%)", 5.0, 200.0,
                                float(md["inflation"].get("zimbabwe_pct", 55)), 5.0)
        with col3:
            esc_months = st.number_input("Duration (months)", min_value=1, value=24)

        if st.button("Calculate Escalation"):
            curve = generate_cost_escalation_curve(esc_base, esc_inf, esc_months / 12,
                                                   steps=int(esc_months))
            total = curve["escalated_cost"].iloc[-1]
            st.metric("Escalated Total Cost",
                      f"${total:,.0f}",
                      f"+{((total / esc_base) - 1) * 100:.0f}% over {esc_months} months")
            if esc_inf >= 50:
                st.error("**High-inflation alert:** Costs escalate sharply. "
                         "Adjust project budget and funding accordingly.")
            fig_esc = px.line(curve, x="date", y="escalated_cost",
                              title="Cost Escalation Curve")
            fig_esc.update_layout(height=350)
            st.plotly_chart(fig_esc, use_container_width=True)

        st.markdown("---")
        st.subheader("Data Source Transparency")
        st.json(md)
    else:
        st.error("Macro data not available.")

# ---------------------------------------------------------------------------
# REPORTS
# ---------------------------------------------------------------------------
elif nav == "Reports":
    st.title("Automated Reports & Decision Support")
    st.caption("Generate deliverables for decision-makers")

    report_type = st.radio("Report type:",
                           ["Project Risk Brief", "Credit Memo", "Portfolio Dashboard"],
                           horizontal=True)

    if report_type == "Project Risk Brief":
        ensure_macro()
        if st.session_state.predictions is None or st.session_state.valuation is None:
            st.info("Run overrun prediction and valuation first, then generate the brief.")
        else:
            preds = st.session_state.predictions
            val = st.session_state.valuation
            pid = val.get("project_id", "")
            row = preds[preds["project_id"].astype(str) == pid].iloc[0] if pid else preds.iloc[0]

            project = {
                "project_id": row.get("project_id"), "sector": row.get("sector"),
                "baseline_capex": float(row.get("baseline_capex", 0)),
                "baseline_duration_days": int(row.get("baseline_duration_days", 0)),
                "province": row.get("province"),
            }
            ovr = {
                "risk_category": str(row.get("risk_category", "MEDIUM")),
                "overrun_probability": float(row.get("overrun_probability", 0.5)),
                "top_risk_drivers": row.get("top_risk_drivers", []),
            }
            if st.button("Generate Project Brief", type="primary"):
                brief = generate_project_brief(project, val, ovr)
                st.markdown("---")
                st.text(brief)
                st.download_button("Download Brief (.txt)", brief,
                                   file_name=f"project_brief_{row.get('project_id')}.txt")

    elif report_type == "Credit Memo":
        ensure_macro()
        if st.session_state.valuation is None:
            st.info("Run a valuation in the 'Capital Budgeting' tab first.")
        else:
            val = st.session_state.valuation
            preds = st.session_state.predictions
            pid = val.get("project_id", "")
            row = preds[preds["project_id"].astype(str) == pid].iloc[0] if preds is not None and pid else preds.iloc[0]

            project = {
                "project_id": row.get("project_id"), "sector": row.get("sector"),
                "baseline_capex": float(row.get("baseline_capex", 0)),
                "baseline_duration_days": int(row.get("baseline_duration_days", 0)),
                "province": row.get("province"), "complexity_score": row.get("complexity_score"),
                "contractor_type": row.get("contractor_type"),
                "procurement_method": row.get("procurement_method"),
                "loan_tenor_years": 5, "duration_years": 3,
            }
            ovr = {
                "risk_category": str(row.get("risk_category", "MEDIUM")),
                "overrun_probability": float(row.get("overrun_probability", 0.5)),
            }
            if st.button("Generate Credit Memo", type="primary"):
                memo = generate_credit_memo(project, val, ovr, st.session_state.macro_data)
                st.markdown("---")
                st.text(memo)
                st.download_button("Download Credit Memo (.txt)", memo,
                                   file_name=f"credit_memo_{row.get('project_id')}.txt")

    else:
        ensure_macro()
        if st.session_state.opt_result is None:
            st.info("Run portfolio optimization first, then generate the dashboard.")
        else:
            if st.button("Generate Portfolio Dashboard", type="primary"):
                summary = generate_portfolio_summary(st.session_state.opt_result,
                                                     st.session_state.macro_data)
                st.markdown("---")
                st.text(summary)
                st.download_button("Download Portfolio Dashboard (.txt)", summary,
                                   file_name="portfolio_dashboard.txt")

    st.markdown("---")
    st.caption("Ethical guideline: This agent provides decision support, not final "
               "decisions. Credit committees, Cabinet, and investment committees "
               "should review all recommendations considering qualitative and "
               "political factors.")
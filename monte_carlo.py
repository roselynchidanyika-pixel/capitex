"""
Monte Carlo simulation engine for capital project NPV uncertainty quantification.
USD-based cash flows with local-cost escalation, matching capital_budgeting logic.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def monte_carlo_npv(
    base_capex: float,
    duration_years: int,
    annual_revenues_base: float,
    annual_opex_base: float,
    discount_rate: float,
    inflation_rate: float,
    base_exchange_rate: float,
    n_simulations: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Run Monte Carlo simulation to quantify NPV uncertainty.

    Distributions:
      - Capex: Triangular (min, mode, max) reflecting overrun risk
      - Revenue: Normal (mean, std)
      - Opex: Normal (mean, std)
      - Inflation: Triangular (Zim CPI forecast)
      - Discount rate: Normal (risk-adjusted, std ~3%)
    Cash flows are in constant USD; local cost share escalates at inflation.
    """
    rng = np.random.RandomState(seed)

    capex_low = base_capex * 0.90
    capex_mode = base_capex
    capex_high = base_capex * 1.75
    capex_samples = rng.triangular(capex_low, capex_mode, capex_high, n_simulations)

    rev_std = annual_revenues_base * 0.25
    rev_samples = rng.normal(annual_revenues_base, rev_std, (n_simulations, duration_years))
    rev_samples = np.maximum(rev_samples, 0)

    opex_std = annual_opex_base * 0.15
    opex_samples = rng.normal(annual_opex_base, opex_std, (n_simulations, duration_years))
    opex_samples = np.maximum(opex_samples, 0)

    inf_low = max(inflation_rate * 0.5, 0.05)
    inf_high = inflation_rate * 2.0
    inf_samples = rng.triangular(inf_low, inflation_rate, inf_high, n_simulations)

    dr_std = 0.03
    dr_samples = rng.normal(discount_rate, dr_std, n_simulations)
    dr_samples = np.maximum(dr_samples, 0.05)

    local_share = 0.60

    npv_results = np.zeros(n_simulations)
    payback_results = np.full(n_simulations, np.nan)

    for i in range(n_simulations):
        inf = inf_samples[i]
        cfs = [-capex_samples[i]]
        for t in range(duration_years):
            esc = (1 + inf) ** t
            rev = rev_samples[i, t]
            cost = opex_samples[i, t] * (local_share * esc + (1 - local_share))
            cfs.append(rev - cost)

        npv_val = sum(cf / (1 + dr_samples[i]) ** t for t, cf in enumerate(cfs))
        npv_results[i] = npv_val

        cum = 0
        for t, cf in enumerate(cfs):
            cum += cf
            if cum >= 0 and t > 0:
                payback_results[i] = t
                break

    prob_positive = float((npv_results > 0).mean())
    var_95 = float(np.percentile(npv_results, 5))
    cvar_95 = float(np.mean(npv_results[npv_results <= var_95]))
    expected_npv = float(np.mean(npv_results))
    median_npv = float(np.median(npv_results))

    hist_counts, hist_edges = np.histogram(npv_results, bins=50)

    return {
        "n_simulations": n_simulations,
        "expected_npv": round(expected_npv, 2),
        "median_npv": round(median_npv, 2),
        "prob_npv_positive": round(prob_positive * 100, 2),
        "var_95": round(var_95, 2),
        "cvar_95": round(cvar_95, 2),
        "npv_std": round(float(np.std(npv_results)), 2),
        "npv_min": round(float(np.min(npv_results)), 2),
        "npv_max": round(float(np.max(npv_results)), 2),
        "avg_payback_years": round(float(np.nanmean(payback_results)), 1),
        "histogram": {
            "counts": hist_counts.tolist(),
            "edges": hist_edges.tolist(),
        },
        "npv_samples": npv_results.tolist(),
        "percentiles": {
            "p5": round(float(np.percentile(npv_results, 5)), 2),
            "p25": round(float(np.percentile(npv_results, 25)), 2),
            "p50": round(float(np.percentile(npv_results, 50)), 2),
            "p75": round(float(np.percentile(npv_results, 75)), 2),
            "p95": round(float(np.percentile(npv_results, 95)), 2),
        },
    }


def sensitivity_tornado(
    base_capex: float,
    duration_years: int,
    annual_revenues: float,
    annual_opex: float,
    discount_rate: float,
    inflation_rate: float,
    base_exchange_rate: float,
    base_npv: float = None,
) -> pd.DataFrame:
    """Tornado chart: vary each variable ±20% and measure NPV impact."""
    params = dict(
        base_capex=base_capex, duration_years=duration_years,
        annual_revenues_base=annual_revenues, annual_opex_base=annual_opex,
        discount_rate=discount_rate, inflation_rate=inflation_rate,
        base_exchange_rate=base_exchange_rate, n_simulations=800, seed=42,
    )

    if base_npv is None:
        base_result = monte_carlo_npv(**params)
        base_npv = base_result["expected_npv"]

    variations = [
        ("Capex", "base_capex", 0.20),
        ("Revenue", "annual_revenues_base", 0.20),
        ("Opex", "annual_opex_base", 0.20),
        ("Discount Rate", "discount_rate", 0.20),
        ("Inflation", "inflation_rate", 0.20),
    ]

    rows = []
    for name, param, pct in variations:
        low_params = params.copy()
        high_params = params.copy()
        low_params[param] = params[param] * (1 - pct)
        high_params[param] = params[param] * (1 + pct)

        low_npv = monte_carlo_npv(**low_params)["expected_npv"]
        high_npv = monte_carlo_npv(**high_params)["expected_npv"]

        impact = abs(high_npv - low_npv)
        rows.append({
            "variable": name,
            "low_value": round(low_params[param], 2),
            "high_value": round(high_params[param], 2),
            "npv_at_low": round(low_npv, 2),
            "npv_at_high": round(high_npv, 2),
            "impact_range": round(impact, 2),
        })

    return pd.DataFrame(rows).sort_values("impact_range", ascending=False)


def portfolio_monte_carlo(
    projects: List[Dict],
    n_simulations: int = 5000,
    seed: int = 42,
) -> Dict:
    """Monte Carlo for a portfolio of projects."""
    rng = np.random.RandomState(seed)
    n_projects = len(projects)
    project_npvs = np.zeros((n_simulations, n_projects))

    for j, proj in enumerate(projects):
        capex = proj.get("baseline_capex", 10e6)
        revenue = proj.get("annual_revenue", capex * 0.3)
        opex = proj.get("annual_opex", capex * 0.1)
        years = int(proj.get("duration_years", 3))
        overrun = proj.get("overrun_probability", 0.4)

        for i in range(n_simulations):
            overrun_mult = 1 + rng.triangular(0, overrun * 0.30, 0.60)
            adj_capex = capex * overrun_mult
            inflation = rng.triangular(0.20, 0.55, 1.20)

            cfs = [-adj_capex]
            for t in range(1, years + 1):
                esc = (1 + inflation) ** t
                rev = revenue * rng.normal(1.0, 0.25)
                cost = opex * (0.60 * esc + 0.40)
                cfs.append(rev - cost)

            dr = rng.normal(0.20, 0.03)
            dr = max(dr, 0.05)
            project_npvs[i, j] = sum(cf / (1 + dr) ** t for t, cf in enumerate(cfs))

    portfolio_npv = project_npvs.sum(axis=1)

    return {
        "n_simulations": n_simulations,
        "portfolio_expected_npv": round(float(np.mean(portfolio_npv)), 2),
        "portfolio_std": round(float(np.std(portfolio_npv)), 2),
        "portfolio_var_95": round(float(np.percentile(portfolio_npv, 5)), 2),
        "prob_portfolio_positive": round(float((portfolio_npv > 0).mean()) * 100, 2),
        "project_expected_npvs": {
            str(projects[j].get("project_id", f"P{j}")): round(float(project_npvs[:, j].mean()), 2)
            for j in range(n_projects)
        },
    }
"""
Portfolio optimization using Mixed-Integer Linear Programming (MILP).
Selects optimal project portfolio under budget, risk, and sector constraints.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    from pulp import LpMaximize, LpProblem, LpVariable, LpBinary, lpSum, LpStatus, value
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def optimize_portfolio(
    projects: pd.DataFrame,
    budget: float,
    max_risk_exposure: float = None,
    min_strategic_score: float = None,
    sector_quotas: Dict[str, float] = None,
    strategic_weights: Dict[str, float] = None,
) -> Dict:
    """
    Solve portfolio optimization using MILP.

    Parameters
    ----------
    projects : DataFrame with columns:
        - project_id
        - baseline_capex
        - expected_npv (or npv_usd)
        - risk_score (overrun_probability)
        - sector
        - strategic_score (optional)
    budget : Total available capital
    max_risk_exposure : Maximum allowable weighted risk (sum of risk * capex)
    min_strategic_score : Minimum total strategic score
    sector_quotas : Min allocation per sector as fraction of budget

    Returns
    -------
    dict with selected projects, objective value, shadow prices
    """
    if not HAS_PULP:
        return _greedy_optimize(projects, budget)

    n = len(projects)
    prob = LpProblem("PortfolioOptimization", LpMaximize)

    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]

    capex = projects["baseline_capex"].values
    npvs = projects.get("expected_npv", projects.get("npv_usd", pd.Series([0] * n))).values
    risks = projects.get("risk_score", projects.get("overrun_probability", pd.Series([0.5] * n))).values
    sectors = projects.get("sector", pd.Series(["other"] * n)).values

    if strategic_weights is None:
        strategic_weights = {}
    strategic_scores = projects.get("strategic_score", pd.Series([1.0] * n)).values
    weights = np.array([strategic_weights.get(s, 1.0) for s in sectors])

    weighted_npvs = weights * npvs

    prob += lpSum(weighted_npvs[i] * x[i] for i in range(n))

    prob += lpSum(capex[i] * x[i] for i in range(n)) <= budget

    if max_risk_exposure is not None:
        prob += lpSum(risks[i] * capex[i] * x[i] for i in range(n)) <= max_risk_exposure

    if min_strategic_score is not None:
        prob += lpSum(strategic_scores[i] * x[i] for i in range(n)) >= min_strategic_score

    if sector_quotas:
        unique_sectors = projects["sector"].unique()
        for sector, min_alloc in sector_quotas.items():
            if sector in unique_sectors:
                indices = [i for i in range(n) if sectors[i] == sector]
                if indices:
                    prob += lpSum(capex[i] * x[i] for i in indices) >= min_alloc * budget

    try:
        from pulp import PULP_CBC_CMD
        prob.solve(PULP_CBC_CMD(msg=False))
    except Exception:
        prob.solve()

    if LpStatus[prob.status] != "Optimal":
        return _greedy_optimize(projects, budget)

    selected = []
    for i in range(n):
        if value(x[i]) and value(x[i]) > 0.5:
            selected.append({
                "project_id": projects.iloc[i].get("project_id", f"P{i}"),
                "sector": sectors[i],
                "capex": float(capex[i]),
                "expected_npv": float(npvs[i]),
                "risk_score": float(risks[i]),
                "strategic_score": float(strategic_scores[i]),
                "decision": "FUND",
            })

    total_capex = sum(p["capex"] for p in selected)
    total_npv = sum(p["expected_npv"] for p in selected)
    avg_risk = np.mean([p["risk_score"] for p in selected]) if selected else 0

    deferred = []
    for i in range(n):
        if not (value(x[i]) and value(x[i]) > 0.5):
            deferred.append({
                "project_id": projects.iloc[i].get("project_id", f"P{i}"),
                "sector": sectors[i],
                "capex": float(capex[i]),
                "expected_npv": float(npvs[i]),
                "risk_score": float(risks[i]),
                "decision": "DEFER",
            })

    return {
        "status": "optimal",
        "objective_value": round(float(value(prob.objective)), 2),
        "selected_projects": selected,
        "deferred_projects": deferred,
        "total_capex_funded": round(total_capex, 2),
        "total_capex_requested": round(float(capex.sum()), 2),
        "budget_utilization": round(total_capex / budget * 100, 1) if budget > 0 else 0,
        "total_expected_npv": round(total_npv, 2),
        "average_risk": round(float(avg_risk), 3),
        "n_funded": len(selected),
        "n_deferred": len(deferred),
    }


def _greedy_optimize(projects: pd.DataFrame, budget: float) -> Dict:
    """Greedy fallback when PuLP is not available."""
    df = projects.copy()
    capex = df["baseline_capex"].values
    npvs = df.get("expected_npv", df.get("npv_usd", pd.Series([0] * len(df)))).values
    ratios = npvs / np.maximum(capex, 1)
    df["pi_ratio"] = ratios
    df_sorted = df.sort_values("pi_ratio", ascending=False)

    selected = []
    deferred = []
    remaining = budget

    for _, row in df_sorted.iterrows():
        c = row["baseline_capex"]
        if c <= remaining:
            selected.append({
                "project_id": row.get("project_id", ""),
                "sector": row.get("sector", ""),
                "capex": float(c),
                "expected_npv": float(row.get("expected_npv", 0)),
                "risk_score": float(row.get("risk_score", row.get("overrun_probability", 0.5))),
                "strategic_score": float(row.get("strategic_score", 1.0)),
                "decision": "FUND",
            })
            remaining -= c
        else:
            deferred.append({
                "project_id": row.get("project_id", ""),
                "sector": row.get("sector", ""),
                "capex": float(c),
                "expected_npv": float(row.get("expected_npv", 0)),
                "risk_score": float(row.get("risk_score", row.get("overrun_probability", 0.5))),
                "decision": "DEFER",
            })

    total_capex = sum(p["capex"] for p in selected)
    total_npv = sum(p["expected_npv"] for p in selected)

    return {
        "status": "greedy",
        "objective_value": round(total_npv, 2),
        "selected_projects": selected,
        "deferred_projects": deferred,
        "total_capex_funded": round(total_capex, 2),
        "total_capex_requested": round(float(capex.sum()), 2),
        "budget_utilization": round(total_capex / budget * 100, 1) if budget > 0 else 0,
        "total_expected_npv": round(total_npv, 2),
        "average_risk": round(float(np.mean([p["risk_score"] for p in selected])) if selected else 0, 3),
        "n_funded": len(selected),
        "n_deferred": len(deferred),
    }


def sensitivity_budget(
    projects: pd.DataFrame,
    budget_range: List[float],
    max_risk_exposure: float = None,
) -> pd.DataFrame:
    """Run optimization across a range of budgets."""
    results = []
    for budget in budget_range:
        res = optimize_portfolio(projects, budget, max_risk_exposure)
        results.append({
            "budget": budget,
            "n_funded": res["n_funded"],
            "total_npv": res["total_expected_npv"],
            "avg_risk": res["average_risk"],
            "budget_utilization": res["budget_utilization"],
        })
    return pd.DataFrame(results)


def efficient_frontier(
    projects: pd.DataFrame,
    n_points: int = 10,
    max_risk_range: List[float] = None,
) -> pd.DataFrame:
    """Generate efficient frontier by varying risk tolerance."""
    capex = projects["baseline_capex"].values
    total_capex = capex.sum()
    budget = total_capex

    if max_risk_range is None:
        max_risk_range = np.linspace(total_capex * 0.2, total_capex * 0.8, n_points).tolist()

    rows = []
    for max_risk in max_risk_range:
        res = optimize_portfolio(projects, budget, max_risk_exposure=max_risk)
        rows.append({
            "max_risk_exposure": max_risk,
            "total_npv": res["total_expected_npv"],
            "avg_risk": res["average_risk"],
            "n_funded": res["n_funded"],
        })
    return pd.DataFrame(rows)

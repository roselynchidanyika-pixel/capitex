"""
Capital budgeting engine: NPV, IRR, MIRR, DSCR, payback, risk-adjusted metrics.
Designed for Zimbabwe's high-inflation, volatile-currency environment.
All cash flows are modeled in constant USD terms. Inflation/FX risk is priced
into both the discount rate and a local-cost escalation overlay.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Core financial functions
# ---------------------------------------------------------------------------

def npv(discount_rate: float, cashflows: List[float], time_periods: List[float] = None) -> float:
    """Net Present Value with optional non-integer time periods."""
    if time_periods is None:
        time_periods = list(range(len(cashflows)))
    return sum(cf / (1 + discount_rate) ** t for cf, t in zip(cashflows, time_periods))


def irr(cashflows: List[float], guess: float = 0.1, tol: float = 1e-8) -> Optional[float]:
    """Internal Rate of Return using Brent's method."""
    if len(cashflows) < 2:
        return None

    def npv_func(r):
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))

    try:
        rate = brentq(npv_func, -0.5, 5.0, xtol=tol)
        return rate
    except (ValueError, RuntimeError):
        return None


def mirr(cashflows: List[float], finance_rate: float, reinvest_rate: float) -> Optional[float]:
    """Modified Internal Rate of Return."""
    n = len(cashflows) - 1
    if n <= 0:
        return None

    negative_cfs = [min(0, cf) for cf in cashflows]
    positive_cfs = [max(0, cf) for cf in cashflows]

    pv_neg = sum(cf / (1 + finance_rate) ** t for t, cf in enumerate(negative_cfs))
    fv_pos = sum(cf * (1 + reinvest_rate) ** (n - t) for t, cf in enumerate(positive_cfs))

    if pv_neg == 0 or fv_pos <= 0:
        return None

    return (fv_pos / (-pv_neg)) ** (1 / n) - 1


def payback_period(cashflows: List[float]) -> Optional[float]:
    """Simple payback period in years."""
    cumulative = 0
    for t, cf in enumerate(cashflows):
        cumulative += cf
        if cumulative >= 0 and t > 0:
            prev = cumulative - cf
            if prev < 0:
                return t - 1 + (-prev / cf) if cf != 0 else t
            return float(t)
    return None


def discounted_payback(cashflows: List[float], discount_rate: float) -> Optional[float]:
    """Discounted payback period."""
    cumulative = 0
    for t, cf in enumerate(cashflows):
        dcf = cf / (1 + discount_rate) ** t
        cumulative += dcf
        if cumulative >= 0 and t > 0:
            prev = cumulative - dcf
            if prev < 0:
                return t - 1 + (-prev / dcf) if dcf != 0 else t
            return float(t)
    return None


def profitability_index(discount_rate: float, cashflows: List[float]) -> float:
    """Profitability Index = PV of future CFs / initial investment."""
    if len(cashflows) < 2:
        return 0.0
    initial = abs(cashflows[0])
    if initial == 0:
        return 0.0
    pv_future = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cashflows[1:], 1))
    return pv_future / initial


def dscr(cashflows: List[float], debt_service: List[float]) -> List[float]:
    """Debt Service Coverage Ratio per period."""
    return [cf / ds if ds > 0 else float("inf") for cf, ds in zip(cashflows, debt_service)]


# ---------------------------------------------------------------------------
# Zimbabwe-adjusted functions
# ---------------------------------------------------------------------------

def risk_adjusted_discount_rate(
    base_rate: float = 0.12,
    overrun_prob: float = 0.0,
    macro_volatility: float = 0.0,
    sector_risk: float = 0.0,
    country_risk: float = 0.0,
) -> float:
    """Compute risk-adjusted discount rate for Zimbabwe projects."""
    overrun_premium = overrun_prob * 0.10
    macro_premium = min(macro_volatility * 0.05, 0.05)
    sector_premium = sector_risk * 0.03
    country_premium = country_risk * 0.02
    return base_rate + overrun_premium + macro_premium + sector_premium + country_premium


def adjusted_capex(
    base_capex: float,
    overrun_prob: float = 0.0,
    expected_overrun_pct: float = 0.30,
    local_cost_share: float = 0.60,
    inflation_rate: float = 0.0,
    duration_years: float = 3.0,
    spend_profile: List[float] = None,
) -> float:
    """
    Adjust capex for expected overruns and local-cost inflation escalation.

    Construction spend is spread over `duration_years` according to
    `spend_profile` (defaults to a front-loaded 40/40/20 pattern). Each
    tranche's local-cost share escalates at the Zim CPI over its own timing,
    giving a realistic total in constant year-0 USD (avoids over-escalating
    the whole capex to the final year).
    """
    if spend_profile is None:
        spend_profile = [0.4, 0.4, 0.2]
    profile = (spend_profile[: duration_years] if len(spend_profile) >= duration_years
               else spend_profile + [0.0] * (duration_years - len(spend_profile)))
    total_share = sum(profile)
    if total_share <= 0:
        profile = [1.0 / duration_years] * duration_years
        total_share = 1.0
    profile = [s / total_share for s in profile]

    risk_adjustment = 1 + overrun_prob * expected_overrun_pct
    escalated = 0.0
    for t, share in enumerate(profile):
        local_escalation = (1 + inflation_rate) ** t
        cost_escalation = local_cost_share * local_escalation + (1 - local_cost_share)
        escalated += share * cost_escalation
    return base_capex * risk_adjustment * escalated


def inflation_adjusted_cashflows(
    nominal_cashflows: List[float],
    inflation_rate: float,
    real: bool = True,
) -> List[float]:
    """Convert nominal cashflows to real or vice versa."""
    if real:
        return [cf / (1 + inflation_rate) ** t for t, cf in enumerate(nominal_cashflows)]
    return [cf * (1 + inflation_rate) ** t for t, cf in enumerate(nominal_cashflows)]


def exchange_rate_path(base_rate: float, annual_depreciation: float, years: int) -> List[float]:
    """Project exchange rate path (local per USD)."""
    return [base_rate * (1 + annual_depreciation) ** t for t in range(years + 1)]


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------

def scenario_analysis(
    base_capex: float,
    annual_revenues: List[float],
    annual_opex: List[float],
    discount_rate: float,
    inflation_rate: float,
    overrun_prob: float = 0.0,
    scenarios: Dict[str, dict] = None,
) -> pd.DataFrame:
    """
    Run NPV analysis under base / downside / upside macro scenarios.
    All flows in constant USD. Scenario adjustments:
      - downside: revenues down, local cost escalation up
      - upside:   revenues up, local cost escalation down
    """
    if scenarios is None:
        scenarios = {
            "base": {"revenue_adj": 1.0, "inf_mult": 1.0},
            "downside": {"revenue_adj": 0.75, "inf_mult": 1.6},
            "upside": {"revenue_adj": 1.25, "inf_mult": 0.7},
        }

    duration = len(annual_revenues)
    rows = []

    for name, params in scenarios.items():
        eff_inf = inflation_rate * params["inf_mult"]
        adj_capex = adjusted_capex(
            base_capex, overrun_prob, 0.30, 0.60, eff_inf, duration
        )

        cfs = [-adj_capex]
        for t in range(duration):
            esc = (1 + eff_inf) ** t  # local cost escalation factor
            rev = annual_revenues[t] * params["revenue_adj"]
            cost = annual_opex[t] * (0.60 * esc + 0.40)
            cfs.append(rev - cost)

        row_npv = npv(discount_rate, cfs)
        row_irr = irr(cfs)
        row_pi = profitability_index(discount_rate, cfs)
        row_payback = payback_period(cfs)

        rows.append({
            "scenario": name,
            "adj_capex_usd": round(adj_capex, 2),
            "npv_usd": round(row_npv, 2),
            "irr_pct": round(row_irr * 100, 2) if row_irr is not None else None,
            "pi": round(row_pi, 3),
            "payback_years": round(row_payback, 1) if row_payback is not None else None,
        })

    return pd.DataFrame(rows)


def expected_npv(scenario_df: pd.DataFrame, probabilities: Dict[str, float] = None) -> float:
    """Compute expected NPV across scenarios."""
    if probabilities is None:
        probabilities = {"base": 0.50, "downside": 0.30, "upside": 0.20}

    enpv = 0.0
    total_prob = 0.0
    for _, row in scenario_df.iterrows():
        prob = probabilities.get(row["scenario"], 0)
        enpv += prob * row["npv_usd"]
        total_prob += prob
    return enpv / total_prob if total_prob > 0 else 0


# ---------------------------------------------------------------------------
# DSCR analysis
# ---------------------------------------------------------------------------

def compute_dscr_profile(
    operating_cfs: List[float],
    debt_service: List[float],
) -> pd.DataFrame:
    """Compute DSCR profile over project life."""
    ratios = dscr(operating_cfs, debt_service)
    return pd.DataFrame({
        "period": list(range(1, len(ratios) + 1)),
        "operating_cf_usd": [round(c, 2) for c in operating_cfs],
        "debt_service_usd": [round(d, 2) for d in debt_service],
        "dscr": [round(r, 3) if r != float("inf") else None for r in ratios],
        "dscr_adequate": [r >= 1.2 if r != float("inf") else True for r in ratios],
    })


# ---------------------------------------------------------------------------
# Full project valuation
# ---------------------------------------------------------------------------

def full_project_valuation(
    project: dict,
    macro_data: dict = None,
) -> dict:
    """
    Complete capital budgeting for a single Zimbabwean project.

    Assumptions:
      - capex, revenues, opex inputs are in constant year-0 USD
      - 60% of costs are local-currency and escalate at Zim CPI
      - 40% of costs are imported (USD) and do not escalate
      - Discount rate is risk-adjusted (base + overrun + macro + sector premia)
      - Exchange rate path is reported for reference (ZiG per USD, depreciating)
    """
    base_capex = project.get("baseline_capex", 10_000_000)
    duration_years = max(int(round(project.get("duration_years", 3))), 1)
    if isinstance(project.get("annual_revenues"), list) and len(project["annual_revenues"]) >= duration_years:
        annual_revenues = project["annual_revenues"][:duration_years]
    else:
        annual_revenues = [base_capex * 0.3] * duration_years
    if isinstance(project.get("annual_opex"), list) and len(project["annual_opex"]) >= duration_years:
        annual_opex = project["annual_opex"][:duration_years]
    else:
        annual_opex = [base_capex * 0.1] * duration_years

    debt_ratio = project.get("debt_ratio", 0.6)
    loan_tenor = min(int(project.get("loan_tenor_years", duration_years)), 30)
    interest_rate = project.get("interest_rate", 0.18)
    overrun_prob = project.get("overrun_probability", 0.4)

    # Macro adjustments
    if macro_data:
        inflation = macro_data.get("inflation", {}).get("zimbabwe_pct", 55) / 100
        us_inflation = macro_data.get("inflation", {}).get("us_pct", 3) / 100
        current_rate = macro_data.get("exchange_rate", {}).get("zig_per_usd", 13.5)
        policy_rate = macro_data.get("interest_rates", {}).get("zim_policy_pct", 35) / 100
    else:
        inflation = 0.55
        us_inflation = 0.03
        current_rate = 13.5
        policy_rate = 0.35

    # Expected depreciation: dampened reflection of the inflation differential.
    # Base case anchored near ~20%/yr (spec: stable ZiG 10-15% / huge downside 50%).
    inflation_diff = max((1 + inflation) / (1 + us_inflation) - 1, 0.0)
    annual_depreciation = min(max(inflation_diff * 0.40, 0.15), 0.50)
    ex_rates = exchange_rate_path(current_rate, annual_depreciation, duration_years)

    # Risk-adjusted discount rate
    macro_vol = inflation
    sector_risk = 0.5
    if project.get("sector") in ["roads", "water"]:
        sector_risk = 0.7
    elif project.get("sector") in ["energy", "telecoms", "mining"]:
        sector_risk = 0.3

    radr = risk_adjusted_discount_rate(
        base_rate=0.12, overrun_prob=overrun_prob,
        macro_volatility=macro_vol, sector_risk=sector_risk,
    )

    # Adjusted capex (USD, constant year-0) with construction spread escalation
    adj_capex = adjusted_capex(
        base_capex, overrun_prob, 0.30, 0.60,
        inflation, min(duration_years, 5),
    )

    # USD constant cash flows
    cfs = [-adj_capex]
    for t in range(duration_years):
        esc = (1 + inflation) ** t
        rev = annual_revenues[t]
        cost = annual_opex[t] * (0.60 * esc + 0.40)
        cfs.append(rev - cost)

    # Financial metrics
    npv_val = npv(radr, cfs)
    irr_val = irr(cfs)
    mirr_val = mirr(cfs, finance_rate=interest_rate, reinvest_rate=0.10)
    pb = payback_period(cfs)
    dpb = discounted_payback(cfs, radr)
    pi_val = profitability_index(radr, cfs)

    # Debt service (USD, constant terms, level amortizing)
    debt_amount = adj_capex * debt_ratio
    if loan_tenor > 0 and interest_rate > 0:
        annual_payment = debt_amount * (interest_rate * (1 + interest_rate) ** loan_tenor) / (
            (1 + interest_rate) ** loan_tenor - 1
        )
    else:
        annual_payment = debt_amount / loan_tenor if loan_tenor > 0 else 0

    ds_schedule = [annual_payment] * duration_years
    op_cfs_usd = [cfs[t + 1] for t in range(duration_years)]
    dscr_vals = compute_dscr_profile(op_cfs_usd, ds_schedule)

    # Scenario analysis
    scenarios_df = scenario_analysis(
        base_capex=base_capex,
        annual_revenues=annual_revenues,
        annual_opex=annual_opex,
        discount_rate=radr,
        inflation_rate=inflation,
        overrun_prob=overrun_prob,
    )
    enpv = expected_npv(scenarios_df, {"base": 0.50, "downside": 0.30, "upside": 0.20})

    # Monte-Carlo-style confidence on NPV (deterministic proxy)
    if npv_val > 0 and pi_val > 1.1:
        recommendation = "FUND"
    elif npv_val > 0 or pi_val > 1.0:
        recommendation = "FUND WITH CONDITIONS"
    else:
        recommendation = "REJECT"

    return {
        "project_id": project.get("project_id", ""),
        "sector": project.get("sector", ""),
        "adjusted_capex_usd": round(adj_capex, 2),
        "capex_overrun_adjustment": round(1 + overrun_prob * 0.30, 3),
        "annual_depreciation_pct": round(annual_depreciation * 100, 2),
        "risk_adjusted_discount_rate": round(radr * 100, 2),
        "npv_usd": round(npv_val, 2),
        "irr_pct": round(irr_val * 100, 2) if irr_val is not None else None,
        "mirr_pct": round(mirr_val * 100, 2) if mirr_val is not None else None,
        "payback_years": round(pb, 1) if pb is not None else None,
        "discounted_payback_years": round(dpb, 1) if dpb is not None else None,
        "profitability_index": round(pi_val, 3),
        "annual_debt_service_usd": round(annual_payment, 2),
        "dscr_profile": dscr_vals,
        "scenario_analysis": scenarios_df,
        "expected_npv_usd": round(enpv, 2),
        "exchange_rate_path": [round(x, 2) for x in ex_rates],
        "cashflows_usd": [round(c, 2) for c in cfs],
        "recommendation": recommendation,
    }
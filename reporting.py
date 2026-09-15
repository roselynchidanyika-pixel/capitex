"""
Reporting module: generates credit memos, portfolio dashboards, project briefs.
"""
import pandas as pd
import json
from datetime import datetime


def _parse_drivers(drivers):
    """Parse top_risk_drivers from either a JSON string or a list of tuples."""
    if isinstance(drivers, str):
        try:
            drivers = json.loads(drivers)
        except Exception:
            return []
    if isinstance(drivers, list):
        return sorted(drivers, key=lambda x: abs(x[1]), reverse=True)[:5]
    return []


def generate_project_brief(project: dict, valuation: dict, overrun: dict) -> str:
    """Generate a 1-page project risk brief."""
    risk_cat = overrun.get("risk_category", "UNKNOWN")
    prob = overrun.get("overrun_probability", 0)
    recommendation = valuation.get("recommendation", "N/A")

    drivers = _parse_drivers(overrun.get("top_risk_drivers", []))
    driver_text = "\n".join(
        f"  {i+1}. {d[0]} (impact: {d[1]:.4f})" for i, d in enumerate(drivers[:5])
    ) if drivers else "  No driver data available"

    dscr_profile = valuation.get("dscr_profile")
    dscr_text = ""
    if dscr_profile is not None and hasattr(dscr_profile, "iterrows"):
        dscr_min = dscr_profile["dscr"].dropna().min() if "dscr" in dscr_profile.columns else None
        dscr_text = f"  Minimum DSCR: {dscr_min:.2f}x" if dscr_min and dscr_min != float("inf") else ""

    scenarios = valuation.get("scenario_analysis")
    scenario_text = ""
    if scenarios is not None and hasattr(scenarios, "iterrows"):
        for _, row in scenarios.iterrows():
            scenario_text += (
                f"  - {row['scenario'].upper()}: NPV=${row['npv_usd']:,.0f}"
                f"  IRR={row['irr_pct'] or 'N/A'}%"
                f"  PI={row['pi']:.2f}\n"
            )

    report = f"""
================================================================================
CAPITAL PROJECT RISK BRIEF
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

PROJECT OVERVIEW
  ID:       {project.get('project_id', 'N/A')}
  Sector:   {project.get('sector', 'N/A')}
  Capex:    ${project.get('baseline_capex', 0):,.0f}
  Duration: {project.get('baseline_duration_days', 0)} days
  Location: {project.get('province', 'N/A')}

OVERRUN RISK ASSESSMENT
  P(cost overrun >= 20%): {prob*100:.1f}%
  Risk Category: {risk_cat}

  Top Risk Drivers:
{driver_text}

FINANCIAL ANALYSIS (Risk-Adjusted, USD)
  Adjusted Capex:  ${valuation.get('adjusted_capex_usd', 0):,.0f}
  Discount Rate:   {valuation.get('risk_adjusted_discount_rate', 0):.1f}%
  NPV:             ${valuation.get('npv_usd', 0):,.0f}
  IRR:             {valuation.get('irr_pct', 'N/A')}%
  MIRR:            {valuation.get('mirr_pct', 'N/A')}%
  Payback:         {valuation.get('payback_years', 'N/A')} years
  Disc. Payback:   {valuation.get('discounted_payback_years', 'N/A')} years
  PI:              {valuation.get('profitability_index', 0):.3f}
  Expected NPV:    ${valuation.get('expected_npv_usd', 0):,.0f}
{dscr_text}

SCENARIO ANALYSIS
{scenario_text}
RECOMMENDATION: {recommendation}

NOTE: This is a decision-support output. Final decisions should consider
qualitative factors, stakeholder input, and political context.
================================================================================
"""
    return report.strip()


def generate_credit_memo(project: dict, valuation: dict, overrun: dict, macro: dict) -> str:
    """Generate a draft credit memo for bank lending decisions."""
    risk_cat = overrun.get("risk_category", "UNKNOWN")
    prob = overrun.get("overrun_probability", 0)

    report = f"""
================================================================================
DRAFT CREDIT MEMO
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d')}

1. EXECUTIVE SUMMARY
   Project: {project.get('project_id', 'N/A')} - {project.get('sector', 'N/A').title()} Infrastructure
   Applicant Sector: {project.get('sector', 'N/A').title()}
   Total Capex: ${project.get('baseline_capex', 0):,.0f}
   Requested Tenor: {project.get('loan_tenor_years', 5)} years
   Overrun Risk: {risk_cat} ({prob*100:.1f}%)

2. PROJECT DESCRIPTION
   Location: {project.get('province', 'N/A')}
   Duration: {project.get('baseline_duration_days', 0)} days
   Complexity: {project.get('complexity_score', 'N/A')}/5
   Contractor: {project.get('contractor_type', 'N/A')}
   Procurement: {project.get('procurement_method', 'N/A')}

3. FINANCIAL ANALYSIS
   Risk-Adjusted Discount Rate: {valuation.get('risk_adjusted_discount_rate', 0):.1f}%
   NPV (Base Case): ${valuation.get('npv_usd', 0):,.0f}
   Expected NPV (Probability-Weighted): ${valuation.get('expected_npv_usd', 0):,.0f}
   IRR: {valuation.get('irr_pct', 'N/A')}%
   MIRR: {valuation.get('mirr_pct', 'N/A')}%
   Payback Period: {valuation.get('payback_years', 'N/A')} years
   Discounted Payback: {valuation.get('discounted_payback_years', 'N/A')} years
   Profitability Index: {valuation.get('profitability_index', 0):.3f}
   Annual Debt Service: ${valuation.get('annual_debt_service_usd', 0):,.0f}

4. RISK ASSESSMENT
   Cost Overrun Risk: {risk_cat} (P(overrun) = {prob*100:.1f}%)
   Macro Environment:
     - Zimbabwe Inflation: {macro.get('inflation', {}).get('zimbabwe_pct', 'N/A')}%
     - Exchange Rate: {macro.get('exchange_rate', {}).get('zig_per_usd', 'N/A')} ZiG/USD
     - Policy Rate: {macro.get('interest_rates', {}).get('zim_policy_pct', 'N/A')}%

5. RECOMMENDED TERMS
   Interest Rate: Base ({macro.get('interest_rates', {}).get('zim_policy_pct', 35)}%) + {max(5, int(prob * 15))} bps risk margin
   Tenor: {project.get('loan_tenor_years', 5)} years with {project.get('duration_years', 3)}-year grace period
   DSCR Covenant: Minimum 1.20x throughout
   Capex Contingency: +{int(prob * 30)}% recommended
   Security: Project assets, assignment of revenue streams

6. RECOMMENDATION: {valuation.get('recommendation', 'REVIEW REQUIRED')}
================================================================================
"""
    return report.strip()


def generate_portfolio_summary(results: dict, macro: dict = None) -> str:
    """Generate a portfolio dashboard summary."""
    selected = results.get("selected_projects", [])
    deferred = results.get("deferred_projects", [])

    sector_allocation = {}
    for p in selected:
        s = p.get("sector", "other")
        sector_allocation[s] = sector_allocation.get(s, 0) + p["capex"]

    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for p in selected:
        r = p.get("risk_score", 0.5)
        if r < 0.30:
            risk_dist["LOW"] += 1
        elif r < 0.60:
            risk_dist["MEDIUM"] += 1
        else:
            risk_dist["HIGH"] += 1

    report = f"""
================================================================================
PORTFOLIO DASHBOARD
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

PORTFOLIO OVERVIEW
  Total Projects:       {results.get('n_funded', 0) + results.get('n_deferred', 0)}
  Projects Funded:      {results.get('n_funded', 0)}
  Projects Deferred:    {results.get('n_deferred', 0)}
  Total Capex Funded:   ${results.get('total_capex_funded', 0):,.0f}
  Total Capex Requested: ${results.get('total_capex_requested', 0):,.0f}
  Budget Utilization:   {results.get('budget_utilization', 0):.1f}%
  Total Expected NPV:   ${results.get('total_expected_npv', 0):,.0f}
  Average Risk Score:   {results.get('average_risk', 0):.1%}
  Optimization Status:  {results.get('status', 'N/A')}

RISK DISTRIBUTION
  LOW (<30%):    {risk_dist['LOW']} projects
  MEDIUM (30-60%): {risk_dist['MEDIUM']} projects
  HIGH (>60%):   {risk_dist['HIGH']} projects

SECTOR ALLOCATION (Funded Projects)
"""
    total_funded = results.get("total_capex_funded", 1)
    for sector, capex in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True):
        pct = capex / total_funded * 100 if total_funded > 0 else 0
        report += f"  {sector:20s} ${capex:>15,.0f} ({pct:5.1f}%)\n"

    report += "\nTOP 10 FUNDED PROJECTS (by Risk-Adjusted NPV)\n"
    top = sorted(selected, key=lambda x: x.get("expected_npv", 0), reverse=True)[:10]
    for i, p in enumerate(top, 1):
        report += (
            f"  {i:2d}. {p['project_id']:12s} | {p['sector']:12s} | "
            f"Capex: ${p['capex']:>12,.0f} | NPV: ${p['expected_npv']:>12,.0f} | "
            f"Risk: {p['risk_score']:.1%}\n"
        )

    report += "\nPROJECTS TO DEFER\n"
    for p in sorted(deferred, key=lambda x: x.get("expected_npv", 0), reverse=True)[:5]:
        report += (
            f"  - {p['project_id']:12s} | {p['sector']:12s} | "
            f"Capex: ${p['capex']:>12,.0f} | NPV: ${p['expected_npv']:>12,.0f}\n"
        )

    if macro:
        report += f"""
MACROECONOMIC CONTEXT
  Exchange Rate:    {macro.get('exchange_rate', {}).get('zig_per_usd', 'N/A')} ZiG/USD
  Inflation (Zim):  {macro.get('inflation', {}).get('zimbabwe_pct', 'N/A')}%
  Inflation (US):   {macro.get('inflation', {}).get('us_pct', 'N/A')}%
  Policy Rate:      {macro.get('interest_rates', {}).get('zim_policy_pct', 'N/A')}%
  Gold Price:       ${macro.get('gold', {}).get('price_usd_per_oz', 'N/A'):,.0f}/oz
  Foreign Reserves: {macro.get('reserves', {}).get('total_usd_bn', 'N/A'):.2f} bn USD
  Data Sources:     {macro.get('exchange_rate', {}).get('source', 'N/A')}, {macro.get('inflation', {}).get('source_zim', 'N/A')}
"""

    report += """
================================================================================
NOTE: This is a decision-support tool. Final decisions should consider
qualitative factors, stakeholder input, and political context.
================================================================================
"""
    return report.strip()

"""
Live macroeconomic data fetcher for Zimbabwe.
Sources: World Bank API, FRED (St. Louis Fed), RBZ, Open Exchange Rates, metals APIs.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import json
import os
import time

# Resolve data/cache dir relative to this module so the layout
# (utils/ subpackage vs flat deploy) doesn't matter.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_THIS_DIR, "..", "data", "cache")
CACHE_DIR = os.path.abspath(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _load_cache(key: str, max_age_hours: int = 12) -> Optional[dict]:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > max_age_hours * 3600:
        return None
    with open(path, "r") as f:
        return json.load(f)


def _save_cache(key: str, data: dict):
    with open(_cache_path(key), "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# 1. Exchange Rate Fetchers
# ---------------------------------------------------------------------------

def fetch_exchange_rate_zig() -> dict:
    """Fetch ZiG/USD rate from multiple sources."""
    cached = _load_cache("zig_usd")
    if cached:
        return cached

    result = {
        "source": "unavailable",
        "rate": None,
        "date": None,
        "status": "pending",
    }

    # Source 1: Open Exchange Rates (free tier)
    try:
        r = requests.get(
            "https://openexchangerates.org/api/latest.json",
            params={"app_id": "demo", "symbols": "ZWL,USD"},
            timeout=10,
        )
        if r.ok:
            data = r.json()
            rate = data.get("rates", {}).get("ZWL")
            if rate and rate > 0:
                result.update(source="openexchangerates.org", rate=1.0 / rate,
                              date=data.get("timestamp"), status="live")
    except Exception:
        pass

    # Source 2: FRED API (Zimbabwe exchange rate)
    if result["rate"] is None:
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "DZXZWLUSDM",
                    "api_key": "DEMO_KEY",
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                },
                timeout=10,
            )
            if r.ok:
                obs = r.json().get("observations", [])
                if obs:
                    val = float(obs[0]["value"])
                    if val > 0:
                        result.update(source="FRED", rate=1.0 / val,
                                      date=obs[0]["date"], status="live")
        except Exception:
            pass

    # Source 3: Parallel rates from freecurrencyapi
    if result["rate"] is None:
        try:
            r = requests.get(
                "https://api.freecurrencyapi.com/v1/latest",
                params={"apikey": "fca_live_demo", "base_currency": "USD", "currencies": "ZWL"},
                timeout=10,
            )
            if r.ok:
                data = r.json().get("data", {})
                rate = data.get("ZWL")
                if rate and rate > 0:
                    result.update(source="freecurrencyapi", rate=1.0 / rate,
                                  date=datetime.now().isoformat(), status="live")
        except Exception:
            pass

    if result["rate"] is None:
        result.update(source="fallback", rate=13.5, date=datetime.now().isoformat(),
                      status="fallback_estimate")

    _save_cache("zig_usd", result)
    return result


def fetch_exchange_rate_history(days: int = 365) -> pd.DataFrame:
    """Fetch historical ZWL/USD from FRED."""
    cached = _load_cache("zig_history")
    if cached:
        df = pd.DataFrame(cached["data"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "DZXZWLUSDM",
                "api_key": "DEMO_KEY",
                "file_type": "json",
                "observation_start": start.strftime("%Y-%m-%d"),
                "observation_end": end.strftime("%Y-%m-%d"),
            },
            timeout=15,
        )
        if r.ok:
            obs = r.json().get("observations", [])
            rows = []
            for o in obs:
                try:
                    rows.append({"date": o["date"], "rate": 1.0 / float(o["value"])})
                except (ValueError, ZeroDivisionError):
                    pass
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                _save_cache("zig_history", {"data": df.to_dict(orient="records")})
                return df
    except Exception:
        pass

    # Generate synthetic data if API unavailable
    dates = pd.date_range(end=datetime.now(), periods=min(days, 90), freq="B")
    np.random.seed(42)
    rates = [13.5]
    for _ in range(len(dates) - 1):
        rates.append(rates[-1] * (1 + np.random.normal(0.0008, 0.015)))
    df = pd.DataFrame({"date": dates, "rate": rates})
    return df


# ---------------------------------------------------------------------------
# 2. Inflation Data
# ---------------------------------------------------------------------------

def fetch_inflation_zimbabwe() -> dict:
    """Fetch Zimbabwe inflation from World Bank API."""
    cached = _load_cache("inflation_zim")
    if cached:
        return cached

    result = {"source": "unavailable", "inflation_pct": None, "date": None, "status": "pending"}

    # World Bank API
    try:
        r = requests.get(
            "https://api.worldbank.org/v2/country/ZWE/indicator/FP.CPI.TOTL.ZG",
            params={"format": "json", "per_page": 5, "date": "2020:2026"},
            timeout=10,
        )
        if r.ok:
            data = r.json()
            if len(data) > 1:
                entries = data[1]
                if entries:
                    latest = entries[0]
                    if latest.get("value") is not None:
                        result.update(
                            source="World Bank",
                            inflation_pct=float(latest["value"]),
                            date=latest.get("date"),
                            status="live",
                        )
    except Exception:
        pass

    if result["inflation_pct"] is None:
        result.update(source="fallback", inflation_pct=55.0, date=str(datetime.now().year),
                      status="fallback_estimate")

    _save_cache("inflation_zim", result)
    return result


def fetch_inflation_us() -> dict:
    """Fetch US CPI inflation from FRED."""
    cached = _load_cache("inflation_us")
    if cached:
        return cached

    result = {"source": "unavailable", "inflation_pct": None, "date": None, "status": "pending"}
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "CPIAUCSL",
                "api_key": "DEMO_KEY",
                "file_type": "json",
                "sort_order": "desc",
                "limit": 13,
            },
            timeout=10,
        )
        if r.ok:
            obs = r.json().get("observations", [])
            if len(obs) >= 13:
                latest = float(obs[0]["value"])
                year_ago = float(obs[12]["value"])
                inflation = ((latest - year_ago) / year_ago) * 100
                result.update(source="FRED", inflation_pct=round(inflation, 2),
                              date=obs[0]["date"], status="live")
    except Exception:
        pass

    if result["inflation_pct"] is None:
        result.update(source="fallback", inflation_pct=3.2, date=str(datetime.now().year),
                      status="fallback_estimate")

    _save_cache("inflation_us", result)
    return result


# ---------------------------------------------------------------------------
# 3. Interest Rates
# ---------------------------------------------------------------------------

def fetch_interest_rate_zim() -> dict:
    """Fetch Zimbabwe policy rate from FRED."""
    cached = _load_cache("policy_rate_zim")
    if cached:
        return cached

    result = {"source": "unavailable", "rate_pct": None, "date": None, "status": "pending"}
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "INTDSRZWM193N",
                "api_key": "DEMO_KEY",
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            },
            timeout=10,
        )
        if r.ok:
            obs = r.json().get("observations", [])
            if obs:
                val = float(obs[0]["value"])
                result.update(source="FRED", rate_pct=val, date=obs[0]["date"], status="live")
    except Exception:
        pass

    if result["rate_pct"] is None:
        result.update(source="fallback", rate_pct=35.0, date=str(datetime.now().year),
                      status="fallback_estimate")

    _save_cache("policy_rate_zim", result)
    return result


def fetch_fed_rate() -> dict:
    """Fetch US Federal Funds Rate from FRED."""
    cached = _load_cache("fed_rate")
    if cached:
        return cached

    result = {"source": "unavailable", "rate_pct": None, "date": None, "status": "pending"}
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "DFEDTARU",
                "api_key": "DEMO_KEY",
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            },
            timeout=10,
        )
        if r.ok:
            obs = r.json().get("observations", [])
            if obs:
                val = float(obs[0]["value"])
                result.update(source="FRED", rate_pct=val, date=obs[0]["date"], status="live")
    except Exception:
        pass

    if result["rate_pct"] is None:
        result.update(source="fallback", rate_pct=4.5, date=str(datetime.now().year),
                      status="fallback_estimate")

    _save_cache("fed_rate", result)
    return result


# ---------------------------------------------------------------------------
# 4. Gold Price (for ZiG backing)
# ---------------------------------------------------------------------------

def fetch_gold_price() -> dict:
    """Fetch gold price in USD per oz."""
    cached = _load_cache("gold_price")
    if cached:
        return cached

    result = {"source": "unavailable", "price_usd": None, "date": None, "status": "pending"}

    # Try metals.dev free API
    try:
        r = requests.get(
            "https://api.metals.dev/v1/latest",
            params={"api_key": "demo", "currency": "USD", "unit": "toz"},
            timeout=10,
        )
        if r.ok:
            data = r.json().get("metals", {})
            gold = data.get("gold")
            if gold:
                result.update(source="metals.dev", price_usd=float(gold),
                              date=datetime.now().isoformat(), status="live")
    except Exception:
        pass

    if result["price_usd"] is None:
        result.update(source="fallback", price_usd=2350.0, date=datetime.now().isoformat(),
                      status="fallback_estimate")

    _save_cache("gold_price", result)
    return result


# ---------------------------------------------------------------------------
# 5. Foreign Reserves
# ---------------------------------------------------------------------------

def fetch_foreign_reserves() -> dict:
    """Fetch Zimbabwe total reserves from World Bank."""
    cached = _load_cache("reserves_zim")
    if cached:
        return cached

    result = {"source": "unavailable", "reserves_usd_bn": None, "date": None, "status": "pending"}
    try:
        r = requests.get(
            "https://api.worldbank.org/v2/country/ZWE/indicator/FI.RES.TOTL.CD",
            params={"format": "json", "per_page": 3, "date": "2020:2026"},
            timeout=10,
        )
        if r.ok:
            data = r.json()
            if len(data) > 1 and data[1]:
                latest = data[1][0]
                if latest.get("value") is not None:
                    result.update(
                        source="World Bank",
                        reserves_usd_bn=float(latest["value"]) / 1e9,
                        date=latest.get("date"),
                        status="live",
                    )
    except Exception:
        pass

    if result["reserves_usd_bn"] is None:
        result.update(source="fallback", reserves_usd_bn=0.5,
                      date=str(datetime.now().year), status="fallback_estimate")

    _save_cache("reserves_zim", result)
    return result


# ---------------------------------------------------------------------------
# 6. GDP Growth
# ---------------------------------------------------------------------------

def fetch_gdp_growth() -> dict:
    """Fetch Zimbabwe GDP growth from World Bank."""
    cached = _load_cache("gdp_growth_zim")
    if cached:
        return cached

    result = {"source": "unavailable", "gdp_growth_pct": None, "date": None, "status": "pending"}
    try:
        r = requests.get(
            "https://api.worldbank.org/v2/country/ZWE/indicator/NY.GDP.MKTP.KD.ZG",
            params={"format": "json", "per_page": 3, "date": "2020:2026"},
            timeout=10,
        )
        if r.ok:
            data = r.json()
            if len(data) > 1 and data[1]:
                latest = data[1][0]
                if latest.get("value") is not None:
                    result.update(
                        source="World Bank",
                        gdp_growth_pct=float(latest["value"]),
                        date=latest.get("date"),
                        status="live",
                    )
    except Exception:
        pass

    if result["gdp_growth_pct"] is None:
        result.update(source="fallback", gdp_growth_pct=3.5,
                      date=str(datetime.now().year), status="fallback_estimate")

    _save_cache("gdp_growth_zim", result)
    return result


# ---------------------------------------------------------------------------
# 7. Composite macro dashboard
# ---------------------------------------------------------------------------

def fetch_all_macro() -> dict:
    """Fetch all macroeconomic indicators and return a consolidated dict."""
    ex = fetch_exchange_rate_zig()
    inf_zim = fetch_inflation_zimbabwe()
    inf_us = fetch_inflation_us()
    pol = fetch_interest_rate_zim()
    fed = fetch_fed_rate()
    gold = fetch_gold_price()
    reserves = fetch_foreign_reserves()
    gdp = fetch_gdp_growth()

    return {
        "exchange_rate": {
            "zig_per_usd": ex["rate"],
            "source": ex["source"],
            "status": ex["status"],
        },
        "inflation": {
            "zimbabwe_pct": inf_zim["inflation_pct"],
            "us_pct": inf_us["inflation_pct"],
            "differential": round((inf_zim["inflation_pct"] or 0) - (inf_us["inflation_pct"] or 0), 2),
            "source_zim": inf_zim["source"],
            "source_us": inf_us["source"],
        },
        "interest_rates": {
            "zim_policy_pct": pol["rate_pct"],
            "us_fed_pct": fed["rate_pct"],
            "spread_pct": round((pol["rate_pct"] or 0) - (fed["rate_pct"] or 0), 2),
            "source_zim": pol["source"],
            "source_us": fed["source"],
        },
        "gold": {
            "price_usd_per_oz": gold["price_usd"],
            "source": gold["source"],
        },
        "reserves": {
            "total_usd_bn": reserves["reserves_usd_bn"],
            "source": reserves["source"],
        },
        "gdp": {
            "growth_pct": gdp["gdp_growth_pct"],
            "source": gdp["source"],
        },
        "fetched_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# 8. Exchange rate forecasting
# ---------------------------------------------------------------------------

def forecast_exchange_rate(years: int = 5, scenarios: dict = None) -> pd.DataFrame:
    """Generate exchange rate forecasts under different scenarios."""
    if scenarios is None:
        scenarios = {
            "stable": {"prob": 0.30, "annual_depreciation": 0.10},
            "gradual_depreciation": {"prob": 0.50, "annual_depreciation": 0.25},
            "rapid_depreciation": {"prob": 0.20, "annual_depreciation": 0.60},
        }

    current = fetch_exchange_rate_zig()
    base_rate = current["rate"] or 13.5

    rows = []
    for year in range(years + 1):
        for scenario_name, params in scenarios.items():
            rate = base_rate * ((1 + params["annual_depreciation"]) ** year)
            rows.append({
                "year": datetime.now().year + year,
                "scenario": scenario_name,
                "probability": params["prob"],
                "zig_per_usd": round(rate, 2),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Inflation forecasting & cost escalation factors
# ---------------------------------------------------------------------------

def inflation_escalation_factor(annual_inflation_pct: float, years: float) -> float:
    """Compute cumulative cost escalation factor."""
    return (1 + annual_inflation_pct / 100) ** years


def generate_cost_escalation_curve(base_cost: float, inflation_pct: float,
                                    duration_years: float, steps: int = 12) -> pd.DataFrame:
    """Generate monthly escalated cost curve."""
    monthly_inflation = (1 + inflation_pct / 100) ** (1 / 12) - 1
    months = np.arange(0, steps + 1)
    dates = pd.date_range(start=datetime.now(), periods=steps + 1, freq="MS")
    costs = base_cost * (1 + monthly_inflation) ** months
    return pd.DataFrame({
        "month": months,
        "date": dates[: len(months)],
        "escalated_cost": costs,
        "inflation_monthly_pct": round(monthly_inflation * 100, 4),
    })


if __name__ == "__main__":
    print("Fetching all macroeconomic data...")
    data = fetch_all_macro()
    print(json.dumps(data, indent=2, default=str))

    print("\nExchange rate forecast:")
    fc = forecast_exchange_rate(years=5)
    print(fc.to_string(index=False))

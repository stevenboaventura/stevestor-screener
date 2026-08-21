#!/usr/bin/env python3
"""
SteVestor Screener v2 — Algoritmo robusto de descoberta de tickers

Two-stage pipeline:
  Stage 1 — Quick scan (.info) → pré-filtro + classificação
  Stage 2 — Deep analysis (financial statements) → tendências, SBC, quality

Scoring por percentil dentro de cada fase, com penalizações e bónus.

Uso:
  python screener_v2.py                          # Scan completo US + Europa
  python screener_v2.py --universo us            # Só US
  python screener_v2.py --dry-run                # CSV em vez de Supabase
  python screener_v2.py --tickers UBER,CRM,RKLB  # Testar tickers específicos
"""

import os
import sys
import json
import argparse
import time
import logging
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("screener")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = (
    os.environ.get("SUPABASE_SERVICE_KEY", "")
    or os.environ.get("SUPABASE_KEY", "")
)

# ── Pré-filtros globais ──
MIN_MARKET_CAP = 300_000_000   # $300M
MIN_AVG_VOLUME = 100_000
EXCLUDED_SECTORS = {"Financial Services"}

# ── Thresholds mínimos por fase ──
THRESHOLDS = {
    "pre_lucro": {
        "revenue_growth": 0.15,
        "gross_margin": 0.30,
        "min_score": 45,
        "max_debt_equity": 4.0,
    },
    "crescimento": {
        "revenue_growth": 0.08,
        "gross_margin": 0.25,
        "min_score": 40,
        "max_debt_equity": 2.5,
    },
    "madura": {
        "revenue_growth": 0.02,
        "gross_margin": 0.20,
        "min_score": 38,
        "max_debt_equity": 2.0,
    },
}

# ── Pesos do scoring por fase ──
WEIGHTS = {
    "pre_lucro": {
        "revenue_growth":     0.20,
        "rev_acceleration":   0.10,
        "gross_margin":       0.15,
        "gross_margin_trend": 0.05,
        "op_margin_trend":    0.10,
        "liquidity":          0.10,
        "momentum":           0.10,
        "rule_of_40":         0.05,
        "sbc_penalty":        0.05,
        "earnings_quality":   0.05,
        "rev_cagr_3y":        0.05,
    },
    "crescimento": {
        "revenue_growth":     0.15,
        "rev_acceleration":   0.08,
        "rev_cagr_3y":        0.07,
        "gross_margin":       0.10,
        "gross_margin_trend": 0.05,
        "fcf_margin":         0.10,
        "roe":                0.10,
        "rule_of_40":         0.08,
        "earnings_quality":   0.07,
        "sbc_penalty":        0.05,
        "valuation":          0.08,
        "momentum":           0.07,
    },
    "madura": {
        "roe":                0.15,
        "fcf_margin":         0.15,
        "fcf_yield":          0.10,
        "earnings_quality":   0.10,
        "revenue_growth":     0.08,
        "rev_cagr_3y":        0.05,
        "gross_margin":       0.08,
        "gross_margin_trend": 0.04,
        "valuation":          0.10,
        "sbc_penalty":        0.05,
        "health":             0.05,
        "momentum":           0.05,
    },
}

MAX_WORKERS = 10
REQUEST_DELAY = 0.3


# ═══════════════════════════════════════════════════════════════════════
# 1. UNIVERSO DE TICKERS
# ═══════════════════════════════════════════════════════════════════════

def get_sp500_tickers() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"S&P 500: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.error(f"Erro S&P 500: {e}")
        return []


def get_nasdaq100_tickers() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            if "Ticker" in t.columns:
                tickers = t["Ticker"].str.replace(".", "-", regex=False).tolist()
                log.info(f"NASDAQ 100: {len(tickers)} tickers")
                return tickers
        return []
    except Exception as e:
        log.error(f"Erro NASDAQ 100: {e}")
        return []


def get_russell2000_sample() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Russell_2000_Index")
        for t in tables:
            for col in ("Ticker", "Symbol"):
                if col in t.columns:
                    tickers = t[col].dropna().str.replace(".", "-", regex=False).tolist()
                    log.info(f"Russell 2000 sample: {len(tickers)} tickers")
                    return tickers
        return []
    except Exception:
        log.info("Russell 2000 não disponível via Wikipedia")
        return []


def get_european_tickers() -> list[str]:
    eu = [
        "ASML.AS", "MC.PA", "SAP.DE", "SIE.DE", "OR.PA", "SU.PA", "AIR.PA",
        "SAN.PA", "TTE.PA", "BNP.PA", "DTE.DE", "ALV.DE", "BAS.DE", "MBG.DE",
        "ADS.DE", "MUV2.DE", "IFX.DE", "VOW3.DE", "BMW.DE", "HEN3.DE",
        "NOVN.SW", "ROG.SW", "NESN.SW", "UHR.SW", "SREN.SW", "ABBN.SW",
        "LONN.SW", "GIVN.SW",
        "NOVO-B.CO", "CARL-B.CO", "DSV.CO", "VWS.CO", "MAERSK-B.CO",
        "AZN.L", "SHEL.L", "ULVR.L", "RIO.L", "GSK.L", "BP.L", "HSBA.L",
        "DGE.L", "REL.L", "LSEG.L", "AAL.L", "AHT.L", "CRH.L",
        "INGA.AS", "PHIA.AS", "AD.AS", "WKL.AS", "HEIA.AS", "UNA.AS",
        "RACE.MI", "ENI.MI", "UCG.MI", "ISP.MI", "ENEL.MI", "STLAM.MI",
        "EL.PA", "KER.PA", "RMS.PA", "CDI.PA", "BN.PA", "RI.PA",
        "ABI.BR", "UCB.BR", "SOLB.BR",
        "NOKIA.HE", "NESTE.HE",
        "TEL2-B.ST", "VOLV-B.ST", "ATCO-A.ST", "SAND.ST", "SEB.ST",
        "FPE3.DE", "RHM.DE", "DB1.DE", "MRK.DE", "SHL.DE",
        "NVO", "NVS", "AZN", "SAP", "SHOP", "SE", "MELI", "BABA",
        "SPOT", "UL", "DEO", "BUD", "ASML", "TM", "HMC",
        "SNN", "GSK", "BTI", "RHHBY", "LOGI", "STM",
    ]
    log.info(f"Europa: {len(eu)} tickers")
    return eu


def build_universe(scope: str = "all") -> list[str]:
    tickers = set()
    if scope in ("all", "us"):
        tickers.update(get_sp500_tickers())
        tickers.update(get_nasdaq100_tickers())
        extras = get_russell2000_sample()
        if extras:
            tickers.update(extras)
    if scope in ("all", "europa"):
        tickers.update(get_european_tickers())
    tickers = {t.strip() for t in tickers if t and isinstance(t, str)}
    log.info(f"Universo total: {len(tickers)} tickers únicos")
    return sorted(tickers)


# ═══════════════════════════════════════════════════════════════════════
# 2. STAGE 1 — QUICK SCAN  (.info only)
# ═══════════════════════════════════════════════════════════════════════

def fetch_ticker_info(ticker: str) -> Optional[dict]:
    """Busca dados básicos via yfinance .info (stage 1)."""
    try:
        time.sleep(REQUEST_DELAY)
        t = yf.Ticker(ticker)
        info = t.info

        if not info or info.get("quoteType") in ("ETF", "MUTUALFUND", "INDEX"):
            return None

        market_cap = info.get("marketCap")
        if not market_cap or market_cap < MIN_MARKET_CAP:
            return None

        avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day", 0)
        if avg_volume < MIN_AVG_VOLUME:
            return None

        sector = info.get("sector", "")
        if sector in EXCLUDED_SECTORS:
            return None

        data = {
            "ticker": ticker,
            "nome": info.get("shortName") or info.get("longName", ""),
            "exchange": info.get("exchange", ""),
            "sector": sector,
            "industry": info.get("industry", ""),
            "country": info.get("country", ""),
            "market_cap": market_cap,
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "forward_pe": info.get("forwardPE"),
            "trailing_pe": info.get("trailingPE"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "pb_ratio": info.get("priceToBook"),
            "free_cashflow": info.get("freeCashflow"),
            "revenue": info.get("totalRevenue"),
            "net_income": info.get("netIncomeToCommon"),
            "debt_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "price": info.get("currentPrice") or info.get("previousClose"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "beta": info.get("beta"),
        }

        # Métricas derivadas
        if data["free_cashflow"] and data["revenue"] and data["revenue"] > 0:
            data["fcf_margin"] = data["free_cashflow"] / data["revenue"]
        else:
            data["fcf_margin"] = None

        if data["free_cashflow"] and data["market_cap"] and data["market_cap"] > 0:
            data["fcf_yield"] = data["free_cashflow"] / data["market_cap"]
        else:
            data["fcf_yield"] = None

        return data
    except Exception as e:
        log.debug(f"Erro ao buscar {ticker}: {e}")
        return None


def fetch_all_stage1(tickers: list[str], workers: int = MAX_WORKERS) -> list[dict]:
    """Stage 1: busca .info de todos os tickers em paralelo."""
    results = []
    total = len(tickers)
    completed = 0
    failed = 0

    log.info(f"[STAGE 1] A buscar dados de {total} tickers ({workers} threads)...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_ticker_info, t): t for t in tickers}
        for future in as_completed(futures):
            completed += 1
            try:
                data = future.result()
                if data:
                    results.append(data)
                else:
                    failed += 1
            except Exception:
                failed += 1
            if completed % 50 == 0:
                log.info(f"  Progresso: {completed}/{total} ({len(results)} válidos)")

    log.info(f"[STAGE 1] Completo: {len(results)} válidos de {total}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# 3. STAGE 2 — DEEP ANALYSIS  (financial statements)
# ═══════════════════════════════════════════════════════════════════════

def _safe_val(df, label, col_idx=0):
    """Extrai valor de um DataFrame de financial statement de forma segura."""
    if df is None or df.empty:
        return None
    # Tentar vários nomes possíveis para o mesmo campo
    aliases = {
        "Total Revenue": ["Total Revenue", "TotalRevenue", "Revenue"],
        "Gross Profit": ["Gross Profit", "GrossProfit"],
        "Operating Income": ["Operating Income", "OperatingIncome", "EBIT"],
        "Net Income": ["Net Income", "NetIncome", "Net Income Common Stockholders",
                        "NetIncomeCommonStockholders"],
        "EBITDA": ["EBITDA", "Normalized EBITDA", "NormalizedEBITDA"],
        "Stock Based Compensation": ["Stock Based Compensation",
                                      "StockBasedCompensation",
                                      "Share Based Compensation"],
        "Free Cash Flow": ["Free Cash Flow", "FreeCashFlow"],
        "Operating Cash Flow": ["Operating Cash Flow", "OperatingCashFlow",
                                 "Cash Flow From Continuing Operating Activities",
                                 "CashFlowFromContinuingOperatingActivities"],
        "Capital Expenditure": ["Capital Expenditure", "CapitalExpenditure"],
        "Total Debt": ["Total Debt", "TotalDebt"],
        "Stockholders Equity": ["Stockholders Equity", "StockholdersEquity",
                                 "Total Stockholder Equity", "Common Stock Equity",
                                 "CommonStockEquity"],
        "Total Assets": ["Total Assets", "TotalAssets"],
        "Research And Development": ["Research And Development",
                                      "ResearchAndDevelopment",
                                      "Research Development"],
    }

    names_to_try = aliases.get(label, [label])
    for name in names_to_try:
        if name in df.index:
            try:
                val = df.iloc[df.index.get_loc(name), col_idx]
                if pd.notna(val):
                    return float(val)
            except (IndexError, KeyError):
                pass
    return None


def _get_revenue_series(income_stmt) -> list[Optional[float]]:
    """Extrai série de receita (mais recente primeiro) do income statement."""
    if income_stmt is None or income_stmt.empty:
        return []
    revs = []
    for i in range(min(4, income_stmt.shape[1])):
        revs.append(_safe_val(income_stmt, "Total Revenue", i))
    return revs


def _calc_cagr(newest: float, oldest: float, years: int) -> Optional[float]:
    """Calcula CAGR dado valor mais recente, mais antigo e nº de anos."""
    if oldest is None or newest is None or oldest <= 0 or years <= 0:
        return None
    return (newest / oldest) ** (1 / years) - 1


def _calc_trend(values: list[Optional[float]]) -> Optional[str]:
    """Dado uma lista de valores (mais recente primeiro), retorna tendência."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    # Comparar média dos mais recentes vs mais antigos
    mid = len(clean) // 2
    recent_avg = np.mean(clean[:mid]) if mid > 0 else clean[0]
    older_avg = np.mean(clean[mid:])
    if older_avg == 0:
        return "stable"
    change = (recent_avg - older_avg) / abs(older_avg)
    if change > 0.05:
        return "improving"
    elif change < -0.05:
        return "declining"
    return "stable"


def fetch_deep_data(ticker: str) -> Optional[dict]:
    """Stage 2: busca demonstrações financeiras para análise profunda."""
    try:
        time.sleep(REQUEST_DELAY)
        t = yf.Ticker(ticker)

        # Tentar APIs mais recentes, fallback para as antigas
        try:
            income = t.income_stmt
        except Exception:
            income = getattr(t, "financials", None)

        try:
            cashflow = t.cash_flow
        except Exception:
            cashflow = getattr(t, "cashflow", None)

        try:
            balance = t.balance_sheet
        except Exception:
            balance = getattr(t, "balance_sheet", None)

        result = {"ticker": ticker}

        # ── Série de receita (3 anos) ──
        rev_series = _get_revenue_series(income)

        if len(rev_series) >= 2 and rev_series[0] and rev_series[1] and rev_series[1] > 0:
            result["rev_growth_prior"] = (rev_series[0] / rev_series[1]) - 1
        else:
            result["rev_growth_prior"] = None

        if len(rev_series) >= 3 and rev_series[1] and rev_series[2] and rev_series[2] > 0:
            result["rev_growth_2y_ago"] = (rev_series[1] / rev_series[2]) - 1
        else:
            result["rev_growth_2y_ago"] = None

        # CAGR 3 anos
        if len(rev_series) >= 4 and rev_series[0] and rev_series[3]:
            result["revenue_cagr_3y"] = _calc_cagr(rev_series[0], rev_series[3], 3)
        elif len(rev_series) >= 3 and rev_series[0] and rev_series[2]:
            result["revenue_cagr_3y"] = _calc_cagr(rev_series[0], rev_series[2], 2)
        else:
            result["revenue_cagr_3y"] = None

        # Aceleração: crescimento recente vs crescimento anterior
        rg_current = result.get("rev_growth_prior")
        rg_prior = result.get("rev_growth_2y_ago")
        if rg_current is not None and rg_prior is not None:
            result["rev_acceleration"] = rg_current - rg_prior
        else:
            result["rev_acceleration"] = None

        # ── Tendência margem bruta ──
        gp_series = []
        for i in range(min(4, income.shape[1] if income is not None else 0)):
            gp = _safe_val(income, "Gross Profit", i)
            rev = _safe_val(income, "Total Revenue", i)
            if gp is not None and rev and rev > 0:
                gp_series.append(gp / rev)
            else:
                gp_series.append(None)
        result["gross_margin_trend"] = _calc_trend(gp_series)

        # ── Tendência margem operacional ──
        op_series = []
        for i in range(min(4, income.shape[1] if income is not None else 0)):
            op = _safe_val(income, "Operating Income", i)
            rev = _safe_val(income, "Total Revenue", i)
            if op is not None and rev and rev > 0:
                op_series.append(op / rev)
            else:
                op_series.append(None)
        result["op_margin_trend"] = _calc_trend(op_series)

        # ── SBC como % da receita ──
        sbc = _safe_val(cashflow, "Stock Based Compensation", 0)
        rev_latest = rev_series[0] if rev_series else None
        if sbc is not None and rev_latest and rev_latest > 0:
            result["sbc_revenue_pct"] = sbc / rev_latest
        else:
            result["sbc_revenue_pct"] = None

        # ── Earnings quality: FCF / Net Income ──
        fcf = _safe_val(cashflow, "Free Cash Flow", 0)
        ni = _safe_val(income, "Net Income", 0)
        if fcf is not None and ni is not None and ni != 0:
            result["earnings_quality"] = fcf / ni
        else:
            result["earnings_quality"] = None

        # ── FCF margin trend ──
        fcf_margin_series = []
        cf_cols = cashflow.shape[1] if cashflow is not None and not cashflow.empty else 0
        is_cols = income.shape[1] if income is not None and not income.empty else 0
        for i in range(min(4, cf_cols, is_cols)):
            f = _safe_val(cashflow, "Free Cash Flow", i)
            r = _safe_val(income, "Total Revenue", i)
            if f is not None and r and r > 0:
                fcf_margin_series.append(f / r)
            else:
                fcf_margin_series.append(None)
        result["fcf_margin_trend"] = _calc_trend(fcf_margin_series)

        # ── Rule of 40 ──
        # Rev growth + FCF margin (ambos como percentagem)
        rg = result.get("rev_growth_prior")
        fcf_m = None
        if fcf is not None and rev_latest and rev_latest > 0:
            fcf_m = fcf / rev_latest
        if rg is not None and fcf_m is not None:
            result["rule_of_40"] = (rg * 100) + (fcf_m * 100)
        else:
            result["rule_of_40"] = None

        # ── R&D como % da receita ──
        rd = _safe_val(income, "Research And Development", 0)
        if rd is not None and rev_latest and rev_latest > 0:
            result["rd_revenue_pct"] = abs(rd) / rev_latest
        else:
            result["rd_revenue_pct"] = None

        # ── Debt trend ──
        debt_series = []
        equity_series = []
        bs_cols = balance.shape[1] if balance is not None and not balance.empty else 0
        for i in range(min(4, bs_cols)):
            debt_series.append(_safe_val(balance, "Total Debt", i))
            equity_series.append(_safe_val(balance, "Stockholders Equity", i))
        result["debt_trend"] = _calc_trend(
            [d / e if d and e and e > 0 else None for d, e in zip(debt_series, equity_series)]
        )

        return result

    except Exception as e:
        log.debug(f"Deep fetch erro {ticker}: {e}")
        return None


def fetch_all_stage2(tickers_data: list[dict], workers: int = MAX_WORKERS) -> list[dict]:
    """Stage 2: busca dados profundos para candidatos que passaram o stage 1."""
    ticker_list = [d["ticker"] for d in tickers_data]
    total = len(ticker_list)
    log.info(f"[STAGE 2] Deep analysis de {total} candidatos ({workers} threads)...")

    deep_map = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_deep_data, t): t for t in ticker_list}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                if result:
                    deep_map[result["ticker"]] = result
            except Exception:
                pass
            if completed % 50 == 0:
                log.info(f"  Deep progress: {completed}/{total}")

    # Merge deep data into stage 1 data
    enriched = 0
    for row in tickers_data:
        deep = deep_map.get(row["ticker"], {})
        for key in (
            "revenue_cagr_3y", "rev_acceleration", "rev_growth_prior",
            "gross_margin_trend", "op_margin_trend", "fcf_margin_trend",
            "sbc_revenue_pct", "earnings_quality", "rule_of_40",
            "rd_revenue_pct", "debt_trend",
        ):
            if key in deep:
                row[key] = deep[key]
        if deep:
            enriched += 1

    log.info(f"[STAGE 2] Completo: {enriched}/{total} tickers enriched")
    return tickers_data


# ═══════════════════════════════════════════════════════════════════════
# 4. CLASSIFICAÇÃO DE FASE
# ═══════════════════════════════════════════════════════════════════════

def classify_phase(row: dict) -> str:
    net_income = row.get("net_income")
    revenue_growth = row.get("revenue_growth")

    if net_income is not None and net_income < 0:
        return "pre_lucro"

    if revenue_growth is not None and revenue_growth > 0.15:
        return "crescimento"

    return "madura"


# ═══════════════════════════════════════════════════════════════════════
# 5. SCORING ENGINE — PERCENTILE-BASED
# ═══════════════════════════════════════════════════════════════════════

def _percentile_rank(values: list[Optional[float]], value: Optional[float],
                      higher_is_better: bool = True) -> float:
    """Calcula o percentile rank (0-100) de um valor dentro de uma distribuição."""
    if value is None:
        return 25  # Score neutro-baixo para dados em falta
    clean = [v for v in values if v is not None]
    if not clean:
        return 50
    arr = np.array(clean)
    if higher_is_better:
        pct = np.sum(arr <= value) / len(arr) * 100
    else:
        pct = np.sum(arr >= value) / len(arr) * 100
    return min(100, max(0, pct))


def _trend_bonus(trend: Optional[str]) -> float:
    """Bónus/penalização por tendência: improving=+10, stable=0, declining=-10."""
    if trend == "improving":
        return 10
    elif trend == "declining":
        return -10
    return 0


def _sbc_score(sbc_pct: Optional[float]) -> float:
    """Score para SBC: quanto menor melhor. >20% = terrível, <5% = excelente."""
    if sbc_pct is None:
        return 50  # Neutro
    if sbc_pct <= 0.03:
        return 100
    elif sbc_pct <= 0.05:
        return 85
    elif sbc_pct <= 0.10:
        return 65
    elif sbc_pct <= 0.15:
        return 40
    elif sbc_pct <= 0.20:
        return 20
    else:
        return 5


def _earnings_quality_score(eq: Optional[float]) -> float:
    """Score para earnings quality (FCF/NI). >1.2 = excelente, <0.5 = suspeito."""
    if eq is None:
        return 40
    if eq < 0:
        return 15  # FCF e NI com sinais opostos
    if eq > 1.5:
        return 95
    elif eq > 1.0:
        return 80
    elif eq > 0.7:
        return 60
    elif eq > 0.4:
        return 40
    else:
        return 20


def score_phase_percentile(data: list[dict], fase: str) -> list[dict]:
    """Scoring por percentil para uma fase inteira."""
    phase_data = [d for d in data if d.get("fase") == fase]
    if not phase_data:
        return data

    weights = WEIGHTS.get(fase, {})

    # Extrair distribuições para cada métrica
    distributions = {}
    metric_keys = {
        "revenue_growth": ("revenue_growth", True),
        "rev_acceleration": ("rev_acceleration", True),
        "rev_cagr_3y": ("revenue_cagr_3y", True),
        "gross_margin": ("gross_margin", True),
        "fcf_margin": ("fcf_margin", True),
        "roe": ("roe", True),
        "fcf_yield": ("fcf_yield", True),
        "rule_of_40": ("rule_of_40", True),
        "liquidity": ("current_ratio", True),
        "health": ("current_ratio", True),
    }

    for metric_name, (data_key, _) in metric_keys.items():
        distributions[metric_name] = [d.get(data_key) for d in phase_data]

    # Valuation (lower is better)
    distributions["valuation_pe"] = [d.get("forward_pe") for d in phase_data]
    distributions["valuation_ev"] = [d.get("ev_ebitda") for d in phase_data]

    for row in phase_data:
        component_scores = {}

        # Métricas baseadas em percentil (higher is better)
        for metric_name in ("revenue_growth", "rev_acceleration", "rev_cagr_3y",
                            "gross_margin", "fcf_margin", "roe", "fcf_yield",
                            "rule_of_40", "liquidity", "health"):
            if metric_name not in weights:
                continue
            data_key = metric_keys.get(metric_name, (metric_name, True))[0]
            val = row.get(data_key)
            pct = _percentile_rank(distributions.get(metric_name, []), val, higher_is_better=True)
            component_scores[metric_name] = pct * weights[metric_name]

        # Valuation (lower is better) — combinar PE e EV/EBITDA
        if "valuation" in weights:
            pe_score = _percentile_rank(distributions["valuation_pe"],
                                         row.get("forward_pe"), higher_is_better=False)
            ev_score = _percentile_rank(distributions["valuation_ev"],
                                         row.get("ev_ebitda"), higher_is_better=False)
            # Média dos dois, ou só o disponível
            val_scores = [s for s in [pe_score, ev_score] if s != 25]  # 25 = default missing
            val_score = np.mean(val_scores) if val_scores else 50
            component_scores["valuation"] = val_score * weights["valuation"]

        # Momentum (posição no range 52w)
        if "momentum" in weights:
            price = row.get("price")
            low52 = row.get("fifty_two_week_low")
            high52 = row.get("fifty_two_week_high")
            if price and low52 and high52 and high52 > low52:
                position = (price - low52) / (high52 - low52)
                # 0.3-0.7 é ideal, penalizar extremos
                mom_score = max(0, 100 - abs(position - 0.5) * 200)
            else:
                mom_score = 50
            component_scores["momentum"] = mom_score * weights["momentum"]

        # SBC penalty
        if "sbc_penalty" in weights:
            sbc_s = _sbc_score(row.get("sbc_revenue_pct"))
            component_scores["sbc_penalty"] = sbc_s * weights["sbc_penalty"]

        # Earnings quality
        if "earnings_quality" in weights:
            eq_s = _earnings_quality_score(row.get("earnings_quality"))
            component_scores["earnings_quality"] = eq_s * weights["earnings_quality"]

        # Trend bonuses (margem bruta e operacional)
        if "gross_margin_trend" in weights:
            gm_trend = row.get("gross_margin_trend")
            base = component_scores.get("gross_margin", 50 * weights.get("gross_margin", 0))
            trend_adj = _trend_bonus(gm_trend) * weights["gross_margin_trend"]
            component_scores["gross_margin_trend"] = max(0, 50 + trend_adj) * weights["gross_margin_trend"]

        if "op_margin_trend" in weights:
            op_trend = row.get("op_margin_trend")
            trend_adj = _trend_bonus(op_trend)
            component_scores["op_margin_trend"] = max(0, 50 + trend_adj) * weights["op_margin_trend"]

        # Score total
        total_score = sum(component_scores.values())
        row["score"] = round(min(100, max(0, total_score)), 1)

        # Sub-scores para o frontend
        growth_keys = ["revenue_growth", "rev_acceleration", "rev_cagr_3y", "rule_of_40"]
        quality_keys = ["earnings_quality", "sbc_penalty", "gross_margin", "gross_margin_trend"]
        value_keys = ["valuation", "fcf_yield", "momentum"]

        row["growth_score"] = round(min(100, sum(
            component_scores.get(k, 0) / weights.get(k, 0.01) * (weights.get(k, 0) / sum(weights.get(k, 0.01) for k in growth_keys if k in weights))
            for k in growth_keys if k in weights and k in component_scores
        )), 1) if any(k in weights for k in growth_keys) else None

        row["quality_score"] = round(min(100, sum(
            component_scores.get(k, 0) / weights.get(k, 0.01) * (weights.get(k, 0) / sum(weights.get(k, 0.01) for k in quality_keys if k in weights))
            for k in quality_keys if k in weights and k in component_scores
        )), 1) if any(k in weights for k in quality_keys) else None

        row["value_score"] = round(min(100, sum(
            component_scores.get(k, 0) / weights.get(k, 0.01) * (weights.get(k, 0) / sum(weights.get(k, 0.01) for k in value_keys if k in weights))
            for k in value_keys if k in weights and k in component_scores
        )), 1) if any(k in weights for k in value_keys) else None

    return data


def calculate_scores(data: list[dict]) -> list[dict]:
    """Classifica fases e calcula scores por percentil."""
    # Classificar fases
    for row in data:
        row["fase"] = classify_phase(row)

    # Scoring por percentil dentro de cada fase
    for fase in ("pre_lucro", "crescimento", "madura"):
        data = score_phase_percentile(data, fase)

    return data


# ═══════════════════════════════════════════════════════════════════════
# 6. FILTRAGEM FINAL
# ═══════════════════════════════════════════════════════════════════════

def apply_thresholds(data: list[dict]) -> list[dict]:
    passed = []
    for row in data:
        fase = row["fase"]
        th = THRESHOLDS[fase]

        if row.get("score", 0) < th["min_score"]:
            continue

        rg = row.get("revenue_growth")
        if rg is not None and rg < th["revenue_growth"]:
            continue

        gm = row.get("gross_margin")
        if gm is not None and gm < th["gross_margin"]:
            continue

        de = row.get("debt_equity")
        if de is not None and de / 100 > th["max_debt_equity"]:
            continue

        passed.append(row)

    log.info(f"Após thresholds: {len(passed)} tickers passaram (de {len(data)})")
    return passed


# ═══════════════════════════════════════════════════════════════════════
# 7. SUPABASE
# ═══════════════════════════════════════════════════════════════════════

def _r_pct(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    return round(val * 100, 2)


def _r_val(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    return round(val, 2)


def save_to_supabase(results: list[dict], scan_date: date):
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning("SUPABASE_URL/SUPABASE_KEY não configurados. A saltar gravação.")
        return False

    import requests as req

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    url = f"{SUPABASE_URL}/rest/v1/screener_results"

    rows = []
    for r in results:
        rows.append({
            "ticker": r["ticker"],
            "nome": r.get("nome", "")[:200],
            "exchange": r.get("exchange", ""),
            "sector": r.get("sector", ""),
            "industry": r.get("industry", ""),
            "country": r.get("country", ""),
            "market_cap": r.get("market_cap"),
            "fase": r["fase"],
            "score": r.get("score"),
            "revenue_growth": _r_pct(r.get("revenue_growth")),
            "gross_margin": _r_pct(r.get("gross_margin")),
            "operating_margin": _r_pct(r.get("operating_margin")),
            "fcf_margin": _r_pct(r.get("fcf_margin")),
            "roe": _r_pct(r.get("roe")),
            "debt_equity": _r_val(r.get("debt_equity")),
            "forward_pe": _r_val(r.get("forward_pe")),
            "ev_ebitda": _r_val(r.get("ev_ebitda")),
            "ps_ratio": _r_val(r.get("ps_ratio")),
            "fcf_yield": _r_pct(r.get("fcf_yield")),
            "current_ratio": _r_val(r.get("current_ratio")),
            # Novos campos v2
            "sbc_revenue_pct": _r_pct(r.get("sbc_revenue_pct")),
            "rule_of_40": _r_val(r.get("rule_of_40")),
            "revenue_cagr_3y": _r_pct(r.get("revenue_cagr_3y")),
            "rev_acceleration": _r_pct(r.get("rev_acceleration")),
            "earnings_quality": _r_val(r.get("earnings_quality")),
            "gross_margin_trend": r.get("gross_margin_trend"),
            "fcf_margin_trend": r.get("fcf_margin_trend"),
            "growth_score": _r_val(r.get("growth_score")),
            "quality_score": _r_val(r.get("quality_score")),
            "value_score": _r_val(r.get("value_score")),
            "scan_date": str(scan_date),
            "raw_data": json.dumps({
                k: v for k, v in r.items()
                if k not in ("ticker", "nome", "fase", "score")
            }, default=str),
        })

    batch_size = 50
    total_inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        resp = req.post(url, headers=headers, json=batch)
        if resp.status_code in (200, 201):
            total_inserted += len(batch)
        else:
            log.error(f"Erro Supabase batch {i}: {resp.status_code} — {resp.text[:200]}")

    log.info(f"Gravados {total_inserted}/{len(rows)} resultados no Supabase")
    return True


# ═══════════════════════════════════════════════════════════════════════
# 8. CSV & RESUMO
# ═══════════════════════════════════════════════════════════════════════

def export_csv(results: list[dict], filename: str = "screener_results_v2.csv"):
    if not results:
        log.warning("Sem resultados para exportar")
        return None
    df = pd.DataFrame(results)
    cols_order = [
        "ticker", "nome", "fase", "score", "growth_score", "quality_score",
        "value_score", "sector", "country", "market_cap",
        "revenue_growth", "revenue_cagr_3y", "rev_acceleration",
        "gross_margin", "gross_margin_trend",
        "fcf_margin", "fcf_margin_trend",
        "roe", "sbc_revenue_pct", "earnings_quality",
        "rule_of_40", "forward_pe", "ev_ebitda", "fcf_yield",
        "debt_equity", "current_ratio",
    ]
    cols = [c for c in cols_order if c in df.columns]
    df = df[cols].sort_values("score", ascending=False)

    pct_cols = ["revenue_growth", "revenue_cagr_3y", "rev_acceleration",
                "gross_margin", "fcf_margin", "roe", "fcf_yield",
                "sbc_revenue_pct"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")

    df.to_csv(filename, index=False)
    log.info(f"Exportado para {filename} ({len(df)} linhas)")
    return filename


def print_summary(results: list[dict]):
    if not results:
        print("\n❌ Nenhum ticker passou os filtros.")
        return

    df = pd.DataFrame(results)

    print("\n" + "═" * 78)
    print(f"  SCREENER STEVESTOR v2 — {date.today().isoformat()}")
    print(f"  {len(results)} tickers encontrados")
    print("═" * 78)

    for fase in ["pre_lucro", "crescimento", "madura"]:
        subset = df[df["fase"] == fase].sort_values("score", ascending=False)
        if subset.empty:
            continue

        fase_label = {
            "pre_lucro": "PRÉ-LUCRO",
            "crescimento": "CRESCIMENTO",
            "madura": "MADURA",
        }[fase]

        print(f"\n{'─' * 78}")
        print(f"  {fase_label} ({len(subset)} tickers)")
        print(f"  {'Ticker':>8s}  {'Score':>5s}  {'Growth':>6s}  {'Quality':>7s}  "
              f"{'RevGr':>5s}  {'GM':>5s}  {'SBC':>5s}  {'R40':>5s}  Nome")
        print(f"{'─' * 78}")

        top = subset.head(20)
        for _, row in top.iterrows():
            rg = f"{row['revenue_growth']*100:.0f}%" if pd.notna(row.get("revenue_growth")) else " n/a"
            gm = f"{row['gross_margin']*100:.0f}%" if pd.notna(row.get("gross_margin")) else " n/a"
            sbc = f"{row['sbc_revenue_pct']*100:.0f}%" if pd.notna(row.get("sbc_revenue_pct")) else " n/a"
            r40 = f"{row['rule_of_40']:.0f}" if pd.notna(row.get("rule_of_40")) else " n/a"
            gs = f"{row['growth_score']:.0f}" if pd.notna(row.get("growth_score")) else " n/a"
            qs = f"{row['quality_score']:.0f}" if pd.notna(row.get("quality_score")) else " n/a"

            print(
                f"  {row['ticker']:>8s}  {row['score']:5.1f}  {gs:>6s}  {qs:>7s}  "
                f"{rg:>5s}  {gm:>5s}  {sbc:>5s}  {r40:>5s}  "
                f"{row.get('nome', '')[:25]}"
            )

    print(f"\n{'═' * 78}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SteVestor Screener v2")
    parser.add_argument("--universo", choices=["all", "us", "europa"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tickers", type=str, default="")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--skip-deep", action="store_true",
                        help="Saltar stage 2 (deep analysis) — mais rápido, menos dados")
    args = parser.parse_args()

    workers = args.workers
    start_time = time.time()

    # ── 1. Universo ──
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        log.info(f"Modo teste: {len(tickers)} tickers")
    else:
        tickers = build_universe(args.universo)

    if not tickers:
        log.error("Nenhum ticker no universo. A abortar.")
        sys.exit(1)

    # ── 2. Stage 1 — Quick scan ──
    data = fetch_all_stage1(tickers, workers=workers)

    if not data:
        log.error("Nenhum dado válido. A abortar.")
        sys.exit(1)

    # ── 3. Classificar fases (pre-scoring, para decidir quem vai ao stage 2) ──
    for row in data:
        row["fase"] = classify_phase(row)

    # ── 4. Stage 2 — Deep analysis (financial statements) ──
    if not args.skip_deep:
        data = fetch_all_stage2(data, workers=workers)
    else:
        log.info("[STAGE 2] Saltado (--skip-deep)")

    # ── 5. Scoring por percentil ──
    data = calculate_scores(data)

    # ── 6. Override score mínimo ──
    if args.min_score is not None:
        for fase in THRESHOLDS:
            THRESHOLDS[fase]["min_score"] = args.min_score

    # ── 7. Filtrar ──
    results = apply_thresholds(data)
    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # ── 8. Output ──
    print_summary(results)

    today = date.today()

    if args.dry_run:
        filename = export_csv(results)
        if filename:
            print(f"\n📄 Resultados exportados para: {filename}")
    else:
        saved = save_to_supabase(results, today)
        if saved:
            print(f"\n✅ Resultados gravados no Supabase ({len(results)} tickers)")
        else:
            filename = export_csv(results)
            print(f"\n⚠️  Supabase indisponível. Exportado para: {filename}")

    elapsed = time.time() - start_time
    log.info(f"Tempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()

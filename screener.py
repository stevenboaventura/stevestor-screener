#!/usr/bin/env python3
"""
SteVestor Screener — Algoritmo de descoberta de tickers
Corre semanalmente, varre US + Europa, pontua por crescimento + qualidade,
grava resultados no Supabase.

Uso:
  python screener.py                    # Corre o scan completo
  python screener.py --universo us      # Só US
  python screener.py --universo europa  # Só Europa
  python screener.py --dry-run          # Não grava no Supabase, exporta CSV
  python screener.py --tickers AAPL,RKLB,NVO  # Testar tickers específicos
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

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("screener")

# Supabase — ler das env vars (Railway injeta estas)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "") or os.environ.get("SUPABASE_KEY", "")

# Thresholds mínimos por fase
THRESHOLDS = {
    "pre_lucro": {
        "revenue_growth": 0.20,
        "gross_margin": 0.35,
        "min_score": 55,
        "max_debt_equity": 3.0,
    },
    "crescimento": {
        "revenue_growth": 0.10,
        "gross_margin": 0.30,
        "min_score": 50,
        "max_debt_equity": 2.0,
    },
    "madura": {
        "revenue_growth": 0.03,
        "gross_margin": 0.25,
        "min_score": 45,
        "max_debt_equity": 1.5,
    },
}

# Pré-filtros globais
MIN_MARKET_CAP = 300_000_000  # $300M
MIN_AVG_VOLUME = 100_000

# Sectores a excluir (métricas fundamentais não comparáveis)
EXCLUDED_SECTORS = {"Financial Services"}

# Máximo de threads para yfinance
MAX_WORKERS = 10

# Delay entre requests para não ser rate-limited
REQUEST_DELAY = 0.3  # segundos


# ---------------------------------------------------------------------------
# 1. Obter universo de tickers
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """S&P 500 via Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"S&P 500: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.error(f"Erro ao obter S&P 500: {e}")
        return []


def get_nasdaq100_tickers() -> list[str]:
    """NASDAQ 100 via Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        # A tabela com os tickers normalmente é a que tem coluna "Ticker"
        for t in tables:
            if "Ticker" in t.columns:
                tickers = t["Ticker"].str.replace(".", "-", regex=False).tolist()
                log.info(f"NASDAQ 100: {len(tickers)} tickers")
                return tickers
        log.warning("Tabela NASDAQ 100 não encontrada")
        return []
    except Exception as e:
        log.error(f"Erro ao obter NASDAQ 100: {e}")
        return []


def get_stoxx600_tickers() -> list[str]:
    """STOXX 600 — lista estática das maiores empresas europeias.
    Wikipedia nem sempre tem a tabela completa, por isso usamos uma
    abordagem de exchanges europeias principais."""
    try:
        # Tentar Wikipedia primeiro
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/STOXX_Europe_600",
            match="Ticker",
        )
        if tables:
            for t in tables:
                for col in ["Ticker", "Symbol", "Bloomberg ticker"]:
                    if col in t.columns:
                        tickers = t[col].dropna().tolist()
                        if len(tickers) > 100:
                            log.info(f"STOXX 600: {len(tickers)} tickers")
                            return tickers
    except Exception:
        pass

    # Fallback: lista curada das maiores europeias acessíveis via yfinance
    # Sufixos yfinance: .L (London), .AS (Amsterdam), .PA (Paris),
    # .DE (Frankfurt), .ST (Stockholm), .HE (Helsinki), .CO (Copenhagen),
    # .MI (Milan), .MC (Madrid), .SW (Zurich)
    european_blue_chips = [
        # UK
        "AZN.L", "SHEL.L", "ULVR.L", "HSBA.L", "BP.L", "GSK.L", "RIO.L",
        "LSEG.L", "DGE.L", "REL.L", "BATS.L", "PRU.L", "NG.L", "VOD.L",
        "AAL.L", "LLOY.L", "BARC.L", "STAN.L", "ABF.L", "CRH.L",
        # Netherlands
        "ASML.AS", "UNA.AS", "INGA.AS", "PHIA.AS", "AD.AS", "WKL.AS",
        "RAND.AS", "HEIA.AS", "AGN.AS", "NN.AS",
        # France
        "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AI.PA", "SU.PA", "BN.PA",
        "CS.PA", "AIR.PA", "SAF.PA", "DSY.PA", "HO.PA", "RI.PA", "KER.PA",
        "BNP.PA", "ACA.PA", "GLE.PA", "VIV.PA", "CAP.PA", "SGO.PA",
        # Germany
        "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "MRK.DE",
        "BMW.DE", "VOW3.DE", "ADS.DE", "MUV2.DE", "DHL.DE", "HEN3.DE",
        "RWE.DE", "IFX.DE", "BEI.DE", "FRE.DE",
        # Switzerland
        "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ABBN.SW", "SIKA.SW",
        "ZURN.SW", "LONN.SW", "GIVN.SW",
        # Denmark / Sweden / Norway
        "NVO", "NOVO-B.CO", "DSV.CO", "VWS.CO", "CARL-B.CO",
        "VOLV-B.ST", "ERIC-B.ST", "ASSA-B.ST", "SAND.ST", "HEXA-B.ST",
        "EQNR.OL", "DNB.OL", "TEL.OL",
        # Spain / Italy
        "SAN.MC", "IBE.MC", "ITX.MC", "TEF.MC", "BBVA.MC",
        "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "STM.MI",
    ]
    log.info(f"Europa (blue chips fallback): {len(european_blue_chips)} tickers")
    return european_blue_chips


def get_russell2000_sample() -> list[str]:
    """Amostra do Russell 2000 — difícil obter a lista completa grátis.
    Usa o ETF IWM holdings como proxy, ou fallback para um subset."""
    try:
        # Tentar obter via Wikipedia
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Russell_2000_Index",
            match="Ticker",
        )
        if tables:
            for t in tables:
                for col in ["Ticker", "Symbol"]:
                    if col in t.columns:
                        tickers = t[col].dropna().tolist()
                        if len(tickers) > 100:
                            log.info(f"Russell 2000: {len(tickers)} tickers")
                            return tickers
    except Exception:
        pass

    log.info("Russell 2000: lista completa não disponível, a usar S&P 500 + NASDAQ como base")
    return []


def build_universe(scope: str = "all") -> list[str]:
    """Constrói o universo de tickers sem duplicados."""
    tickers = set()

    if scope in ("all", "us"):
        tickers.update(get_sp500_tickers())
        tickers.update(get_nasdaq100_tickers())
        extras_us = get_russell2000_sample()
        if extras_us:
            tickers.update(extras_us)

    if scope in ("all", "europa"):
        tickers.update(get_stoxx600_tickers())

    # Limpar
    tickers = {t.strip() for t in tickers if t and isinstance(t, str)}
    log.info(f"Universo total: {len(tickers)} tickers únicos")
    return sorted(tickers)


# ---------------------------------------------------------------------------
# 2. Fetch de dados fundamentais via yfinance
# ---------------------------------------------------------------------------

def fetch_ticker_data(ticker: str) -> Optional[dict]:
    """Busca dados fundamentais de um ticker via yfinance."""
    try:
        time.sleep(REQUEST_DELAY)
        t = yf.Ticker(ticker)
        info = t.info

        if not info or info.get("quoteType") in ("ETF", "MUTUALFUND", "INDEX"):
            return None

        # Verificar se tem dados mínimos
        market_cap = info.get("marketCap")
        if not market_cap or market_cap < MIN_MARKET_CAP:
            return None

        avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day", 0)
        if avg_volume < MIN_AVG_VOLUME:
            return None

        sector = info.get("sector", "")
        if sector in EXCLUDED_SECTORS:
            return None

        # Extrair métricas
        data = {
            "ticker": ticker,
            "nome": info.get("shortName") or info.get("longName", ""),
            "exchange": info.get("exchange", ""),
            "sector": sector,
            "industry": info.get("industry", ""),
            "country": info.get("country", ""),
            "market_cap": market_cap,
            # Crescimento
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            # Margens
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            # Rentabilidade
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            # Valuation
            "forward_pe": info.get("forwardPE"),
            "trailing_pe": info.get("trailingPE"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "pb_ratio": info.get("priceToBook"),
            # FCF
            "free_cashflow": info.get("freeCashflow"),
            "revenue": info.get("totalRevenue"),
            "net_income": info.get("netIncomeToCommon"),
            # Saúde financeira
            "debt_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            # Preço
            "price": info.get("currentPrice") or info.get("previousClose"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "beta": info.get("beta"),
        }

        # Calcular métricas derivadas
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


def fetch_all_data(tickers: list[str], workers: int = MAX_WORKERS) -> list[dict]:
    """Busca dados de todos os tickers em paralelo."""
    results = []
    total = len(tickers)
    completed = 0
    failed = 0

    log.info(f"A buscar dados de {total} tickers ({workers} threads)...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_ticker_data, t): t for t in tickers}

        for future in as_completed(futures):
            completed += 1
            ticker = futures[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
                else:
                    failed += 1
            except Exception:
                failed += 1

            if completed % 50 == 0:
                log.info(
                    f"Progresso: {completed}/{total} "
                    f"({len(results)} válidos, {failed} excluídos)"
                )

    log.info(
        f"Fetch completo: {len(results)} tickers válidos de {total} "
        f"({failed} excluídos no pré-filtro)"
    )
    return results


# ---------------------------------------------------------------------------
# 3. Classificação de maturidade
# ---------------------------------------------------------------------------

def classify_phase(row: dict) -> str:
    """Classifica o ticker por fase de maturidade."""
    net_income = row.get("net_income")
    revenue_growth = row.get("revenue_growth")

    # Pré-lucro: net income negativo
    if net_income is not None and net_income < 0:
        return "pre_lucro"

    # Crescimento: revenue growth > 15% e lucro positivo
    if revenue_growth is not None and revenue_growth > 0.15:
        return "crescimento"

    # Madura: o resto com lucro positivo
    return "madura"


# ---------------------------------------------------------------------------
# 4. Scoring
# ---------------------------------------------------------------------------

def normalize_score(value: Optional[float], low: float, high: float) -> float:
    """Normaliza um valor para 0-100. Valores fora do range são clipped."""
    if value is None:
        return 0
    clipped = max(low, min(high, value))
    return ((clipped - low) / (high - low)) * 100 if high != low else 50


def score_pre_lucro(row: dict) -> float:
    """Scoring para empresas pré-lucro."""
    scores = {}

    # Crescimento receita (peso 35) — 10% a 80%+
    scores["revenue_growth"] = normalize_score(row.get("revenue_growth"), 0.10, 0.80) * 0.35

    # Margem bruta (peso 20) — 30% a 80%
    scores["gross_margin"] = normalize_score(row.get("gross_margin"), 0.30, 0.80) * 0.20

    # Melhoria margem operacional (peso 15) — difícil de calcular sem histórico
    # Usamos a margem operacional atual como proxy (menos negativa = melhor)
    op_margin = row.get("operating_margin")
    if op_margin is not None:
        # Para pré-lucro, -50% é terrível, -5% é quase breakeven
        scores["op_margin_trend"] = normalize_score(op_margin, -0.50, 0.0) * 0.15
    else:
        scores["op_margin_trend"] = 0

    # Liquidez (peso 15) — current ratio
    scores["liquidity"] = normalize_score(row.get("current_ratio"), 0.5, 5.0) * 0.15

    # Momento preço (peso 15) — posição no range 52w
    price = row.get("price")
    low52 = row.get("fifty_two_week_low")
    high52 = row.get("fifty_two_week_high")
    if price and low52 and high52 and high52 > low52:
        position = (price - low52) / (high52 - low52)
        # Preferimos meio do range (0.3-0.7), penalizamos extremos
        momentum_score = 100 - abs(position - 0.5) * 200
        scores["momentum"] = max(0, momentum_score) * 0.15
    else:
        scores["momentum"] = 0

    return sum(scores.values())


def score_crescimento(row: dict) -> float:
    """Scoring para empresas em crescimento."""
    scores = {}

    # Crescimento receita (peso 25) — 10% a 60%
    scores["revenue_growth"] = normalize_score(row.get("revenue_growth"), 0.10, 0.60) * 0.25

    # Margem bruta (peso 15) — 30% a 75%
    scores["gross_margin"] = normalize_score(row.get("gross_margin"), 0.30, 0.75) * 0.15

    # Margem FCF (peso 15) — 0% a 30%
    scores["fcf_margin"] = normalize_score(row.get("fcf_margin"), 0.0, 0.30) * 0.15

    # ROE (peso 15) — 5% a 40%
    scores["roe"] = normalize_score(row.get("roe"), 0.05, 0.40) * 0.15

    # Aceleração — usamos earnings growth como proxy (peso 10)
    scores["acceleration"] = normalize_score(row.get("earnings_growth"), 0.0, 0.50) * 0.10

    # Valorização relativa (peso 10) — forward PE (mais baixo = melhor)
    fwd_pe = row.get("forward_pe")
    if fwd_pe and fwd_pe > 0:
        # PE de 10 = excelente, PE de 60 = caro
        scores["valuation"] = normalize_score(1 / fwd_pe, 1 / 60, 1 / 10) * 0.10
    else:
        scores["valuation"] = 0

    # Saúde financeira (peso 10) — debt/equity (mais baixo = melhor)
    de = row.get("debt_equity")
    if de is not None and de >= 0:
        scores["health"] = normalize_score(1 / (1 + de / 100), 0.3, 1.0) * 0.10
    else:
        scores["health"] = 5  # Sem dívida = bom, score base

    return sum(scores.values())


def score_madura(row: dict) -> float:
    """Scoring para empresas maduras."""
    scores = {}

    # ROE (peso 25) — 8% a 35%
    scores["roe"] = normalize_score(row.get("roe"), 0.08, 0.35) * 0.25

    # Margem FCF (peso 20) — 5% a 30%
    scores["fcf_margin"] = normalize_score(row.get("fcf_margin"), 0.05, 0.30) * 0.20

    # Crescimento receita (peso 15) — 3% a 20%
    scores["revenue_growth"] = normalize_score(row.get("revenue_growth"), 0.03, 0.20) * 0.15

    # FCF yield (peso 15) — 2% a 10%
    scores["fcf_yield"] = normalize_score(row.get("fcf_yield"), 0.02, 0.10) * 0.15

    # Saúde financeira (peso 10)
    de = row.get("debt_equity")
    if de is not None and de >= 0:
        scores["health"] = normalize_score(1 / (1 + de / 100), 0.4, 1.0) * 0.10
    else:
        scores["health"] = 5

    # Valorização relativa (peso 10)
    fwd_pe = row.get("forward_pe")
    if fwd_pe and fwd_pe > 0:
        scores["valuation"] = normalize_score(1 / fwd_pe, 1 / 30, 1 / 8) * 0.10
    else:
        scores["valuation"] = 0

    # Consistência (peso 5) — margem de lucro positiva como proxy
    pm = row.get("profit_margin")
    if pm and pm > 0.05:
        scores["consistency"] = 5.0
    elif pm and pm > 0:
        scores["consistency"] = 2.5
    else:
        scores["consistency"] = 0

    return sum(scores.values())


def calculate_scores(data: list[dict]) -> list[dict]:
    """Calcula fase e score para cada ticker."""
    score_fns = {
        "pre_lucro": score_pre_lucro,
        "crescimento": score_crescimento,
        "madura": score_madura,
    }

    for row in data:
        fase = classify_phase(row)
        row["fase"] = fase
        row["score"] = round(score_fns[fase](row), 1)

    return data


# ---------------------------------------------------------------------------
# 5. Filtragem final
# ---------------------------------------------------------------------------

def apply_thresholds(data: list[dict]) -> list[dict]:
    """Aplica thresholds mínimos por fase."""
    passed = []

    for row in data:
        fase = row["fase"]
        th = THRESHOLDS[fase]

        # Score mínimo
        if row["score"] < th["min_score"]:
            continue

        # Revenue growth mínimo
        rg = row.get("revenue_growth")
        if rg is not None and rg < th["revenue_growth"]:
            continue

        # Gross margin mínimo
        gm = row.get("gross_margin")
        if gm is not None and gm < th["gross_margin"]:
            continue

        # Debt/equity máximo
        de = row.get("debt_equity")
        if de is not None and de / 100 > th["max_debt_equity"]:
            # yfinance reporta debt_equity como percentagem (ex: 150 = 1.5x)
            continue

        passed.append(row)

    log.info(f"Após thresholds: {len(passed)} tickers passaram (de {len(data)})")
    return passed


# ---------------------------------------------------------------------------
# 6. Gravação no Supabase
# ---------------------------------------------------------------------------

def save_to_supabase(results: list[dict], scan_date: date):
    """Grava resultados no Supabase via REST API."""
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

    # Preparar rows
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
            "score": r["score"],
            "revenue_growth": _round_pct(r.get("revenue_growth")),
            "gross_margin": _round_pct(r.get("gross_margin")),
            "operating_margin": _round_pct(r.get("operating_margin")),
            "fcf_margin": _round_pct(r.get("fcf_margin")),
            "roic": None,  # yfinance não dá ROIC directo
            "roe": _round_pct(r.get("roe")),
            "debt_equity": _round_val(r.get("debt_equity")),
            "forward_pe": _round_val(r.get("forward_pe")),
            "ev_ebitda": _round_val(r.get("ev_ebitda")),
            "ps_ratio": _round_val(r.get("ps_ratio")),
            "fcf_yield": _round_pct(r.get("fcf_yield")),
            "current_ratio": _round_val(r.get("current_ratio")),
            "scan_date": str(scan_date),
            "raw_data": json.dumps({
                k: v for k, v in r.items()
                if k not in ("ticker", "nome", "fase", "score")
            }, default=str),
        })

    # Inserir em batches de 50
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


def _round_pct(val: Optional[float]) -> Optional[float]:
    """Arredonda percentagem para 2 casas (ex: 0.2534 → 25.34)."""
    if val is None:
        return None
    return round(val * 100, 2)


def _round_val(val: Optional[float]) -> Optional[float]:
    """Arredonda valor para 2 casas."""
    if val is None:
        return None
    return round(val, 2)


# ---------------------------------------------------------------------------
# 7. Exportação CSV (dry-run)
# ---------------------------------------------------------------------------

def export_csv(results: list[dict], filename: str = "screener_results.csv"):
    """Exporta resultados para CSV."""
    if not results:
        log.warning("Sem resultados para exportar")
        return

    df = pd.DataFrame(results)

    # Selecionar e ordenar colunas
    cols_order = [
        "ticker", "nome", "fase", "score", "sector", "industry", "country",
        "market_cap", "revenue_growth", "gross_margin", "operating_margin",
        "fcf_margin", "roe", "forward_pe", "ev_ebitda", "ps_ratio",
        "fcf_yield", "debt_equity", "current_ratio",
    ]
    cols = [c for c in cols_order if c in df.columns]
    df = df[cols].sort_values("score", ascending=False)

    # Formatar percentagens para leitura
    pct_cols = ["revenue_growth", "gross_margin", "operating_margin",
                "fcf_margin", "roe", "fcf_yield"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
            )

    df.to_csv(filename, index=False)
    log.info(f"Exportado para {filename} ({len(df)} linhas)")
    return filename


# ---------------------------------------------------------------------------
# 8. Resumo
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Imprime resumo dos resultados."""
    if not results:
        print("\n❌ Nenhum ticker passou os filtros.")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print(f"  SCREENER STEVESTOR — {date.today().isoformat()}")
    print(f"  {len(results)} tickers encontrados")
    print("=" * 70)

    # Por fase
    for fase in ["pre_lucro", "crescimento", "madura"]:
        subset = df[df["fase"] == fase].sort_values("score", ascending=False)
        if subset.empty:
            continue

        fase_label = {
            "pre_lucro": "PRÉ-LUCRO",
            "crescimento": "CRESCIMENTO",
            "madura": "MADURA",
        }[fase]

        print(f"\n{'─' * 70}")
        print(f"  {fase_label} ({len(subset)} tickers)")
        print(f"{'─' * 70}")

        top = subset.head(15)
        for _, row in top.iterrows():
            rg = f"{row['revenue_growth']*100:.0f}%" if pd.notna(row.get("revenue_growth")) else "n/a"
            gm = f"{row['gross_margin']*100:.0f}%" if pd.notna(row.get("gross_margin")) else "n/a"
            score = f"{row['score']:.0f}"

            print(
                f"  {row['ticker']:>8s}  Score: {score:>3s}  "
                f"Rev Growth: {rg:>5s}  Gross Margin: {gm:>5s}  "
                f"| {row.get('nome', '')[:30]}"
            )

    print(f"\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SteVestor Screener")
    parser.add_argument(
        "--universo",
        choices=["all", "us", "europa"],
        default="all",
        help="Universo de tickers (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Não grava no Supabase, exporta CSV",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Lista de tickers separada por vírgulas (para teste)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Override do score mínimo global",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Número de threads (default: {MAX_WORKERS})",
    )
    args = parser.parse_args()

    workers = args.workers

    start_time = time.time()

    # 1. Universo
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        log.info(f"Modo teste: {len(tickers)} tickers")
    else:
        tickers = build_universe(args.universo)

    if not tickers:
        log.error("Nenhum ticker no universo. A abortar.")
        sys.exit(1)

    # 2. Fetch dados
    data = fetch_all_data(tickers, workers=workers)

    if not data:
        log.error("Nenhum dado válido. A abortar.")
        sys.exit(1)

    # 3. Scoring
    data = calculate_scores(data)

    # 4. Override score mínimo se pedido
    if args.min_score is not None:
        for fase in THRESHOLDS:
            THRESHOLDS[fase]["min_score"] = args.min_score

    # 5. Filtrar
    results = apply_thresholds(data)

    # 6. Ordenar por score
    results.sort(key=lambda x: x["score"], reverse=True)

    # 7. Output
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
            # Fallback para CSV
            filename = export_csv(results)
            print(f"\n⚠️  Supabase indisponível. Exportado para: {filename}")

    elapsed = time.time() - start_time
    log.info(f"Tempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()

"""
LLM commentary module — calls local Ollama (Mistral) REST API directly via requests
so that corporate proxies are bypassed (proxies={"http": None, "https": None}).

No data leaves your machine. Requires Ollama running at localhost:11434.
Model: mistral:latest

To disable: set ENABLE_AI = False in app.py.
To revert: git checkout HEAD~1 -- src/llm.py app.py requirements.txt
"""

from __future__ import annotations

import json
from typing import Generator

import requests

OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "mistral:latest"

# Explicitly bypass any system / corporate proxy for all localhost calls
_NO_PROXY = {"http": None, "https": None}


def _check_ollama() -> tuple[bool, str]:
    """Return (available, error_message). Calls Ollama's /api/tags endpoint."""
    try:
        resp = requests.get(
            f"{OLLAMA_HOST}/api/tags",
            proxies=_NO_PROXY,
            timeout=5,
        )
        resp.raise_for_status()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def build_seasonal_prompt(
    sym_name: str,
    win_label: str,
    summary: dict,
    today_features: dict | None = None,
    similar_years: list[dict] | None = None,
) -> str:
    """
    Build the prompt sent to Mistral for seasonal analysis commentary.

    Parameters
    ----------
    sym_name        : human-readable stock name
    win_label       : e.g. "01-Apr → 90 days"
    summary         : dict from seasonal_analysis() (win_rate, avg_return, etc.)
    today_features  : dict from similar_years_analysis() today_features key (optional)
    similar_years   : list of dicts for top similar years (optional)
    """
    total      = summary.get("total_instances", 0)
    win_rate   = summary.get("win_rate_pct", 0)
    avg_ret    = summary.get("avg_return_pct", 0)
    median_ret = summary.get("median_return_pct", 0)
    best_ret   = summary.get("best_return_pct", 0)
    best_yr    = summary.get("best_year", "")
    worst_ret  = summary.get("worst_return_pct", 0)
    worst_yr   = summary.get("worst_year", "")
    losing_yrs = summary.get("losing_years", [])
    avg_mae    = summary.get("avg_mae_pct")
    avg_mfe    = summary.get("avg_mfe_pct")

    lines = [
        "You are a quantitative analyst assistant. Interpret the following historical seasonal",
        "analysis data for a swing trader. Write 3-4 concise sentences in plain English.",
        "Do NOT give buy/sell recommendations. Focus on pattern strength, caveats (small sample",
        "size if < 10 years), and any conditions that seem to influence outcomes.",
        "Do not repeat the numbers verbatim — synthesise them into insight.",
        "",
        "--- ANALYSIS DATA ---",
        f"Stock: {sym_name}",
        f"Seasonal window: {win_label}",
        f"Years of data: {total}",
        f"Win rate: {win_rate:.0f}%",
        f"Average return: {avg_ret:+.2f}%",
        f"Median return: {median_ret:+.2f}%",
        f"Best year: {best_yr} ({best_ret:+.2f}%)",
        f"Worst year: {worst_yr} ({worst_ret:+.2f}%)",
    ]

    if losing_yrs:
        lines.append(f"Losing years: {', '.join(str(y) for y in losing_yrs)}")
    if avg_mae is not None:
        lines.append(f"Average intraday drawdown from entry (MAE): {avg_mae:+.2f}%")
    if avg_mfe is not None:
        lines.append(f"Average intraday peak from entry (MFE): {avg_mfe:+.2f}%")

    if today_features:
        lines.append("")
        lines.append("Current market conditions at today's comparable entry date:")
        lines.append(f"  India VIX: {today_features.get('vix_level', '—')}")
        nifty_mom = today_features.get("nifty_mom_20d")
        if nifty_mom is not None:
            lines.append(f"  NIFTY 20-day momentum: {nifty_mom:+.2f}%")
        nifty_dma = today_features.get("nifty_200dma_dist")
        if nifty_dma is not None:
            lines.append(f"  NIFTY vs 200-DMA: {nifty_dma:+.2f}%")
        rsi = today_features.get("stock_rsi14")
        if rsi is not None:
            lines.append(f"  Stock RSI-14: {rsi:.1f}")
        dma = today_features.get("stock_200dma_dist")
        if dma is not None:
            lines.append(f"  Stock vs 200-DMA: {dma:+.2f}%")

    if similar_years:
        lines.append("")
        top = similar_years[:3]
        sim_desc = ", ".join(
            f"{r.get('Year')} (returned {r.get('Final Return %', 0):+.1f}%)"
            for r in top
        )
        lines.append(f"Most similar historical years to today: {sim_desc}")

    lines.append("")
    lines.append("--- YOUR COMMENTARY (3-4 sentences) ---")
    return "\n".join(lines)


def stream_commentary(prompt: str) -> Generator[str, None, None]:
    """
    Stream Mistral commentary from local Ollama via direct REST API call.
    Bypasses corporate proxies by setting proxies={"http": None, "https": None}.
    Yields text chunks as they arrive (for use with st.write_stream).

    Raises RuntimeError if Ollama is not reachable.
    """
    available, err = _check_ollama()
    if not available:
        raise RuntimeError(
            f"Ollama is not running or not reachable at {OLLAMA_HOST}.\n"
            f"Start it with: ollama serve\nDetail: {err}"
        )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    with requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json=payload,
        proxies=_NO_PROXY,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if raw_line:
                try:
                    data    = json.loads(raw_line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    pass


def build_seasonal_prompt(
    sym_name: str,
    win_label: str,
    summary: dict,
    today_features: dict | None = None,
    similar_years: list[dict] | None = None,
) -> str:
    """
    Build the prompt sent to Mistral for seasonal analysis commentary.

    Parameters
    ----------
    sym_name        : human-readable stock name
    win_label       : e.g. "01-Apr → 90 days"
    summary         : dict from seasonal_analysis() (win_rate, avg_return, etc.)
    today_features  : dict from similar_years_analysis() today_features key (optional)
    similar_years   : list of dicts for top similar years (optional)
    """
    total      = summary.get("total_instances", 0)
    win_rate   = summary.get("win_rate_pct", 0)
    avg_ret    = summary.get("avg_return_pct", 0)
    median_ret = summary.get("median_return_pct", 0)
    best_ret   = summary.get("best_return_pct", 0)
    best_yr    = summary.get("best_year", "")
    worst_ret  = summary.get("worst_return_pct", 0)
    worst_yr   = summary.get("worst_year", "")
    losing_yrs = summary.get("losing_years", [])
    avg_mae    = summary.get("avg_mae_pct")
    avg_mfe    = summary.get("avg_mfe_pct")

    lines = [
        "You are a quantitative analyst assistant. Interpret the following historical seasonal",
        "analysis data for a swing trader. Write 3-4 concise sentences in plain English.",
        "Do NOT give buy/sell recommendations. Focus on pattern strength, caveats (small sample",
        "size if < 10 years), and any conditions that seem to influence outcomes.",
        "Do not repeat the numbers verbatim — synthesise them into insight.",
        "",
        "--- ANALYSIS DATA ---",
        f"Stock: {sym_name}",
        f"Seasonal window: {win_label}",
        f"Years of data: {total}",
        f"Win rate: {win_rate:.0f}%",
        f"Average return: {avg_ret:+.2f}%",
        f"Median return: {median_ret:+.2f}%",
        f"Best year: {best_yr} ({best_ret:+.2f}%)",
        f"Worst year: {worst_yr} ({worst_ret:+.2f}%)",
    ]

    if losing_yrs:
        lines.append(f"Losing years: {', '.join(str(y) for y in losing_yrs)}")

    if avg_mae is not None:
        lines.append(f"Average intraday drawdown from entry (MAE): {avg_mae:+.2f}%")
    if avg_mfe is not None:
        lines.append(f"Average intraday peak from entry (MFE): {avg_mfe:+.2f}%")

    if today_features:
        lines.append("")
        lines.append("Current market conditions at today's comparable entry date:")
        lines.append(f"  India VIX: {today_features.get('vix_level', '—')}")
        nifty_mom = today_features.get('nifty_mom_20d')
        if nifty_mom is not None:
            lines.append(f"  NIFTY 20-day momentum: {nifty_mom:+.2f}%")
        nifty_dma = today_features.get('nifty_200dma_dist')
        if nifty_dma is not None:
            lines.append(f"  NIFTY vs 200-DMA: {nifty_dma:+.2f}%")
        rsi = today_features.get('stock_rsi14')
        if rsi is not None:
            lines.append(f"  Stock RSI-14: {rsi:.1f}")
        dma = today_features.get('stock_200dma_dist')
        if dma is not None:
            lines.append(f"  Stock vs 200-DMA: {dma:+.2f}%")

    if similar_years:
        lines.append("")
        top = similar_years[:3]
        sim_desc = ", ".join(
            f"{r.get('Year')} (returned {r.get('Final Return %', 0):+.1f}%)"
            for r in top
        )
        lines.append(f"Most similar historical years to today: {sim_desc}")

    lines.append("")
    lines.append("--- YOUR COMMENTARY (3-4 sentences) ---")

    return "\n".join(lines)


def stream_commentary(prompt: str) -> Generator[str, None, None]:
    """
    Stream Mistral commentary from local Ollama.
    Yields text chunks as they arrive (for use with st.write_stream).

    Raises RuntimeError if Ollama is not reachable.
    """
    import ollama

    available, err = _check_ollama()
    if not available:
        raise RuntimeError(
            f"Ollama is not running or not reachable at localhost:11434.\n"
            f"Start it with: ollama serve\nDetail: {err}"
        )

    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content

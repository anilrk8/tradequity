"""
NSE Swing Trading Assistant — Streamlit app entry point.

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import datetime

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st

from src.analyzer import (
    best_windows_for_stock, seasonal_analysis, sector_seasonal_analysis, universe_screener,
    monthly_return_heatmap, excess_return_vs_nifty, sector_rotation_analysis, mae_analysis,
)
from src.fetcher import (
    bulk_download, get_data_status, has_any_data,
    fetch_indices, get_index_status, has_index_data, INDEX_TICKERS,
)
from src.universe import get_sectors, get_stocks, get_symbol_to_name
from src.db import init_db

# ─── Constants ────────────────────────────────────────────────────────────────

UNIVERSE = "NIFTY500"

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE Swing Trader Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; }
        div[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 600; }
        div[data-testid="stMetricDelta"] svg { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Shared UI helpers ────────────────────────────────────────────────────────

def _stock_selector(key_prefix: str = "") -> tuple[str, str]:
    stocks = get_stocks(UNIVERSE)
    display_opts = [f"{s['name']}  ({s['symbol']})" for s in stocks]
    sym_map      = {f"{s['name']}  ({s['symbol']})": s["symbol"] for s in stocks}
    choice = st.selectbox("Select Stock", display_opts, key=f"{key_prefix}_stock")
    return choice, sym_map[choice]


def _entry_inputs(key_prefix: str = "") -> tuple[int, int, int, float]:
    """
    Returns (start_month, start_day, holding_days, min_return_pct).
    Uses a date_input for a calendar picker; user picks any date and we
    extract month + day (year is ignored — it's a seasonal window).
    """
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        default_date = datetime.date(datetime.date.today().year, 4, 1)
        picked = st.date_input(
            "Entry Date (month & day matter; year ignored)",
            value=default_date,
            format="DD/MM/YYYY",
            key=f"{key_prefix}_date",
        )
        start_month = picked.month
        start_day   = picked.day

    with c2:
        holding_days = st.number_input(
            "Holding Period (calendar days)",
            min_value=5,
            max_value=365,
            value=90,
            step=1,
            key=f"{key_prefix}_hold",
            help="Number of calendar days to hold from entry date.",
        )

    with c3:
        min_return = st.number_input(
            "Target Return %  (min)",
            min_value=0.0,
            max_value=200.0,
            value=12.0,
            step=0.5,
            format="%.1f",
            key=f"{key_prefix}_ret",
            help="Only count years where the stock returned at least this much.",
        )

    return start_month, start_day, int(holding_days), float(min_return)


def _window_label(sm: int, sd: int, days: int) -> str:
    abbr = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return f"{abbr[sm-1]} {sd}  +{days} days"


def _no_data_warning():
    st.warning(
        "No data found in the database. "
        "Go to the **Data Management** tab and click **Download / Update Now** first."
    )


def _year_range_filter(results_df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Render a year-range slider and return the filtered DataFrame."""
    all_years = sorted(results_df["year"].unique().tolist())
    if len(all_years) < 2:
        return results_df
    min_y, max_y = int(all_years[0]), int(all_years[-1])
    selected = st.slider(
        "Filter Year Range",
        min_value=min_y,
        max_value=max_y,
        value=(min_y, max_y),
        step=1,
        key=key,
    )
    return results_df[(results_df["year"] >= selected[0]) & (results_df["year"] <= selected[1])]


# ─── Chart builders ───────────────────────────────────────────────────────────

def _fan_chart(
    results_df: pd.DataFrame,
    norm_map: dict,
    title: str,
    min_return_pct: float,
) -> go.Figure:
    """
    Readable fan chart replacing the spaghetti overlay.

    Shows P25-P75 band, median line, and best/worst years as dotted lines.
    Much easier to read than 15 individual lines.
    """
    if results_df.empty:
        return go.Figure()

    all_years = sorted(norm_map.keys())
    max_len = max(len(norm_map[y]) for y in all_years)
    matrix = []
    for y in all_years:
        s = norm_map[y].values
        padded = list(s) + [s[-1]] * (max_len - len(s))
        matrix.append(padded)

    mat   = np.array(matrix, dtype=float)
    p25   = np.percentile(mat, 25, axis=0)
    p50   = np.percentile(mat, 50, axis=0)
    p75   = np.percentile(mat, 75, axis=0)

    best_year  = results_df.loc[results_df["return_pct"].idxmax(), "year"]
    worst_year = results_df.loc[results_df["return_pct"].idxmin(), "year"]
    best_vals  = norm_map[best_year].values
    worst_vals = norm_map[worst_year].values
    best_ret   = float(results_df.loc[results_df["year"] == best_year, "return_pct"].iloc[0])
    worst_ret  = float(results_df.loc[results_df["year"] == worst_year, "return_pct"].iloc[0])

    x = list(range(max_len))

    fig = go.Figure()

    # P25-P75 filled band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(p75) + list(p25[::-1]),
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Middle 50% of years (P25–P75)",
        hoverinfo="skip",
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=x, y=p50,
        mode="lines",
        line=dict(color="#2980b9", width=2.5),
        name="Median year",
        hovertemplate="Day %{x}<br>Median index: %{y:.1f}<extra></extra>",
    ))

    # Best year
    fig.add_trace(go.Scatter(
        x=list(range(len(best_vals))), y=best_vals,
        mode="lines",
        line=dict(color="#27ae60", width=1.8, dash="dot"),
        name=f"Best: {best_year} ({best_ret:+.1f}%)",
        hovertemplate=f"<b>{best_year}</b><br>Day %{{x}}<br>%{{y:.1f}}<extra></extra>",
    ))

    # Worst year
    fig.add_trace(go.Scatter(
        x=list(range(len(worst_vals))), y=worst_vals,
        mode="lines",
        line=dict(color="#e74c3c", width=1.8, dash="dot"),
        name=f"Worst: {worst_year} ({worst_ret:+.1f}%)",
        hovertemplate=f"<b>{worst_year}</b><br>Day %{{x}}<br>%{{y:.1f}}<extra></extra>",
    ))

    if min_return_pct > 0:
        fig.add_hline(
            y=100 + min_return_pct,
            line_dash="dash",
            line_color="#f39c12",
            opacity=0.85,
            annotation_text=f"Target {min_return_pct:+.0f}%",
            annotation_position="top right",
        )

    fig.add_hline(y=100, line_dash="dot", line_color="#777", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Trading Days into Window",
        yaxis_title="Indexed Price (Entry = 100)",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=80),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#efefef")
    fig.update_yaxes(showgrid=True, gridcolor="#efefef")
    return fig


def _bar_chart(results_df: pd.DataFrame, min_return_pct: float, title: str) -> go.Figure:
    """Bar chart coloured by target: green = met, orange = positive but below, red = negative."""
    colors = [
        "#27ae60" if r >= min_return_pct else ("#e67e22" if r >= 0 else "#e74c3c")
        for r in results_df["return_pct"]
    ]

    fig = go.Figure(go.Bar(
        x=results_df["year"].astype(str),
        y=results_df["return_pct"],
        marker_color=colors,
        text=[f"{r:+.1f}%" for r in results_df["return_pct"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>",
        name="",
        showlegend=False,
    ))

    if min_return_pct > 0:
        fig.add_hline(
            y=min_return_pct,
            line_dash="dash",
            line_color="#f39c12",
            opacity=0.85,
            annotation_text=f"Target {min_return_pct:+.0f}%",
            annotation_position="top right",
        )
    fig.add_hline(y=0, line_color="#333", line_width=1)

    # Legend indicators
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#27ae60", name="Target met"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#e67e22", name="Positive, below target"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#e74c3c", name="Negative"))

    fig.update_layout(
        title=title,
        xaxis_title="Year", yaxis_title="Return (%)",
        height=410,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickangle=-45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        showlegend=True,
        barmode="relative",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#efefef")
    return fig


def _metric_row(summary: dict):
    """Render the key metric tiles."""
    tgt     = summary["min_return_pct"]
    avg_ret = summary["avg_return_pct"]
    avg_met = summary["avg_return_when_met"]
    never   = summary["target_never_met"]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        f"Target ≥{tgt:.0f}% Met",
        summary["target_met_label"],
        help="Number of years the stock returned at or above your target "
             "out of all historical instances.",
    )
    c2.metric("Avg Return (all years)", f"{avg_ret:+.2f}%")

    if never:
        c3.metric(f"Avg when ≥{tgt:.0f}%", "Never met")
    else:
        c3.metric(
            f"Avg when ≥{tgt:.0f}%",
            f"{avg_met:+.2f}%" if avg_met is not None else "—",
        )

    c4.metric(
        "Best Year",
        str(summary["best_year"]),
        delta=f"{summary['best_return_pct']:+.1f}%",
    )
    c5.metric(
        "Worst Year",
        str(summary["worst_year"]),
        delta=f"{summary['worst_return_pct']:+.1f}%",
    )


# ─── Tab 1 — Stock Analysis ───────────────────────────────────────────────────

def tab_stock_analysis():
    st.subheader("Stock Seasonal Analysis")
    st.caption(
        "Pick a stock, an entry date (month & day — year is irrelevant for seasonality), "
        "how many days you plan to hold, and your target return. "
        "See how the stock has performed in that exact window across all historical years."
    )

    col_s, col_w = st.columns([1, 3])
    with col_s:
        _, symbol = _stock_selector("sa")
    with col_w:
        sm, sd, holding_days, min_return = _entry_inputs("sa")

    if st.button("Analyse →", type="primary", key="sa_go"):
        with st.spinner("Running analysis…"):
            results_df, summary = seasonal_analysis(symbol, sm, sd, holding_days, min_return)

        if results_df is None:
            _no_data_warning()
            return

        st.session_state["sa_results_df"] = results_df
        st.session_state["sa_summary"]    = summary
        st.session_state["_res_sa_symbol"]  = symbol
        st.session_state["_res_sa_sm"]      = sm
        st.session_state["_res_sa_sd"]      = sd
        st.session_state["_res_sa_hold"]    = holding_days
        st.session_state["_res_sa_minret"]  = min_return

    # Render from session state so slider / tab interactions don't lose the results
    if "sa_results_df" not in st.session_state:
        return

    results_df  = st.session_state["sa_results_df"]
    summary     = dict(st.session_state["sa_summary"])   # copy so pop doesn't corrupt state
    symbol      = st.session_state["_res_sa_symbol"]
    sm          = st.session_state["_res_sa_sm"]
    sd          = st.session_state["_res_sa_sd"]
    holding_days = st.session_state["_res_sa_hold"]
    min_return  = st.session_state["_res_sa_minret"]

    norm_map  = summary.pop("norm_series_map")
    sym_name  = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
    win_label = _window_label(sm, sd, holding_days)

    st.divider()
    st.markdown(f"### {sym_name} &nbsp;·&nbsp; {win_label}")

    if summary["target_never_met"]:
        st.error(
            f"⚠  The target of **{min_return:.0f}%** was **never** reached in this window "
            f"across all {summary['total_instances']} historical years. "
            f"The average return in this window was **{summary['avg_return_pct']:+.2f}%**."
        )
    else:
        st.caption(
            f"Historical window across **{summary['total_instances']} years**  ·  "
            f"Target ≥{min_return:.0f}% met in **{summary['target_met_label']}**"
        )

    _metric_row(summary)
    st.divider()

    t_fan, t_bar, t_table = st.tabs(
        ["📈 Price Trend Chart", "📊 Year-by-Year Returns", "🗃 Raw Data Table"]
    )

    with t_fan:
        st.caption(
            "**Blue band** = middle 50% of all years (P25–P75).  "
            "**Blue line** = median year.  "
            "**Green dotted** = best year.  "
            "**Red dotted** = worst year.  "
            "**Orange dashed** = your return target."
        )
        fig = _fan_chart(
            results_df, norm_map,
            f"{sym_name} — Price Trend  ·  {win_label}",
            min_return,
        )
        st.plotly_chart(fig, use_container_width=True)

    with t_bar:
        st.caption("Green = target met · Orange = positive but below target · Red = negative")
        filtered_bar = _year_range_filter(results_df, key="sa_bar_yr")
        fig = _bar_chart(filtered_bar, min_return, f"{sym_name} — Annual Returns")
        st.plotly_chart(fig, use_container_width=True)

    with t_table:
        display = results_df.copy()
        display["return_pct"]  = display["return_pct"].apply(lambda x: f"{x:+.2f}%")
        display["target_met"]  = display["target_met"].map({True: "✓ Yes", False: "✗ No"})
        st.dataframe(display, use_container_width=True)


# ─── Tab 2 — Sector Analysis ──────────────────────────────────────────────────

def tab_sector_analysis():
    st.subheader("Sector Seasonal Analysis")
    st.caption(
        "Rank all stocks in a sector by the number of years they met your return target "
        "during a given entry window."
    )

    sectors = get_sectors(UNIVERSE)
    sector_names = sorted(sectors.keys())

    col_s, col_w = st.columns([1, 3])
    with col_s:
        chosen_sector = st.selectbox("Select Sector", sector_names, key="sec_select")
    with col_w:
        sm, sd, holding_days, min_return = _entry_inputs("sec")

    if st.button("Analyse Sector →", type="primary", key="sec_go"):
        with st.spinner(f"Analysing {chosen_sector}…"):
            result_df = sector_seasonal_analysis(
                chosen_sector, sm, sd, holding_days, min_return, UNIVERSE
            )
        if result_df is None:
            _no_data_warning()
        else:
            st.session_state["sec_result_df"]  = result_df
            st.session_state["_res_sec_sector"] = chosen_sector
            st.session_state["_res_sec_sm"]     = sm
            st.session_state["_res_sec_sd"]     = sd
            st.session_state["_res_sec_hold"]   = holding_days
            st.session_state["_res_sec_minret"] = min_return

    if "sec_result_df" not in st.session_state:
        return

    result_df    = st.session_state["sec_result_df"]
    chosen_sector = st.session_state["_res_sec_sector"]
    sm           = st.session_state["_res_sec_sm"]
    sd           = st.session_state["_res_sec_sd"]
    holding_days = st.session_state["_res_sec_hold"]
    min_return   = st.session_state["_res_sec_minret"]

    win_label = _window_label(sm, sd, holding_days)
    st.divider()
    st.markdown(
        f"### {chosen_sector} Sector &nbsp;·&nbsp; {win_label}  ·  Target ≥{min_return:.0f}%"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks Analysed", len(result_df))
    c2.metric("Avg Sector Return", f"{result_df['Avg Return %'].mean():+.2f}%")
    best_row = result_df.iloc[0]
    c3.metric(
        "Top Stock (by target met)",
        best_row["Symbol"],
        delta=f"{best_row['Target Met (yrs)']} of {best_row['Out of (yrs)']} yrs",
    )
    avg_met_col = result_df["Avg When Target Met"].dropna()
    c4.metric(
        f"Avg return when ≥{min_return:.0f}%",
        f"{avg_met_col.mean():+.2f}%" if len(avg_met_col) > 0 else "—",
    )

    st.divider()

    st.dataframe(
        result_df.style
            .background_gradient(
                subset=["Target Met (yrs)"], cmap="RdYlGn",
                vmin=0, vmax=result_df["Out of (yrs)"].max()
            )
            .background_gradient(subset=["Avg Return %"], cmap="RdYlGn", vmin=-20, vmax=20)
            .format({
                "Avg Return %":        "{:+.2f}%",
                "Avg When Target Met": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                "Best Return %":       "{:+.2f}%",
                "Worst Return %":      "{:+.2f}%",
            }),
        use_container_width=True,
    )

    fig = px.bar(
        result_df, x="Symbol", y="Target Met (yrs)",
        color="Target Met (yrs)",
        color_continuous_scale="RdYlGn",
        range_color=[0, result_df["Out of (yrs)"].max()],
        title=f"{chosen_sector} — Years Target ≥{min_return:.0f}% Met  ·  {win_label}",
        text="Target Met (yrs)",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


# ─── Tab 3 — Best Windows ─────────────────────────────────────────────────────

def _render_stock_best_windows():
    """Sub-tab: pick a stock, see its best entry windows."""
    st.caption(
        "Pick a stock and see which calendar entry month has historically "
        "produced the most years meeting your return target."
    )
    col_s, col_h, col_r = st.columns([2, 1, 1])
    with col_s:
        _, symbol = _stock_selector("bw")
    with col_h:
        holding_days = st.number_input(
            "Holding Period (calendar days)", min_value=5, max_value=365,
            value=90, step=1, key="bw_hold",
        )
    with col_r:
        min_return = st.number_input(
            "Target Return % (min)", min_value=0.0, max_value=200.0,
            value=12.0, step=0.5, format="%.1f", key="bw_ret",
        )

    if st.button("Scan All Windows →", type="primary", key="bw_go"):
        sym_name = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
        with st.spinner(f"Scanning {sym_name}…"):
            result_df = best_windows_for_stock(symbol, int(holding_days), float(min_return))
        if result_df is None:
            _no_data_warning()
        else:
            st.session_state["bw_result_df"]   = result_df
            st.session_state["_res_bw_symbol"] = symbol
            st.session_state["_res_bw_hold"]   = holding_days
            st.session_state["_res_bw_minret"] = min_return

    if "bw_result_df" not in st.session_state:
        return

    result_df    = st.session_state["bw_result_df"]
    symbol       = st.session_state["_res_bw_symbol"]
    holding_days = st.session_state["_res_bw_hold"]
    min_return   = st.session_state["_res_bw_minret"]
    sym_name     = get_symbol_to_name(UNIVERSE).get(symbol, symbol)

    st.divider()
    st.markdown(
        f"### {sym_name} — Best Windows  ·  +{holding_days} days hold  ·  "
        f"Target ≥{min_return:.0f}%"
    )

    top3 = result_df.head(3)
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
        col.metric(
            f"{medals[i]} {row['Window']}",
            f"{row['Target Met (yrs)']} of {row['Out of (yrs)']} years",
            delta=f"Avg {row['Avg Return %']:+.2f}%",
        )

    st.divider()

    st.dataframe(
        result_df.style
            .background_gradient(
                subset=["Target Met (yrs)"], cmap="RdYlGn",
                vmin=0, vmax=result_df["Out of (yrs)"].max()
            )
            .background_gradient(subset=["Avg Return %"], cmap="RdYlGn", vmin=-20, vmax=20)
            .format({
                "Avg Return %":   "{:+.2f}%",
                "Best Return %":  "{:+.2f}%",
                "Worst Return %": "{:+.2f}%",
            }),
        use_container_width=True,
    )

    fig = px.bar(
        result_df, x="Window", y="Target Met (yrs)",
        color="Target Met (yrs)",
        color_continuous_scale="RdYlGn",
        range_color=[0, result_df["Out of (yrs)"].max()],
        title=f"{sym_name} — Years Target ≥{min_return:.0f}% Met, by Entry Month",
        text="Target Met (yrs)",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(
        height=420, xaxis_tickangle=-30,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_window_best_stocks():
    """Sub-tab: pick a window, see ALL stocks ranked for that window."""
    st.caption(
        "Fix a calendar entry window and see every stock in the universe ranked "
        "by how many years it met your return target in that window."
    )
    sm, sd, holding_days, min_return = _entry_inputs("ws")

    if st.button("Screen All Stocks →", type="primary", key="ws_go"):
        win_label = _window_label(sm, sd, holding_days)
        with st.spinner(f"Scanning all {UNIVERSE} stocks for {win_label}…"):
            result_df = universe_screener(sm, sd, holding_days, min_return, UNIVERSE)
        if result_df is None:
            _no_data_warning()
        else:
            st.session_state["ws_result_df"]   = result_df
            st.session_state["_res_ws_sm"]     = sm
            st.session_state["_res_ws_sd"]     = sd
            st.session_state["_res_ws_hold"]   = holding_days
            st.session_state["_res_ws_minret"] = min_return

    if "ws_result_df" not in st.session_state:
        return

    result_df    = st.session_state["ws_result_df"]
    sm           = st.session_state["_res_ws_sm"]
    sd           = st.session_state["_res_ws_sd"]
    holding_days = st.session_state["_res_ws_hold"]
    min_return   = st.session_state["_res_ws_minret"]
    win_label    = _window_label(sm, sd, holding_days)

    st.divider()
    st.markdown(f"### All Stocks · {win_label}  ·  Target ≥{min_return:.0f}%")

    max_met = result_df["Out of (yrs)"].max()
    top3 = result_df.head(3)
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
        col.metric(
            f"{medals[i]} {row['Name']}  ({row['Symbol']})",
            f"{row['Target Met (yrs)']} of {row['Out of (yrs)']} years met",
            delta=f"Avg {row['Avg Return %']:+.2f}%",
        )

    st.divider()

    all_sectors = sorted(result_df["Sector"].unique().tolist())
    chosen = st.multiselect(
        "Filter by Sector (leave empty = show all)",
        options=all_sectors,
        default=[],
        key="ws_sector_filter",
    )
    display_df = result_df[result_df["Sector"].isin(chosen)] if chosen else result_df

    st.dataframe(
        display_df.style
            .background_gradient(
                subset=["Target Met (yrs)"], cmap="RdYlGn",
                vmin=0, vmax=max_met
            )
            .background_gradient(
                subset=["Avg Return %"], cmap="RdYlGn", vmin=-20, vmax=20
            )
            .format({
                "Avg Return %":        "{:+.2f}%",
                "Avg When Target Met": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                "Best Return %":       "{:+.2f}%",
                "Worst Return %":      "{:+.2f}%",
            }),
        use_container_width=True,
        height=540,
    )

    fig = px.bar(
        display_df, x="Symbol", y="Target Met (yrs)",
        color="Target Met (yrs)",
        color_continuous_scale="RdYlGn",
        range_color=[0, max_met],
        hover_data=["Name", "Sector", "Avg Return %"],
        title=f"All Stocks — Years Target ≥{min_return:.0f}% Met  ·  {win_label}",
        text="Target Met (yrs)",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(
        height=460, xaxis_tickangle=-45,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def tab_best_windows():
    sub1, sub2 = st.tabs(["Stock → Best Entry Windows", "Window → Best Stocks"])
    with sub1:
        _render_stock_best_windows()
    with sub2:
        _render_window_best_stocks()


# ─── Tab 4 — Deep Insights ────────────────────────────────────────────────────

def _render_monthly_heatmap():
    st.caption(
        "Shows the stock's return for every calendar month across all years in the database. "
        "Green = positive month, Red = negative. Spot months that are consistently green."
    )
    _, symbol = _stock_selector("mh")

    if st.button("Show Monthly Heatmap →", type="primary", key="mh_go"):
        sym_name = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
        with st.spinner(f"Computing heatmap for {sym_name}…"):
            pivot = monthly_return_heatmap(symbol)
        if pivot is None:
            _no_data_warning()
        else:
            st.session_state["mh_pivot"]        = pivot
            st.session_state["_res_mh_symbol"]  = symbol

    if "mh_pivot" not in st.session_state:
        return

    pivot    = st.session_state["mh_pivot"]
    symbol   = st.session_state["_res_mh_symbol"]
    sym_name = get_symbol_to_name(UNIVERSE).get(symbol, symbol)

    st.divider()
    st.markdown(f"### {sym_name} — Monthly Return Heatmap")

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        aspect="auto",
        text_auto=".1f",
        title=f"{sym_name} — Monthly Return % (each cell = that month's closing return)",
        labels={"x": "Month", "y": "Year", "color": "Return %"},
    )
    fig.update_layout(
        height=max(420, len(pivot) * 25 + 120),
        paper_bgcolor="white",
        coloraxis_colorbar=dict(title="Return %"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Average Monthly Return  (all years combined)")
    avg_by_month = pivot.mean(axis=0).reset_index()
    avg_by_month.columns = ["Month", "Avg Return %"]
    avg_by_month["Avg Return %"] = avg_by_month["Avg Return %"].round(2)

    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in avg_by_month["Avg Return %"]]
    fig2 = go.Figure(go.Bar(
        x=avg_by_month["Month"],
        y=avg_by_month["Avg Return %"],
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in avg_by_month["Avg Return %"]],
        textposition="outside",
    ))
    fig2.add_hline(y=0, line_color="#333", line_width=1)
    fig2.update_layout(
        title=f"{sym_name} — Avg Return by Month (across all years)",
        xaxis_title="Month", yaxis_title="Avg Return %",
        height=380, plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig2, use_container_width=True)

    total_years = len(pivot)
    pos_count = (pivot > 0).sum(axis=0).reset_index()
    pos_count.columns = ["Month", "Positive Years"]
    st.markdown(f"#### Consistency — how often each month closes positive  ({total_years} years total)")
    c1, c2 = st.columns([3, 1])
    with c1:
        fig3 = go.Figure(go.Bar(
            x=pos_count["Month"],
            y=pos_count["Positive Years"],
            marker_color=[
                "#27ae60" if v >= total_years * 0.6
                else ("#e67e22" if v >= total_years * 0.4 else "#e74c3c")
                for v in pos_count["Positive Years"]
            ],
            text=[f"{v}/{total_years}" for v in pos_count["Positive Years"]],
            textposition="outside",
        ))
        fig3.update_layout(
            title="Positive closes per month",
            height=340, plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        st.dataframe(
            pos_count.assign(**{
                "% Positive": (pos_count["Positive Years"] / total_years * 100).round(1)
            }),
            use_container_width=True,
        )


def _render_excess_vs_nifty():
    st.caption(
        "Is the stock's seasonal edge genuine alpha, or just market tailwind? "
        "Compares the stock's return against NIFTY 50 over the same exact dates each year. "
        "India VIX at entry is overlaid to flag elevated-risk environments."
    )

    col_s, col_w = st.columns([1, 3])
    with col_s:
        _, symbol = _stock_selector("ev")
    with col_w:
        sm, sd, holding_days, min_return = _entry_inputs("ev")

    if not has_index_data():
        st.warning(
            "Market index data (NIFTY, VIX) not yet downloaded.  \n"
            "Go to **Data Management → Market Indices & Macro** and download first.",
            icon="⚠",
        )

    if st.button("Compare vs NIFTY →", type="primary", key="ev_go"):
        sym_name  = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
        win_label = _window_label(sm, sd, holding_days)
        with st.spinner(f"Comparing {sym_name} vs NIFTY for {win_label}…"):
            results_df, summary = excess_return_vs_nifty(symbol, sm, sd, holding_days, min_return)
        if results_df is None:
            _no_data_warning()
        else:
            st.session_state["ev_results_df"]  = results_df
            st.session_state["ev_summary"]     = summary
            st.session_state["_res_ev_symbol"] = symbol
            st.session_state["_res_ev_sm"]     = sm
            st.session_state["_res_ev_sd"]     = sd
            st.session_state["_res_ev_hold"]   = holding_days
            st.session_state["_res_ev_minret"] = min_return

    if "ev_results_df" not in st.session_state:
        return

    results_df   = st.session_state["ev_results_df"]
    summary      = st.session_state["ev_summary"]
    symbol       = st.session_state["_res_ev_symbol"]
    sm           = st.session_state["_res_ev_sm"]
    sd           = st.session_state["_res_ev_sd"]
    holding_days = st.session_state["_res_ev_hold"]
    min_return   = st.session_state["_res_ev_minret"]
    sym_name     = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
    win_label    = _window_label(sm, sd, holding_days)

    st.divider()
    st.markdown(f"### {sym_name} vs NIFTY 50  ·  {win_label}")

    n_avail = summary["nifty_available"]
    v_avail = summary["vix_available"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Stock Return", f"{summary['avg_stock_return']:+.2f}%")
    if n_avail:
        c2.metric("Avg NIFTY Return",  f"{summary['avg_nifty_return']:+.2f}%")
        c3.metric(
            "Avg Excess Return",
            f"{summary['avg_excess_return']:+.2f}%" if summary["avg_excess_return"] is not None else "—",
            help="Stock return minus NIFTY return over the same window",
        )
        c4.metric("Beat NIFTY", summary["beat_index_label"])
    else:
        c2.metric("NIFTY Data", "Not downloaded")
        c3.metric("→ See", "Data Management")
        c4.metric("", "")

    st.divider()

    if n_avail:
        # Grouped bar: stock vs NIFTY per year
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=results_df["year"].astype(str),
            y=results_df["stock_return"],
            name=sym_name,
            marker_color="#2980b9",
            offsetgroup=0,
        ))
        fig.add_trace(go.Bar(
            x=results_df["year"].astype(str),
            y=results_df["nifty_return"],
            name="NIFTY 50",
            marker_color="#bdc3c7",
            offsetgroup=1,
        ))
        fig.add_hline(y=0, line_color="#333", line_width=1)
        fig.update_layout(
            title=f"{sym_name} vs NIFTY 50 — Annual Window Returns  ·  {win_label}",
            barmode="group",
            xaxis_title="Year", yaxis_title="Return %",
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Excess return bar
        exc = results_df.dropna(subset=["excess_return"])
        exc_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in exc["excess_return"]]
        fig2 = go.Figure(go.Bar(
            x=exc["year"].astype(str),
            y=exc["excess_return"],
            marker_color=exc_colors,
            text=[f"{v:+.1f}%" for v in exc["excess_return"]],
            textposition="outside",
        ))
        fig2.add_hline(y=0, line_color="#333", line_width=1)
        if summary["avg_excess_return"] is not None:
            fig2.add_hline(
                y=summary["avg_excess_return"],
                line_dash="dash", line_color="#9b59b6",
                annotation_text=f"Avg excess: {summary['avg_excess_return']:+.1f}%",
                annotation_position="top right",
            )
        fig2.update_layout(
            title=f"Excess Return (Stock − NIFTY)  ·  {win_label}",
            xaxis_title="Year", yaxis_title="Excess Return %",
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # VIX overlay
    if v_avail and results_df["vix_at_entry"].notna().sum() >= 3:
        st.divider()
        st.markdown("#### India VIX at Entry vs Outcome")
        st.caption(
            "Each dot = one year.  X = VIX when you would have entered.  "
            "Y = stock return.  High VIX (>20) is elevated-risk territory."
        )
        vix_data = results_df.dropna(subset=["vix_at_entry"]).copy()
        fig3 = px.scatter(
            vix_data,
            x="vix_at_entry",
            y="stock_return",
            text="year",
            color="stock_return",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title=f"VIX at Entry vs Stock Return  ·  {win_label}",
            labels={"vix_at_entry": "India VIX at Entry", "stock_return": "Stock Return %"},
        )
        fig3.update_traces(textposition="top center", marker_size=10)
        fig3.add_vline(
            x=20, line_dash="dash", line_color="#e74c3c",
            annotation_text="VIX = 20  (elevated threshold)",
            annotation_position="top right",
        )
        fig3.update_layout(height=430, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

        # VIX regime summary
        vix_data["vix_regime"] = pd.cut(
            vix_data["vix_at_entry"],
            bins=[0, 14, 20, 100],
            labels=["Calm  (<14)", "Normal  (14–20)", "Elevated  (>20)"],
        )
        regime_tbl = (
            vix_data.groupby("vix_regime", observed=True)["stock_return"]
            .agg(Count="count", Avg_Return="mean", Min="min", Max="max")
            .round(2)
        )
        regime_tbl.columns = ["Count", "Avg Return %", "Min %", "Max %"]
        st.markdown("**Average outcome by VIX regime at entry**")
        st.dataframe(regime_tbl, use_container_width=True)


def _render_sector_rotation():
    st.caption(
        "Which sectors are structurally strong in which months? "
        "Average monthly return for each sector — computed across all constituent stocks "
        "and all years in the database.  Takes ~20-30 seconds."
    )

    if st.button("Compute Sector Rotation →", type="primary", key="sr_go"):
        with st.spinner("Scanning all sectors across all years — please wait…"):
            pivot = sector_rotation_analysis(UNIVERSE)
        if pivot is None:
            _no_data_warning()
        else:
            st.session_state["sr_pivot"] = pivot

    if "sr_pivot" not in st.session_state:
        return

    pivot = st.session_state["sr_pivot"]

    st.divider()
    st.markdown("### Sector Rotation — Monthly Return Heatmap")
    st.caption(
        "Each cell = average return for that sector in that calendar month, "
        "across all constituent stocks and all years in the database."
    )

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        aspect="auto",
        text_auto=".1f",
        title="Average Monthly Return by Sector (%)",
        labels={"x": "Month", "y": "Sector", "color": "Avg Return %"},
    )
    fig.update_layout(
        height=max(480, len(pivot) * 38 + 120),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Quarterly View")
    quarterly = pd.DataFrame({
        "Q1 (Jan–Mar)": pivot[["Jan", "Feb", "Mar"]].mean(axis=1),
        "Q2 (Apr–Jun)": pivot[["Apr", "May", "Jun"]].mean(axis=1),
        "Q3 (Jul–Sep)": pivot[["Jul", "Aug", "Sep"]].mean(axis=1),
        "Q4 (Oct–Dec)": pivot[["Oct", "Nov", "Dec"]].mean(axis=1),
    }).round(2)

    fig2 = px.imshow(
        quarterly,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text_auto=".2f",
        title="Average Quarterly Return by Sector (%)",
        labels={"x": "Quarter", "y": "Sector", "color": "Avg Return %"},
    )
    fig2.update_layout(
        height=max(420, len(quarterly) * 38 + 120),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        pivot.style
            .background_gradient(cmap="RdYlGn", axis=None, vmin=-5, vmax=5)
            .format(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"),
        use_container_width=True,
    )


def _render_mae_analysis():
    st.caption(
        "**Maximum Adverse Excursion (MAE):** how far against you the trade went "
        "before recovering — critical for calibrating stop-loss levels.  "
        "**MFE:** the best intraday peak reached during the window."
    )
    col_s, col_w = st.columns([1, 3])
    with col_s:
        _, symbol = _stock_selector("mae")
    with col_w:
        sm, sd, holding_days, _ = _entry_inputs("mae")

    if st.button("Analyse Stop-Loss →", type="primary", key="mae_go"):
        sym_name  = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
        win_label = _window_label(sm, sd, holding_days)
        with st.spinner(f"Computing MAE/MFE for {sym_name}…"):
            mae_df = mae_analysis(symbol, sm, sd, holding_days)
        if mae_df is None:
            _no_data_warning()
        else:
            st.session_state["mae_df"]          = mae_df
            st.session_state["_res_mae_symbol"] = symbol
            st.session_state["_res_mae_sm"]     = sm
            st.session_state["_res_mae_sd"]     = sd
            st.session_state["_res_mae_hold"]   = holding_days

    if "mae_df" not in st.session_state:
        return

    mae_df       = st.session_state["mae_df"]
    symbol       = st.session_state["_res_mae_symbol"]
    sm           = st.session_state["_res_mae_sm"]
    sd           = st.session_state["_res_mae_sd"]
    holding_days = st.session_state["_res_mae_hold"]
    sym_name     = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
    win_label    = _window_label(sm, sd, holding_days)

    st.divider()
    st.markdown(f"### {sym_name} — MAE & MFE  ·  {win_label}")

    valid_mae = mae_df["MAE % (worst intraday dip)"].dropna()
    valid_mfe = mae_df["MFE % (best intraday peak)"].dropna()

    # Suggested stop: 25th-percentile MAE (most adverse quarter of years) × 1.2 cushion
    suggested_sl = round(float(valid_mae.quantile(0.25)) * 1.2, 1) if len(valid_mae) > 0 else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg MAE",     f"{valid_mae.mean():+.2f}%" if len(valid_mae) > 0 else "—",
              help="Average deepest intraday dip from entry across all years")
    c2.metric("Worst MAE",   f"{valid_mae.min():+.2f}%"  if len(valid_mae) > 0 else "—",
              help="Single worst intraday dip from entry across all history")
    c3.metric("Suggested SL",
              f"{suggested_sl:+.1f}%" if suggested_sl is not None else "—",
              help="25th-pct MAE × 1.2 — tight enough to cut losses, "
                   "wide enough not to get stopped on normal noise")
    c4.metric("Avg MFE",     f"{valid_mfe.mean():+.2f}%" if len(valid_mfe) > 0 else "—",
              help="Average highest intraday peak reached during the window")

    st.divider()

    t1, t2, t3 = st.tabs(["MAE by Year", "MAE vs Final Return", "Raw Data"])

    with t1:
        bar_colors = [
            "#27ae60" if p else "#e74c3c"
            for p in mae_df["Profitable"]
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=mae_df["Year"].astype(str),
            y=mae_df["MAE % (worst intraday dip)"],
            name="MAE (intraday low vs entry)",
            marker_color=bar_colors,
            text=[f"{v:+.1f}%" if pd.notna(v) else "" for v in mae_df["MAE % (worst intraday dip)"]],
            textposition="outside",
        ))
        if len(valid_mae) > 0:
            fig.add_hline(
                y=float(valid_mae.mean()),
                line_dash="dash", line_color="#e67e22",
                annotation_text=f"Avg MAE: {valid_mae.mean():+.1f}%",
                annotation_position="bottom right",
            )
        if suggested_sl is not None:
            fig.add_hline(
                y=suggested_sl,
                line_dash="dot", line_color="#9b59b6",
                annotation_text=f"Suggested SL: {suggested_sl:+.1f}%",
                annotation_position="bottom left",
            )
        fig.update_layout(
            title=f"{sym_name} — MAE per Year  ·  {win_label}"
                  "  (green bar = trade ended profitable)",
            xaxis_title="Year", yaxis_title="MAE %",
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.caption(
            "Each dot = one year.  X = how far the trade went against you (MAE).  "
            "Y = where it ended up.  A cluster in top-left is ideal: small MAE, big gain."
        )
        scatter_df = mae_df.dropna(subset=["MAE % (worst intraday dip)", "Final Return %"])
        fig2 = px.scatter(
            scatter_df,
            x="MAE % (worst intraday dip)",
            y="Final Return %",
            text="Year",
            color="Final Return %",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title=f"{sym_name} — MAE vs Final Return  ·  {win_label}",
        )
        fig2.update_traces(textposition="top center", marker_size=10)
        fig2.add_hline(y=0, line_color="#888", line_dash="dot")
        fig2.add_vline(x=0, line_color="#888", line_dash="dot")
        fig2.update_layout(height=430, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        st.dataframe(
            mae_df.style.format({
                "Final Return %":              "{:+.2f}%",
                "MAE % (worst intraday dip)":  lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                "MFE % (best intraday peak)":  lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                "Risk/Reward (Final÷|MAE|)":   lambda x: f"{x:.2f}"   if pd.notna(x) else "—",
            }),
            use_container_width=True,
        )


def tab_deep_insights():
    st.subheader("Deep Insights")
    st.caption(
        "Advanced analysis built on top of the historical OHLCV data. "
        "Download market indices first (Data Management tab) to unlock "
        "Excess vs NIFTY and the VIX overlay."
    )
    sub1, sub2, sub3, sub4 = st.tabs([
        "📅 Monthly Heatmap",
        "📊 Excess vs NIFTY",
        "🔄 Sector Rotation",
        "🛡 MAE & Stop-Loss",
    ])
    with sub1:
        _render_monthly_heatmap()
    with sub2:
        _render_excess_vs_nifty()
    with sub3:
        _render_sector_rotation()
    with sub4:
        _render_mae_analysis()


# ─── Tab 5 — Data Management ──────────────────────────────────────────────────

def tab_data_management():
    st.subheader("Data Management")
    st.caption(
        "First run: downloads ~15 years of daily OHLCV for all Nifty 50 stocks "
        "(takes roughly 5-10 minutes). Every run after that is incremental — "
        "only missing days are fetched, which takes seconds."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Download / Update")
        st.info(
            "Click the button below any time to bring all data up to today.  \n"
            "The system automatically skips dates already in the database.",
        )
        run_btn = st.button(
            "⬇  Download / Update Now", type="primary", use_container_width=True, key="dm_run"
        )

        if run_btn:
            progress_bar = st.progress(0.0, text="Starting…")
            log_lines: list[str] = []
            recent_log = st.empty()

            def on_progress(done: int, total: int, symbol: str, rows: int, status: str):
                progress_bar.progress(done / total, text=f"[{done}/{total}]  {symbol}")
                msg = (
                    f"✓  {symbol}  — already up to date"
                    if status == "already_up_to_date"
                    else f"↓  {symbol}  — {rows} rows fetched"
                )
                log_lines.append(msg)
                recent_log.text("\n".join(log_lines[-12:]))

            bulk_download(universe=UNIVERSE, progress_callback=on_progress)
            progress_bar.progress(1.0, text="Complete ✓")
            st.success("Data update finished!", icon=None)

    with col_right:
        st.markdown("#### Data Status")
        if st.button("🔄  Refresh Status", key="dm_refresh"):
            with st.spinner("Reading database…"):
                status_df = get_data_status(UNIVERSE)
            complete = (status_df["Trading Days"] > 0).sum()
            total    = len(status_df)
            st.caption(f"**{complete} / {total}** stocks have data in the database")
            st.dataframe(status_df, use_container_width=True, height=520)
        else:
            st.caption("Click **Refresh Status** to see what's stored.")

    # ── Market Indices & Macro ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Market Indices & Macro Data")
    st.caption(
        "Downloads NIFTY 50 Index, NIFTY Bank, India VIX, and USD/INR from Yahoo Finance. "
        "Required to unlock the **Excess vs NIFTY** and **VIX overlay** features in Deep Insights."
    )

    c_idx_left, c_idx_right = st.columns([1, 2])

    with c_idx_left:
        tickers_md = "  \n".join(
            f"• **{sym}** — {name}" for sym, name in INDEX_TICKERS.items()
        )
        st.markdown(tickers_md)
        idx_btn = st.button(
            "⬇  Download / Update Indices",
            type="primary", use_container_width=True, key="dm_idx_run",
        )

        if idx_btn:
            idx_progress = st.progress(0.0, text="Fetching indices…")
            idx_log_lines: list[str] = []
            idx_log = st.empty()

            def on_idx_progress(done: int, total: int, symbol: str, rows: int, status: str):
                idx_progress.progress(done / total, text=f"[{done}/{total}]  {symbol}")
                iname = INDEX_TICKERS.get(symbol, symbol)
                msg = (
                    f"✓  {iname}  — already up to date"
                    if status == "already_up_to_date"
                    else f"↓  {iname}  — {rows} rows fetched"
                )
                idx_log_lines.append(msg)
                idx_log.text("\n".join(idx_log_lines))

            fetch_indices(progress_callback=on_idx_progress)
            idx_progress.progress(1.0, text="Complete ✓")
            st.success("Index data updated!", icon=None)

    with c_idx_right:
        st.markdown("#### Index Data Status")
        if st.button("🔄  Refresh Index Status", key="dm_idx_refresh"):
            idx_status = get_index_status()
            if idx_status.empty:
                st.caption("No index data downloaded yet.")
            else:
                st.dataframe(idx_status, use_container_width=True)
        else:
            st.caption("Click **Refresh Index Status** to check stored index data.")


# ─── App shell ────────────────────────────────────────────────────────────────

def main():
    init_db()  # Ensure both ohlcv and indices_ohlcv tables exist on every startup

    st.title("📈  NSE Swing Trading Assistant")
    st.caption(
        "Historical seasonal analysis for Nifty 50 &nbsp;|&nbsp; "
        "Data: Yahoo Finance via yfinance &nbsp;|&nbsp; "
        "All analysis runs locally on your machine"
    )

    if not has_any_data():
        st.warning(
            "**Welcome! No data in the database yet.**  \n"
            "Head to the **Data Management** tab below and click "
            "**Download / Update Now** to fetch ~15 years of Nifty 50 history.",
            icon="⚠",
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊  Stock Analysis", "🏭  Sector Analysis",
         "🔍  Best Windows",  "📉  Deep Insights",
         "🗄  Data Management"]
    )

    with tab1:
        tab_stock_analysis()
    with tab2:
        tab_sector_analysis()
    with tab3:
        tab_best_windows()
    with tab4:
        tab_deep_insights()
    with tab5:
        tab_data_management()


if __name__ == "__main__":
    main()

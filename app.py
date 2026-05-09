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
    monthly_return_heatmap, excess_return_vs_nifty, sector_rotation_analysis,
    mae_analysis, stop_loss_survival, similar_years_analysis, volume_analysis,
)
from src.fetcher import (
    bulk_download, get_data_status, has_any_data,
    fetch_indices, get_index_status, has_index_data, INDEX_TICKERS,
    fetch_custom_ticker, get_custom_tickers, update_custom_tickers,
)
from src.universe import get_sectors, get_stocks, get_symbol_to_name
from src.db import init_db

# ─── Feature flags ────────────────────────────────────────────────────────────
# Set ENABLE_AI = False to hide the AI Commentary button entirely (e.g. on machines
# without Ollama, or to quickly revert the feature without git changes).
ENABLE_AI = True

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
    # Append any custom tickers registered via Data Management
    custom = get_custom_tickers()
    if custom:
        stocks = stocks + [{"name": c["name"], "symbol": c["symbol"], "sector": "Custom"} for c in custom]
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
        default_date = datetime.date.today()
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


def _days_to_target_chart(
    results_df: pd.DataFrame,
    norm_map: dict,
    min_return_pct: float,
    title: str,
) -> tuple:
    """
    For each year where the target was ultimately met, find the first calendar
    day the indexed price crossed 100 + min_return_pct.
    Returns (go.Figure, avg_days) or (None, None) if no years met the target.
    """
    threshold = 100 + min_return_pct
    records = []
    for _, row in results_df.iterrows():
        year = row["year"]
        if not row["target_met"] or year not in norm_map:
            continue
        series = norm_map[year].values
        crossed = [i for i, v in enumerate(series) if v >= threshold]
        if crossed:
            records.append({"Year": str(year), "Days": crossed[0]})

    if not records:
        return None, None

    df_d = pd.DataFrame(records)
    avg_days = df_d["Days"].mean()
    min_days = df_d["Days"].min()
    max_days = df_d["Days"].max()

    colors = [
        "#27ae60" if d <= avg_days else "#e67e22"
        for d in df_d["Days"]
    ]

    fig = go.Figure(go.Bar(
        x=df_d["Year"],
        y=df_d["Days"],
        marker_color=colors,
        text=df_d["Days"].astype(str),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>First touched target on day %{y}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(
        y=avg_days,
        line_dash="dash",
        line_color="#2980b9",
        annotation_text=f"Avg {avg_days:.0f} days",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Calendar Days from Entry",
        height=410,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(tickangle=-45),
        margin=dict(t=80),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#efefef")
    return fig, (avg_days, min_days, max_days)


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
            if summary and summary.get("error") == "insufficient_years":
                yrs = summary.get("years_found", 0)
                st.warning(
                    f"Not enough historical data for seasonal analysis — only **{yrs}** completed "
                    f"window(s) found (need at least 3). This stock may be recently listed. "
                    f"Try again after more history accumulates."
                )
            else:
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

    if summary.get("low_sample_warning"):
        st.warning(
            f"⚠ **Very limited history** — only **{summary['total_instances']} completed window(s)** found. "
            f"This stock may be recently listed. Results are indicative only and have low statistical confidence. "
            f"Treat with caution."
        )

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

    t_fan, t_bar, t_days, t_table = st.tabs(
        ["📈 Price Trend Chart", "📊 Year-by-Year Returns", "⏱ Days to Target", "🗃 Raw Data Table"]
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

    with t_days:
        fig_d, d_stats = _days_to_target_chart(
            results_df, norm_map, min_return,
            f"{sym_name} — Days to First Touch {min_return:.0f}%  ·  {win_label}",
        )
        if fig_d is not None:
            avg_d, min_d, max_d = d_stats
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Avg Days to Target", f"{avg_d:.0f} days",
                       help="Average calendar days from entry to first crossing the target, in years where target was met.")
            dc2.metric("Fastest Hit", f"{min_d:.0f} days")
            dc3.metric("Slowest Hit", f"{max_d:.0f} days")
            st.caption(
                "Green bars = at-or-below average speed · Orange bars = slower than average.  "
                "Only years where the target was **ultimately met** at window close are shown."
            )
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info(f"Target of {min_return:.0f}% was never met in this window — no days-to-target data.")

    with t_table:
        display = results_df.copy()
        display["return_pct"]  = display["return_pct"].apply(lambda x: f"{x:+.2f}%")
        display["target_met"]  = display["target_met"].map({True: "✓ Yes", False: "✗ No"})
        st.dataframe(display, use_container_width=True)

    # ── AI Commentary ─────────────────────────────────────────────────────────
    if ENABLE_AI:
        st.divider()
        if st.button("✨ Generate AI Commentary", key="sa_ai", use_container_width=False):
            from src.llm import build_seasonal_prompt, stream_commentary

            # Gather today's conditions from Similar Years session state if available
            today_feats = None
            sim_rows    = None
            sim_res = st.session_state.get("sim_result")
            if sim_res and not sim_res.get("missing_indices"):
                today_feats = sim_res.get("today_features")
                sim_df = sim_res.get("similar_years")
                if sim_df is not None and not sim_df.empty:
                    sim_rows = sim_df[["Year", "Final Return %"]].to_dict("records")

            prompt = build_seasonal_prompt(
                sym_name     = sym_name,
                win_label    = win_label,
                summary      = summary,
                today_features = today_feats,
                similar_years  = sim_rows,
            )

            with st.spinner("Mistral is thinking…"):
                commentary_box = st.empty()
                try:
                    full_text = ""
                    for chunk in stream_commentary(prompt):
                        full_text += chunk
                        commentary_box.markdown(
                            f"> {full_text}▌",
                            unsafe_allow_html=False,
                        )
                    commentary_box.markdown(f"> {full_text}")
                except RuntimeError as e:
                    st.error(str(e))


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

    # ── Entry Date Sensitivity ─────────────────────────────────────────────
    st.divider()
    st.markdown("### 📅 Entry Date Sensitivity")
    st.caption(
        "Shows how each top stock's **'Target Met' count** changes when you shift "
        "the entry date by ±3 days. A stock with identical or near-identical counts "
        "across all 7 dates is a **robust** seasonal signal. Wide variation means it's "
        "sitting right on the edge of your threshold — treat it with caution."
    )

    top_n = min(20, len(display_df))
    top_symbols = display_df.head(top_n)["Symbol"].tolist()
    top_names   = {
        row["Symbol"]: row["Name"]
        for _, row in display_df.head(top_n).iterrows()
    }

    if st.button("Run Sensitivity Check →", key="ws_sensitivity"):
        base_date = datetime.date(2000, sm, sd)   # year doesn't matter — just for arithmetic
        offsets   = list(range(-3, 4))            # -3 to +3
        col_labels = []
        sensitivity_rows = {sym: [] for sym in top_symbols}

        with st.spinner("Running sensitivity check for ±3 days…"):
            for offset in offsets:
                shifted     = base_date + datetime.timedelta(days=offset)
                o_sm, o_sd  = shifted.month, shifted.day
                label       = f"{'+'if offset>=0 else ''}{offset}d\n({_window_label(o_sm, o_sd, holding_days).split('+')[0].strip()})"
                col_labels.append(label)

                for sym in top_symbols:
                    _, summary = seasonal_analysis(sym, o_sm, o_sd, holding_days, min_return)
                    sensitivity_rows[sym].append(
                        summary.get("target_met_count", 0) if (summary and "error" not in summary) else 0
                    )

        # Build DataFrame
        sens_df = pd.DataFrame(
            sensitivity_rows,
            index=col_labels,
        ).T   # rows = stocks, cols = date offsets

        sens_df.insert(0, "Stock", [f"{top_names.get(s,s)}" for s in top_symbols])
        base_col = col_labels[3]   # the "+0d" column = user's chosen date
        sens_df["Wobble (max−min)"] = sens_df[col_labels].max(axis=1) - sens_df[col_labels].min(axis=1)
        sens_df["Stability"] = sens_df["Wobble (max−min)"].apply(
            lambda w: "🟢 Robust" if w == 0 else ("🟡 Minor" if w <= 1 else "🔴 Fragile")
        )

        st.session_state["ws_sensitivity_df"] = sens_df
        st.session_state["ws_sensitivity_cols"] = col_labels
        st.session_state["ws_sensitivity_base"] = base_col

    if "ws_sensitivity_df" in st.session_state:
        sens_df   = st.session_state["ws_sensitivity_df"]
        col_labels = st.session_state["ws_sensitivity_cols"]
        base_col   = st.session_state["ws_sensitivity_base"]

        total_years = int(display_df["Out of (yrs)"].max())

        def _color_cell(val):
            """Green if high hit rate, fading to red."""
            try:
                v = int(val)
            except (ValueError, TypeError):
                return ""
            pct = v / max(total_years, 1)
            if pct >= 0.7:   return "background-color: #c7f2c7; color: #1a5c1a"
            if pct >= 0.5:   return "background-color: #f7f7c7; color: #5c5c00"
            if pct >= 0.3:   return "background-color: #f2ddc7; color: #7a3e00"
            return                  "background-color: #f2c7c7; color: #7a0000"

        def _color_wobble(val):
            try: w = int(val)
            except (ValueError, TypeError): return ""
            if w == 0: return "background-color: #c7f2c7; font-weight:600"
            if w == 1: return "background-color: #f7f7c7"
            return             "background-color: #f2c7c7"

        styled = sens_df.style.applymap(_color_cell, subset=col_labels)
        styled = styled.applymap(_color_wobble, subset=["Wobble (max−min)"])

        # Bold the base (0d) column header
        styled = styled.set_properties(
            subset=[base_col],
            **{"font-weight": "bold", "border-left": "2px solid #2980b9", "border-right": "2px solid #2980b9"}
        )

        st.dataframe(styled, use_container_width=True, height=min(60 + 35 * len(sens_df), 600))

        robust  = (sens_df["Wobble (max−min)"] == 0).sum()
        fragile = (sens_df["Wobble (max−min)"] >= 2).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Robust (no wobble)",       robust)
        c2.metric("🟡 Minor wobble (±1 yr)",     len(sens_df) - robust - fragile)
        c3.metric("🔴 Fragile (≥2 yr change)",   fragile)

        st.info(
            "**Bold column** = your chosen entry date.  "
            "Each cell = number of years out of the historical sample where the "
            f"≥{min_return:.0f}% target was met.  "
            "**Wobble** = max – min across all 7 dates; zero means the stock is "
            "indifferent to which exact day you enter."
        )


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
        min_ret_ev = st.session_state.get("_res_ev_minret", 0)
        vix_data["target_met"] = vix_data["stock_return"] >= min_ret_ev

        def _regime_agg(g):
            return pd.Series({
                "Years":          len(g),
                f"Target ≥{min_ret_ev:.0f}% Met": int(g["target_met"].sum()),
                "Hit Rate %":     round(g["target_met"].mean() * 100, 1),
                "Avg Return %":   round(g["stock_return"].mean(), 2),
                "Min %":          round(g["stock_return"].min(), 2),
                "Max %":          round(g["stock_return"].max(), 2),
            })

        regime_tbl = (
            vix_data.groupby("vix_regime", observed=True)
            .apply(_regime_agg)
        )
        st.markdown("**Target hit rate & outcomes by VIX regime at entry**")
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


def _render_similar_years():
    st.caption(
        "Finds the historical years whose **market conditions at entry** most closely resemble today — "
        "then shows what actually happened in those years.  "
        "Similarity is measured across VIX level, NIFTY momentum, NIFTY vs 200-DMA, "
        "Stock RSI-14, and Stock vs 200-DMA."
    )
    st.info(
        "Requires **Market Indices** data (^NSEI + ^INDIAVIX). "
        "Download them in the Data Management tab if not done yet.",
        icon="ℹ️",
    )

    col_s, col_w = st.columns([1, 3])
    with col_s:
        _, symbol = _stock_selector("sim")
    with col_w:
        sm, sd, holding_days, _ = _entry_inputs("sim")

    n_sim = st.slider("Number of similar years to show", min_value=3, max_value=10, value=5, key="sim_n")

    if st.button("🔭 Find Similar Years", type="primary", key="sim_go"):
        sym_name = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
        with st.spinner(f"Analysing conditions for {sym_name}…"):
            result = similar_years_analysis(symbol, sm, sd, holding_days, n_similar=n_sim)
        if result is None:
            _no_data_warning()
        else:
            st.session_state["sim_result"]      = result
            st.session_state["_res_sim_symbol"] = symbol
            st.session_state["_res_sim_sm"]     = sm
            st.session_state["_res_sim_sd"]     = sd
            st.session_state["_res_sim_hold"]   = holding_days

    if "sim_result" not in st.session_state:
        return

    result       = st.session_state["sim_result"]
    symbol       = st.session_state["_res_sim_symbol"]
    sm           = st.session_state["_res_sim_sm"]
    sd           = st.session_state["_res_sim_sd"]
    holding_days = st.session_state["_res_sim_hold"]
    sym_name     = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
    win_label    = _window_label(sm, sd, holding_days)

    st.divider()
    st.markdown(f"### {sym_name}  ·  {win_label}")

    if result["missing_indices"]:
        st.warning(
            "Index data (NIFTY / VIX) not found in database. "
            "Go to **Data Management → Download Indices** and retry."
        )
        return

    if result["today_features"] is None:
        st.warning("Could not compute today's conditions — not enough history in DB.")
        return

    # ── Today's conditions ────────────────────────────────────────────────────
    tf = result["today_features"]
    st.markdown(f"**Today's market conditions** *(reference date: {result['today_entry']})*")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("India VIX",         f"{tf['vix_level']:.1f}")
    c2.metric("NIFTY 20d Mom",      f"{tf['nifty_mom_20d']:+.2f}%")
    c3.metric("NIFTY vs 200DMA",    f"{tf['nifty_200dma_dist']:+.2f}%" if tf['nifty_200dma_dist'] is not None else "—")
    c4.metric("Stock RSI-14",       f"{tf['stock_rsi14']:.1f}")
    c5.metric("Stock vs 200DMA",    f"{tf['stock_200dma_dist']:+.2f}%" if tf['stock_200dma_dist'] is not None else "—")

    st.divider()

    similar_df = result["similar_years"]
    if similar_df is None or similar_df.empty:
        st.warning("Not enough historical data with complete features to compute similarity.")
        return

    # ── Similar years summary ─────────────────────────────────────────────────
    st.markdown("#### Most Similar Historical Years")
    st.caption(
        "These are the years that entered this window under the most similar market conditions to today. "
        "The **Final Return %** column shows what actually happened — your base-rate expectation."
    )

    feature_cols = ["VIX at Entry", "NIFTY 20d Mom %", "NIFTY vs 200DMA %", "Stock RSI-14", "Stock vs 200DMA %"]
    display_cols = ["Year", "Similarity Score", "Final Return %"] + feature_cols
    display_cols = [c for c in display_cols if c in similar_df.columns]

    def _color_return(val):
        if pd.isna(val):
            return ""
        return "color: #27ae60; font-weight:600" if val > 0 else "color: #e74c3c; font-weight:600"

    fmt = {"Final Return %": "{:+.2f}%", "Similarity Score": "{:.1f}"}
    for c in feature_cols:
        if c in similar_df.columns:
            fmt[c] = "{:.1f}"

    st.dataframe(
        similar_df[display_cols]
        .style
        .format(fmt)
        .map(_color_return, subset=["Final Return %"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Outcome bar chart ─────────────────────────────────────────────────────
    fig = go.Figure()
    bar_colors = ["#27ae60" if p else "#e74c3c" for p in similar_df["Profitable"]]
    fig.add_trace(go.Bar(
        x=similar_df["Year"].astype(str),
        y=similar_df["Final Return %"],
        marker_color=bar_colors,
        text=[f"{v:+.1f}%" for v in similar_df["Final Return %"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="#888", line_dash="dot")
    avg = similar_df["Final Return %"].mean()
    fig.add_hline(
        y=avg, line_dash="dash", line_color="#e67e22",
        annotation_text=f"Avg: {avg:+.1f}%",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"Returns in the {len(similar_df)} most similar years  ·  {win_label}",
        xaxis_title="Year", yaxis_title="Final Return %",
        height=380, plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Radar chart — today vs avg of similar years ───────────────────────────
    radar_cols = [c for c in feature_cols if c in similar_df.columns and similar_df[c].notna().any()]
    if len(radar_cols) >= 3:
        st.markdown("#### Conditions: Today vs Average of Similar Years")
        sim_avg = similar_df[radar_cols].mean()
        today_vals = [tf.get({
            "VIX at Entry": "vix_level",
            "NIFTY 20d Mom %": "nifty_mom_20d",
            "NIFTY vs 200DMA %": "nifty_200dma_dist",
            "Stock RSI-14": "stock_rsi14",
            "Stock vs 200DMA %": "stock_200dma_dist",
        }.get(c, c), 0) or 0 for c in radar_cols]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=today_vals + [today_vals[0]],
            theta=radar_cols + [radar_cols[0]],
            fill="toself", name="Today",
            line_color="#2980b9", fillcolor="rgba(41,128,185,0.15)",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=sim_avg.tolist() + [sim_avg.iloc[0]],
            theta=radar_cols + [radar_cols[0]],
            fill="toself", name="Avg of similar years",
            line_color="#e67e22", fillcolor="rgba(230,126,34,0.15)",
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            height=400, showlegend=True,
        )
        st.plotly_chart(fig_r, use_container_width=True)


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
            st.session_state["mae_df"]           = mae_df
            st.session_state["mae_survival_df"]  = stop_loss_survival(mae_df)
            st.session_state["_res_mae_symbol"]  = symbol
            st.session_state["_res_mae_sm"]      = sm
            st.session_state["_res_mae_sd"]      = sd
            st.session_state["_res_mae_hold"]    = holding_days

    if "mae_df" not in st.session_state:
        return

    mae_df       = st.session_state["mae_df"]
    survival_df  = st.session_state.get("mae_survival_df")
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

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg MAE",   f"{valid_mae.mean():+.2f}%" if len(valid_mae) > 0 else "—",
              help="Average deepest intraday dip from entry across all years")
    c2.metric("Worst MAE", f"{valid_mae.min():+.2f}%"  if len(valid_mae) > 0 else "—",
              help="Single worst intraday dip from entry across all history")
    c3.metric("Avg MFE",   f"{valid_mfe.mean():+.2f}%" if len(valid_mfe) > 0 else "—",
              help="Average highest intraday peak reached during the window")

    st.divider()

    t1, t2, t3, t4 = st.tabs(["🎯 Stop-Loss Survival", "📊 MAE by Year", "🔵 MAE vs Final Return", "🗃 Raw Data"])

    with t1:
        st.caption(
            "At each stop-loss level, how many trades survive — and crucially, "
            "how many **winning trades** are preserved?  "
            "The sweet spot: a stop where losers start getting cut while winners are still left intact."
        )
        if survival_df is None or survival_df.empty:
            st.warning("No survival data available.")
        else:
            fig_surv = go.Figure()
            fig_surv.add_trace(go.Scatter(
                x=survival_df["Stop Level %"],
                y=survival_df["Winner Preservation %"],
                mode="lines+markers",
                name="Winner Preservation %",
                line=dict(color="#27ae60", width=2),
                marker=dict(size=4),
            ))
            fig_surv.add_trace(go.Scatter(
                x=survival_df["Stop Level %"],
                y=survival_df["Survival Rate %"],
                mode="lines+markers",
                name="Overall Survival Rate %",
                line=dict(color="#2980b9", width=2, dash="dash"),
                marker=dict(size=4),
            ))
            fig_surv.add_hline(
                y=80,
                line_dash="dot", line_color="#e74c3c",
                annotation_text="80% winner preservation threshold",
                annotation_position="bottom right",
            )
            fig_surv.update_layout(
                title=f"{sym_name} — Stop-Loss Survival Analysis  ·  {win_label}",
                xaxis_title="Stop-Loss Level (%)",
                yaxis_title="% of Trades",
                yaxis=dict(range=[0, 105]),
                height=430,
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(x=0.01, y=0.05),
            )
            st.plotly_chart(fig_surv, use_container_width=True)

            summary_levels = [round(-x * 2.0, 1) for x in range(1, 16)]  # -2% to -30% in 2% steps
            summary = survival_df[survival_df["Stop Level %"].isin(summary_levels)][
                ["Stop Level %", "Trades Stopped", "Winners Stopped", "Losers Stopped",
                 "Survival Rate %", "Winner Preservation %"]
            ].reset_index(drop=True)
            st.dataframe(summary.style.format({
                "Stop Level %":          "{:.1f}%",
                "Survival Rate %":       "{:.1f}%",
                "Winner Preservation %": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
            }), use_container_width=True)

    with t2:
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
        fig.update_layout(
            title=f"{sym_name} — MAE per Year  ·  {win_label}"
                  "  (green bar = trade ended profitable)",
            xaxis_title="Year", yaxis_title="MAE %",
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    with t3:
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

    with t4:
        st.dataframe(
            mae_df.style.format({
                "Final Return %":              "{:+.2f}%",
                "MAE % (worst intraday dip)":  lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                "MFE % (best intraday peak)":  lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
            }),
            use_container_width=True,
        )


def _render_volume_analysis():
    st.caption(
        "Volume-based analytics for the selected stock across its full history. "
        "Pick a stock and an entry window to see whether volume confirms seasonal price moves."
    )

    col_s, col_w = st.columns([1, 3])
    with col_s:
        _, symbol = _stock_selector("va")
    with col_w:
        sm, sd, holding_days, _ = _entry_inputs("va")

    if st.button("Run Volume Analysis →", type="primary", key="va_go"):
        with st.spinner("Computing volume metrics…"):
            result = volume_analysis(symbol, sm, sd, holding_days)
        if result is None:
            st.warning("No volume data available for this stock.")
            return
        st.session_state["va_result"]   = result
        st.session_state["_va_symbol"]  = symbol
        st.session_state["_va_sm"]      = sm
        st.session_state["_va_sd"]      = sd
        st.session_state["_va_hold"]    = holding_days

    if "va_result" not in st.session_state:
        return

    result       = st.session_state["va_result"]
    symbol       = st.session_state["_va_symbol"]
    sm           = st.session_state["_va_sm"]
    sd           = st.session_state["_va_sd"]
    holding_days = st.session_state["_va_hold"]

    sym_name     = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
    win_label    = _window_label(sm, sd, holding_days)
    summary      = result["summary"]
    window_df    = result["window_df"]
    monthly_df   = result["monthly_vol_df"]
    obv_by_year  = result["obv_by_year"]

    st.divider()
    st.markdown(f"### {sym_name} &nbsp;·&nbsp; {win_label}")

    # ── Summary metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Vol-Confirmed Entries",
        f"{summary['vol_confirmed_entries']} / {summary['total_years']}",
        delta=f"{summary['vol_confirmed_pct']}% of years",
        help="Entry days where volume was above its 20-day average (strong conviction signal)",
    )
    c2.metric(
        "Avg Return — Vol Confirmed",
        f"{summary['avg_return_vol_confirmed']:+.2f}%" if summary["avg_return_vol_confirmed"] is not None else "—",
        help="Average window return in years where entry volume was above average",
    )
    c3.metric(
        "Avg Return — Low Volume Entry",
        f"{summary['avg_return_unconfirmed']:+.2f}%" if summary["avg_return_unconfirmed"] is not None else "—",
        help="Average window return in years where entry volume was below average",
    )
    c4.metric(
        "Weak-Vol Rallies (Divergence)",
        summary["divergence_count"],
        help="Years where price went UP but window avg volume was <85% of baseline — unreliable rally",
    )

    st.divider()
    t1, t2, t3, t4 = st.tabs([
        "📅 Seasonal Volume Rhythm",
        "📊 Volume vs Return by Year",
        "📈 OBV During Window",
        "🗃 Raw Data",
    ])

    # ── Tab 1: Seasonal volume rhythm ─────────────────────────────────────────
    with t1:
        st.caption(
            "Average normalised trading volume for each calendar month across all history.  "
            "**1.0 = normal volume.** Values above 1 mean that month is unusually active.  "
            "Useful for spotting which months have the highest market participation — "
            "high-volume months tend to produce more reliable price moves."
        )
        monthly_valid = monthly_df.dropna(subset=["Avg Normalised Volume"])
        fig_month = px.bar(
            monthly_valid,
            x="Month", y="Avg Normalised Volume",
            color="Avg Normalised Volume",
            color_continuous_scale="RdYlGn",
            range_color=[0.5, 1.5],
            text=monthly_valid["Avg Normalised Volume"].apply(lambda x: f"{x:.2f}"),
            title=f"{sym_name} — Seasonal Volume Rhythm (monthly avg, normalised to 90-day baseline)",
        )
        fig_month.add_hline(y=1.0, line_dash="dash", line_color="#888",
                            annotation_text="Baseline (1.0)", annotation_position="right")
        fig_month.update_traces(textposition="outside")
        fig_month.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                                coloraxis_showscale=False)
        st.plotly_chart(fig_month, use_container_width=True)

    # ── Tab 2: Volume vs Return by year ───────────────────────────────────────
    with t2:
        st.caption(
            "Each bar = one historical year.  "
            "**Entry Vol Ratio** = entry-day volume ÷ its 20-day average (>1 = high-volume entry).  "
            "**Window Avg Vol Ratio** = mean daily volume during the window ÷ 90-day baseline (>1 = accumulation).  "
            "Green return = price went up. Orange = price went down."
        )
        fig_ev = go.Figure()
        colors_ret = ["#27ae60" if d == "UP" else "#e74c3c" for d in window_df["Direction"]]
        fig_ev.add_trace(go.Bar(
            name="Return %",
            x=window_df["Year"].astype(str),
            y=window_df["Return %"],
            marker_color=colors_ret,
            yaxis="y1",
            text=window_df["Return %"].apply(lambda x: f"{x:+.1f}%"),
            textposition="outside",
        ))
        fig_ev.add_trace(go.Scatter(
            name="Entry Vol Ratio",
            x=window_df["Year"].astype(str),
            y=window_df["Entry Vol Ratio"],
            mode="lines+markers",
            line=dict(color="#2980b9", width=2),
            marker=dict(size=7),
            yaxis="y2",
        ))
        fig_ev.add_trace(go.Scatter(
            name="Window Avg Vol Ratio",
            x=window_df["Year"].astype(str),
            y=window_df["Window Avg Vol Ratio"],
            mode="lines+markers",
            line=dict(color="#8e44ad", width=2, dash="dot"),
            marker=dict(size=7),
            yaxis="y2",
        ))
        fig_ev.update_layout(
            title=f"{sym_name} — Return vs Volume Confirmation  ·  {win_label}",
            height=440,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Year"),
            yaxis=dict(title="Return %", side="left"),
            yaxis2=dict(title="Volume Ratio (1.0 = avg)", overlaying="y", side="right",
                        zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_ev.add_hline(y=1.0, line_dash="dash", line_color="#aaa", yref="y2")
        st.plotly_chart(fig_ev, use_container_width=True)

        # Volume confirmation vs return comparison
        if summary["avg_return_vol_confirmed"] is not None and summary["avg_return_unconfirmed"] is not None:
            diff = summary["avg_return_vol_confirmed"] - summary["avg_return_unconfirmed"]
            if diff > 0:
                st.success(
                    f"**Volume-confirmed entries outperformed** by **{diff:+.2f}%** on average "
                    f"({summary['avg_return_vol_confirmed']:+.2f}% vs {summary['avg_return_unconfirmed']:+.2f}%). "
                    f"This suggests watching for high-volume entry days in this window."
                )
            else:
                st.info(
                    f"Low-volume entries had similar or better returns "
                    f"({summary['avg_return_unconfirmed']:+.2f}% vs {summary['avg_return_vol_confirmed']:+.2f}%). "
                    f"Volume confirmation may not add edge for this stock/window."
                )

        if summary["divergence_count"] > 0:
            div_years = window_df.loc[window_df["Vol-Price Divergence"], "Year"].tolist()
            st.warning(
                f"**Price-Volume Divergence detected in {summary['divergence_count']} year(s): "
                f"{', '.join(str(y) for y in div_years)}** — price rose but volume was well below average. "
                f"These rallies may have been less sustainable."
            )

    # ── Tab 3: OBV within window ──────────────────────────────────────────────
    with t3:
        st.caption(
            "**On-Balance Volume (OBV)** accumulated *within* the entry window for each historical year.  "
            "Rising OBV = buying pressure dominated. Falling OBV = selling pressure.  "
            "Normalised to avg daily volume so different absolute price levels are comparable.  "
            "Green = profitable year, Red = losing year."
        )
        fig_obv = go.Figure()
        for year, obv_vals in obv_by_year.items():
            row = window_df[window_df["Year"] == year]
            direction = row["Direction"].iloc[0] if not row.empty else "UP"
            is_up = direction == "UP"
            fig_obv.add_trace(go.Scatter(
                x=list(range(len(obv_vals))),
                y=obv_vals,
                mode="lines",
                name=str(year),
                line=dict(
                    color="rgba(39,174,96,0.55)"  if is_up else "rgba(231,76,60,0.45)",
                    width=1.5,
                ),
                hovertemplate=f"<b>{year}</b><br>Day %{{x}}<br>OBV %{{y:.1f}}x avg vol<extra></extra>",
            ))
        fig_obv.add_hline(y=0, line_dash="dash", line_color="#888")
        fig_obv.update_layout(
            title=f"{sym_name} — OBV within Window (normalised)  ·  {win_label}",
            height=440,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Trading Day in Window",
            yaxis_title="Cumulative OBV (× avg daily vol)",
            showlegend=True,
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig_obv, use_container_width=True)
        st.caption("**Green lines** = years that ended profitable · **Red lines** = years that ended in a loss")

    # ── Tab 4: Raw data ────────────────────────────────────────────────────────
    with t4:
        st.dataframe(
            window_df.style
                .background_gradient(subset=["Return %"], cmap="RdYlGn", vmin=-30, vmax=30)
                .background_gradient(subset=["Entry Vol Ratio", "Window Avg Vol Ratio"],
                                     cmap="Blues", vmin=0.5, vmax=2.0)
                .format({
                    "Return %":             "{:+.2f}%",
                    "Entry Vol Ratio":      "{:.2f}x",
                    "Window Avg Vol Ratio": "{:.2f}x",
                })
                .applymap(lambda v: "background-color: #fff3cd" if v is True else "",
                          subset=["Vol-Price Divergence"]),
            use_container_width=True,
            height=500,
        )


def tab_deep_insights():
    st.subheader("Deep Insights")
    st.caption(
        "Advanced analysis built on top of the historical OHLCV data. "
        "Download market indices first (Data Management tab) to unlock "
        "Excess vs NIFTY and the VIX overlay."
    )
    sub1, sub2, sub3, sub4, sub5, sub6 = st.tabs([
        "📅 Monthly Heatmap",
        "📊 Excess vs NIFTY",
        "🔄 Sector Rotation",
        "🛡 MAE & Stop-Loss",
        "🔭 Similar Years",
        "📦 Volume Analysis",
    ])
    with sub1:
        _render_monthly_heatmap()
    with sub2:
        _render_excess_vs_nifty()
    with sub3:
        _render_sector_rotation()
    with sub4:
        _render_mae_analysis()
    with sub5:
        _render_similar_years()
    with sub6:
        _render_volume_analysis()


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

    # ── Custom Tickers ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Add Custom Ticker")
    st.caption(
        "Add any stock or index that isn't in the Nifty 500 universe. "
        "Use the Yahoo Finance ticker format: NSE stocks end with **.NS** "
        "(e.g. `TATAPOWER.NS`, `IRFC.NS`). Once added, the ticker appears in "
        "all stock dropdowns for analysis."
    )

    c_ct_left, c_ct_right = st.columns([1, 2])

    with c_ct_left:
        ct_symbol = st.text_input(
            "Yahoo Finance Ticker",
            placeholder="e.g. TATAPOWER.NS",
            key="ct_symbol",
        )
        ct_name = st.text_input(
            "Friendly Name (optional)",
            placeholder="e.g. Tata Power",
            key="ct_name",
        )
        ct_btn = st.button(
            "⬇  Fetch & Add",
            type="primary",
            use_container_width=True,
            key="ct_fetch",
            disabled=not ct_symbol.strip(),
        )

        if ct_btn and ct_symbol.strip():
            with st.spinner(f"Fetching {ct_symbol.strip().upper()}…"):
                rows, err = fetch_custom_ticker(ct_symbol, ct_name)
            if err:
                st.error(f"Failed: {err}")
            elif rows == 0:
                st.success(
                    f"**{ct_symbol.strip().upper()}** registered and already up to date."
                )
            else:
                st.success(
                    f"**{ct_symbol.strip().upper()}** added — {rows} rows fetched."
                )

    with c_ct_right:
        st.markdown("#### Custom Tickers in Database")
        custom_list = get_custom_tickers()
        if custom_list:
            st.dataframe(
                pd.DataFrame(custom_list).rename(
                    columns={"symbol": "Ticker", "name": "Name"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            ct_update_btn = st.button(
                "🔄  Update Custom Tickers Only",
                use_container_width=True,
                key="ct_update",
            )
            if ct_update_btn:
                ct_prog = st.progress(0.0, text="Updating custom tickers…")
                ct_log_lines: list[str] = []
                ct_log = st.empty()

                def on_ct_progress(done: int, total: int, symbol: str, rows: int, status: str):
                    ct_prog.progress(done / total, text=f"[{done}/{total}]  {symbol}")
                    msg = (
                        f"✓  {symbol}  — already up to date"
                        if status == "already_up_to_date"
                        else f"↓  {symbol}  — {rows} rows fetched"
                    )
                    ct_log_lines.append(msg)
                    ct_log.text("\n".join(ct_log_lines))

                update_custom_tickers(progress_callback=on_ct_progress)
                ct_prog.progress(1.0, text="Complete ✓")
                st.success("Custom tickers updated!")
        else:
            st.caption("No custom tickers added yet.")


# ─── Dashboard Tab ────────────────────────────────────────────────────────────

def tab_dashboard():
    st.subheader("Trade Dashboard")
    st.caption(
        "One-stop view of all key analysis for a stock and entry window. "
        "Select your stock, entry date, holding period and target — then click **Run All**."
    )

    col_s, col_w = st.columns([1, 3])
    with col_s:
        _, symbol = _stock_selector("db")
    with col_w:
        sm, sd, holding_days, min_return = _entry_inputs("db")

    if st.button("▶ Run All Analysis", type="primary", key="db_run"):
        sym_name  = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
        win_label = _window_label(sm, sd, holding_days)
        errors    = {}

        with st.spinner(f"Running full analysis for {sym_name}  ·  {win_label}…"):
            try:
                db_res_df, db_summary = seasonal_analysis(symbol, sm, sd, holding_days, min_return)
            except Exception as e:
                db_res_df, db_summary = None, None
                errors["seasonal"] = str(e)

            try:
                db_bw_df = best_windows_for_stock(symbol, int(holding_days), float(min_return))
            except Exception as e:
                db_bw_df = None
                errors["best_windows"] = str(e)

            try:
                db_pivot = monthly_return_heatmap(symbol)
            except Exception as e:
                db_pivot = None
                errors["heatmap"] = str(e)

            try:
                db_ev_df, db_ev_sum = excess_return_vs_nifty(symbol, sm, sd, holding_days, min_return)
            except Exception as e:
                db_ev_df, db_ev_sum = None, None
                errors["excess"] = str(e)

            try:
                db_vol = volume_analysis(symbol, sm, sd, holding_days)
            except Exception as e:
                db_vol = None
                errors["volume"] = str(e)

            try:
                db_sim = similar_years_analysis(symbol, sm, sd, holding_days, 5)
            except Exception as e:
                db_sim = None
                errors["similar"] = str(e)

            # Latest India VIX from DB
            try:
                from src.db import get_connection as _gc
                from src.analyzer import _load_index_closes as _lic
                _conn = _gc()
                _vix_series = _lic("^INDIAVIX", _conn)
                _conn.close()
                today_vix = round(float(_vix_series["close"].iloc[-1]), 2) if not _vix_series.empty else None
            except Exception:
                today_vix = None

        st.session_state["db_res_df"]  = db_res_df
        st.session_state["db_summary"] = db_summary
        st.session_state["db_bw_df"]   = db_bw_df
        st.session_state["db_pivot"]   = db_pivot
        st.session_state["db_ev_df"]   = db_ev_df
        st.session_state["db_ev_sum"]  = db_ev_sum
        st.session_state["db_vol"]     = db_vol
        st.session_state["db_sim"]     = db_sim
        st.session_state["db_today_vix"] = today_vix
        st.session_state["_db_symbol"] = symbol
        st.session_state["_db_sm"]     = sm
        st.session_state["_db_sd"]     = sd
        st.session_state["_db_hold"]   = holding_days
        st.session_state["_db_minret"] = min_return
        st.session_state["_db_errors"] = errors

    if "_db_symbol" not in st.session_state:
        return

    db_res_df    = st.session_state["db_res_df"]
    db_summary   = st.session_state["db_summary"]
    db_bw_df     = st.session_state["db_bw_df"]
    db_pivot     = st.session_state["db_pivot"]
    db_ev_df     = st.session_state["db_ev_df"]
    db_ev_sum    = st.session_state["db_ev_sum"]
    db_vol       = st.session_state["db_vol"]
    db_sim       = st.session_state.get("db_sim")
    today_vix    = st.session_state.get("db_today_vix")
    symbol       = st.session_state["_db_symbol"]
    sm           = st.session_state["_db_sm"]
    sd           = st.session_state["_db_sd"]
    holding_days = st.session_state["_db_hold"]
    min_return   = st.session_state["_db_minret"]
    errors       = st.session_state["_db_errors"]

    sym_name  = get_symbol_to_name(UNIVERSE).get(symbol, symbol)
    win_label = _window_label(sm, sd, holding_days)

    for err_key, err_msg in errors.items():
        st.warning(f"⚠ {err_key}: {err_msg}")

    st.divider()
    st.markdown(f"## {sym_name}  ·  {win_label}  ·  Target ≥{min_return:.0f}%")

    # ── Plain-English Snapshot (numbered, match user-readable format) ─────────
    _snap = []

    # 1. Seasonal performance
    if db_summary and "error" not in db_summary:
        _tot     = db_summary.get("total_instances", 0)
        _hit_ct  = db_summary.get("target_met_count", 0)
        _avg_all = db_summary.get("avg_return_pct")
        _avg_met = db_summary.get("avg_return_when_met")
        _never   = db_summary.get("target_never_met", False)
        if db_summary.get("low_sample_warning"):
            _snap.append(f"⚠ **Limited data** — only {_tot} year(s) available; treat results as indicative.")
        if not _never and _avg_met is not None:
            _snap.append(
                f"Target met **{_hit_ct} out of {_tot} years** with an average return of "
                f"**{_avg_met:+.2f}%** when the target was met "
                f"(avg across all years: **{_avg_all:+.2f}%**)."
            )
        elif _never:
            _snap.append(
                f"Target of ≥{min_return:.0f}% was **never met** in this window "
                f"across all {_tot} years.  Avg return: **{_avg_all:+.2f}%**."
            )

    # 2. Days to target (computed from norm_map in summary)
    if db_summary and "error" not in db_summary and db_res_df is not None:
        _norm_map = db_summary.get("norm_series_map", {})
        _threshold = 100.0 + min_return
        _dtd = []
        for _, _row in db_res_df.iterrows():
            _yr = _row["year"]
            if not _row["target_met"] or _yr not in _norm_map:
                continue
            _series = _norm_map[_yr].values
            _crossed = [i for i, v in enumerate(_series) if v >= _threshold]
            if _crossed:
                _dtd.append(_crossed[0])
        if _dtd:
            _avg_d = round(sum(_dtd) / len(_dtd))
            _min_d = min(_dtd)
            _max_d = max(_dtd)
            _snap.append(
                f"On average, it took **{_avg_d} days** to hit the ≥{min_return:.0f}% target "
                f"historically whenever the target was met "
                f"(fastest: **{_min_d}d**, slowest: **{_max_d}d**)."
            )

    # 3. Best entry windows (top-3)
    if db_bw_df is not None and not db_bw_df.empty:
        _medals = ["🥇", "🥈", "🥉"]
        _cur_abbr = datetime.date(2000, sm, 1).strftime("%b")
        _top3_parts = []
        for _i, (_, _r) in enumerate(db_bw_df.head(3).iterrows()):
            _w = _r["Window"]
            _suffix = " ← *your entry month*" if _w.startswith(_cur_abbr) else ""
            _top3_parts.append(
                f"{_medals[_i]} **{_w}** — {_r['Target Met (yrs)']} of {_r['Out of (yrs)']} yrs"
                f", avg {_r['Avg Return %']:+.2f}%{_suffix}"
            )
        # Check if selected month is not in top 3
        _cur_rows = db_bw_df[db_bw_df["Window"].str.startswith(_cur_abbr)]
        _cur_not_in_top3 = _cur_rows.empty or (list(db_bw_df.index).index(_cur_rows.iloc[0].name) + 1 > 3)
        if not _cur_rows.empty and _cur_not_in_top3:
            _cr = _cur_rows.iloc[0]
            _cur_rank = list(db_bw_df.index).index(_cr.name) + 1
            _top3_parts.append(
                f"*(Your month **{_cur_abbr}** ranks #{_cur_rank} — "
                f"{_cr['Target Met (yrs)']} of {_cr['Out of (yrs)']} yrs, avg {_cr['Avg Return %']:+.2f}%)*"
            )
        _snap.append("Best entry windows for this stock:  \n" + "  \n".join(_top3_parts))

    # 4. Beat NIFTY
    if db_ev_sum and db_ev_sum.get("nifty_available"):
        _avg_exc   = db_ev_sum.get("avg_excess_return")
        _beat_lbl  = db_ev_sum.get("beat_index_label", "")
        _direction = "outperforms" if (_avg_exc or 0) > 0 else "underperforms"
        _snap.append(
            f"Stock **{_direction}** NIFTY in **{_beat_lbl}** in this window "
            f"(avg excess return: **{_avg_exc:+.2f}%**)."
        )

    # 5. India VIX regime at today's level
    if db_ev_df is not None and today_vix is not None:
        _vix_data = db_ev_df.dropna(subset=["vix_at_entry"]).copy()
        if len(_vix_data) >= 2:
            # Determine today's regime
            if today_vix < 14:
                _regime_label = f"Calm (<14)"
                _regime_filter = _vix_data["vix_at_entry"] < 14
            elif today_vix <= 20:
                _regime_label = f"Normal (14–20)"
                _regime_filter = (_vix_data["vix_at_entry"] >= 14) & (_vix_data["vix_at_entry"] <= 20)
            else:
                _regime_label = f"Elevated (>20)"
                _regime_filter = _vix_data["vix_at_entry"] > 20
            _regime_data = _vix_data[_regime_filter]
            if len(_regime_data) >= 1:
                _rv_avg = round(float(_regime_data["stock_return"].mean()), 2)
                _rv_max = round(float(_regime_data["stock_return"].max()), 2)
                _rv_met = int(_regime_data["target_met"].sum())
                _rv_tot = len(_regime_data)
                _snap.append(
                    f"India VIX today is **{today_vix}** ({_regime_label} regime).  \n"
                    f"In {_regime_label}-VIX years: avg return **{_rv_avg:+.2f}%**, "
                    f"max **{_rv_max:+.2f}%**, target met **{_rv_met} of {_rv_tot} times**."
                )
            else:
                _snap.append(f"India VIX today is **{today_vix}** — no historical entries at this regime level yet.")
        elif today_vix is not None:
            _snap.append(f"India VIX today: **{today_vix}** (insufficient VIX history to segment by regime).")
    elif today_vix is not None and db_ev_df is None:
        _snap.append(f"India VIX today: **{today_vix}** — download indices for VIX regime breakdown.")

    # 6. Most similar historical year
    if db_sim is not None and not db_sim.get("missing_indices", True):
        _sim_years = db_sim.get("similar_years")
        if _sim_years is not None and not _sim_years.empty:
            _top_sim = _sim_years.iloc[0]
            _sim_yr  = int(_top_sim["Year"])
            _sim_ret = float(_top_sim["Final Return %"])
            _snap.append(
                f"Most similar historical year to **{datetime.date.today().year}** is "
                f"**{_sim_yr}**, when the stock returned **{_sim_ret:+.2f}%** in this window."
            )

    if _snap:
        with st.expander("📋 Key Takeaways — Plain-English Summary", expanded=True):
            for _i, _pt in enumerate(_snap, 1):
                st.markdown(f"**{_i}.** {_pt}")

    # ── Seasonal summary metrics ───────────────────────────────────────────────
    if db_summary and "error" not in db_summary:
        if db_summary.get("low_sample_warning"):
            st.warning(
                f"⚠ Limited history — only {db_summary['total_instances']} completed "
                f"window(s). Treat results with caution."
            )
        _metric_row(db_summary)
    elif db_res_df is None:
        st.info("No seasonal data found for this stock/window.")

    st.divider()

    # ── Days to Target  +  Year-by-Year Returns ─────────────────────────────
    st.markdown("### ⏱ Days to Hit Target  &  📊 Year-by-Year Returns")

    if db_res_df is not None and db_summary and "error" not in db_summary:
        summary_copy = dict(db_summary)
        norm_map = summary_copy.pop("norm_series_map")
        col_days, col_bar = st.columns(2)
        with col_days:
            fig_d, d_stats = _days_to_target_chart(
                db_res_df, norm_map, min_return,
                f"{sym_name} — Days to First Touch {min_return:.0f}%  ·  {win_label}",
            )
            if fig_d is not None:
                avg_d, min_d, max_d = d_stats
                fig_d.update_layout(height=380, margin=dict(t=60))
                st.plotly_chart(fig_d, use_container_width=True)
                st.caption(f"Avg **{avg_d:.0f} days** to hit target · fastest {min_d:.0f}d · slowest {max_d:.0f}d")
            else:
                st.info(f"Target of {min_return:.0f}% was never met in this window.")
        with col_bar:
            fig_bar = _bar_chart(
                db_res_df, min_return,
                f"{sym_name} — Returns by Year",
            )
            fig_bar.update_layout(height=380, margin=dict(t=60))
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Seasonal analysis not available for this stock/window.")

    st.divider()

    # ── Best Entry Windows ────────────────────────────────────────────────────
    st.markdown("### 🔍 Best Entry Windows (All 12 Months)")

    if db_bw_df is not None:
        fig_bw = px.bar(
            db_bw_df, x="Window", y="Target Met (yrs)",
            color="Target Met (yrs)",
            color_continuous_scale="RdYlGn",
            range_color=[0, db_bw_df["Out of (yrs)"].max()],
            title=f"{sym_name} — Years ≥{min_return:.0f}% Met by Entry Month  ·  +{holding_days}d hold",
            text="Target Met (yrs)",
        )
        fig_bw.update_traces(texttemplate="%{text}", textposition="outside")
        fig_bw.update_layout(
            height=340, margin=dict(t=60),
            xaxis_tickangle=-30,
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bw, use_container_width=True)

        top3 = db_bw_df.head(3)
        medals = ["🥇", "🥈", "🥉"]
        cols = st.columns(3)
        for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
            col.metric(
                f"{medals[i]} {row['Window']}",
                f"{row['Target Met (yrs)']} of {row['Out of (yrs)']} yrs",
                delta=f"Avg {row['Avg Return %']:+.2f}%",
            )
    else:
        st.info("Best windows data not available.")

    st.divider()

    # ── Monthly Rhythm  +  Excess vs NIFTY ───────────────────────────────────
    st.markdown("### 📅 Monthly Rhythm  &  📊 Excess vs NIFTY")

    col_heat, col_excess = st.columns(2)

    with col_heat:
        if db_pivot is not None:
            avg_by_month = db_pivot.mean(axis=0).reset_index()
            avg_by_month.columns = ["Month", "Avg Return %"]
            avg_by_month["Avg Return %"] = avg_by_month["Avg Return %"].round(2)
            colors_m = ["#27ae60" if v >= 0 else "#e74c3c" for v in avg_by_month["Avg Return %"]]
            fig_heat = go.Figure(go.Bar(
                x=avg_by_month["Month"],
                y=avg_by_month["Avg Return %"],
                marker_color=colors_m,
                text=[f"{v:+.1f}%" for v in avg_by_month["Avg Return %"]],
                textposition="outside",
            ))
            fig_heat.add_hline(y=0, line_color="#333", line_width=1)
            fig_heat.update_layout(
                title=f"{sym_name} — Avg Monthly Return",
                height=340, margin=dict(t=60),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Month", yaxis_title="Avg Return %",
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Monthly heatmap data not available.")

    with col_excess:
        if db_ev_df is not None and db_ev_sum and db_ev_sum.get("nifty_available"):
            exc = db_ev_df.dropna(subset=["excess_return"])
            exc_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in exc["excess_return"]]
            avg_exc = db_ev_sum.get("avg_excess_return")
            fig_exc = go.Figure(go.Bar(
                x=exc["year"].astype(str),
                y=exc["excess_return"],
                marker_color=exc_colors,
                text=[f"{v:+.1f}%" for v in exc["excess_return"]],
                textposition="outside",
            ))
            fig_exc.add_hline(y=0, line_color="#333", line_width=1)
            if avg_exc is not None:
                fig_exc.add_hline(
                    y=avg_exc, line_dash="dash", line_color="#9b59b6",
                    annotation_text=f"Avg {avg_exc:+.1f}%",
                    annotation_position="top right",
                )
            fig_exc.update_layout(
                title=f"Excess Return vs NIFTY  ·  {win_label}",
                height=340, margin=dict(t=60),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Year", yaxis_title="Excess Return %",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_exc, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Stock Return",  f"{db_ev_sum['avg_stock_return']:+.2f}%")
            c2.metric("Avg NIFTY Return",  f"{db_ev_sum['avg_nifty_return']:+.2f}%")
            c3.metric("Avg Excess",
                      f"{avg_exc:+.2f}%" if avg_exc is not None else "—",
                      delta=db_ev_sum.get("beat_index_label"))
        elif not has_index_data():
            st.info("Download market indices (Data Management → Market Indices) to see Excess vs NIFTY.")
        else:
            st.info("Excess vs NIFTY data not available.")

    st.divider()

    # ── Volume Analysis ───────────────────────────────────────────────────────
    st.markdown("### 📦 Volume Analysis")

    if db_vol is not None:
        vol_sum     = db_vol["summary"]
        monthly_vol = db_vol["monthly_vol_df"]
        window_rows = db_vol["window_df"]

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            ordered_vol = monthly_vol.set_index("Month").reindex(month_order).reset_index()
            vol_colors  = ["#27ae60" if (v or 0) >= 1 else "#e74c3c"
                           for v in ordered_vol["Avg Normalised Volume"]]
            fig_vrhy = go.Figure(go.Bar(
                x=ordered_vol["Month"],
                y=ordered_vol["Avg Normalised Volume"],
                marker_color=vol_colors,
                text=[f"{v:.2f}×" if pd.notna(v) else "—"
                      for v in ordered_vol["Avg Normalised Volume"]],
                textposition="outside",
            ))
            fig_vrhy.add_hline(y=1.0, line_dash="dash", line_color="#888",
                               annotation_text="Baseline (1.0)", annotation_position="right")
            fig_vrhy.update_layout(
                title=f"{sym_name} — Seasonal Volume Rhythm",
                height=320, margin=dict(t=60),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Month", yaxis_title="Normalised Volume (×)",
            )
            st.plotly_chart(fig_vrhy, use_container_width=True)

        with col_v2:
            conf_ret   = vol_sum.get("avg_return_vol_confirmed")
            unconf_ret = vol_sum.get("avg_return_unconfirmed")
            if conf_ret is not None and unconf_ret is not None:
                comp_df = pd.DataFrame([
                    {"Category": "High-Vol Entry", "Avg Return %": conf_ret},
                    {"Category": "Low-Vol Entry",  "Avg Return %": unconf_ret},
                ])
                fig_vcomp = go.Figure(go.Bar(
                    x=comp_df["Category"],
                    y=comp_df["Avg Return %"],
                    marker_color=["#2980b9", "#95a5a6"],
                    text=[f"{v:+.2f}%" for v in comp_df["Avg Return %"]],
                    textposition="outside",
                    width=0.4,
                ))
                fig_vcomp.add_hline(y=0, line_color="#333", line_width=1)
                fig_vcomp.update_layout(
                    title="Avg Return: High vs Low Volume Entry",
                    height=320, margin=dict(t=60),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis_title="Avg Return %",
                )
                st.plotly_chart(fig_vcomp, use_container_width=True)

        cv1, cv2, cv3, cv4 = st.columns(4)
        cv1.metric("Vol-Confirmed Entries",
                   f"{vol_sum['vol_confirmed_entries']} / {vol_sum['total_years']}",
                   delta=f"{vol_sum['vol_confirmed_pct']}% of years")
        cv2.metric("Accumulation Windows",
                   f"{vol_sum['accumulation_windows']} / {vol_sum['total_years']}")
        cv3.metric("Avg Return (High-Vol Entry)",
                   f"{conf_ret:+.2f}%" if conf_ret is not None else "—")
        cv4.metric("Price-Vol Divergence Yrs", vol_sum["divergence_count"])

        div_years = window_rows.loc[window_rows["Vol-Price Divergence"], "Year"].tolist()
        if div_years:
            st.warning(
                f"⚠ Price-Volume Divergence in {len(div_years)} year(s): "
                f"{', '.join(str(y) for y in div_years)} — "
                f"price went up on below-average volume. Weaker-conviction rallies."
            )
    else:
        st.info("Volume analysis data not available for this stock/window.")


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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["🎯  Dashboard",
         "📊  Stock Analysis", "🏭  Sector Analysis",
         "🔍  Best Windows",  "📉  Deep Insights",
         "🆚  Compare",
         "🗄  Data Management"]
    )

    with tab1:
        tab_dashboard()
    with tab2:
        tab_stock_analysis()
    with tab3:
        tab_sector_analysis()
    with tab4:
        tab_best_windows()
    with tab5:
        tab_deep_insights()
    with tab6:
        tab_compare()
    with tab7:
        tab_data_management()


if __name__ == "__main__":
    main()

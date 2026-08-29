import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from model_utils import FrequencyEncoder

ART = "artifacts"

st.set_page_config(
    page_title="Malaysia Housing Price Intelligence",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8b5cf6', '#e0507a']
HOVERLABEL = dict(bgcolor="white", font_size=12, font_family="Segoe UI, Inter, sans-serif")

# Applied to every st.plotly_chart(): keeps the default view = full data
# (system autosize), removes the "pan" tool so users can't drag the plot
# into empty space, and disables scroll-wheel zoom drift. Zoom-box + reset
# (double-click) still work.
PLOTLY_CONFIG = dict(
    displayModeBar=True,
    displaylogo=False,
    scrollZoom=False,
    doubleClick="reset+autosize",
    modeBarButtonsToRemove=["pan2d", "lasso2d", "select2d", "autoScale2d"],
)

# ---------- Icon set (line-style SVG, no emoji) ----------
ICON_PATHS = {
    "building": '<rect x="4" y="3" width="16" height="18" rx="1"></rect>'
                '<line x1="9" y1="7" x2="9" y2="7.01"></line><line x1="15" y1="7" x2="15" y2="7.01"></line>'
                '<line x1="9" y1="11" x2="9" y2="11.01"></line><line x1="15" y1="11" x2="15" y2="11.01"></line>'
                '<line x1="9" y1="15" x2="9" y2="15.01"></line><line x1="15" y1="15" x2="15" y2="15.01"></line>'
                '<path d="M9 21v-4h6v4"></path>',
    "award": '<circle cx="12" cy="8" r="6"></circle><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"></path>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>'
              '<polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline>',
    "bars": '<line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line>'
            '<line x1="6" y1="20" x2="6" y2="14"></line>',
}

def icon(name, size=18, stroke_width=1.8, margin_right=6):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="{stroke_width}" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align:-3px;display:inline-block;margin-right:{margin_right}px;flex-shrink:0;">'
            f'{ICON_PATHS[name]}</svg>')

# ---------- Theme ----------
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    header { background: transparent; }
    /* keep the sidebar collapse/expand control visible & clickable */
    [data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; }

    html, body, [class*="css"]  { font-family: 'Segoe UI', 'Inter', sans-serif; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    .app-header {
        background: linear-gradient(135deg, #0f2740 0%, #1c4b7c 100%);
        padding: 28px 36px; border-radius: 14px; color: white; margin-bottom: 24px;
        animation: fadeInUp .5s ease both;
    }
    .app-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; display:flex; align-items:center; }
    .app-header p { margin: 6px 0 0 0; opacity: 0.85; font-size: 0.95rem; }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* NOTE: intentionally no blanket entrance-animation on stPlotlyChart/stDataFrame here —
       that used to re-fire on every rerun and looked like a "fade" whenever a chart's
       data changed, masking the actual value-transition animations we want instead. */

    /* pill-style tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; margin-bottom: 20px; border-bottom: none; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px; padding: 8px 20px; background-color: transparent;
        border: 0.5px solid #e0e3e8; color: #6b7280; font-size: 13px; font-weight: 500;
        transition: all .2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #f3f6fa; color: #1c4b7c; }
    .stTabs [aria-selected="true"] {
        background-color: #eaf1fb !important; color: #1c4b7c !important; border: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    .metric-card {
        background: #ffffff; border: 1px solid #e6e9ee; border-radius: 12px;
        padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform .22s ease, box-shadow .22s ease;
        animation: fadeInUp .5s ease both; cursor: default;
    }
    .metric-card:hover { transform: translateY(-4px) scale(1.03); box-shadow: 0 12px 26px rgba(15,39,64,.14); }
    .metric-card .label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #0f2740; margin-top: 4px; display:flex; align-items:center; }

    .chart-label { font-size: 13px; color: #6b7280; margin-bottom: 8px; }

    .stButton>button { transition: transform .15s ease, box-shadow .15s ease; }
    .stButton>button:hover { transform: translateY(-2px) scale(1.015); box-shadow: 0 6px 16px rgba(15,39,64,.18); }

    div[data-testid="stMetricValue"] { color: #0f2740; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2740 0%, #173a5e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #eaf0f7 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }
    .sidebar-card {
        background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px; padding: 12px 16px; margin-bottom: 10px;
        transition: transform .2s ease, background .2s ease;
    }
    .sidebar-card:hover { transform: translateX(2px); background: rgba(255,255,255,0.11); }
    .sidebar-card .label { font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em; opacity: .65; }
    .sidebar-card .value { font-size: 1.05rem; font-weight: 700; margin-top: 2px; display:flex; align-items:center; }
    .legend-row { display: flex; align-items: center; gap: 8px; font-size: 13px; padding: 5px 2px; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(255,255,255,.06); }
    .legend-crown { display:inline-flex; opacity:.9; }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #eaf0f7 !important; font-size: 1.1rem; }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #b9c6d6 !important; }

    /* sidebar expander ("About this app") -> match the dark theme in EVERY interaction
       state (default / hover / focus / active / open), so it can't flash back to
       Streamlit's own white background once the mouse leaves the header. */
    section[data-testid="stSidebar"] [data-testid="stExpander"],
    section[data-testid="stSidebar"] .streamlit-expander,
    section[data-testid="stSidebar"] [data-testid="stExpander"] details,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:active,
    section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    section[data-testid="stSidebar"] [data-testid="stExpanderDetails"],
    section[data-testid="stSidebar"] .streamlit-expanderHeader,
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background: rgba(255,255,255,0.06) !important;
        background-color: rgba(255,255,255,0.06) !important;
        outline: none !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
    section[data-testid="stSidebar"] [data-testid="stExpander"] p,
    section[data-testid="stSidebar"] [data-testid="stExpander"] div,
    section[data-testid="stSidebar"] [data-testid="stExpander"] span,
    section[data-testid="stSidebar"] [data-testid="stExpander"] svg {
        color: #eaf0f7 !important;
        fill: #eaf0f7 !important;
    }

    /* bordered containers (st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important; border-color: #e6e9ee !important;
        transition: box-shadow .2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 6px 18px rgba(15,39,64,.08);
    }

    /* multiselect pill tags */
    span[data-baseweb="tag"] {
        background-color: #1c4b7c !important; border-radius: 999px !important;
    }

    /* ---------- Ranking leaderboard ---------- */
    .rank-list { display:flex; flex-direction:column; gap:8px; }
    .rank-row {
        display:flex; align-items:center; gap:14px; background:#fff;
        border:1px solid #e6e9ee; border-radius:12px; padding:12px 16px;
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .rank-row:hover { transform: translateX(4px); box-shadow: 0 6px 16px rgba(15,39,64,.08); }
    .rank-badge {
        width:28px; height:28px; border-radius:50%; color:white; font-weight:700; font-size:13px;
        display:flex; align-items:center; justify-content:center; flex-shrink:0;
    }
    .rank-model { flex:1; font-weight:600; color:#0f2740; display:flex; align-items:center; gap:8px; font-size:14px; }
    .rank-score { font-size:13px; color:#6b7280; font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)

# ---------- Load artifacts ----------
@st.cache_resource
def load_models(names):
    return {n: joblib.load(f"{ART}/model_{n.lower().replace(' ', '_')}.pkl") for n in names}

@st.cache_data
def load_data():
    meta = json.load(open(f"{ART}/metadata.json"))
    comparison = pd.read_csv(f"{ART}/comparison.csv", index_col=0)
    predictions = pd.read_csv(f"{ART}/predictions.csv")
    cv_folds = pd.read_csv(f"{ART}/cv_folds.csv")
    return meta, comparison, predictions, cv_folds

meta, comparison, predictions, cv_folds = load_data()
models = load_models(meta["model_names"])
BEST = meta["best_model_name"]
MODEL_COLORS = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(meta["model_names"])}

# ---------- Helpers ----------
def normalize(series, invert=False):
    lo, hi = series.min(), series.max()
    norm = (series - lo) / (hi - lo + 1e-9)
    return 1 - norm if invert else norm

def build_radar_df():
    radar_specs = [
        ("RMSE (inv)", "Test RMSE", True),
        ("MAE (inv)", "Test MAE", True),
        ("RMSLE (inv)", "Test RMSLE", True) if "Test RMSLE" in comparison.columns else None,
        ("R2", "Test R2", False),
        ("Adj R2", "Test Adjusted R2", False) if "Test Adjusted R2" in comparison.columns else None,
    ]
    radar_specs = [s for s in radar_specs if s is not None]
    out = pd.DataFrame(index=comparison.index)
    for label, col, invert in radar_specs:
        out[label] = normalize(comparison[col], invert=invert)
    return out

def build_stock_chart_df(sample_n=45):
    pivot, actual_ref = {}, None
    for m in meta["model_names"]:
        sub = predictions[predictions["Model"] == m].reset_index(drop=True)
        if actual_ref is None:
            actual_ref = sub["Actual"]
        pivot[m] = sub["Predicted"]
    df = pd.DataFrame(pivot)
    df["Actual"] = actual_ref
    df = df.sort_values("Actual").reset_index(drop=True)
    if len(df) > sample_n:
        idx = np.linspace(0, len(df) - 1, sample_n).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    df["x"] = range(len(df))
    return df

def build_geo_sunburst():
    """Sunburst of dataset geographic coverage: Malaysia -> State -> Area."""
    ids, labels, parents, values = [], [], [], []
    for state, areas in meta["areas_by_state"].items():
        for area in areas:
            ids.append(f"{state}/{area}")
            labels.append(area)
            parents.append(state)
            values.append(1)
    for state, areas in meta["areas_by_state"].items():
        ids.append(state)
        labels.append(state)
        parents.append("Malaysia")
        values.append(len(areas))
    total_areas = sum(len(a) for a in meta["areas_by_state"].values())
    ids.append("Malaysia")
    labels.append("Malaysia")
    parents.append("")
    values.append(total_areas)

    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total", insidetextorientation="radial",
        marker=dict(colorscale="Blues", line=dict(width=1, color="white")),
        hovertemplate="<b>%{label}</b><br>Areas covered: %{value}<extra></extra>",
    ))
    fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10),
                       paper_bgcolor="rgba(0,0,0,0)", hoverlabel=HOVERLABEL)
    return fig

def build_error_ridgeline():
    """Ridgeline (stacked KDE) of % prediction error per model."""
    models_order = list(meta["model_names"])
    fig = go.Figure()
    offset_step = 1.0
    tick_vals, tick_text = [], []
    for i, m in enumerate(models_order):
        sub = predictions[predictions["Model"] == m]
        denom = sub["Actual"].replace(0, np.nan)
        pct_err = ((sub["Predicted"] - sub["Actual"]) / denom * 100).replace([np.inf, -np.inf], np.nan).dropna()
        if len(pct_err) < 5:
            continue
        lo, hi = pct_err.quantile(0.02), pct_err.quantile(0.98)
        pct_err = pct_err.clip(lo, hi)
        kde = gaussian_kde(pct_err)
        xs = np.linspace(pct_err.min(), pct_err.max(), 200)
        ys = kde(xs)
        ys = ys / ys.max() * 0.85

        y_base = i * offset_step
        color = MODEL_COLORS[m]
        rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.35)"

        fig.add_trace(go.Scatter(x=xs, y=[y_base] * len(xs), mode="lines",
                                  line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=ys + y_base, mode="lines",
                                  line=dict(color=color, width=2), fill="tonexty",
                                  fillcolor=rgba, name=m, showlegend=False,
                                  hovertemplate=f"<b>{m}</b><br>Error: %{{x:.1f}}%<extra></extra>"))
        tick_vals.append(y_base)
        tick_text.append(m)

    fig.add_vline(x=0, line_dash="dash", line_color="#9aa1ab", line_width=1)
    fig.update_layout(
        height=380, margin=dict(t=20, b=50, l=90, r=20),
        xaxis=dict(title="Prediction error (%)", zeroline=False, gridcolor="#eef0f3", automargin=True),
        yaxis=dict(tickvals=tick_vals, ticktext=tick_text, showgrid=False, zeroline=False, automargin=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, hoverlabel=HOVERLABEL, dragmode="zoom",
    )
    return fig

def build_facet_scatter(models_to_show):
    """Small-multiples Actual vs Predicted — one clean panel per model."""
    n = len(models_to_show)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=models_to_show,
                         horizontal_spacing=0.08, vertical_spacing=0.16)
    lims = [predictions[["Actual", "Predicted"]].min().min(), predictions[["Actual", "Predicted"]].max().max()]
    for i, m in enumerate(models_to_show):
        r, c = divmod(i, cols)
        sub = predictions[predictions["Model"] == m]
        fig.add_trace(go.Scatter(x=sub["Actual"], y=sub["Predicted"], mode="markers",
                                  marker=dict(color=MODEL_COLORS[m], opacity=0.45, size=6),
                                  name=m, showlegend=False), row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                  line=dict(color="#0b0b0b", dash="dash", width=1.2),
                                  showlegend=False), row=r + 1, col=c + 1)
    fig.update_layout(height=280 * rows, margin=dict(t=50, b=40, l=60, r=20),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       hoverlabel=HOVERLABEL, dragmode="zoom")
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f3", automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f3", automargin=True)
    return fig

def build_performance_bubble():
    """2D bubble chart replacing the old hard-to-use 3D scatter:
    RMSE x MAE positions, bubble size = R2."""
    x_col, y_col, z_col = "Test RMSE", "Test MAE", "Test R2"
    r2 = comparison[z_col]
    sizes = 26 + (r2 - r2.min()) / (r2.max() - r2.min() + 1e-9) * 34

    fig = go.Figure()
    for m in meta["model_names"]:
        is_best = m == BEST
        fig.add_trace(go.Scatter(
            x=[comparison.loc[m, x_col]], y=[comparison.loc[m, y_col]],
            mode="markers+text", text=[m], textposition="top center",
            textfont=dict(size=12, color="#0f2740"),
            marker=dict(size=sizes[m], color=MODEL_COLORS[m],
                        line=dict(width=3 if is_best else 1.5, color="#0f2740" if is_best else "white"),
                        opacity=0.85),
            name=m, showlegend=False,
            hovertemplate=(f"<b>{m}</b><br>{x_col}: RM %{{x:,.0f}}<br>{y_col}: RM %{{y:,.0f}}"
                            f"<br>{z_col}: {comparison.loc[m, z_col]:.3f}<extra></extra>"),
        ))
    fig.update_layout(
        height=420, margin=dict(t=40, b=50, l=70, r=20),
        xaxis=dict(title=f"{x_col} (RM) — lower is better", gridcolor="#eef0f3", automargin=True),
        yaxis=dict(title=f"{y_col} (RM) — lower is better", gridcolor="#eef0f3", automargin=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel=HOVERLABEL,
        dragmode="zoom",
        annotations=[dict(text="Bubble size = Test R² (bigger is better)",
                           x=1, y=1.1, xref="paper", yref="paper", xanchor="right",
                           showarrow=False, font=dict(size=11, color="#6b7280"))],
    )
    return fig

def build_metrics_heatmap():
    """Model x metric heatmap. Each column is normalized so that 1 = best
    model on that metric (regardless of whether the metric is 'lower is
    better' or 'higher is better'), so color always means the same thing —
    darker is better — while the actual value is annotated per cell with
    a text color chosen for contrast against its own cell."""
    metric_cols = [c for c in comparison.columns if c.startswith("Test")]
    model_order = meta["model_names"]
    lower_better = {c: any(k in c for k in ["RMSE", "MAE", "RMSLE", "MAPE"]) for c in metric_cols}

    z, text = [], []
    for m in model_order:
        row_z, row_text = [], []
        for c in metric_cols:
            col_vals = comparison[c]
            lo, hi = col_vals.min(), col_vals.max()
            norm = (comparison.loc[m, c] - lo) / (hi - lo + 1e-9)
            score = 1 - norm if lower_better[c] else norm
            row_z.append(score); row_text.append(f"{comparison.loc[m, c]:,.3f}")
        z.append(row_z); text.append(row_text)

    # Lighter scale so annotated numbers stay legible without needing dark text everywhere.
    scale = [[0.0, "#fbfaf7"], [0.25, "#eef2f8"], [0.5, "#cfdcec"],
             [0.75, "#9db6d6"], [1.0, "#5c85b8"]]

    fig = go.Figure(go.Heatmap(
        z=z, x=[c.replace("Test ", "") for c in metric_cols], y=model_order,
        text=text, colorscale=scale, zmin=0, zmax=1, showscale=False, xgap=4, ygap=4,
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    ))

    # Per-cell annotation with contrast-aware text color.
    annotations = []
    for i in range(len(model_order)):
        for j in range(len(metric_cols)):
            score = z[i][j]
            txt_color = "#ffffff" if score > 0.62 else "#0f2740"
            annotations.append(dict(
                x=j, y=i, text=text[i][j], showarrow=False,
                font=dict(size=12.5, color=txt_color, family="Georgia, 'Times New Roman', serif"),
            ))

    fig.update_layout(
        height=110 + 60 * len(model_order), margin=dict(t=40, b=20, l=140, r=20),
        xaxis=dict(side="top", showgrid=False, automargin=True,
                   tickfont=dict(size=12, color="#0f2740", family="Georgia, 'Times New Roman', serif")),
        yaxis=dict(showgrid=False, autorange="reversed", automargin=True,
                   tickfont=dict(size=12.5, color="#0f2740", family="Georgia, 'Times New Roman', serif")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel=HOVERLABEL,
        annotations=annotations, dragmode="zoom",
    )
    for j in range(len(metric_cols)):
        best_row = int(np.argmax([z[i][j] for i in range(len(model_order))]))
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=best_row - 0.5, y1=best_row + 0.5,
                      line=dict(color="#0f2740", width=2), fillcolor="rgba(0,0,0,0)")
    return fig

def build_cv_stability_chart():
    """Vertical box + jittered points per model — replaces the earlier
    tilted raincloud, which got cluttered with overlapping tooltips."""
    fig = go.Figure()
    for m in meta["model_names"]:
        sub = cv_folds[cv_folds["Model"] == m]
        c = MODEL_COLORS[m]
        fig.add_trace(go.Box(
            y=sub["Fold RMSE"], x=[m] * len(sub), name=m,
            marker=dict(color=c, size=6, opacity=0.55, line=dict(width=1, color="white")),
            line=dict(color=c),
            fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.18)",
            boxpoints="all", jitter=0.35, pointpos=0, width=0.5, showlegend=False,
            hovertemplate=f"<b>{m}</b><br>Fold RMSE: RM %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        height=400, margin=dict(t=20, b=70, l=80, r=20),
        yaxis=dict(title="Fold RMSE (RM)", gridcolor="#eef0f3", tickformat=",.0f", automargin=True),
        xaxis=dict(showgrid=False, automargin=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest", hoverlabel=HOVERLABEL, dragmode="zoom",
    )
    return fig

def run_budget_search(model, meta, budget_lo, budget_hi, states_scope):
    """Vectorized reverse-search: for every State x Area x Type x Tenure combo,
    predict the price with the champion model and keep the ones inside the
    user's budget range."""
    types = meta["all_types"]
    tenures = meta["all_tenures"]
    combos = [
        (s, a, t, tn)
        for s in states_scope
        for a in meta["areas_by_state"][s]
        for t in types
        for tn in tenures
    ]
    if not combos:
        return pd.DataFrame()

    base = pd.DataFrame(combos, columns=["State", "Area", "_Type", "_Tenure"])
    base["Transactions"] = 10
    base["n_types"] = 1
    base["is_aggregated_type"] = 0
    base["n_tenure"] = 1
    base["is_aggregated_tenure"] = 0
    for t in types:
        base[f"Type_{t}"] = (base["_Type"] == t).astype(int)
    for tn in tenures:
        base[f"Tenure_{tn}"] = (base["_Tenure"] == tn).astype(int)

    X = base[meta["feature_cols"]]
    base["PredictedPrice"] = model.predict(X)

    matches = base[(base["PredictedPrice"] >= budget_lo) & (base["PredictedPrice"] <= budget_hi)].copy()
    budget_mid = (budget_lo + budget_hi) / 2
    matches["DistanceFromBudget"] = (matches["PredictedPrice"] - budget_mid).abs()
    matches = matches.sort_values("DistanceFromBudget")
    return matches[["State", "Area", "_Type", "_Tenure", "PredictedPrice"]].rename(
        columns={"_Type": "Type", "_Tenure": "Tenure"}
    )

def build_budget_matches_chart(top, budget_lo, budget_hi):
    """Replaces the old dumbbell chart (which overlapped same-area/same-type
    combos). Groups rows by Area, jitters same-area points apart, colors by
    Type and uses marker symbol for Tenure so overlapping combos stay legible."""
    areas_order = list(dict.fromkeys(top["Area"]))
    row_of = {a: i for i, a in enumerate(areas_order)}
    type_list = sorted(top["Type"].unique())
    type_color = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(type_list)}
    tenure_symbol = {t: s for t, s in zip(sorted(top["Tenure"].unique()),
                                           ["circle", "diamond", "square", "triangle-up", "star"])}

    base_rows = [row_of[a] for a in top["Area"]]
    seen = {}
    y_jittered = []
    for base_row in base_rows:
        n_in_row = base_rows.count(base_row)
        i_in_row = seen.get(base_row, 0)
        seen[base_row] = i_in_row + 1
        offset = 0 if n_in_row == 1 else -0.16 + 0.32 * i_in_row / (n_in_row - 1)
        y_jittered.append(base_row + offset)

    fig = go.Figure()
    fig.add_vrect(x0=budget_lo, x1=budget_hi, fillcolor="#eb6834", opacity=0.08,
                  line_width=0, annotation_text="Your budget range", annotation_position="top left")

    for t in type_list:
        mask = (top["Type"] == t).values
        fig.add_trace(go.Scatter(
            x=top["PredictedPrice"][mask],
            y=[y for y, m in zip(y_jittered, mask) if m],
            mode="markers",
            marker=dict(size=13, color=type_color[t],
                        symbol=[tenure_symbol[tn] for tn, m in zip(top["Tenure"], mask) if m],
                        line=dict(width=1.5, color="white")),
            name=t,
            customdata=top.loc[mask, ["Area", "Tenure"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>Type: " + t +
                          "<br>Tenure: %{customdata[1]}<br>Predicted: RM %{x:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        height=max(360, 46 * len(areas_order)), margin=dict(t=40, b=60, l=140, r=30),
        yaxis=dict(tickmode="array", tickvals=list(range(len(areas_order))),
                   ticktext=areas_order, autorange="reversed", gridcolor="#f3f5f8", automargin=True),
        xaxis=dict(title="Predicted price (RM)", gridcolor="#eef0f3", automargin=True),
        legend=dict(orientation="h", y=-0.18, title="Property type"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel=HOVERLABEL,
        dragmode="zoom",
    )
    return fig

def render_plotly_hover_dim(fig, key, height=340, dim_opacity=0.15, protect_index=None, enable_fullscreen=False):
    """Like st.plotly_chart, but dims all non-hovered traces on hover.
    Also applies the same 'default = full view, no pan-to-blank-space'
    behavior as PLOTLY_CONFIG since this bypasses st.plotly_chart.

    protect_index: index of a trace (e.g. the "Actual" line) that:
      - is never dimmed, and hovering it doesn't dim the others,
      - can't be hidden or isolated alone via its own legend entry,
      - stays visible whenever ANOTHER trace is isolated via legend double-click
        (so isolating a model's line keeps it comparable against Actual).
    enable_fullscreen: adds a real Plotly modebar button (via
    modeBarButtonsToAdd) so it renders as one more icon inside Plotly's own
    toolbar — same row, same background, same hover behavior as pan/zoom/
    reset — instead of a separately positioned HTML button next to it.
    """
    fig_json = fig.to_json()
    protect_js = "null" if protect_index is None else str(protect_index)
    enable_fs_js = "true" if enable_fullscreen else "false"

    html = f"""
    <div id="{key}_wrap" style="position:relative;width:100%;background:#ffffff;">
      <div id="{key}" style="width:100%;height:{height}px;"></div>
    </div>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <script>
      (function() {{
        const figData = {fig_json};
        const gd = document.getElementById("{key}");
        const wrap = document.getElementById("{key}_wrap");
        const protectIdx = {protect_js};
        const baseHeight = {height};
        const n = figData.data.length;
        const enableFullscreen = {enable_fs_js};
        let isolatedIdx = null;

        // Material Design "fullscreen" glyph — a 4-corner-bracket icon that's
        // symmetric top/bottom and left/right, so Plotly's internal icon
        // y-flip (used for its own built-in icons) can't distort it.
        const fullscreenIcon = {{
          width: 24, height: 24, ascent: 24, descent: 0,
          path: 'M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z',
        }};

        const config = {{
          displayModeBar: true, displaylogo: false, scrollZoom: false,
          doubleClick: 'reset+autosize', responsive: true,
          modeBarButtonsToRemove: ['pan2d','lasso2d','select2d','autoScale2d'],
          modeBarButtonsToAdd: enableFullscreen ? [{{
            name: 'toggleFullscreen',
            title: 'Toggle fullscreen',
            icon: fullscreenIcon,
            click: function() {{
              if (!document.fullscreenElement) {{
                (wrap.requestFullscreen && wrap.requestFullscreen()) ||
                (wrap.webkitRequestFullscreen && wrap.webkitRequestFullscreen());
              }} else {{
                (document.exitFullscreen && document.exitFullscreen()) ||
                (document.webkitExitFullscreen && document.webkitExitFullscreen());
              }}
            }},
          }}] : [],
        }};

        function init() {{
          Plotly.newPlot(gd, figData.data, figData.layout, config).then(function() {{
            Plotly.relayout(gd, {{hoverdistance: 30}});

            if (enableFullscreen) {{
              document.addEventListener('fullscreenchange', function() {{
                if (document.fullscreenElement === wrap) {{
                  wrap.style.padding = '18px';
                  gd.style.height = 'calc(100vh - 36px)';
                }} else {{
                  wrap.style.padding = '';
                  gd.style.height = baseHeight + 'px';
                }}
                setTimeout(function() {{ Plotly.Plots.resize(gd); }}, 60);
              }});
            }}

            gd.on('plotly_hover', function(evt) {{
              const idx = evt.points[0].curveNumber;
              if (protectIdx !== null && idx === protectIdx) {{
                Plotly.restyle(gd, {{'opacity': new Array(n).fill(1)}});
                return;
              }}
              Plotly.restyle(gd, {{'opacity': Array.from({{length:n}}, (_,i)=> (i===idx || i===protectIdx) ? 1 : {dim_opacity})}});
            }});
            gd.on('plotly_unhover', function() {{
              Plotly.restyle(gd, {{'opacity': new Array(n).fill(1)}});
            }});

            if (protectIdx !== null) {{
              gd.on('plotly_legendclick', function(evt) {{
                if (evt.curveNumber === protectIdx) return false;
              }});
              gd.on('plotly_legenddoubleclick', function(evt) {{
                const idx = evt.curveNumber;
                if (idx === protectIdx) return false; // Actual can't be isolated alone
                if (isolatedIdx === idx) {{
                  Plotly.restyle(gd, {{visible: new Array(n).fill(true)}});
                  isolatedIdx = null;
                }} else {{
                  const vis = Array.from({{length:n}}, (_,i) => (i === idx || i === protectIdx) ? true : 'legendonly');
                  Plotly.restyle(gd, {{visible: vis}});
                  isolatedIdx = idx;
                }}
                return false;
              }});
            }}
          }});
        }}
        requestAnimationFrame(function() {{ requestAnimationFrame(init); }});
      }})();
    </script>
    """
    components.html(html, height=height + 10)

def render_prediction_carousel(all_preds, model_colors, best_model, default_model):
    """Swipeable / draggable card carousel for browsing each model's prediction —
    drag with mouse or touch, arrow buttons, or dot indicators. Autoplays
    until the user interacts (drag/click), then stops until the component
    is rebuilt (new page visit or a fresh Estimate Price click)."""
    model_names = list(all_preds.keys())
    default_index = model_names.index(default_model) if default_model in model_names else 0

    cards_html, dots_html = "", ""
    for i, name in enumerate(model_names):
        price = all_preds[name]
        badge = f'<div class="pbadge">{icon("award", size=12, margin_right=4)}Champion</div>' if name == best_model else ""
        cards_html += f"""
        <div class="pcard">
          <div class="pcard-inner">
            {badge}
            <div class="pcard-model">{name}</div>
            <div class="pcard-price">RM {price:,.0f}</div>
            <div class="pcard-sub">Predicted median price</div>
          </div>
        </div>"""
        dots_html += f'<div class="pdot" data-i="{i}"></div>'

    template = """
    <style>
      .pcarousel { position:relative; width:100%; user-select:none; padding: 4px 30px; box-sizing:border-box; }
      .pcarousel-viewport { overflow:hidden; border-radius:16px; }
      .pcarousel-track { display:flex; transition: transform .45s cubic-bezier(.22,.9,.32,1); cursor:grab; touch-action: pan-y; }
      .pcarousel-track.dragging { transition:none; cursor:grabbing; }
      .pcard { flex: 0 0 100%; box-sizing:border-box; padding: 0 4px; }
      .pcard-inner {
        position:relative; background: linear-gradient(135deg, #0f2740, #1c4b7c);
        border-radius:16px; padding: 30px; color:white; text-align:center;
        box-shadow: 0 10px 28px rgba(15,39,64,.28);
      }
      .pbadge {
        position:absolute; top:12px; right:16px; font-size:11px; font-weight:600;
        background: rgba(255,255,255,.15); padding: 4px 10px; border-radius:999px;
        display:flex; align-items:center;
      }
      .pcard-model { font-size:.9rem; opacity:.85; font-weight:600; letter-spacing:.02em; }
      .pcard-price { font-size:2.4rem; font-weight:800; margin-top:6px; }
      .pcard-sub { font-size:.82rem; opacity:.75; margin-top:4px; }
      .parrow {
        position:absolute; top:50%; transform:translateY(-50%); width:36px; height:36px;
        border-radius:50%; background:#ffffff; color:#1c4b7c; border:1px solid #e6e9ee;
        display:flex; align-items:center; justify-content:center; cursor:pointer;
        box-shadow: 0 4px 14px rgba(15,39,64,.16);
        transition: background .2s ease, transform .15s ease, box-shadow .2s ease; z-index:2;
      }
      .parrow:hover { background:#eaf1fb; transform:translateY(-50%) scale(1.08); box-shadow: 0 6px 18px rgba(15,39,64,.22); }
      .parrow.left { left:-6px; } .parrow.right { right:-6px; }
      .parrow svg { width:16px; height:16px; }
      .pdots { display:flex; justify-content:center; gap:6px; margin-top:14px; }
      .pdot { width:7px; height:7px; border-radius:50%; background:#d7dde5; cursor:pointer; transition: all .2s ease; }
      .pdot.active { background:#1c4b7c; transform:scale(1.4); }
    </style>
    <div class="pcarousel">
      <div class="pcarousel-viewport">
        <div class="pcarousel-track" id="ptrack">__CARDS__</div>
      </div>
      <button class="parrow left" id="pprev"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><polyline points="15 18 9 12 15 6"></polyline></svg></button>
      <button class="parrow right" id="pnext"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><polyline points="9 18 15 12 9 6"></polyline></svg></button>
      <div class="pdots" id="pdots">__DOTS__</div>
    </div>
    <script>
      (function() {
        const track = document.getElementById('ptrack');
        const dots = document.querySelectorAll('.pdot');
        const n = __N__;
        let idx = __START__;
        let startX = 0, deltaX = 0, dragging = false;
        let autoplayTimer = null;

        function render() {
          track.style.transform = `translateX(${-idx * 100}%)`;
          dots.forEach((d, i) => d.classList.toggle('active', i === idx));
        }
        function go(i) { idx = (i + n) % n; render(); }
        function startAutoplay() {
          if (n <= 1) return;
          stopAutoplay();
          autoplayTimer = setInterval(() => go(idx + 1), 2800);
        }
        function stopAutoplay() { if (autoplayTimer) { clearInterval(autoplayTimer); autoplayTimer = null; } }

        document.getElementById('pprev').onclick = () => { stopAutoplay(); go(idx - 1); };
        document.getElementById('pnext').onclick = () => { stopAutoplay(); go(idx + 1); };
        dots.forEach(d => d.onclick = () => { stopAutoplay(); go(parseInt(d.dataset.i)); });

        track.addEventListener('pointerdown', e => {
          stopAutoplay(); dragging = true; startX = e.clientX; track.classList.add('dragging');
        });
        window.addEventListener('pointermove', e => { if (dragging) deltaX = e.clientX - startX; });
        window.addEventListener('pointerup', () => {
          if (!dragging) return;
          dragging = false; track.classList.remove('dragging');
          if (deltaX > 60) go(idx - 1);
          else if (deltaX < -60) go(idx + 1);
          else render();
          deltaX = 0;
        });

        render();
        startAutoplay();
      })();
    </script>
    """
    html = (template.replace("__CARDS__", cards_html)
                     .replace("__DOTS__", dots_html)
                     .replace("__N__", str(len(model_names)))
                     .replace("__START__", str(default_index)))
    components.html(html, height=270)

def render_metric_compare(metric_options, model_order, best_model):
    """Custom HTML dropdown (avoids Plotly's native updatemenus, which
    misplaced itself on first paint inside the Streamlit iframe) driving
    a Plotly.animate() call so the bar-length transition is preserved.

    Value labels are drawn as layout annotations rather than the bar
    trace's own `textposition: 'outside'`. Plotly's automatic "outside"
    text placement is computed from the bar's rendered geometry at the
    moment of the *first* draw, and that computation is what silently
    fails for the initial (default) metric — every later Plotly.animate()
    call goes through a different, already-settled code path, which is
    why only the very first metric was ever affected. Annotations are
    positioned purely from data coordinates and aren't subject to that
    first-draw geometry bug, so this removes the root cause instead of
    trying to time around it.
    """
    lower_better = {m: any(k in m for k in ["RMSE", "MAE", "RMSLE", "MAPE"]) for m in metric_options}
    default_metric = "Test RMSE" if "Test RMSE" in metric_options else metric_options[0]
    y_labels = [f"★ {m}" if m == best_model else m for m in model_order]
    colors = [MODEL_COLORS[m] for m in model_order]

    frames = {}
    for metric in metric_options:
        vals = comparison.loc[model_order, metric].values.astype(float)
        mx = vals.max() or 1.0
        frames[metric] = {"x": (vals / mx).tolist(),
                           "text": [f"{v:,.3f}" for v in vals],
                           "caption": "Lower is better" if lower_better[metric] else "Higher is better"}

    height = 110 + 48 * len(model_order)
    options_html = "".join(f'<div class="mopt" data-m="{m}">{m.replace("Test ", "")}</div>' for m in metric_options)

    html = f"""
    <style>
      .mwrap {{ font-family: 'Segoe UI','Inter',sans-serif; }}
      .mddl {{ position:relative; width:220px; margin-bottom:14px; }}
      .mtrigger {{ display:flex; align-items:center; justify-content:space-between; cursor:pointer;
        background:#eaf1fb; border:1px solid #d7e3f4; border-radius:10px; padding:9px 14px;
        font-size:13px; font-weight:600; color:#1c4b7c; transition: background .2s ease; }}
      .mtrigger:hover {{ background:#dcebfa; }}
      .mtrigger .chev {{ transition: transform .2s ease; }}
      .mtrigger.open .chev {{ transform: rotate(180deg); }}
      .mmenu {{ position:absolute; top:calc(100% + 6px); left:0; width:100%; z-index:5;
        background:white; border:1px solid #e6e9ee; border-radius:10px;
        box-shadow:0 10px 26px rgba(15,39,64,.14); overflow:hidden;
        max-height:0; opacity:0; transition: max-height .22s ease, opacity .18s ease; }}
      .mmenu.show {{ max-height:320px; opacity:1; }}
      .mopt {{ padding:9px 14px; font-size:13px; color:#0f2740; cursor:pointer; transition: background .15s ease; }}
      .mopt:hover {{ background:#f3f6fa; }}
      .mopt.active {{ background:#eaf1fb; color:#1c4b7c; font-weight:600; }}
      .mcaption {{ font-size:12px; color:#6b7280; margin-bottom:8px; }}
    </style>
    <div class="mwrap">
      <div class="mddl">
        <div class="mtrigger" id="mtrig">
          <span id="mtrig-label">{default_metric.replace("Test ", "")}</span>
          <svg class="chev" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polyline points="6 9 12 15 18 9"></polyline></svg>
        </div>
        <div class="mmenu" id="mmenu">{options_html}</div>
      </div>
      <div class="mcaption" id="mcaption">{frames[default_metric]["caption"]}</div>
      <div id="mchart" style="width:100%;height:{height}px;"></div>
    </div>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <script>
      (function() {{
        const frames = {json.dumps(frames)};
        const yLabels = {json.dumps(y_labels)};
        const colors = {json.dumps(colors)};
        const gd = document.getElementById("mchart");
        const defaultMetric = "{default_metric}";
        const initial = frames[defaultMetric];

        // Value labels are laid out ourselves (as annotations, positioned
        // from data coordinates), not via the bar trace's own
        // `textposition: 'outside'` — see the function docstring for why.
        function buildAnnotations(xs, texts) {{
          return xs.map((v, i) => ({{
            x: v, y: yLabels[i], xref: 'x', yref: 'y',
            text: texts[i], showarrow: false,
            xanchor: 'left', xshift: 8, align: 'left',
            font: {{size: 12.5, color: '#0f2740', family: "'Segoe UI','Inter',sans-serif"}},
          }}));
        }}

        function init() {{
          Plotly.newPlot(gd, [{{
            x: initial.x, y: yLabels, orientation: 'h', type: 'bar',
            marker: {{color: colors}}, hovertext: initial.text,
            cliponaxis: false, hovertemplate: '<b>%{{y}}</b><br>%{{hovertext:}}<extra></extra>',
          }}], {{
            height: {height}, margin: {{t:10, b:10, l:160, r:80}},
            xaxis: {{range:[0,1.25], showticklabels:false, gridcolor:'#eef0f3', zeroline:false, automargin:true}},
            yaxis: {{autorange:'reversed', automargin:true}},
            dragmode: 'zoom',
            annotations: buildAnnotations(initial.x, initial.text),
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', showlegend: false,
          }}, {{displayModeBar:false, responsive:true, scrollZoom:false}});
        }}
        // Kept as a general first-paint safety net (waits for the iframe's
        // layout to settle before the initial draw); the RMSE label bug
        // itself is now fixed by the annotations approach above, not by
        // this delay.
        requestAnimationFrame(function() {{ requestAnimationFrame(init); }});

        const trig = document.getElementById("mtrig");
        const menu = document.getElementById("mmenu");
        const label = document.getElementById("mtrig-label");
        const caption = document.getElementById("mcaption");

        trig.onclick = () => {{ trig.classList.toggle('open'); menu.classList.toggle('show'); }};
        document.addEventListener('click', (e) => {{
          if (!trig.contains(e.target) && !menu.contains(e.target)) {{
            trig.classList.remove('open'); menu.classList.remove('show');
          }}
        }});

        menu.querySelectorAll('.mopt').forEach(opt => {{
          if (opt.dataset.m === defaultMetric) opt.classList.add('active');
          opt.onclick = () => {{
            const metric = opt.dataset.m;
            menu.querySelectorAll('.mopt').forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
            label.textContent = metric.replace('Test ', '');
            caption.textContent = frames[metric].caption;
            trig.classList.remove('open'); menu.classList.remove('show');
            Plotly.animate(gd, {{data: [{{x: frames[metric].x, hovertext: frames[metric].text}}], traces:[0]}},
                {{transition: {{duration:500, easing:'cubic-in-out'}}, frame: {{duration:500, redraw:false}}}});
            Plotly.relayout(gd, {{annotations: buildAnnotations(frames[metric].x, frames[metric].text)}});
          }};
        }});
      }})();
    </script>
    """
    components.html(html, height=height + 90)

# ---------- Header ----------
st.markdown(f"""
<div class="app-header">
    <h1>{icon("building", size=26, margin_right=10)}Malaysia Housing Price Intelligence</h1>
    <p>Automated valuation model (AVM) &nbsp;·&nbsp; Trained on township-level transaction data across Malaysia</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
n_states = len(meta["states"])
n_areas = sum(len(v) for v in meta["areas_by_state"].values())
n_types = len(meta["all_types"])
n_tenures = len(meta["all_tenures"])

st.sidebar.markdown(f"""
<div style="display:flex;align-items:center;margin-bottom:2px;">
  {icon("building", size=22, margin_right=8)}<span style="font-size:1.25rem;font-weight:700;">AVM Dashboard</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.caption("Automated valuation model for Malaysian residential properties")
st.sidebar.divider()

st.sidebar.markdown(f"""
<div class="sidebar-card">
  <div class="label">Champion Model</div>
  <div class="value">{icon("award", size=16, margin_right=6)}{BEST}</div>
</div>
<div class="sidebar-card">
  <div class="label">Test RMSE</div>
  <div class="value">RM {comparison.loc[BEST, 'Test RMSE']:,.0f}</div>
</div>
<div class="sidebar-card">
  <div class="label">Test R²</div>
  <div class="value">{comparison.loc[BEST, 'Test R2']:.3f}</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;font-weight:600;margin-bottom:6px;">'
    f'{icon("bars", size=16, margin_right=8)}Dataset coverage</div>',
    unsafe_allow_html=True
)
sc1, sc2 = st.sidebar.columns(2)
sc1.metric("States", n_states)
sc2.metric("Areas", n_areas)
sc3, sc4 = st.sidebar.columns(2)
sc3.metric("Property types", n_types)
sc4.metric("Tenure types", n_tenures)

st.sidebar.divider()
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;font-weight:600;margin-bottom:4px;">'
    f'{icon("layers", size=16, margin_right=8)}Model legend</div>',
    unsafe_allow_html=True
)
for m in meta["model_names"]:
    crown = f'<span class="legend-crown">{icon("award", size=13, margin_right=4)}</span>' if m == BEST else ""
    st.sidebar.markdown(
        f'<div class="legend-row"><span class="legend-dot" style="background:{MODEL_COLORS[m]}"></span>{crown}{m}</div>',
        unsafe_allow_html=True
    )

st.sidebar.divider()
with st.sidebar.expander("About this app", icon=":material/info:"):
    st.markdown(
        f"This dashboard compares **{len(meta['model_names'])} regression models** trained to "
        f"estimate median transaction prices for residential properties across Malaysia. "
        f"Use **Estimate by Property** to value a specific property, or **Estimate by Budget** "
        f"to reverse-search which areas & property types fit a given budget range. "
        f"Model diagnostics and cross-validation stability live under **Model performance**."
    )

tab_overview, tab_estimator, tab_performance = st.tabs([
    ":material/dashboard: Overview",
    ":material/payments: Price estimator",
    ":material/monitoring: Model performance",
])

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab_overview:
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in [
        (k1, "Champion model", BEST),
        (k2, "Test RMSE", f"RM {comparison.loc[BEST, 'Test RMSE']:,.0f}"),
        (k3, "Test R²", f"{comparison.loc[BEST, 'Test R2']:.3f}"),
        (k4, "Models compared", str(len(meta["model_names"]))),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card"><div class="label">{label}</div>
                <div class="value">{value}</div></div>""", unsafe_allow_html=True)

    st.markdown("###  ")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-label">Metric profile — all models</div>', unsafe_allow_html=True)
        radar_df = build_radar_df()
        fig = go.Figure()
        for m in meta["model_names"]:
            color = MODEL_COLORS[m]
            rgba_fill = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.22)"
            fig.add_trace(go.Scatterpolar(
                r=radar_df.loc[m].values.tolist() + [radar_df.loc[m].values[0]],
                theta=radar_df.columns.tolist() + [radar_df.columns[0]],
                name=m, mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color, line=dict(width=1.5, color="white")),
                fill='toself', fillcolor=rgba_fill,
                hoveron='points',
                hovertemplate=f"<b>{m}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 1]), bgcolor="rgba(0,0,0,0)"),
            showlegend=True, legend=dict(orientation="h", y=-0.1),
            height=340, margin=dict(t=10, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)", hovermode="closest",
            hoverlabel=HOVERLABEL,
        )
        # unique key: works around the known Streamlit bug where a Plotly
        # chart doesn't restore its original size after exiting fullscreen
        # (github.com/streamlit/streamlit/issues/11327)
        st.plotly_chart(fig, use_container_width=True, key="overview_radar", config=PLOTLY_CONFIG)

    with c2:
        st.markdown('<div class="chart-label">Predicted vs actual — all models</div>', unsafe_allow_html=True)
        stock_df = build_stock_chart_df()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_df["x"], y=stock_df["Actual"], name="Actual",
            mode="lines", line=dict(color="#0b0b0b", width=2, shape="spline"),
            fill="tozeroy", fillcolor="rgba(11,11,11,0.04)",
            hovertemplate="RM %{y:,.0f}<extra>Actual</extra>",
        ))
        for m in meta["model_names"]:
            fig.add_trace(go.Scatter(
                x=stock_df["x"], y=stock_df[m], name=m,
                mode="lines", line=dict(color=MODEL_COLORS[m], width=1.6, shape="spline"),
                customdata=stock_df["Actual"],
                hovertemplate=(f"<b>{m}</b><br>Predicted: RM %{{y:,.0f}}"
                                "<br>Actual: RM %{customdata:,.0f}<extra></extra>"),
            ))
        fig.update_layout(
            height=340, margin=dict(t=20, b=40, l=70, r=20),
            xaxis=dict(showticklabels=False, showgrid=False, showspikes=True,
                       spikemode="across", spikethickness=1, automargin=True),
            yaxis=dict(tickprefix="RM ", tickformat=",.0f", gridcolor="#eef0f3", automargin=True),
            legend=dict(orientation="h", y=-0.15),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hovermode="closest", hoverlabel=HOVERLABEL, dragmode="zoom",
        )
        # protect_index=0 -> the "Actual" trace: never dims, can't be isolated
        # or hidden alone via legend, and stays visible when a model is isolated.
        render_plotly_hover_dim(fig, "overview_predicted_vs_actual", height=340,
                                 protect_index=0, enable_fullscreen=True)

    st.markdown("###  ")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="chart-label">Geographic coverage of training data</div>', unsafe_allow_html=True)
        st.plotly_chart(build_geo_sunburst(), use_container_width=True, key="overview_geo_sunburst", config=PLOTLY_CONFIG)
    with c4:
        st.markdown('<div class="chart-label">Prediction error distribution (ridgeline)</div>', unsafe_allow_html=True)
        st.plotly_chart(build_error_ridgeline(), use_container_width=True, key="overview_error_ridgeline", config=PLOTLY_CONFIG)

    st.markdown("###  ")
    st.markdown('<div class="chart-label">Performance landscape — RMSE × MAE, bubble size = R²</div>', unsafe_allow_html=True)
    st.plotly_chart(build_performance_bubble(), use_container_width=True, key="overview_perf_bubble", config=PLOTLY_CONFIG)

# =========================================================
# TAB 2 — PRICE ESTIMATOR
# =========================================================
with tab_estimator:
    subtab_property, subtab_budget = st.tabs([
        ":material/home_work: Estimate by Property",
        ":material/savings: Estimate by Budget",
    ])

    # ----- Sub-tab: estimate the price of a specific property -----
    with subtab_property:
        st.markdown(
            f'<h3 style="display:flex;align-items:center;margin:0 0 0.5rem 0;'
                f'font-size:1.5rem;font-weight:600;color:#0f2740;">'
            f'{icon("building", size=20, margin_right=8)}Property Details</h3>',
            unsafe_allow_html=True
        )
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            state = c1.selectbox("State", meta["states"])
            area = c2.selectbox("Area", meta["areas_by_state"][state])
            transactions = c3.number_input("Recent Transactions", min_value=0, value=10, step=1)
            c4, c5 = st.columns(2)
            types_selected = c4.multiselect("Property Type(s)", meta["all_types"], default=[meta["all_types"][0]])
            tenures_selected = c5.multiselect("Tenure(s)", meta["all_tenures"], default=[meta["all_tenures"][0]])
            predict_btn = st.button("Estimate Price", type="primary", use_container_width=True,
                                     icon=":material/calculate:")

        if predict_btn:
            if not types_selected or not tenures_selected:
                st.error("Please select at least one Property Type and Tenure.")
            else:
                row = {"State": state, "Area": area, "Transactions": transactions,
                       "n_types": len(types_selected), "is_aggregated_type": int(len(types_selected) > 1),
                       "n_tenure": len(tenures_selected), "is_aggregated_tenure": int(len(tenures_selected) > 1)}
                for t in meta["all_types"]:
                    row[f"Type_{t}"] = int(t in types_selected)
                for t in meta["all_tenures"]:
                    row[f"Tenure_{t}"] = int(t in tenures_selected)
                input_df = pd.DataFrame([row])[meta["feature_cols"]]

                # always compute all models, champion shown first / highlighted
                all_preds = {n: m.predict(input_df)[0] for n, m in models.items()}

                st.markdown("###  ")
                st.markdown(
                    f'<h3 style="display:flex;align-items:center;margin:0 0 0.5rem 0;'
                        f'font-size:1.5rem;font-weight:600;color:#0f2740;">'
                    f'{icon("bars", size=20, margin_right=8)}Estimated Value — All Models</h3>',
                    unsafe_allow_html=True
                )
                st.caption("Drag, swipe, or use the arrows to browse each model's estimate")
                render_prediction_carousel(all_preds, MODEL_COLORS, BEST, BEST)

                st.markdown('<div class="chart-label">Model comparison</div>', unsafe_allow_html=True)
                cmp_fig = go.Figure()
                for m in meta["model_names"]:
                    val = all_preds[m]
                    cmp_fig.add_trace(go.Bar(
                        x=[m], y=[val], marker_color=MODEL_COLORS[m], showlegend=False,
                        marker_line=dict(width=3 if m == BEST else 0, color="#0f2740"),
                        text=[f"RM {val:,.0f}"], textposition="outside",
                        hovertemplate=f"<b>{m}</b><br>RM %{{y:,.0f}}<extra></extra>",
                    ))
                cmp_fig.update_layout(
                    height=320, margin=dict(t=30, b=20, l=60, r=20),
                    yaxis=dict(tickprefix="RM ", tickformat=",.0f", gridcolor="#eef0f3", automargin=True),
                    xaxis=dict(categoryorder="array", categoryarray=meta["model_names"], automargin=True),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    hoverlabel=HOVERLABEL, dragmode="zoom",
                )
                st.plotly_chart(cmp_fig, use_container_width=True, key="estimator_model_comparison", config=PLOTLY_CONFIG)
        else:
            st.info("Fill in the property details and click **Estimate Price**.")

    # ----- Sub-tab: reverse-search by budget -----
    with subtab_budget:
        st.markdown(
            f'<h3 style="display:flex;align-items:center;margin:0 0 0.5rem 0;'
                f'font-size:1.5rem;font-weight:600;color:#0f2740;">'
            f'{icon("layers", size=20, margin_right=8)}Budget-Based Property Finder</h3>',
            unsafe_allow_html=True
        )
        st.caption("Tell us your budget range — we'll scan matching areas & property types using the champion model.")

        # clamp the floor at 0 so the slider never offers a negative budget
        min_price = max(0, int(np.floor(predictions["Predicted"].min() / 10000) * 10000))
        max_price = max(min_price + 10000, int(np.ceil(predictions["Predicted"].max() / 10000) * 10000))
        default_hi = min(max_price, min_price + (max_price - min_price) // 3)

        with st.container(border=True):
            budget_lo, budget_hi = st.slider(
                "Budget range (RM)",
                min_value=min_price, max_value=max_price,
                value=(min_price, default_hi), step=10000, format="RM %d",
            )
            scope_state = st.selectbox("Search in", ["All States"] + meta["states"])
            find_btn = st.button("Find Matching Options", use_container_width=True,
                                  icon=":material/search:")

        if find_btn:
            states_scope = meta["states"] if scope_state == "All States" else [scope_state]
            with st.spinner("Scanning the market for matches..."):
                matches = run_budget_search(models[BEST], meta, budget_lo, budget_hi, states_scope)

            if matches.empty:
                st.warning("No matches found within this budget range — try widening it.")
            else:
                top = matches.head(15).reset_index(drop=True)
                st.success(f"Found {len(matches):,} matching combinations between "
                           f"RM {budget_lo:,.0f} – RM {budget_hi:,.0f}. Showing closest {len(top)}.")

                display_df = top.copy()
                display_df["Predicted Price"] = display_df["PredictedPrice"].map(lambda v: f"RM {v:,.0f}")
                st.dataframe(
                    display_df[["State", "Area", "Type", "Tenure", "Predicted Price"]],
                    hide_index=True, use_container_width=True
                )

                st.markdown('<div class="chart-label">Closest matches within your budget range</div>', unsafe_allow_html=True)
                st.plotly_chart(build_budget_matches_chart(top, budget_lo, budget_hi),
                                 use_container_width=True, key="budget_matches", config=PLOTLY_CONFIG)

# =========================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================
with tab_performance:
    k1, k2, k3, k4 = st.columns(4)
    for col, label, key, fmt in [
        (k1, "Test RMSE", "Test RMSE", "RM {:,.0f}"),
        (k2, "Test MAE", "Test MAE", "RM {:,.0f}"),
        (k3, "Test R²", "Test R2", "{:.3f}"),
        (k4, "Adjusted R²", "Test Adjusted R2", "{:.3f}"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card"><div class="label">{label} ({BEST})</div>
                <div class="value">{fmt.format(comparison.loc[BEST, key])}</div></div>""", unsafe_allow_html=True)

    st.markdown("###  ")
    st.markdown('<div class="chart-label">Compare a metric across models — pick a metric, watch the bars animate</div>', unsafe_allow_html=True)
    metric_options = [c for c in comparison.columns if c.startswith("Test")]
    render_metric_compare(metric_options, meta["model_names"], BEST)

    st.divider()
    st.markdown('<div class="chart-label">Actual vs Predicted (small multiples — one panel per model)</div>', unsafe_allow_html=True)
    with st.container(border=True):
        filter_models = st.multiselect("Compare models", meta["model_names"], default=meta["model_names"],
                                        label_visibility="collapsed")
        if filter_models:
            st.plotly_chart(build_facet_scatter(filter_models), use_container_width=True, key="perf_facet_scatter", config=PLOTLY_CONFIG)
        else:
            st.info("Select at least one model to display.")

    st.divider()
    st.markdown('<div class="chart-label">All metrics at a glance — darker is better for each model</div>', unsafe_allow_html=True)
    st.plotly_chart(build_metrics_heatmap(), use_container_width=True, key="perf_metrics_heatmap", config=PLOTLY_CONFIG)

    st.divider()
    st.markdown('<div class="chart-label">Model stability — 5-fold CV RMSE (lower & tighter = more stable)</div>', unsafe_allow_html=True)
    st.plotly_chart(build_cv_stability_chart(), use_container_width=True, key="perf_cv_stability", config=PLOTLY_CONFIG)

    st.divider()
    st.markdown('<div class="chart-label">Prediction error spread (Predicted − Actual, RM)</div>', unsafe_allow_html=True)
    resid_df = predictions.copy()
    resid_df["Residual"] = resid_df["Predicted"] - resid_df["Actual"]
    resid_fig = go.Figure()
    for m in meta["model_names"]:
        sub = resid_df[resid_df["Model"] == m]
        resid_fig.add_trace(go.Violin(
            y=sub["Residual"], name=m, line_color=MODEL_COLORS[m],
            fillcolor=MODEL_COLORS[m], opacity=0.5, box_visible=True,
            meanline_visible=True, points=False,
        ))
    resid_fig.add_hline(y=0, line_dash="dash", line_color="#9aa1ab")
    resid_fig.update_layout(
        height=380, margin=dict(t=20, b=50, l=90, r=20), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Residual (RM)", tickformat=",.0f", gridcolor="#eef0f3", automargin=True),
        xaxis=dict(automargin=True),
        hoverlabel=HOVERLABEL, dragmode="zoom",
    )
    st.plotly_chart(resid_fig, use_container_width=True, key="perf_residual_violin", config=PLOTLY_CONFIG)

    st.divider()
    st.markdown('<div class="chart-label">Overall ranking (average rank across all metrics)</div>', unsafe_allow_html=True)
    rank_df = pd.DataFrame(index=comparison.index)
    for col in metric_options:
        lb = any(k in col for k in ["RMSE", "MAE", "RMSLE", "MAPE"])
        rank_df[col] = comparison[col].rank(ascending=lb)
    rank_df["Average Rank"] = rank_df.mean(axis=1)
    rank_df = rank_df.sort_values("Average Rank")

    rank_colors = {0: "#d4af37", 1: "#a8a8a8", 2: "#b08d57"}
    rows_html = ""
    for i, (model, row) in enumerate(rank_df.iterrows()):
        badge_color = rank_colors.get(i, "#c9d2dc")
        rows_html += f"""
        <div class="rank-row">
          <div class="rank-badge" style="background:{badge_color};">{i + 1}</div>
          <div class="rank-model"><span class="legend-dot" style="background:{MODEL_COLORS[model]}"></span>{model}</div>
          <div class="rank-score">{row['Average Rank']:.2f} avg rank</div>
        </div>"""
    st.markdown(f'<div class="rank-list">{rows_html}</div>', unsafe_allow_html=True)
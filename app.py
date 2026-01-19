import os
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
import numpy as np

st.markdown(
    """
<style>
    /* Force header text to be white */
    div[data-testid="stMarkdownContainer"] h1 {
        color: #ffffff !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


DEFAULT_CSV_PATH = os.path.join("data", "orders.csv")


@st.cache_data
def load_data(
    csv_path: str,
    database_url: str | None,
    table_name: str | None,
    sql_query: str | None,
) -> pd.DataFrame:
    if database_url:
        engine = create_engine(database_url)
        query = sql_query or f"SELECT * FROM {table_name or 'orders'}"
        df = pd.read_sql(query, engine)
    else:
        df = pd.read_csv(csv_path)

    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def quarter_sort_key(quarter_label: str) -> tuple[int, int]:
    parts = quarter_label.split()
    if len(parts) != 2:
        return (0, 0)
    q = int(parts[0].replace("Q", ""))
    year = int(parts[1])
    return (year, q)


def apply_refresh_variation(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if seed <= 0 or df.empty:
        return df.copy()
    rng = np.random.default_rng(seed)
    out = df.copy()
    revenue_noise = rng.normal(1.0, 0.03, len(out))
    orders_noise = rng.normal(1.0, 0.02, len(out))
    rating_noise = rng.normal(0.0, 0.05, len(out))
    out["revenue"] = (out["revenue"] * revenue_noise).round(2)
    out["orders"] = (out["orders"] * orders_noise).round().astype(int).clip(lower=0)
    out["rating"] = (out["rating"] + rating_noise).clip(1.0, 5.0).round(2)
    return out


def build_events_table(filtered_df: pd.DataFrame) -> pd.DataFrame:
    quarter_labels = sorted(filtered_df["quarter"].unique(), key=quarter_sort_key)
    if not quarter_labels:
        return pd.DataFrame(columns=["Account/Plan", "Event", "Q1", "Q2"])

    years = [int(label.split()[1]) for label in quarter_labels]
    latest_year = max(years)
    q1_label = f"Q1 {latest_year}"
    q2_label = f"Q2 {latest_year}"
    if q1_label not in quarter_labels or q2_label not in quarter_labels:
        q1_label = quarter_labels[0]
        q2_label = quarter_labels[1] if len(quarter_labels) > 1 else quarter_labels[0]

    q1_orders = int(
        filtered_df.loc[filtered_df["quarter"] == q1_label, "orders"].sum()
    )
    q2_orders = int(
        filtered_df.loc[filtered_df["quarter"] == q2_label, "orders"].sum()
    )

    plans = {"Basic": 0.6, "Business": 0.25, "Premium": 0.15}
    events = {"Button Clicked": 2.4, "Page Viewed": 8.1}
    rows = []
    for plan, weight in plans.items():
        plan_q1_total = 0
        plan_q2_total = 0
        for event, multiplier in events.items():
            q1_value = int(q1_orders * weight * multiplier)
            q2_value = int(q2_orders * weight * multiplier)
            rows.append(
                {
                    "Account/Plan": plan,
                    "Event": event,
                    "Q1": q1_value,
                    "Q2": q2_value,
                }
            )
            plan_q1_total += q1_value
            plan_q2_total += q2_value
        rows.append(
            {
                "Account/Plan": f"Totals for {plan}",
                "Event": "",
                "Q1": plan_q1_total,
                "Q2": plan_q2_total,
            }
        )

    grand_q1 = sum(row["Q1"] for row in rows if row["Event"])
    grand_q2 = sum(row["Q2"] for row in rows if row["Event"])
    rows.append(
        {
            "Account/Plan": "Grand totals",
            "Event": "",
            "Q1": grand_q1,
            "Q2": grand_q2,
        }
    )
    return pd.DataFrame(rows)


def enrich_data(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rng = np.random.default_rng(seed + 17)
    city_map = {
        "California": ["San Francisco", "Los Angeles", "San Diego"],
        "Texas": ["Austin", "Dallas", "Houston"],
        "New York": ["New York", "Buffalo", "Rochester"],
        "Florida": ["Miami", "Orlando", "Tampa"],
        "Washington": ["Seattle", "Spokane", "Tacoma"],
        "Illinois": ["Chicago", "Naperville", "Aurora"],
        "Georgia": ["Atlanta", "Savannah", "Augusta"],
        "Arizona": ["Phoenix", "Tucson", "Mesa"],
        "Pennsylvania": ["Philadelphia", "Pittsburgh", "Harrisburg"],
        "North Carolina": ["Charlotte", "Raleigh", "Durham"],
        "Colorado": ["Denver", "Boulder", "Colorado Springs"],
        "Ohio": ["Columbus", "Cleveland", "Cincinnati"],
        "New Jersey": ["Newark", "Jersey City", "Trenton"],
    }
    segment_map = {
        "Doohickey": "SMB",
        "Widget": "Enterprise",
        "Gadget": "Mid-Market",
        "Gizmo": "SMB",
    }
    enriched = df.copy()
    enriched["city"] = enriched["state_name"].map(
        lambda state: rng.choice(city_map.get(state, ["Metro"]))
    )
    enriched["segment"] = enriched["product_category"].map(segment_map).fillna("SMB")
    enriched["growth_pct"] = rng.normal(0.06, 0.08, len(enriched)).clip(-0.15, 0.25)
    enriched["cancel_rate"] = rng.normal(0.03, 0.015, len(enriched)).clip(0.0, 0.12)
    enriched["refund_rate"] = rng.normal(0.02, 0.01, len(enriched)).clip(0.0, 0.1)
    enriched["opportunity_score"] = (
        (enriched["revenue"] / enriched["revenue"].max()) * 55
        + (enriched["growth_pct"] * 100).clip(0, 25)
        - (enriched["refund_rate"] * 100)
        + rng.normal(0, 4, len(enriched))
    ).clip(0, 100)
    return enriched


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color="#1f2937"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        transition=dict(duration=500, easing="cubic-in-out"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_skeleton_layout() -> None:
    def skeleton_card() -> None:
        st.markdown('<div class="card skeleton"></div>', unsafe_allow_html=True)

    top_cols = st.columns(4, gap="large")
    for col in top_cols:
        with col:
            skeleton_card()

    mid_cols = st.columns(2, gap="large")
    for col in mid_cols:
        with col:
            skeleton_card()

    bottom_cols = st.columns(3, gap="large")
    for col in bottom_cols:
        with col:
            skeleton_card()


st.set_page_config(page_title="BizPulse Dashboard", layout="wide")

st.markdown(
    """
<style>
/* Main background */
.main {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.stApp {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

[data-testid="stSidebar"] { background-color: #ffffff; }

/* Header banner */
.dashboard-header {
  background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
  padding: 2rem 1.5rem;
  border-radius: 0 0 16px 16px;
  margin: 0 0 1.5rem 0;
  text-align: center;
  color: white;
}

.dashboard-header h1 {
  margin: 0;
  font-size: 2.5rem;
  font-weight: 700;
}

.dashboard-header p {
  margin: 0.5rem 0 0 0;
  font-size: 1.1rem;
  opacity: 0.9;
}

.main .block-container {
  max-width: 100%;
  padding: 0 1rem 2rem;
  animation: fadeIn 600ms ease;
}

.block-container { padding-top: 0; padding-bottom: 1.5rem; }

/* Card styling (apply only to columns) */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(10px);
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
}

/* Global text on gradient background */
.main, .main * {
  color: white !important;
}

/* Keep card content readable */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div,
div[data-testid="stVerticalBlock"] > div > div {
  background: rgba(255, 255, 255, 0.95);
  color: #1f2937 !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div *,
div[data-testid="stVerticalBlock"] > div > div * {
  color: #1f2937 !important;
}

/* Keep header text white even with card overrides */
.dashboard-header, .dashboard-header * {
  color: #ffffff !important;
}

div[data-testid="stHorizontalBlock"] { gap: 20px; }
div[data-testid="column"] { min-width: 260px; }

@media (max-width: 1100px) {
  div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  div[data-testid="column"] { flex: 1 1 calc(50% - 20px); }
}
@media (max-width: 700px) {
  div[data-testid="column"] { flex: 1 1 100%; }
}

/* Chart titles */
.chart-title {
  font-size: 18px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 1rem;
}

/* KPI */
.kpi-card {
  border-left: 6px solid #4A90E2;
  padding: 6px 0 6px 12px;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}
.kpi-card.green { border-left-color: #10B981; }
.kpi-card.orange { border-left-color: #F59E0B; }
.kpi-card.purple { border-left-color: #BD10E0; }
.kpi-value { font-size: 36px; font-weight: 700; color: #1f2937; margin: 6px 0; }
.kpi-label { font-size: 0.9rem; color: #6b7280; margin: 0; }
.kpi-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(16, 185, 129, 0.15);
  color: #10B981;
  font-size: 0.75rem;
  font-weight: 600;
}
.kpi-badge.negative { background: rgba(239, 68, 68, 0.15); color: #EF4444; }
.kpi-delta { font-size: 0.85rem; font-weight: 600; color: #10B981; }
.kpi-delta.negative { color: #EF4444; }

/* Skeleton */
.skeleton {
  position: relative;
  overflow: hidden;
  background: #f1f5f9;
  border-radius: 10px;
  height: 220px;
}
.skeleton::after {
  content: "";
  position: absolute;
  top: 0;
  left: -150px;
  height: 100%;
  width: 150px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
  animation: shimmer 1.2s infinite;
}
@keyframes shimmer {
  0% { transform: translateX(0); }
  100% { transform: translateX(500px); }
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.section-divider { border: none; border-top: 1px solid rgba(255,255,255,0.25); margin: 1rem 0; }
.section-header {
  color: white !important;
  font-size: 24px;
  font-weight: 600;
  margin: 2rem 0 1rem 0;
}

/* Force header text to stay white even with card overrides */
.dashboard-header,
.dashboard-header * {
  color: #ffffff !important;
}

/* Remove Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Smooth transitions */
button, input, select, textarea {
  transition: all 0.2s ease;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); padding: 2rem; border-radius: 0 0 16px 16px; margin-bottom: 2rem; text-align: center;">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; color: #ffffff !important;">BizPulse Analytics Dashboard</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; color: #ffffff !important;">Real-time Business Intelligence &amp; Performance Metrics</p>
</div>
""",
    unsafe_allow_html=True,
)

csv_path = os.getenv("CSV_PATH", DEFAULT_CSV_PATH)
database_url = os.getenv("DATABASE_URL")
table_name = os.getenv("DB_TABLE")
sql_query = os.getenv("SQL_QUERY")

try:
    df = load_data(csv_path, database_url, table_name, sql_query)
except Exception as exc:
    st.error(f"Failed to load data source: {exc}")
    st.stop()

st.sidebar.header("Filters")
quarter_options = ["All"] + sorted(df["quarter"].unique(), key=quarter_sort_key)
product_options = ["All"] + sorted(df["product_category"].unique())

selected_quarter = st.sidebar.selectbox("Quarter", quarter_options, index=0)
selected_product = st.sidebar.multiselect(
    "Product category",
    product_options[1:],
    default=product_options[1:],
)
refresh_clicked = st.sidebar.button("Refresh")

if "loading" not in st.session_state:
    st.session_state["loading"] = False

if refresh_clicked:
    st.session_state["refresh_seed"] = st.session_state.get("refresh_seed", 0) + 1
    st.session_state["loading"] = True
    st.rerun()

loading = st.session_state.get("loading", False)

refresh_seed = st.session_state.get("refresh_seed", 0)

base_df = apply_refresh_variation(df, refresh_seed)

if loading:
    render_skeleton_layout()
    time.sleep(0.4)
    st.session_state["loading"] = False
    st.rerun()

filtered = base_df.copy()
if selected_quarter != "All":
    filtered = filtered[filtered["quarter"] == selected_quarter]
if selected_product:
    filtered = filtered[filtered["product_category"].isin(selected_product)]

total_revenue = float(filtered["revenue"].sum())
total_orders = int(filtered["orders"].sum())
avg_rating = float(filtered["rating"].mean()) if not filtered.empty else 0.0

palette = {
    "blue": "#4A90E2",
    "green": "#10B981",
    "purple": "#BD10E0",
    "orange": "#F59E0B",
    "red": "#EF4444",
    "teal": "#14b8a6",
    "slate": "#64748b",
}
color_sequence = [palette["blue"], palette["green"], palette["purple"], palette["orange"]]

orders_by_cat = filtered.groupby("product_category")["orders"].sum().reset_index()
donut_fig = go.Figure(
    data=[
        go.Pie(
            labels=orders_by_cat["product_category"],
            values=orders_by_cat["orders"],
            hole=0.62,
            marker=dict(colors=color_sequence, line=dict(color="#ffffff", width=2)),
            textinfo="percent",
        )
    ]
)
donut_fig.add_annotation(
    text=f"{total_orders:,.0f}<br><span style='font-size:12px;color:#64748b;'>TOTAL</span>",
    x=0.5,
    y=0.5,
    showarrow=False,
    font=dict(size=18, color="#1f2937"),
)
donut_fig = style_figure(donut_fig)
donut_fig.update_layout(height=300)

revenue_per_quarter = filtered.groupby("quarter")["revenue"].sum().reset_index()
if not revenue_per_quarter.empty:
    revenue_per_quarter["sort_key"] = revenue_per_quarter["quarter"].apply(
        quarter_sort_key
    )
    revenue_per_quarter = revenue_per_quarter.sort_values("sort_key")
q_fig = px.bar(
    revenue_per_quarter,
    x="quarter",
    y="revenue",
    text_auto=".2s",
    color_discrete_sequence=[palette["blue"]],
)
q_fig.update_traces(textposition="outside")
q_fig = style_figure(q_fig)
q_fig.update_layout(height=300)

if not revenue_per_quarter.empty:
    latest_row = revenue_per_quarter.iloc[-1]
    latest_quarter = latest_row["quarter"]
    latest_revenue = float(latest_row["revenue"])
    if len(revenue_per_quarter) > 1:
        prev_revenue = float(revenue_per_quarter.iloc[-2]["revenue"])
        change_pct = (latest_revenue - prev_revenue) / prev_revenue if prev_revenue else 0
    else:
        change_pct = 0.0
else:
    latest_quarter = ""
    latest_revenue = 0.0
    change_pct = 0.0

goal_value = 250000
goal_fig = go.Figure(
    go.Indicator(
        mode="number+gauge",
        value=total_revenue,
        number={"prefix": "$", "valueformat": ",.0f"},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, goal_value]},
            "bar": {"color": palette["green"]},
            "steps": [
                {"range": [0, goal_value * 0.5], "color": "#e2f4e8"},
                {"range": [goal_value * 0.5, goal_value], "color": "#c7ecd6"},
            ],
        },
    )
)
goal_fig = style_figure(goal_fig)
goal_fig.update_layout(height=300)

state_rev = (
    filtered.groupby(["state_code", "state_name"])["revenue"]
    .sum()
    .reset_index()
)
map_fig = px.choropleth(
    state_rev,
    locations="state_code",
    locationmode="USA-states",
    color="revenue",
    scope="usa",
    hover_name="state_name",
    color_continuous_scale="Blues",
)
map_fig = style_figure(map_fig)
map_fig.update_layout(coloraxis_colorbar=dict(title="Revenue", thickness=12), height=300)

rating_fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=avg_rating,
        number={"valueformat": ".2f"},
        gauge={
            "axis": {"range": [1, 5]},
            "bar": {"color": palette["green"]},
            "steps": [
                {"range": [1, 2.5], "color": "#fde2e2"},
                {"range": [2.5, 3.5], "color": "#fef3c7"},
                {"range": [3.5, 5], "color": "#dcfce7"},
            ],
        },
    )
)
rating_fig = style_figure(rating_fig)
rating_fig.update_layout(height=300)

time_df = (
    filtered.groupby("quarter")
    .agg(revenue=("revenue", "sum"), orders=("orders", "sum"))
    .reset_index()
)
if not time_df.empty:
    time_df["sort_key"] = time_df["quarter"].apply(quarter_sort_key)
    time_df = time_df.sort_values("sort_key")
combo_fig = go.Figure()
combo_fig.add_trace(
    go.Bar(
        x=time_df["quarter"],
        y=time_df["orders"],
        name="Orders",
        marker_color=palette["purple"],
    )
)
combo_fig.add_trace(
    go.Scatter(
        x=time_df["quarter"],
        y=time_df["revenue"],
        name="Revenue",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color=palette["green"], width=3),
    )
)
combo_fig.update_layout(
    yaxis=dict(title="Orders"),
    yaxis2=dict(title="Revenue", overlaying="y", side="right"),
)
combo_fig = style_figure(combo_fig)
combo_fig.update_layout(height=400)

rev_by_cat = (
    filtered.groupby(["quarter", "product_category"])["revenue"]
    .sum()
    .reset_index()
)
if not rev_by_cat.empty:
    rev_by_cat["sort_key"] = rev_by_cat["quarter"].apply(quarter_sort_key)
    rev_by_cat = rev_by_cat.sort_values("sort_key")
rev_fig = px.bar(
    rev_by_cat,
    x="quarter",
    y="revenue",
    color="product_category",
    barmode="stack",
    text_auto=".2s",
    color_discrete_sequence=color_sequence,
)
rev_fig.update_traces(textposition="inside", insidetextanchor="middle")
rev_fig = style_figure(rev_fig)
rev_fig.update_layout(height=300)

orders_by_cat_q = (
    filtered.groupby(["quarter", "product_category"])["orders"]
    .sum()
    .reset_index()
)
if not orders_by_cat_q.empty:
    orders_by_cat_q["sort_key"] = orders_by_cat_q["quarter"].apply(quarter_sort_key)
    orders_by_cat_q = orders_by_cat_q.sort_values("sort_key")
orders_fig = px.area(
    orders_by_cat_q,
    x="quarter",
    y="orders",
    color="product_category",
    groupnorm=None,
    line_shape="spline",
    color_discrete_sequence=color_sequence,
)
orders_fig = style_figure(orders_fig)
orders_fig.update_layout(height=300)

events_df = build_events_table(filtered)
enriched = enrich_data(filtered, refresh_seed)

def checkbox_group(label: str, options: list[str], key_prefix: str) -> list[str]:
    selected = []
    with st.sidebar.expander(label, expanded=False):
        for option in options:
            if st.checkbox(option, value=True, key=f"{key_prefix}_{option}"):
                selected.append(option)
    return selected

st.sidebar.subheader("Opportunity filters")
product_choices = sorted(enriched["product_category"].unique())
city_choices = sorted(enriched["city"].unique())
segment_choices = sorted(enriched["segment"].unique())

selected_products = checkbox_group("Product", product_choices, "opp_prod") or product_choices
selected_cities = checkbox_group("City", city_choices, "opp_city") or city_choices
selected_segments = checkbox_group("Segment", segment_choices, "opp_seg") or segment_choices

opportunity_df = enriched[
    enriched["product_category"].isin(selected_products)
    & enriched["city"].isin(selected_cities)
    & enriched["segment"].isin(selected_segments)
]

treemap_df = (
    opportunity_df.groupby(["state_name", "city", "product_category"])["revenue"]
    .sum()
    .reset_index()
)
treemap_fig = px.treemap(
    treemap_df,
    path=["state_name", "city", "product_category"],
    values="revenue",
    color="revenue",
    color_continuous_scale="RdYlGn",
)
treemap_fig.update_traces(
    texttemplate="%{label}<br>$%{value:,.0f}",
    textfont=dict(color="#0f172a"),
)
treemap_fig = style_figure(treemap_fig)
treemap_fig.update_layout(height=400, coloraxis_showscale=False)

scatter_df = (
    opportunity_df.groupby(["city", "product_category", "segment"])
    .agg(revenue=("revenue", "sum"), growth_pct=("growth_pct", "mean"), orders=("orders", "sum"))
    .reset_index()
)
scatter_fig = px.scatter(
    scatter_df,
    x="growth_pct",
    y="revenue",
    size="orders",
    color="product_category",
    hover_name="city",
    color_discrete_sequence=color_sequence,
    size_max=40,
)
scatter_fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="#ffffff")))
scatter_fig.update_layout(
    xaxis_tickformat=".0%",
    xaxis_title="Growth %",
    yaxis_title="Revenue",
    height=400,
)
x_mid = scatter_df["growth_pct"].median() if not scatter_df.empty else 0
y_mid = scatter_df["revenue"].median() if not scatter_df.empty else 0
scatter_fig.add_shape(type="line", x0=x_mid, x1=x_mid, y0=0, y1=scatter_df["revenue"].max() if not scatter_df.empty else 1, line=dict(color="#cbd5f5", dash="dot"))
scatter_fig.add_shape(type="line", x0=scatter_df["growth_pct"].min() if not scatter_df.empty else -0.1, x1=scatter_df["growth_pct"].max() if not scatter_df.empty else 0.1, y0=y_mid, y1=y_mid, line=dict(color="#cbd5f5", dash="dot"))
scatter_fig.add_annotation(x=x_mid + 0.02, y=y_mid * 1.15, text="Invest", showarrow=False, font=dict(color="#0f172a"))
scatter_fig.add_annotation(x=x_mid - 0.08, y=y_mid * 1.15, text="Protect", showarrow=False, font=dict(color="#0f172a"))
scatter_fig.add_annotation(x=x_mid + 0.02, y=y_mid * 0.6, text="Monitor", showarrow=False, font=dict(color="#0f172a"))
scatter_fig.add_annotation(x=x_mid - 0.08, y=y_mid * 0.6, text="Fix Later", showarrow=False, font=dict(color="#0f172a"))
scatter_fig = style_figure(scatter_fig)

risk_drivers = [
    "Market share",
    "Product trends",
    "High refunds",
    "Pricing pressure",
    "Late renewals",
    "Low engagement",
]
risk_values = [52000, 47000, 39000, 34000, 28000, 22000]
mitigated_values = [22000, 18000, 12000, 10000, 9000, 7000]
risk_totals = [r + m for r, m in zip(risk_values, mitigated_values)]
risk_fig = go.Figure()
risk_fig.add_trace(
    go.Bar(
        y=risk_drivers,
        x=mitigated_values,
        name="Mitigated",
        orientation="h",
        marker_color=palette["green"],
    )
)
risk_fig.add_trace(
    go.Bar(
        y=risk_drivers,
        x=risk_values,
        name="At Risk",
        orientation="h",
        marker_color=palette["red"],
        text=[f"${total/1000:.1f}k" for total in risk_totals],
        textposition="outside",
    )
)
risk_fig.update_layout(barmode="stack", xaxis_title="Revenue at Risk", height=300)
risk_fig = style_figure(risk_fig)

ranking_fig = go.Figure(
    go.Bar(
        y=["Good", "Average", "Bad"],
        x=[78, 52, 31],
        orientation="h",
        marker_color=[palette["green"], palette["orange"], palette["red"]],
    )
)
ranking_fig.update_layout(height=300, xaxis_visible=False, yaxis_title="")
ranking_fig = style_figure(ranking_fig)

priority_base = (
    opportunity_df.groupby(["city", "product_category", "segment"])
    .agg(
        revenue=("revenue", "sum"),
        growth_pct=("growth_pct", "mean"),
        cancel_rate=("cancel_rate", "mean"),
        refund_rate=("refund_rate", "mean"),
        orders=("orders", "sum"),
        opportunity_score=("opportunity_score", "mean"),
    )
    .reset_index()
)
issues = ["High refunds", "Low growth", "Pricing pressure", "Churn risk"]
priority_rows = []
for idx, row in priority_base.iterrows():
    priority_rows.append(
        {
            "Opportunity ID": f"OPP-{1000 + idx}",
            "Merchant": f"{row['city']} {row['product_category']} Co",
            "Category": row["product_category"],
            "Segment": row["segment"],
            "Top Issue/Driver": issues[idx % len(issues)],
            "Next Touch In": f"{(idx % 5) + 1} days",
            "Revenue": row["revenue"],
            "Growth%": row["growth_pct"],
            "Cancel Rate": row["cancel_rate"],
            "Refund Rate": row["refund_rate"],
            "Opportunity Score": row["opportunity_score"],
        }
    )
priority_df = pd.DataFrame(priority_rows).sort_values(
    "Opportunity Score", ascending=False
)

st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

kpi_cols = st.columns(4, gap="large")
with kpi_cols[0]:
    st.markdown(
        f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">{format_currency(total_revenue)}</div>
            <span class="kpi-badge">YTD</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_cols[1]:
    st.markdown(
        f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{total_orders:,}</div>
            <span class="kpi-badge">Active</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_cols[2]:
    st.markdown(
        f"""
        <div class="kpi-card kpi-orange">
            <div class="kpi-label">Avg Rating</div>
            <div class="kpi-value">{avg_rating:.2f}</div>
            <span class="kpi-badge">Quality</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_cols[3]:
    badge_class = "kpi-badge" if change_pct >= 0 else "kpi-badge negative"
    badge_symbol = "▲" if change_pct >= 0 else "▼"
    st.markdown(
        f"""
        <div class="kpi-card kpi-purple">
            <div class="kpi-label">QoQ Change</div>
            <div class="kpi-value">{change_pct:+.1%}</div>
            <span class="{badge_class}">{badge_symbol} Trend</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

top_cols = st.columns(4, gap="large")
with top_cols[0]:
    st.markdown('<div class="chart-title">🧩 Total orders by product category</div>', unsafe_allow_html=True)
    st.plotly_chart(donut_fig, use_container_width=True)
with top_cols[1]:
    st.markdown('<div class="chart-title">💰 Revenue per quarter</div>', unsafe_allow_html=True)
    if latest_quarter:
        st.caption(latest_quarter)
    delta_class = "positive" if change_pct >= 0 else "negative"
    st.markdown(
        f"<div class='kpi-value'>{format_currency(latest_revenue)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='kpi-delta {delta_class}'>"
        f"{'▲' if change_pct >= 0 else '▼'} {change_pct:+.1%} vs previous quarter"
        "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(q_fig, use_container_width=True)
with top_cols[2]:
    st.markdown('<div class="chart-title">🎯 Revenue goal for this quarter</div>', unsafe_allow_html=True)
    st.caption(f"Goal {format_currency(goal_value)}")
    st.plotly_chart(goal_fig, use_container_width=True)
with top_cols[3]:
    st.markdown('<div class="chart-title">🗺️ Revenue by state</div>', unsafe_allow_html=True)
    st.plotly_chart(map_fig, use_container_width=True)

st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

mid_cols = st.columns(3, gap="large")
with mid_cols[0]:
    st.markdown('<div class="chart-title">🌡️ Opportunity Heatmap (City × Category)</div>', unsafe_allow_html=True)
    st.plotly_chart(treemap_fig, use_container_width=True)
with mid_cols[1]:
    st.markdown('<div class="chart-title">📈 Where is the Opportunity? Revenue vs Growth</div>', unsafe_allow_html=True)
    st.plotly_chart(scatter_fig, use_container_width=True)
with mid_cols[2]:
    st.markdown('<div class="chart-title">📊 Revenue and orders over time</div>', unsafe_allow_html=True)
    st.plotly_chart(combo_fig, use_container_width=True)

st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

lower_cols = st.columns(3, gap="large")
with lower_cols[0]:
    st.markdown('<div class="chart-title">⭐ Average product rating</div>', unsafe_allow_html=True)
    st.plotly_chart(rating_fig, use_container_width=True)
with lower_cols[1]:
    st.markdown('<div class="chart-title">🧱 Revenue by product category</div>', unsafe_allow_html=True)
    st.plotly_chart(rev_fig, use_container_width=True)
with lower_cols[2]:
    st.markdown('<div class="chart-title">🧮 Orders by product category</div>', unsafe_allow_html=True)
    st.plotly_chart(orders_fig, use_container_width=True)

st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

bottom_cols = st.columns(3, gap="large")
with bottom_cols[0]:
    st.markdown('<div class="chart-title">⚠️ Revenue at Risk by Driver</div>', unsafe_allow_html=True)
    st.plotly_chart(risk_fig, use_container_width=True)
with bottom_cols[1]:
    st.markdown('<div class="chart-title">🏅 Ranking</div>', unsafe_allow_html=True)
    st.plotly_chart(ranking_fig, use_container_width=True)
with bottom_cols[2]:
    st.markdown('<div class="chart-title">🧾 Events by quarter</div>', unsafe_allow_html=True)
    q1_max = events_df["Q1"].max() if not events_df.empty else 0
    q2_max = events_df["Q2"].max() if not events_df.empty else 0
    max_cell = max(q1_max, q2_max, 1)

    def color_cells(value: int) -> str:
            intensity = 0.15 + (value / max_cell) * 0.55 if max_cell else 0.2
            return f"background-color: rgba(16,185,129,{intensity:.2f});"

    def bold_totals(row: pd.Series) -> list[str]:
        label = str(row["Account/Plan"])
        if label.startswith("Totals") or label.startswith("Grand"):
            return ["font-weight: 700"] * len(row)
        return [""] * len(row)

    styled = (
        events_df.style.apply(bold_totals, axis=1)
        .applymap(color_cells, subset=["Q1", "Q2"])
        .format(precision=0)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=300)

st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
st.markdown('<p class="section-header">📋 Priority Merchant Action List</p>', unsafe_allow_html=True)

table_data = {
    "Opportunity ID": [f"OPP-{1001 + i}" for i in range(20)],
    "Merchant": [
        "Acme Retail",
        "Northwind Co",
        "Summit Stores",
        "Blue Harbor",
        "BrightMart",
        "Everline",
        "NovaTrade",
        "Urban Goods",
        "Greenfield",
        "Momentum",
        "Lighthouse",
        "Cascade",
        "Peak Supply",
        "Orbit Corp",
        "Atlas Market",
        "Zenith",
        "Vantage",
        "Vertex",
        "Pioneer",
        "Aurora",
    ],
    "Category": [
        "Doohickey",
        "Widget",
        "Gadget",
        "Gizmo",
        "Widget",
        "Gizmo",
        "Doohickey",
        "Gadget",
        "Widget",
        "Doohickey",
        "Gizmo",
        "Widget",
        "Gadget",
        "Doohickey",
        "Widget",
        "Gizmo",
        "Gadget",
        "Widget",
        "Doohickey",
        "Gizmo",
    ],
    "Segment": [
        "SMB",
        "Enterprise",
        "Mid-Market",
        "SMB",
        "Enterprise",
        "SMB",
        "Mid-Market",
        "Enterprise",
        "SMB",
        "Mid-Market",
        "Enterprise",
        "SMB",
        "Mid-Market",
        "Enterprise",
        "SMB",
        "Mid-Market",
        "Enterprise",
        "SMB",
        "Mid-Market",
        "Enterprise",
    ],
    "Top Issue Driver": [
        "High refunds",
        "Low growth",
        "Pricing pressure",
        "Churn risk",
        "High refunds",
        "Low growth",
        "Pricing pressure",
        "Churn risk",
        "High refunds",
        "Low growth",
        "Pricing pressure",
        "Churn risk",
        "High refunds",
        "Low growth",
        "Pricing pressure",
        "Churn risk",
        "High refunds",
        "Low growth",
        "Pricing pressure",
        "Churn risk",
    ],
    "Next Touch In": [
        "1 day",
        "2 days",
        "3 days",
        "4 days",
        "2 days",
        "5 days",
        "1 day",
        "3 days",
        "4 days",
        "2 days",
        "5 days",
        "1 day",
        "3 days",
        "2 days",
        "4 days",
        "5 days",
        "1 day",
        "3 days",
        "2 days",
        "4 days",
    ],
    "Revenue": np.random.default_rng(42).integers(18000, 95000, 20),
    "Avg Orders Growth": np.random.default_rng(7).normal(0.07, 0.05, 20).clip(-0.1, 0.2),
    "Avg Cancel Rate": np.random.default_rng(9).normal(0.03, 0.015, 20).clip(0.0, 0.12),
    "Avg Refund Rate": np.random.default_rng(11).normal(0.02, 0.01, 20).clip(0.0, 0.1),
    "Opportunity Score": np.random.default_rng(5).integers(55, 98, 20),
}

df_table = pd.DataFrame(table_data)
df_table = df_table.sort_values("Opportunity Score", ascending=False)

def table_color_good_bad(value: float, invert: bool = False) -> str:
    val = float(value)
    if invert:
        color = "#EF4444" if val > 0.05 else "#10B981"
    else:
        color = "#10B981" if val >= 0 else "#EF4444"
    return f"color: {color}; font-weight: 600;"

styled_table = (
    df_table.style
    .apply(
        lambda row: ["background-color: #f9fafb;" if row.name % 2 else "" for _ in row],
        axis=1,
    )
    .applymap(lambda v: table_color_good_bad(v), subset=["Avg Orders Growth"])
    .applymap(lambda v: table_color_good_bad(v, invert=True), subset=["Avg Cancel Rate", "Avg Refund Rate"])
    .applymap(lambda v: "color: #0f172a; font-weight: 600;", subset=["Opportunity Score"])
    .format(
        {
            "Revenue": "${:,.0f}",
            "Avg Orders Growth": "{:.1%}",
            "Avg Cancel Rate": "{:.1%}",
            "Avg Refund Rate": "{:.1%}",
            "Opportunity Score": "{:.0f}",
        }
    )
)
st.dataframe(styled_table, use_container_width=True, height=400, hide_index=True)


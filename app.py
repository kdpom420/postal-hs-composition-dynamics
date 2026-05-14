import json

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Country HS Composition Dynamics",
    layout="wide"
)

DATA_DIR = "data"


@st.cache_data
def load_data():
    ctry_country_year_hs = pd.read_csv(f"{DATA_DIR}/ctry_country_year_hs.csv")
    ctry_js_df = pd.read_csv(f"{DATA_DIR}/ctry_js_df.csv")
    ctry_country_summary = pd.read_csv(f"{DATA_DIR}/ctry_country_summary.csv")
    ctry_level_df = pd.read_csv(f"{DATA_DIR}/ctry_level_df.csv")

    with open(f"{DATA_DIR}/partnerAreas.json", "r", encoding="utf-8") as f:
        partner_json = json.load(f)

    if isinstance(partner_json, dict) and "results" in partner_json:
        partner_areas = pd.DataFrame(partner_json["results"])
    else:
        partner_areas = pd.DataFrame(partner_json)

    partner_areas = partner_areas.rename(columns={
        "PartnerCode": "partnerCode",
        "PartnerDesc": "country_name",
        "text": "country_text",
        "PartnerCodeIsoAlpha3": "iso3"
    })

    if "country_name" not in partner_areas.columns:
        partner_areas["country_name"] = partner_areas.get("country_text", "")

    if "iso3" not in partner_areas.columns:
        partner_areas["iso3"] = ""

    partner_areas = partner_areas[
        ["partnerCode", "country_name", "iso3"]
    ].drop_duplicates()

    return (
        ctry_country_year_hs,
        ctry_js_df,
        ctry_country_summary,
        ctry_level_df,
        partner_areas
    )


ctry_country_year_hs, ctry_js_df, ctry_country_summary, ctry_level_df, partner_areas = load_data()

# -------------------------
# Type cleanup
# -------------------------

ctry_country_year_hs["partnerCode"] = ctry_country_year_hs["partnerCode"].astype(int)
ctry_country_year_hs["year"] = ctry_country_year_hs["year"].astype(int)
ctry_country_year_hs["hs_code"] = ctry_country_year_hs["hs_code"].astype(str).str.zfill(4)

ctry_js_df["partnerCode"] = ctry_js_df["partnerCode"].astype(int)
ctry_js_df["prev_year"] = ctry_js_df["prev_year"].astype(int)
ctry_js_df["year"] = ctry_js_df["year"].astype(int)

ctry_country_summary["partnerCode"] = ctry_country_summary["partnerCode"].astype(int)
ctry_level_df["partnerCode"] = ctry_level_df["partnerCode"].astype(int)

partner_areas["partnerCode"] = partner_areas["partnerCode"].astype(int)

# -------------------------
# Sidebar
# -------------------------

st.sidebar.header("Controls")

country_df = (
    ctry_country_year_hs[["partnerCode"]]
    .drop_duplicates()
    .merge(partner_areas, on="partnerCode", how="left")
)

country_df["country_name"] = country_df["country_name"].fillna("Unknown")

country_df["country_label"] = (
    country_df["country_name"].astype(str)
    + " ("
    + country_df["partnerCode"].astype(str)
    + ")"
)

country_df = country_df.sort_values("country_label")

selected_country_label = st.sidebar.selectbox(
    "Country",
    country_df["country_label"]
)

selected_country = int(
    country_df.loc[
        country_df["country_label"] == selected_country_label,
        "partnerCode"
    ].iloc[0]
)

selected_country_name = country_df.loc[
    country_df["country_label"] == selected_country_label,
    "country_name"
].iloc[0]

country_years = sorted(
    ctry_country_year_hs.loc[
        ctry_country_year_hs["partnerCode"] == selected_country,
        "year"
    ].unique()
)

if len(country_years) < 2:
    st.warning("This country does not have enough years for year-pair comparison.")
    st.stop()

year_pair_labels = [
    f"{int(a)} → {int(b)}"
    for a, b in zip(country_years[:-1], country_years[1:])
]

selected_label = st.sidebar.selectbox(
    "Year Pair",
    year_pair_labels
)

prev_year, curr_year = [
    int(x.strip())
    for x in selected_label.split("→")
]

top_n = st.sidebar.slider(
    "Top N HS Changes",
    5,
    30,
    15
)

# -------------------------
# Header
# -------------------------

st.title(f"{selected_country_name} HS Composition Dynamics")

st.markdown(
    f"""
    **Selected country:** `{selected_country_name}`  
    **Year pair:** `{prev_year} → {curr_year}`
    """
)

st.caption(
    "Exploratory visualization of country-level HS share changes "
    "using Korea export data filtered by postal-friendly HS candidate groups."
)

# -------------------------
# Metrics
# -------------------------

metric_row = ctry_js_df[
    (ctry_js_df["partnerCode"] == selected_country)
    & (ctry_js_df["prev_year"] == prev_year)
    & (ctry_js_df["year"] == curr_year)
]

col1, col2, col3, col4 = st.columns(4)

if len(metric_row) > 0:
    m = metric_row.iloc[0]

    col1.metric("JS Similarity", round(m["ctry_hs_js_similarity"], 3))
    col2.metric("New HS", int(m["ctry_new_hs_count"]))
    col3.metric("Lost HS", int(m["ctry_lost_hs_count"]))
    col4.metric("Common HS", int(m["ctry_common_hs_count"]))
else:
    col1.metric("JS Similarity", "N/A")
    col2.metric("New HS", "N/A")
    col3.metric("Lost HS", "N/A")
    col4.metric("Common HS", "N/A")
    
if len(metric_row) > 0:
    if m["ctry_hs_js_similarity"] >= 0.75:
        insight_text = "This country shows relatively stable HS composition for the selected year pair."
    elif m["ctry_hs_js_similarity"] >= 0.45:
        insight_text = "This country shows moderate HS composition change for the selected year pair."
    else:
        insight_text = "This country shows strong HS composition transition for the selected year pair."

    st.info(insight_text)
    
# -------------------------
# Delta Data
# -------------------------

base_cols = ["hs_code", "ctry_hs_share", "hs_desc", "cluster_name"]

prev_df = (
    ctry_country_year_hs[
        (ctry_country_year_hs["partnerCode"] == selected_country)
        & (ctry_country_year_hs["year"] == prev_year)
    ][base_cols]
    .rename(columns={"ctry_hs_share": "share_prev"})
)

curr_df = (
    ctry_country_year_hs[
        (ctry_country_year_hs["partnerCode"] == selected_country)
        & (ctry_country_year_hs["year"] == curr_year)
    ][base_cols]
    .rename(columns={"ctry_hs_share": "share_curr"})
)

delta_df = prev_df.merge(
    curr_df,
    on="hs_code",
    how="outer",
    suffixes=("_prev", "_curr")
)

delta_df["share_prev"] = delta_df["share_prev"].fillna(0)
delta_df["share_curr"] = delta_df["share_curr"].fillna(0)

delta_df["hs_desc"] = delta_df["hs_desc_prev"].combine_first(
    delta_df["hs_desc_curr"]
).fillna("Unknown")

delta_df["cluster_name"] = delta_df["cluster_name_prev"].combine_first(
    delta_df["cluster_name_curr"]
).fillna("Unknown")

delta_df["delta"] = delta_df["share_curr"] - delta_df["share_prev"]
delta_df["abs_delta"] = delta_df["delta"].abs()

delta_df["label"] = (
    delta_df["hs_code"].astype(str)
    + " | "
    + delta_df["hs_desc"].astype(str).str.slice(0, 45)
)

plot_df = (
    delta_df
    .sort_values("abs_delta", ascending=False)
    .head(top_n)
    .sort_values("delta")
)

# -------------------------
# Main Plot
# -------------------------

st.subheader(f"HS Share Delta: {prev_year} → {curr_year}")

plot_df["change_type"] = np.where(
    plot_df["delta"] >= 0,
    "Increase",
    "Decrease"
)

fig = px.bar(
    plot_df,
    x="delta",
    y="label",
    orientation="h",
    color="change_type",
    color_discrete_map={
        "Increase": "#2563eb",
        "Decrease": "#ef4444"
    },
    hover_data={
        "hs_code": True,
        "hs_desc": True,
        "cluster_name": True,
        "share_prev": ":.4f",
        "share_curr": ":.4f",
        "delta": ":.4f",
        "label": False,
        "change_type": False
    },
    title=f"HS Share Delta: {prev_year} → {curr_year}"
)

fig.add_vline(x=0, line_width=1, line_color="black")

fig.update_layout(
    title_text=(
        f"HS Behavior-State Transition Sankey "
        f"- {selected_country_name} "
        f"({2019}-{2024})"
    ),
    font=dict(
        family="Arial Black",
        size=15,
        color="black"
    ),
    height=750,
    margin=dict(l=20, r=20, t=70, b=20),
    paper_bgcolor="white",
    plot_bgcolor="white"
)

fig.update_yaxes(autorange="reversed")

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "toImageButtonOptions": {
            "format": "png",
            "scale": 2
        }
    }
)


# -------------------------
# Tables
# -------------------------

left, right = st.columns(2)

with left:
    st.subheader("Top Increasing HS")
    st.dataframe(
        delta_df.sort_values("delta", ascending=False)
        .head(10)[
            ["hs_code", "hs_desc", "cluster_name", "share_prev", "share_curr", "delta"]
        ],
        use_container_width=True
    )

with right:
    st.subheader("Top Decreasing HS")
    st.dataframe(
        delta_df.sort_values("delta", ascending=True)
        .head(10)[
            ["hs_code", "hs_desc", "cluster_name", "share_prev", "share_curr", "delta"]
        ],
        use_container_width=True
    )

# -------------------------
# Country Summary
# -------------------------

st.subheader("Country Summary")

summary_row = ctry_country_summary[
    ctry_country_summary["partnerCode"] == selected_country
]

st.dataframe(summary_row, use_container_width=True)

# -------------------------
# Year-level Dynamics
# -------------------------

st.subheader("Year-level Dynamics")

level_rows = ctry_level_df[
    ctry_level_df["partnerCode"] == selected_country
]

show_cols = [
    "prev_year",
    "year",
    "ctry_drift",
    "ctry_hs_js_similarity",
    "ctry_new_hs_count",
    "ctry_lost_hs_count",
    "ctry_quadrant_type"
]

show_cols = [c for c in show_cols if c in level_rows.columns]

st.dataframe(
    level_rows[show_cols],
    use_container_width=True
)


# -------------------------
# HS Behavior-State Transition Sankey
# -------------------------

st.subheader("HS Behavior-State Transition Sankey")

st.caption(
    "HS items are clustered by relative share, growth, volatility, and growth persistence. "
    "The Sankey chart shows how each HS item moves across behavior states over time."
)

BEHAVIOR_CLUSTER_NAMES = {
    0: "Small Declining",
    1: "Growing Mid-Core",
    2: "Stable Large Core",
    3: "Volatile Decliner",
    4: "Mixed Stable Niche",
    5: "Surging Volatile",
    6: "Dominant Persistent Core",
}

behavior_start_year = int(min(country_years))
behavior_end_year = int(max(country_years))
behavior_n_clusters = 7

behavior_country = ctry_country_year_hs[
    (ctry_country_year_hs["partnerCode"] == selected_country)
    & (ctry_country_year_hs["year"] >= behavior_start_year)
    & (ctry_country_year_hs["year"] <= behavior_end_year)
].copy()

code_col = "cmdCode_clean" if "cmdCode_clean" in behavior_country.columns else "hs_code"

candidate_weight_cols = [
    "netWgt",
    "primaryValue",
    "fobvalue",
    "cifvalue",
    "trade_value",
    "value",
    "ctry_hs_share",
    "hs_share",
    "share",
    "wgt_share",
]

weight_col = next(
    (col for col in candidate_weight_cols if col in behavior_country.columns),
    None,
)

if weight_col is None:
    st.warning("No valid weight/share column found for behavior-state Sankey.")
elif behavior_country["year"].nunique() < 3:
    st.warning("At least three years are recommended for behavior-state transition analysis.")
else:
    hs_year = (
        behavior_country
        .groupby([code_col, "year"], as_index=False)
        .agg(weight=(weight_col, "sum"))
    )

    hs_year["year_total_weight"] = hs_year.groupby("year")["weight"].transform("sum")
    hs_year["share"] = np.where(
        hs_year["year_total_weight"] > 0,
        hs_year["weight"] / hs_year["year_total_weight"],
        0,
    )

    hs_year["log_weight"] = np.log1p(hs_year["weight"])
    hs_year["log_share"] = np.log1p(hs_year["share"] * 1_000_000)

    hs_year = hs_year.sort_values([code_col, "year"]).reset_index(drop=True)

    hs_year["log_growth"] = hs_year.groupby(code_col)["log_weight"].diff()

    hs_year["volatility_3y"] = (
        hs_year.groupby(code_col)["log_growth"]
        .rolling(3, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    hs_year["recent_growth_3y"] = (
        hs_year.groupby(code_col)["log_growth"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    hs_year["growth_up"] = (hs_year["log_growth"] > 0).astype(float)

    hs_year["growth_persistence_3y"] = (
        hs_year.groupby(code_col)["growth_up"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    feature_cols = [
        "log_share",
        "log_growth",
        "recent_growth_3y",
        "volatility_3y",
        "growth_persistence_3y",
    ]

    X = (
        hs_year[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    if len(X) < behavior_n_clusters:
        st.warning("Not enough HS-year observations to build seven behavior clusters.")
    else:
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(
            n_clusters=behavior_n_clusters,
            random_state=42,
            n_init=20,
        )

        hs_year["hs_behavior_cluster"] = kmeans.fit_predict(X_scaled)
        hs_year["behavior_state"] = hs_year["hs_behavior_cluster"].map(BEHAVIOR_CLUSTER_NAMES)

        hs_year["node"] = (
            hs_year["year"].astype(str)
            + "_"
            + hs_year["behavior_state"]
        )

        # Exclude first year from transitions because growth/volatility features start from lag values.
        sankey_hs_year = hs_year[hs_year["year"] > behavior_start_year].copy()
        sankey_years = list(range(behavior_start_year + 1, behavior_end_year + 1))

        flows = []

        for y1, y2 in zip(sankey_years[:-1], sankey_years[1:]):
            left = (
                sankey_hs_year[sankey_hs_year["year"] == y1]
                [[code_col, "node"]]
                .rename(columns={"node": "source"})
            )

            right = (
                sankey_hs_year[sankey_hs_year["year"] == y2]
                [[code_col, "node"]]
                .rename(columns={"node": "target"})
            )

            pair = left.merge(right, on=code_col, how="inner")

            if len(pair) > 0:
                flow = (
                    pair.groupby(["source", "target"], as_index=False)
                    .size()
                    .rename(columns={"size": "value"})
                )
                flows.append(flow)

        if len(flows) == 0:
            st.warning("Not enough overlapping HS codes to build behavior-state transitions.")
        else:
            flow_df = pd.concat(flows, ignore_index=True)

            labels = pd.Index(
                pd.concat([flow_df["source"], flow_df["target"]]).unique()
            )
            label_to_id = {label: i for i, label in enumerate(labels)}

            sankey_fig = go.Figure(data=[go.Sankey(
                node=dict(
                    label=labels.tolist(),
                    pad=18,
                    thickness=22,
                ),
                link=dict(
                    source=flow_df["source"].map(label_to_id),
                    target=flow_df["target"].map(label_to_id),
                    value=flow_df["value"],
                ),
            )])

            sankey_fig.update_layout(
                title_text=(
                    f"HS Behavior-State Transition Sankey - {selected_country_name} "
                    f"({behavior_start_year + 1}-{behavior_end_year})"
                ),
                font_size=10,
                height=760,
                margin=dict(l=20, r=20, t=60, b=20),
            )

            st.plotly_chart(sankey_fig, use_container_width=True)

            cluster_profile = (
                hs_year
                .groupby(["hs_behavior_cluster", "behavior_state"])[feature_cols]
                .mean()
                .round(3)
                .reset_index()
            )

            with st.expander("Behavior Cluster Profile"):
                st.dataframe(cluster_profile, use_container_width=True)

            with st.expander("Behavior Transition Table"):
                st.dataframe(flow_df, use_container_width=True)

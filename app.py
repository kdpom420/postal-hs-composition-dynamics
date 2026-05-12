import json

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

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
    template="plotly_white",
    height=650,
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis_title="Share Change",
    yaxis_title="HS Code",
    legend_title_text="Change",
)

fig.update_yaxes(autorange="reversed")

st.plotly_chart(fig, use_container_width=True)

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

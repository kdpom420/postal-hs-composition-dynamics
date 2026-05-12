import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    partner_areas = pd.read_json(f"{DATA_DIR}/partnerAreas.json")

    partner_areas = partner_areas.rename(columns={        
        "PartnerCode": "partnerCode",
        "PartnerDesc": "country_name",
        "PartnerCodeIsoAlpha3": "iso3"
    })
    
    partner_areas = partner_areas[
        ["partnerCode", "country_name", "iso3"]
    ].drop_duplicates()

    return ctry_country_year_hs, ctry_js_df, ctry_country_summary, ctry_level_df, partner_areas   


ctry_country_year_hs, ctry_js_df, ctry_country_summary, ctry_level_df, partner_areas = load_data()

st.title(f"{selected_country_name} HS Composition Dynamics")

st.caption(
    "Exploratory visualization of country-level HS share changes "
    "using Korea export data filtered by postal-friendly HS candidate groups."
)

# -------------------------
# Sidebar
# -------------------------

st.sidebar.header("Controls")

country_df = (
    ctry_country_year_hs[["partnerCode"]]
    .drop_duplicates()
    .merge(partner_areas, on="partnerCode", how="left")
)

country_df["country_label"] = (
    country_df["country_name"].fillna("Unknown")
    + " ("
    + country_df["partnerCode"].astype(str)
    + ")"
)

country_df = country_df.sort_values("country_label")

selected_country_label = st.sidebar.selectbox(
    "Country",
    country_df["country_label"]
)

selected_country = country_df.loc[
    country_df["country_label"] == selected_country_label,
    "partnerCode"
].iloc[0]

selected_country_name = country_df.loc[
    country_df["country_label"] == selected_country_label,
    "country_name"
].iloc[0]
# -------------------------
# Year Pair
# -------------------------
top_n = st.sidebar.slider(
    "Top N HS Changes",
    5,
    30,
    15
)
year_pair_labels = [
    f"{int(a)} → {int(b)}"
    for a, b in zip(country_years[:-1], country_years[1:])
]

selected_label = st.sidebar.selectbox(
    "Year Pair",
    year_pair_labels
)

# label 다시 분해
prev_year, curr_year = [
    int(x.strip())
    for x in selected_label.split("→")
]

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

# -------------------------
# Delta Data
# -------------------------

prev_df = (
    ctry_country_year_hs[
        (ctry_country_year_hs["partnerCode"] == selected_country)
        & (ctry_country_year_hs["year"] == prev_year)
    ][["hs_code", "ctry_hs_share", "hs_desc", "cluster_name"]]
    .rename(columns={"ctry_hs_share": "share_prev"})
)

curr_df = (
    ctry_country_year_hs[
        (ctry_country_year_hs["partnerCode"] == selected_country)
        & (ctry_country_year_hs["year"] == curr_year)
    ][["hs_code", "ctry_hs_share", "hs_desc", "cluster_name"]]
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

fig, ax = plt.subplots(figsize=(12, 8))

colors = ["red" if x < 0 else "blue" for x in plot_df["delta"]]

ax.barh(
    plot_df["label"],
    plot_df["delta"],
    color=colors
)

ax.axvline(0, color="black")
ax.set_xlabel("Share Change")
ax.set_ylabel("HS Code")
ax.grid(alpha=0.3)

st.pyplot(fig)

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

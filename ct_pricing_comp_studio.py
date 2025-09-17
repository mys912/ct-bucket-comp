# ct_pricing_comp_studio.py
import re
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Charging & Compensation Studio", layout="wide")

# =========================
# Defaults & helpers
# =========================
SERVICE_PRICES = {
    "Ct Consult - 1 Study": 180.24,
    "Ct Consult - 2 Study": 265.74,
    "Ct Consult - 3 Study": 348.93,
    "Ct Consult - 4 Study": 418.25,
    "Ct Consult - Whole Body": 523.20,
}
SERVICE_POINTS_BASELINE = {
    "Ct Consult - 1 Study": 3.0,
    "Ct Consult - 2 Study": 4.5,
    "Ct Consult - 3 Study": 6.0,
    "Ct Consult - 4 Study": 7.65,
    "Ct Consult - Whole Body": 10.0,
}
USD_PER_POINT_DEFAULT = 25.0
WORD_NUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}

def normalize_ct(service: Optional[str]) -> Optional[str]:
    """Normalize CT service labels into canonical classes."""
    if service is None or (isinstance(service, float) and pd.isna(service)):
        return None
    s = str(service).lower().strip()
    s = re.sub(r"\s*,\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for w, d in WORD_NUM.items():
        s = re.sub(rf"\b{w}\b", d, s)
    if "whole body" in s:
        return "Ct Consult - Whole Body"
    m = re.search(r"([1-4])\s*study", s)
    if m:
        return f"Ct Consult - {m.group(1)} Study"
    m2 = re.search(r"ct.*consult.*([1-4])", s)
    if m2:
        return f"Ct Consult - {m2.group(1)} Study"
    return None

@st.cache_data
def read_excel_one(path_or_buf: Union[str, Path, "BytesIO"], sheet_name: Union[str, int]):
    return pd.read_excel(path_or_buf, sheet_name=sheet_name)

@st.cache_data
def read_csv_any(path_or_buf: Union[str, Path, "BytesIO"]):
    return pd.read_csv(path_or_buf)

def load_any_with_sheet_picker(uploaded_file, typed_sheet: str):
    """Return (DataFrame, chosen_sheet). Supports CSV and Excel (with picker)."""
    name = getattr(uploaded_file, "name", "").lower()
    is_csv = name.endswith(".csv")
    is_excel = name.endswith(".xls") or name.endswith(".xlsx")

    if is_csv:
        df = read_csv_any(uploaded_file)
        return df, None

    if is_excel:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        st.sidebar.subheader("Excel sheet")
        if typed_sheet.strip() and typed_sheet.strip() in sheet_names:
            chosen = typed_sheet.strip()
        else:
            chosen = st.sidebar.selectbox("Pick sheet", options=sheet_names, index=0)
        df = read_excel_one(uploaded_file, sheet_name=chosen)
        return df, chosen

    # Fallback: try first sheet
    df = read_excel_one(uploaded_file, sheet_name=0)
    return df, 0

def parse_thresholds(th_str: str) -> List[float]:
    out = []
    for part in th_str.split(","):
        p = part.strip()
        if p:
            try:
                out.append(float(p))
            except Exception:
                pass
    out = sorted(set(out))
    return out

def build_labels_and_edges(thresholds: List[float]) -> tuple[list[str], list[str], list[float]]:
    """Return (pretty_labels, short_labels, bin_edges) for N buckets where thresholds are N-1 values."""
    edges = [-np.inf] + thresholds + [np.inf]
    pretty = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if np.isneginf(lo) and np.isposinf(hi):
            pretty.append("All")
        elif np.isneginf(lo):
            pretty.append(f"≤ {int(hi)}")
        elif np.isposinf(hi):
            pretty.append(f"> {int(lo)}")
        else:
            pretty.append(f"{int(lo) + 1}–{int(hi)}")
    short = [f"B{i+1}" for i in range(len(pretty))]
    return pretty, short, edges

def auto_quantile_thresholds(x: np.ndarray, n_buckets: int) -> List[float]:
    x = x[~np.isnan(x)]
    qs = np.linspace(0, 1, n_buckets + 1)
    qtiles = np.quantile(x, qs)
    return [float(v) for v in qtiles[1:-1]]  # interior points

# =========================
# Sidebar controls
# =========================
st.sidebar.title("Setup")

uploaded = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"])
typed_sheet = st.sidebar.text_input("Excel sheet name (optional)", "")

st.sidebar.subheader("Columns (rename if needed)")
col_case = st.sidebar.text_input("Casenumber", "Casenumber")
col_imgs = st.sidebar.text_input("TotalImages", "TotalImages")
col_spec = st.sidebar.text_input("Specialist", "Specialist")
col_date = st.sidebar.text_input("dateRequested", "dateRequested")
col_chgd = st.sidebar.text_input("ChargedService", "ChargedService")

st.sidebar.subheader("Optional columns")
col_company = st.sidebar.text_input("Company (optional)", "Company")

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
ct_only = st.sidebar.checkbox("CT-only (filter non-CT rows)", value=True)
use_date_filter = st.sidebar.checkbox("Filter by date range", value=False)
if use_date_filter:
    date_min = st.sidebar.date_input("Start date")
    date_max = st.sidebar.date_input("End date")
else:
    date_min = None
    date_max = None

st.sidebar.markdown("---")
st.sidebar.subheader("Buckets")
n_buckets = st.sidebar.slider("Number of buckets", min_value=2, max_value=6, value=3)
th_input = st.sidebar.text_input(
    f"Thresholds (comma-separated, {n_buckets-1} values). Leave blank to auto-calc.",
    value="1200, 3000" if n_buckets == 3 else ""
)

st.sidebar.markdown("---")
st.sidebar.subheader("Specialist compensation mode (Bucket model)")
comp_mode = st.sidebar.radio(
    "How should specialist comp be calculated for the Bucket model?",
    ["By bucket (points per bucket)", "By study type (service points)"],
    index=0
)

st.sidebar.subheader("Service Points (used when 'By study type' is selected)")
sp_ct1 = st.sidebar.number_input("CT-1 points", value=3.0, step=0.25)
sp_ct2 = st.sidebar.number_input("CT-2 points", value=4.5, step=0.25)
sp_ct3 = st.sidebar.number_input("CT-3 points", value=6.0, step=0.25)
sp_ct4 = st.sidebar.number_input("CT-4 points", value=7.65, step=0.25)
sp_wb  = st.sidebar.number_input("Whole Body points", value=10.0, step=0.25)
CUSTOM_SERVICE_POINTS = {
    "Ct Consult - 1 Study": float(sp_ct1),
    "Ct Consult - 2 Study": float(sp_ct2),
    "Ct Consult - 3 Study": float(sp_ct3),
    "Ct Consult - 4 Study": float(sp_ct4),
    "Ct Consult - Whole Body": float(sp_wb),
}
if comp_mode == "By study type (service points)":
    st.sidebar.caption("Note: Bucket point inputs are ignored; comp uses these service points.")
else:
    st.sidebar.caption("Currently paying specialists by bucket; service point edits are ignored.")

st.sidebar.subheader("$ per point")
usd_per_pt = st.sidebar.number_input("$ per point", value=USD_PER_POINT_DEFAULT, step=1.0)

show_specialist_comp = st.sidebar.checkbox("Show per-specialist comp delta", value=True)

# =========================
# Load data
# =========================
st.title("Charging & Compensation Studio (Buckets vs Site-Based)")

if uploaded is None:
    st.info("Upload your file to begin (columns: Casenumber, TotalImages, Specialist, dateRequested, ChargedService).")
    st.stop()

df, chosen_sheet = load_any_with_sheet_picker(uploaded, typed_sheet)
if isinstance(df, dict):
    if typed_sheet and typed_sheet in df:
        df = df[typed_sheet]
        chosen_sheet = typed_sheet
    else:
        first_key = next(iter(df.keys()))
        df = df[first_key]
        chosen_sheet = first_key

df.columns = df.columns.str.strip()

required = [col_case, col_imgs, col_spec, col_date, col_chgd]
if col_company in df.columns:
    required.append(col_company)

missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Keep & rename (include Company if present)
base_cols = [col_case, col_imgs, col_spec, col_date, col_chgd]
if col_company in df.columns:
    base_cols.append(col_company)
df = df[base_cols].copy()

rename_map = {
    col_case: "Casenumber",
    col_imgs: "TotalImages",
    col_spec: "Specialist",
    col_date: "dateRequested",
    col_chgd: "ChargedService",
}
if col_company in df.columns:
    rename_map[col_company] = "Company"
df.rename(columns=rename_map, inplace=True)

# Parse types & clean
df["dateRequested"] = pd.to_datetime(df["dateRequested"], errors="coerce")
df["TotalImages"] = pd.to_numeric(df["TotalImages"], errors="coerce")
df["ChargedService_clean"] = df["ChargedService"].apply(normalize_ct)

# Date filter
if use_date_filter:
    if pd.notna(pd.to_datetime(date_min, errors="coerce")):
        df = df[df["dateRequested"] >= pd.to_datetime(date_min)]
    if pd.notna(pd.to_datetime(date_max, errors="coerce")):
        df = df[df["dateRequested"] <= pd.to_datetime(date_max)]

# CT-only & drop critical missing
if ct_only:
    df = df[df["ChargedService_clean"].notna()].copy()
df = df.dropna(subset=["TotalImages", "dateRequested", "ChargedService_clean"]).copy()

n_rows = len(df)
st.caption(
    f"Rows included after cleaning/filters: **{n_rows:,}**"
    + (f" | Sheet: **{chosen_sheet}**" if chosen_sheet is not None else "")
)

if n_rows == 0:
    st.warning("No rows after filters.")
    st.stop()

# =========================
# Bucketing (dynamic)
# =========================
user_thresholds = parse_thresholds(th_input)
if len(user_thresholds) != (n_buckets - 1):
    auto_thr = auto_quantile_thresholds(df["TotalImages"].to_numpy(), n_buckets)
    st.info(f"Auto thresholds (quantiles) applied: {', '.join(str(int(t)) for t in auto_thr)}")
    thresholds = auto_thr
else:
    thresholds = user_thresholds

pretty_labels, short_labels, bin_edges = build_labels_and_edges(thresholds)

df["BucketLabel"] = pd.cut(df["TotalImages"], bins=bin_edges, labels=pretty_labels, include_lowest=True)
df["Bucket"] = pd.cut(df["TotalImages"], bins=bin_edges, labels=short_labels, include_lowest=True)

# =========================
# Bucket Prices & Points (for bucket-comp mode)
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Bucket Prices ($) & Points (used when 'By bucket' comp mode selected)")

bucket_price_map = {}
bucket_points_map = {}

# Seed ascending defaults
base_price, price_step = 200.0, 150.0 / max(1, n_buckets - 1)
base_pts, pts_step = 3.5, 5.5 / max(1, n_buckets - 1)

for i, lbl in enumerate(short_labels, start=1):
    default_p = float(base_price + (i - 1) * price_step)
    default_pts = float(base_pts + (i - 1) * pts_step)
    bucket_price_map[lbl] = st.sidebar.number_input(f"Price: {lbl}", value=default_p, step=5.0, key=f"price_{i}")
    bucket_points_map[lbl] = st.sidebar.number_input(f"Points: {lbl}", value=default_pts, step=0.25, key=f"pts_{i}")

# =========================
# Baseline (site-based) vs Bucket model
# =========================
# Baseline (service-based) - revenue & comp (fixed service points)
df["BaselineRevenue"] = df["ChargedService_clean"].map(SERVICE_PRICES)
df["BaselinePoints"] = df["ChargedService_clean"].map(SERVICE_POINTS_BASELINE)
df["BaselineComp"] = df["BaselinePoints"] * usd_per_pt
df["BaselineMargin"] = df["BaselineRevenue"] - df["BaselineComp"]

# Bucket model: client billing by bucket (always)
df["BucketRevenue"] = df["Bucket"].map(bucket_price_map).astype(float)

# Bucket model: specialist comp based on mode
if comp_mode == "By bucket (points per bucket)":
    df["BucketPoints"] = df["Bucket"].map(bucket_points_map).astype(float)
else:  # By study type (service points) -- editable via CUSTOM_SERVICE_POINTS
    df["BucketPoints"] = df["ChargedService_clean"].map(CUSTOM_SERVICE_POINTS).astype(float)

df["BucketComp"] = df["BucketPoints"] * usd_per_pt
df["BucketMargin"] = df["BucketRevenue"] - df["BucketComp"]

# Deltas vs baseline
df["RevenueDelta"] = df["BucketRevenue"] - df["BaselineRevenue"]
df["CompDelta"]    = df["BucketComp"]    - df["BaselineComp"]
df["MarginDelta"]  = df["BucketMargin"]  - df["BaselineMargin"]

# =========================
# KPIs
# =========================
k_rev_base = float(df["BaselineRevenue"].sum())
k_rev_bucket = float(df["BucketRevenue"].sum())
k_rev_delta = k_rev_bucket - k_rev_base

k_comp_base = float(df["BaselineComp"].sum())
k_comp_bucket = float(df["BucketComp"].sum())
k_comp_delta = k_comp_bucket - k_comp_base

k_margin_base = float(df["BaselineMargin"].sum())
k_margin_bucket = float(df["BucketMargin"].sum())
k_margin_delta = k_margin_bucket - k_margin_base

st.caption(f"Compensation mode for Bucket model: **{comp_mode}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue (Baseline)", f"${k_rev_base:,.0f}")
c2.metric("Total Revenue (Bucket)", f"${k_rev_bucket:,.0f}", delta=f"${k_rev_delta:,.0f}")
c3.metric("Total Comp (Baseline)", f"${k_comp_base:,.0f}")
c4.metric("Total Comp (Bucket)", f"${k_comp_bucket:,.0f}", delta=f"${k_comp_delta:,.0f}")

c5, c6 = st.columns(2)
c5.metric("Total Margin (Baseline)", f"${k_margin_base:,.0f}")
c6.metric("Total Margin (Bucket)", f"${k_margin_bucket:,.0f}", delta=f"${k_margin_delta:,.0f}")

st.markdown("---")

# =========================
# Charts
# =========================
# Composition by bucket (100% stacked)
ctab = pd.crosstab(df["BucketLabel"], df["ChargedService_clean"])
ctab_pct = ctab.div(ctab.sum(axis=1).replace(0, np.nan), axis=0) * 100
fig_comp = px.bar(
    ctab_pct.reset_index().melt(id_vars="BucketLabel", var_name="ChargedService", value_name="Percent"),
    x="BucketLabel", y="Percent", color="ChargedService",
    title="Composition by Charged Service within Buckets (100%)"
)
fig_comp.update_layout(barmode="stack", xaxis_title="Bucket", yaxis_title="Percent")
st.plotly_chart(fig_comp, use_container_width=True)

# Revenue / Comp / Margin grouped bars
sum_df = pd.DataFrame({
    "Model": ["Baseline", "Bucket"],
    "Revenue": [k_rev_base, k_rev_bucket],
    "Comp": [k_comp_base, k_comp_bucket],
    "Margin": [k_margin_base, k_margin_bucket],
})
tab1, tab2 = st.tabs(["Revenue vs Comp vs Margin", "Per-Bucket Summary"])

with tab1:
    fig_rc = go.Figure()
    fig_rc.add_bar(name="Revenue", x=sum_df["Model"], y=sum_df["Revenue"])
    fig_rc.add_bar(name="Comp", x=sum_df["Model"], y=sum_df["Comp"])
    fig_rc.add_bar(name="Margin", x=sum_df["Model"], y=sum_df["Margin"])
    fig_rc.update_layout(barmode="group", title="Model Comparison")
    st.plotly_chart(fig_rc, use_container_width=True)

with tab2:
    by_bucket = df.groupby("Bucket")[["BaselineRevenue","BucketRevenue","BaselineComp","BucketComp","BaselineMargin","BucketMargin"]].sum()
    by_bucket["RevenueDelta"] = by_bucket["BucketRevenue"] - by_bucket["BaselineRevenue"]
    by_bucket["CompDelta"] = by_bucket["BucketComp"] - by_bucket["BaselineComp"]
    by_bucket["MarginDelta"] = by_bucket["BucketMargin"] - by_bucket["BaselineMargin"]
    st.dataframe(by_bucket)

st.markdown("---")

# =========================
# Specialist comp deltas
# =========================
if show_specialist_comp:
    spec = (
        df.groupby("Specialist")[["BaselineComp","BucketComp"]].sum()
          .assign(CompDelta=lambda x: x["BucketComp"] - x["BaselineComp"])
          .sort_values("CompDelta", ascending=False)
    )
    st.subheader("Specialist Compensation — Δ ($) by Specialist")
    fig_spec = px.bar(
        spec.reset_index(),
        x="Specialist", y="CompDelta",
        title="Compensation Δ (Bucket – Baseline) per Specialist"
    )
    st.plotly_chart(fig_spec, use_container_width=True)
    st.dataframe(spec)

# =========================
# Client cost deltas (by Company)
# =========================
if "Company" in df.columns:
    st.markdown("---")
    st.subheader("Clinic Cost — Δ ($) by Company (Bucket billing – Baseline billing)")

    client = (
        df.groupby("Company")[["BaselineRevenue", "BucketRevenue"]].sum()
          .assign(CostDelta=lambda x: x["BucketRevenue"] - x["BaselineRevenue"])
          .sort_values("CostDelta", ascending=False)
    )

    top_n = st.slider(
        "Show top N companies by absolute change",
        min_value=5, max_value=50,
        value=min(20, len(client)), step=1
    )

    client_sorted_abs = client.reindex(client["CostDelta"].abs().sort_values(ascending=False).index)
    client_top = client_sorted_abs.head(top_n)

    fig_client = px.bar(
        client_top.reset_index(),
        x="Company",
        y="CostDelta",
        title="Clinic Cost Δ (Bucket – Baseline) per Company",
        labels={"CostDelta": "Δ Cost ($)", "Company": "Company"}
    )
    st.plotly_chart(fig_client, use_container_width=True)

    st.dataframe(client)
else:
    st.info("No 'Company' column detected. Add it in the sidebar mapping if available.")

# =========================
# Export
# =========================
st.markdown("---")
st.subheader("Export")

ledger_cols = [
    "Casenumber","Specialist","dateRequested","ChargedService","ChargedService_clean",
    "TotalImages","Bucket","BucketLabel",
    "BaselineRevenue","BucketRevenue","RevenueDelta",
    "BaselinePoints","BucketPoints","BaselineComp","BucketComp","CompDelta",
    "BaselineMargin","BucketMargin","MarginDelta"
]
if "Company" in df.columns:
    ledger_cols.insert(1, "Company")

ledger = df[ledger_cols].copy()

st.download_button(
    "Download per-study ledger (CSV)",
    data=ledger.to_csv(index=False).encode("utf-8"),
    file_name="charging_comp_ledger.csv",
    mime="text/csv"
)
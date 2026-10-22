"""Data cleaning and feature engineering for the Sri Lanka paddy yield forecasting project.

The prediction is made at SOWING time, so only information available before harvest is used:
the district, season, year, how much land was sown (and under which irrigation type), and the
district's yield history from previous seasons. Anything measured at harvest (harvested area,
per-source yields, production) is excluded because it would leak the answer.
"""
import numpy as np
import pandas as pd

# Irrigation/administrative aggregates that overlap with real districts (would double count)
AGGREGATE_REGIONS = ["Udawalawe", "Mahaweli'h'", "Highland", "North & East"]

NUM_COLS = [
    "Major Sown (Ha)", "Minor Sown (Ha)", "Rainfed Sown (Ha)", "Total Extent (Ha)",
    "Major Harvest (Ha)", "Minor Harvest (Ha)", "Rainfed Harvest (Ha)", "Net Harvest (Ha)",
    "Major Avg Yield (MT/Ha)", "Minor Avg Yield (MT/Ha)", "Rainfed Avg Yield (MT/Ha)",
    "Average Yield (MT/Ha)", "Production (MT)",
]

CAT_FEATURES = ["District", "Season"]
NUM_FEATURES = [
    "Year",
    "log_total_sown",          # log of total sown extent (ha)
    "share_major",             # share of sown area under major irrigation schemes
    "share_minor",             # share under minor irrigation (tanks/anicuts)
    "share_rainfed",           # share that is rain-fed
    "sown_change_pct",         # % change in sown extent vs same season last year
    "yield_prev_season",       # yield of the immediately previous season (kg/ha)
    "yield_same_season_ly",    # yield of the same season last year (kg/ha)
    "yield_avg_last4",         # mean yield of the previous 4 seasons (kg/ha)
    "harvest_ratio_prev",      # harvested / sown in previous season (crop-loss signal)
]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "yield_kg_ha"


def load_and_clean(path: str) -> pd.DataFrame:
    """Read the raw DCS paddy CSV and return a clean district-season table."""
    df = pd.read_csv(path)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.strip().replace({"-": np.nan, "": np.nan}),
            errors="coerce",
        )
    df["District"] = df["District"].str.strip()
    df["Season"] = df["Season"].str.strip().str.upper()
    df = df.drop_duplicates(subset=["Year", "Season", "District"])           # 100 exact duplicate rows (2010-2012)
    df = df[~df["District"].isin(AGGREGATE_REGIONS)]                          # keep the 25 administrative districts
    # The yield column is labelled "MT/Ha" but values (1,150-6,600) are clearly kg/ha
    df = df.rename(columns={"Average Yield (MT/Ha)": TARGET})
    # Chronological season index: Maha (Oct-Mar) precedes Yala (Apr-Sep) of the same DCS year
    df["t"] = df["Year"] * 2 + (df["Season"] == "YALA").astype(int)
    return df.sort_values(["District", "t"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create sowing-time features. Lags are joined on the season index, so gaps stay NaN."""
    df = df.copy()
    sown = df["Total Extent (Ha)"]
    df["log_total_sown"] = np.log1p(sown)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["share_major"] = (df["Major Sown (Ha)"] / sown).where(sown > 0)
        df["share_minor"] = (df["Minor Sown (Ha)"] / sown).where(sown > 0)
        df["share_rainfed"] = (df["Rainfed Sown (Ha)"] / sown).where(sown > 0)
        df["harvest_ratio"] = (df["Net Harvest (Ha)"] / sown).where(sown > 0).clip(0, 1.2)

    # Only seasons that were actually cultivated count as history
    hist = df.loc[(sown > 0) & (df[TARGET] > 0), ["District", "t", TARGET, "harvest_ratio", "Total Extent (Ha)"]]

    def lag(k, col, name):
        h = hist[["District", "t", col]].copy()
        h["t"] = h["t"] + k
        return h.rename(columns={col: name})

    df = df.merge(lag(1, TARGET, "yield_prev_season"), on=["District", "t"], how="left")
    df = df.merge(lag(2, TARGET, "yield_same_season_ly"), on=["District", "t"], how="left")
    df = df.merge(lag(1, "harvest_ratio", "harvest_ratio_prev"), on=["District", "t"], how="left")
    df = df.merge(lag(2, "Total Extent (Ha)", "sown_ly"), on=["District", "t"], how="left")
    lags = [lag(k, TARGET, f"_l{k}") for k in range(1, 5)]
    for l in lags:
        df = df.merge(l, on=["District", "t"], how="left")
    df["yield_avg_last4"] = df[[f"_l{k}" for k in range(1, 5)]].mean(axis=1)  # mean of available lags
    with np.errstate(divide="ignore", invalid="ignore"):
        df["sown_change_pct"] = ((df["Total Extent (Ha)"] - df["sown_ly"]) / df["sown_ly"] * 100).where(df["sown_ly"] > 0)
    df["sown_change_pct"] = df["sown_change_pct"].clip(-100, 300)
    return df.drop(columns=[f"_l{k}" for k in range(1, 5)] + ["sown_ly"])


def modelling_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for modelling: the season was cultivated and a yield was recorded."""
    m = df[(df["Total Extent (Ha)"] > 0) & (df[TARGET] > 0)].copy()
    return m.sort_values(["t", "District"]).reset_index(drop=True)


def make_input_row(district, season, year, major_ha, minor_ha, rainfed_ha,
                   yield_prev_season, yield_same_season_ly, yield_avg_last4,
                   harvest_ratio_prev, sown_same_season_ly=None) -> pd.DataFrame:
    """Build one model-ready row from raw user inputs (used by the front-end app)."""
    total = major_ha + minor_ha + rainfed_ha
    row = {
        "District": district, "Season": season.upper(), "Year": year,
        "log_total_sown": np.log1p(total),
        "share_major": major_ha / total if total else np.nan,
        "share_minor": minor_ha / total if total else np.nan,
        "share_rainfed": rainfed_ha / total if total else np.nan,
        "sown_change_pct": (np.clip((total - sown_same_season_ly) / sown_same_season_ly * 100, -100, 300)
                            if sown_same_season_ly else np.nan),
        "yield_prev_season": yield_prev_season, "yield_same_season_ly": yield_same_season_ly,
        "yield_avg_last4": yield_avg_last4, "harvest_ratio_prev": harvest_ratio_prev,
    }
    return pd.DataFrame([row])[FEATURES]

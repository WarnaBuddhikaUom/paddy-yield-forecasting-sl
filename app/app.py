"""Sri Lanka Paddy Yield Forecaster – Streamlit front end.

Loads the pickled model (models/paddy_yield_model.pkl) and predicts a district's average paddy
yield for a season from sowing-time inputs. Run with:  streamlit run app/app.py
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import make_input_row, FEATURES, NUM_FEATURES  # noqa: E402

st.set_page_config(page_title="Paddy Yield Forecaster – Sri Lanka", page_icon="🌾", layout="wide")


@st.cache_resource
def load_bundle():
    with open(ROOT / "models" / "paddy_yield_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_history():
    h = pd.read_csv(ROOT / "models" / "district_history.csv")
    return h


@st.cache_resource
def load_explainer(_model):
    import shap
    return shap.TreeExplainer(_model.named_steps["model"])


bundle = load_bundle()
model = bundle["model"]
hist = load_history()


def lookup(district, season, year):
    """Pre-fill inputs from the district's history (the user can overwrite every value)."""
    t = year * 2 + (season == "YALA")
    h = hist[hist["District"] == district].set_index("t")
    grown = h[(h["Total Extent (Ha)"] > 0) & (h["yield_kg_ha"] > 0)]

    def y(k):
        return float(grown.loc[t - k, "yield_kg_ha"]) if (t - k) in grown.index else None

    prev, ly = y(1), y(2)
    last4 = [v for v in (y(1), y(2), y(3), y(4)) if v is not None]
    hr = None
    if (t - 1) in grown.index:
        r = grown.loc[t - 1]
        hr = float(min(r["Net Harvest (Ha)"] / r["Total Extent (Ha)"], 1.2))
    # sown extents: use the actual season if known (replay), otherwise same season last year
    src = h.loc[t] if t in h.index else (h.loc[t - 2] if (t - 2) in h.index else None)
    ext = (dict(major=float(src["Major Sown (Ha)"]), minor=float(src["Minor Sown (Ha)"]),
                rainfed=float(src["Rainfed Sown (Ha)"])) if src is not None else dict(major=0.0, minor=0.0, rainfed=0.0))
    sown_ly = float(h.loc[t - 2, "Total Extent (Ha)"]) if (t - 2) in h.index else None
    actual = float(h.loc[t, "yield_kg_ha"]) if t in grown.index else None
    return dict(prev=prev, ly=ly, avg4=float(np.mean(last4)) if last4 else None, hr=hr,
                ext=ext, sown_ly=sown_ly, actual=actual)


# ---------------- Header ----------------
st.title("🌾 Sri Lanka Paddy Yield Forecaster")
st.caption("Predicts the average paddy yield (kg/ha) of a district for a Maha or Yala season at sowing time, "
           f"using a {bundle['model_name']} model trained on Department of Census and Statistics data ({bundle['trained_on']}).")

tab_pred, tab_about = st.tabs(["Forecast", "About the model"])

with tab_pred:
    c1, c2, c3 = st.columns(3)
    district = c1.selectbox("District", bundle["districts"], index=bundle["districts"].index("Polonnaruwa"))
    season = c2.selectbox("Season", ["MAHA", "YALA"], index=1,
                          help="Maha: Oct–Mar (north-east monsoon). Yala: Apr–Sep (south-west monsoon).")
    year = c3.number_input("Year (DCS season year)", min_value=1981, max_value=2035, value=2023, step=1)

    d = lookup(district, season, int(year))

    st.subheader("1. Area sown this season (hectares)")
    e1, e2, e3 = st.columns(3)
    major = e1.number_input("Major irrigation schemes", min_value=0.0, value=d["ext"]["major"], step=100.0, format="%.0f",
                            key=f"maj{district}{season}{year}")
    minor = e2.number_input("Minor irrigation (tanks/anicuts)", min_value=0.0, value=d["ext"]["minor"], step=100.0,
                            format="%.0f", key=f"min{district}{season}{year}")
    rainfed = e3.number_input("Rain-fed", min_value=0.0, value=d["ext"]["rainfed"], step=100.0, format="%.0f",
                              key=f"rf{district}{season}{year}")

    st.subheader("2. Recent history of the district")
    st.caption("Pre-filled from DCS records when available – overwrite them for a new season.")
    h1, h2, h3, h4, h5 = st.columns(5)
    prev = h1.number_input("Previous season yield (kg/ha)", 0.0, 9000.0, d["prev"] or 0.0, 50.0, format="%.0f",
                           key=f"p{district}{season}{year}")
    ly = h2.number_input("Same season last year (kg/ha)", 0.0, 9000.0, d["ly"] or 0.0, 50.0, format="%.0f",
                         key=f"l{district}{season}{year}")
    avg4 = h3.number_input("Average of last 4 seasons (kg/ha)", 0.0, 9000.0, d["avg4"] or 0.0, 50.0, format="%.0f",
                           key=f"a{district}{season}{year}")
    hr = h4.number_input("Last season harvested ÷ sown", 0.0, 1.2, d["hr"] if d["hr"] is not None else 0.9, 0.01,
                         key=f"h{district}{season}{year}")
    sown_ly = h5.number_input("Area sown same season last year (ha)", 0.0, 500000.0, d["sown_ly"] or 0.0, 100.0,
                              format="%.0f", key=f"s{district}{season}{year}")

    total = major + minor + rainfed
    if st.button("Predict yield", type="primary", width="stretch"):
        if total <= 0:
            st.error("Enter the sown area – at least one of the three extents must be above zero.")
        else:
            x = make_input_row(district, season, int(year), major, minor, rainfed,
                               prev or np.nan, ly or np.nan, avg4 or np.nan, hr, sown_ly or None)
            pred = float(model.predict(x)[0])
            mae = bundle["test_mae"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted yield", f"{pred:,.0f} kg/ha", f"{pred/1000:.2f} t/ha", delta_color="off")
            m2.metric("Typical error range", f"{pred-mae:,.0f} – {pred+mae:,.0f}", "± test MAE", delta_color="off")
            m3.metric("Indicative production", f"{pred/1000*total*hr:,.0f} t",
                      f"{total:,.0f} ha sown × {hr:.2f} harvested", delta_color="off")
            if d["actual"] is not None:
                st.info(f"Recorded DCS yield for {district} {season} {int(year)}: **{d['actual']:,.0f} kg/ha** "
                        f"(model error {pred - d['actual']:+,.0f} kg/ha).")

            # Explain this prediction with SHAP (one-hot districts/seasons grouped back together)
            try:
                explainer = load_explainer(model)
                prep = model.named_steps["prep"]
                names = list(prep.get_feature_names_out())
                sv = explainer(pd.DataFrame(np.asarray(prep.transform(x)), columns=names))
                contrib = pd.Series(sv.values[0], index=names)
                grp = contrib.groupby(lambda n: "District" if n.startswith("District_") else
                                      "Season" if n.startswith("Season_") else n).sum()
                grp = grp.reindex(grp.abs().sort_values().index)
                fig, ax = plt.subplots(figsize=(7, 3.8))
                ax.barh(grp.index, grp.values, color=["#2e8b57" if v > 0 else "#c0392b" for v in grp.values])
                ax.axvline(0, color="k", lw=0.6)
                ax.set_xlabel("Contribution to the forecast (kg/ha)")
                ax.set_title(f"Why this forecast? (start: average {float(np.ravel(sv.base_values)[0]):,.0f} kg/ha)")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                st.caption("Green bars raise the forecast, red bars lower it (SHAP values).")
            except Exception as exc:  # explanation is optional – never block the prediction
                st.caption(f"Explanation unavailable: {exc}")

with tab_about:
    st.markdown(f"""
**Model:** {bundle['model_name']} (tuned with time-series cross-validation), test RMSE on 2016–2023
seasons ≈ **{bundle['test_rmse']:,.0f} kg/ha**, MAE ≈ **{bundle['test_mae']:,.0f} kg/ha**.

**Inputs used (all known at sowing time):** district, season, year, sown area by irrigation type,
change in sown area vs last year, previous yields and last season's harvested share.

**Limitations:** the model cannot foresee shocks that are not in its inputs (e.g. the 2021/22 chemical
fertilizer ban, floods or droughts after sowing). Use it as a planning baseline, not a guarantee.
""")
    comp = ROOT / "models" / "model_comparison.csv"
    if comp.exists():
        st.dataframe(pd.read_csv(comp, index_col=0).round(1), width="stretch")
    for img in ["05_model_comparison.png", "07_shap_importance.png"]:
        p = ROOT / "figures" / img
        if p.exists():
            st.image(str(p))

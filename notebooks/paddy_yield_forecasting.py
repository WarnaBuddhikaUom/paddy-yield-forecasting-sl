# %% [markdown]
# # Forecasting District-Level Paddy Yield in Sri Lanka at Sowing Time
# ### Applied Machine Learning – Final Assignment
#
# **Problem (local, Sri Lankan context).** Rice is Sri Lanka's staple food and paddy is grown in two seasons:
# **Maha** (north-east monsoon, roughly Oct–Mar) and **Yala** (south-west monsoon, roughly Apr–Sep).
# The Department of Agriculture, the Paddy Marketing Board and district officers have to plan fertilizer
# distribution, storage, purchasing and imports *before* the harvest is known.
#
# **Goal.** At the time a season is sown, predict the **average paddy yield (kg/ha) of each district** for that season,
# using only information that is already available at sowing time:
# how much land was sown, under which irrigation type, and the district's yield history.
# This is a **supervised regression** problem on tabular data.
#
# **Dataset.** District-wise paddy statistics (1979–2023, Maha & Yala) published by the
# Department of Census and Statistics (DCS), Sri Lanka: sown and harvested extent by irrigation type
# (major irrigation, minor irrigation, rain-fed), average yields and production. 2,612 raw rows × 16 columns.
# The data are **aggregated official statistics at district level – no personal or sensitive data** are involved.
#
# **New algorithm (not covered in lectures): Explainable Boosting Machine (EBM)** – a modern generalised additive model with
# pairwise interactions (GA²M), trained by cyclic gradient boosting. It is a *glass-box* model: every feature's effect is an
# exact, plottable function, so the model is interpretable by design (intrinsic interpretability), not only by post-hoc tools.
#
# **Plan of the notebook (follows the assignment rubric)**
# 1. Data audit & cleaning  2. Exploratory analysis  3. Leakage-free features  4. Train / validation / test design
# 5. Preprocessing  6. Existing approaches (baselines)  7. **The new algorithm: EBM** – why, how it differs, hyperparameters
# 8. Benchmarks taught in lectures (RF, XGBoost, LightGBM, CatBoost, Stacking)  9. Evaluation & comparison
# 10. Explainability: EBM global/local, PDP, SHAP cross-check, LIME  11. Bias & fairness check
# 12. Saving the model & inference on an unseen sample  13. Critical discussion  14. Conclusions

# %%
import sys, time, json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import randint, uniform, loguniform
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor  # used through the CatBoostSK wrapper
import shap
from lime.lime_tabular import LimeTabularExplainer
from interpret.glassbox import ExplainableBoostingRegressor

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from features import (load_and_clean, add_features, modelling_table, AGGREGATE_REGIONS,
                      FEATURES, NUM_FEATURES, CAT_FEATURES, TARGET)
from cat_wrapper import CatBoostSK

FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42
import os; QUICK = os.environ.get('QUICK') == '1'   # debugging only: tiny searches
NI = (lambda n: 2 if QUICK else n)
np.random.seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

def savefig(name):
    plt.tight_layout(); plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight"); plt.show(); plt.close("all")

# %% [markdown]
# ## 1. Data audit
# First look at the raw file **before** cleaning, to understand what problems need fixing.

# %%
raw = pd.read_csv(ROOT / "data" / "Agriculture_Statistics_Paddy.csv")
print("Raw shape:", raw.shape)
print(raw.dtypes.value_counts())
raw.head()

# %%
audit = {
    "numeric columns stored as text (thousand separators)": int((raw.dtypes == object).sum() + (raw.dtypes == "str").sum()) - 2,
    "'-' placeholders in Production": int((raw["Production (MT)"].astype(str).str.strip() == "-").sum()),
    "fully duplicated rows": int(raw.duplicated().sum()),
    "rows for overlapping aggregate regions": int(raw["District"].isin(AGGREGATE_REGIONS).sum()),
    "rows with zero sown extent": int((raw["Total Extent (Ha)"].astype(str).str.strip() == "0").sum()),
}
pd.Series(audit, name="count").to_frame()

# %% [markdown]
# **Findings and decisions**
# * All measurements are stored as text with thousand separators (`"7,111.70"`) and `-` for missing → convert to numbers, `-` → NaN.
# * **100 exact duplicate rows** (years 2010–2012 were entered twice) → drop, otherwise those years get double weight and could leak between folds.
# * `Udawalawe`, `Mahaweli'h'`, `Highland`, `North & East` are **irrigation-system / regional aggregates**, not districts.
#   They overlap with real districts (e.g. Mahaweli areas are inside Polonnaruwa/Anuradhapura), so they are removed and the 25 administrative districts are kept.
# * Seasons with **zero sown extent** (mostly Jaffna, Kilinochchi and other Northern districts during the civil-war years) have no yield to predict → excluded from modelling, but kept as "gaps" in the history.
# * **Unit issue:** the column says "MT/Ha" but values are 1,150–6,600. Sri Lankan paddy yields are about 3–5 tonnes/ha, so the values are actually **kg/ha**. The target is renamed `yield_kg_ha`.
#
# Domain knowledge was needed for the last three points. A model trained blindly on the raw Kaggle-style table would double count, mix regions with districts, and report yields 1000× too high.

# %%
clean = load_and_clean(ROOT / "data" / "Agriculture_Statistics_Paddy.csv")
print("After cleaning:", clean.shape, "| districts:", clean["District"].nunique(), "| years:", clean["Year"].min(), "-", clean["Year"].max())
print("Missing values per column (after converting '-' to NaN):")
print(clean.isna().sum()[lambda s: s > 0])

# %% [markdown]
# ## 2. Exploratory data analysis

# %%
cult = clean[(clean["Total Extent (Ha)"] > 0) & (clean[TARGET] > 0)]
nat = cult.groupby(["Year", "Season"]).apply(
    lambda g: np.average(g[TARGET], weights=g["Net Harvest (Ha)"].fillna(g["Total Extent (Ha)"]))).unstack()
fig, ax = plt.subplots(figsize=(10, 4))
nat.plot(ax=ax, marker="o", ms=3)
ax.axvspan(2020.5, 2022.5, color="red", alpha=0.08)
ax.annotate("2021/22 chemical\nfertilizer ban", xy=(2022, nat.loc[2022].min()), xytext=(2008, 2900),
            arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.set_ylabel("Area-weighted yield (kg/ha)"); ax.set_title("National paddy yield by season, 1979–2023")
savefig("01_national_trend")

# %% [markdown]
# * Yields roughly **doubled** from ~2,600 kg/ha (1980) to ~4,500 kg/ha (2019–2021): a strong long-term trend (improved varieties, irrigation, fertilizer subsidy). `Year` is therefore a useful feature.
# * **Maha and Yala behave differently**; Yala is more variable because it depends on stored tank water.
# * A sharp collapse in **2022 (Maha 2021/22 and Yala 2022)**: the government's sudden ban on chemical fertilizer in 2021. This is a *structural shock* that no historical pattern can predict — important when we judge the models on 2016–2023.

# %%
order = cult.groupby("District")[TARGET].median().sort_values().index
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.boxplot(data=cult[cult.Year >= 2000], x="District", y=TARGET, order=order, ax=ax, color="#8fbc8f", fliersize=2)
ax.tick_params(axis="x", rotation=75); ax.set_ylabel("Yield (kg/ha)"); ax.set_xlabel("")
ax.set_title("District yield distribution (2000–2023)")
savefig("02_district_box")

# %% [markdown]
# The dry-zone districts with large **major irrigation** schemes (Polonnaruwa, Ampara, Hambantota, Anuradhapura) have the highest yields;
# wet-zone, mostly **rain-fed** districts (Colombo, Kalutara, Galle, Kegalle, Ratnapura) have the lowest. District identity carries a lot of information.

# %%
feat_all = add_features(clean)
model_df = modelling_table(feat_all)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, col, lab in zip(axes, ["share_major", "share_rainfed", "yield_same_season_ly"],
                        ["Share under major irrigation", "Share rain-fed", "Same season last year (kg/ha)"]):
    ax.scatter(model_df[col], model_df[TARGET], s=6, alpha=0.35, c=model_df["Year"], cmap="viridis")
    ax.set_xlabel(lab); ax.set_ylabel("Yield (kg/ha)")
fig.suptitle("Irrigation type and yield history vs. yield (colour = year)")
savefig("03_scatter")

# %%
corr = model_df[NUM_FEATURES + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation of engineered features with yield")
savefig("04_corr")

# %% [markdown]
# * Lagged yields correlate strongly with the target (0.65–0.79) – yields are persistent.
# * `share_major` (+0.52) and `share_rainfed` (−0.56) confirm the irrigation story. They are strongly negatively correlated with each other (they sum to 1 with `share_minor`) — tree models handle this collinearity well, but it makes linear coefficients unstable.
# * `harvest_ratio_prev` is close to zero linearly, but it can matter non-linearly (a previous drought/flood season).

# %% [markdown]
# ## 3. Leakage-free feature engineering
# | Feature | Meaning | Why it is allowed |
# |---|---|---|
# | `District`, `Season`, `Year` | where / when | known in advance |
# | `log_total_sown` | log of total sown hectares | sowing is complete before the forecast |
# | `share_major / minor / rainfed` | irrigation mix of the sown area | known at sowing |
# | `sown_change_pct` | sown area vs. same season last year | farmers sow less when water is short – an early drought signal |
# | `yield_prev_season`, `yield_same_season_ly`, `yield_avg_last4` | yield history | previous seasons are already harvested |
# | `harvest_ratio_prev` | harvested ÷ sown area last season | crop loss (flood / drought) last season |
#
# **Excluded on purpose (data leakage):** harvested extents, per-irrigation yields and production of the *current* season.
# `Production = Net Harvest × Yield` exactly (checked below), so including them would let the model "read" the answer and produce a perfect but useless score.
#
# Lags are joined on a chronological season index (`t = 2·Year + [Yala]`), not with `shift()`, so a missing season (e.g. war years in Jaffna) becomes a **missing value** instead of silently using an older season.

# %%
chk = clean.dropna(subset=["Production (MT)"]).query("`Net Harvest (Ha)` > 0")
ratio = chk["Production (MT)"] / (chk["Net Harvest (Ha)"] * chk[TARGET] / 1000)
print(f"Production / (Net harvest x yield/1000): median = {ratio.median():.4f}  -> confirms kg/ha and the leakage risk")
print("Modelling table:", model_df.shape)
model_df[FEATURES + [TARGET]].isna().mean().mul(100).round(1).rename("% missing").to_frame().T

# %% [markdown]
# ## 4. Train / validation / test design
# Random splitting would let the model train on 2020 and be tested on 2019 – unrealistic for a forecast.
# We use a **chronological** design:
# * **Training set: 1979–2015** (1,782 district-seasons).
# * **Validation: forward-chaining `TimeSeriesSplit` (4 folds) inside the training years.** Each fold trains on the past and validates on the next block of years (the last fold validates on ≈2008–2015). All hyperparameters are chosen on this validation score, never on the test set.
# * **Test set: 2016–2023** (391 district-seasons, 8 unseen years, including the 2022 fertilizer shock). Used once, for reporting.
# * **Held-out inference sample:** *Polonnaruwa, Yala 2023* is removed from everything and used only at the end to demonstrate inference with the saved pickle model.

# %%
HOLDOUT = (model_df["District"] == "Polonnaruwa") & (model_df["Year"] == 2023) & (model_df["Season"] == "YALA")
holdout = model_df[HOLDOUT].copy()
data = model_df[~HOLDOUT]
train = data[data["Year"] <= 2015].sort_values("t")
test = data[data["Year"] > 2015].sort_values("t")
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
print(f"train {X_train.shape}, test {X_test.shape}, holdout {holdout.shape}")
tscv = TimeSeriesSplit(n_splits=4)   # forward-chaining CV inside the training period, used for all tuning

# %% [markdown]
# ## 5. Preprocessing pipeline
# All preprocessing lives inside a scikit-learn `Pipeline`, so imputation statistics and scaling are learned **only from the training folds** (no leakage into validation/test), and the exact same steps are saved in the pickle file for inference.
# * **Imputation:** median for numeric lags (robust to the skewed yields); the ~3% missing lags come from gaps after uncultivated seasons.
# * **Scaling:** `StandardScaler` – needed for the linear baselines; harmless for trees.
# * **Encoding:** one-hot for `District` and `Season` (`handle_unknown="ignore"` so a new district does not crash the app). CatBoost uses its own categorical handling (ordered target statistics), and **EBM handles categorical strings and missing values natively** (missing values get their own learned score), so it needs no imputation or scaling at all.
# * **Transformation:** sown area is log-transformed (`log_total_sown`) because district extents range from a few hundred to 100,000+ ha.

# %%
def onehot_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ], verbose_feature_names_out=False)

def catboost_preprocessor():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
        ("cat", "passthrough", CAT_FEATURES),
    ], verbose_feature_names_out=False).set_output(transform="pandas")

def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))

def evaluate(name, y_true, pred, mask=None):
    out = {"model": name,
           "RMSE": rmse(y_true, pred), "MAE": mean_absolute_error(y_true, pred),
           "MAPE %": 100 * mean_absolute_percentage_error(y_true, pred), "R2": r2_score(y_true, pred),
           "within ±10 %": 100 * np.mean(np.abs(pred - y_true) / y_true <= 0.10)}
    no22 = (test["Year"] != 2022).values
    out["RMSE excl. 2022"] = rmse(y_true[no22], pred[no22])
    return out

results, cv_scores, fitted, train_time = [], {}, {}, {}

# %% [markdown]
# ## 6. Existing approaches (baselines)
# Before any advanced model, we measure what simple, commonly used approaches achieve. A new model is only worth deploying if it clearly beats these.
# 1. **Persistence (naïve forecast)** – "this season = same season last year". This is how many planning estimates are made in practice.
# 2. **Moving average** – mean of the last 4 seasons.
# 3. **Linear regression (OLS)** – the classical statistical approach used in many crop-yield studies.

# %%
fill = y_train.median()
for name, col in [("Persistence (same season LY)", "yield_same_season_ly"), ("Moving average (last 4)", "yield_avg_last4")]:
    pred = X_test[col].fillna(fill).values
    results.append(evaluate(name, y_test.values, pred))
    cv_scores[name] = np.mean([rmse(y_train.iloc[v], X_train[col].iloc[v].fillna(fill)) for _, v in tscv.split(X_train)])

lin = Pipeline([("prep", onehot_preprocessor()), ("model", LinearRegression())])
cv_scores["Linear regression"] = -cross_val_score(lin, X_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error").mean()
t0 = time.time(); lin.fit(X_train, y_train); train_time["Linear regression"] = time.time() - t0
fitted["Linear regression"] = lin
results.append(evaluate("Linear regression", y_test.values, lin.predict(X_test)))
pd.DataFrame(results).set_index("model").round(1)

# %% [markdown]
# ## 7. The new algorithm: Explainable Boosting Machine (EBM)
# **What it is.** EBM (Lou, Caruana & Gehrke 2012–2013; implemented in Microsoft's *InterpretML*) is a **generalised additive model with pairwise interactions (GA²M)**:
#
# $$\hat{y} = \beta_0 + \sum_i f_i(x_i) + \sum_{(i,j)} f_{ij}(x_i, x_j)$$
#
# Each $f_i$ is a learned **shape function** for one feature, and each $f_{ij}$ a small 2-D function for one of a few automatically detected feature pairs.
# The functions are learned with **cyclic gradient boosting**: in each round the algorithm visits the features *one at a time* and fits a tiny tree on that single feature to the current residuals, with a very small learning rate, for thousands of rounds. Many bagged copies ("outer bags") are averaged to reduce variance.
#
# **How it differs from the standard models taught in lectures**
# | Model | Form of the model | Interpretability |
# |---|---|---|
# | Linear / logistic regression | $\beta_0 + \sum \beta_i x_i$ – one straight-line weight per feature | Transparent, but cannot represent curves or thresholds |
# | Decision tree | One tree; every split can mix features | Readable only when shallow; unstable |
# | k-NN | No global function – predicts from nearest stored examples | Only "similar cases" explanations |
# | Random Forest / XGBoost / LightGBM / CatBoost | Sums of hundreds of deep, multi-feature trees | Black box – needs post-hoc SHAP/LIME |
# | **EBM** | **Sum of one learned curve per feature + a few pairwise surfaces** | **Glass box – each curve *is* the model; local explanations are exact** |
#
# **Why it was selected for this problem**
# * **Trust in a public-planning setting.** Agricultural officers must be able to see *exactly* how irrigation mix or last year's yield changes a forecast. EBM's shape functions are the model itself, so explanations are exact rather than approximations (lecture: *intrinsic* vs *post-hoc* interpretability).
# * **Accuracy close to boosted trees.** Boosting keeps it competitive with XGBoost/LightGBM on tabular data – verified below.
# * **Fits the data:** handles the categorical `District`/`Season` and the missing lag values natively, and the additive structure is a good match for yield, which is largely "baseline level + irrigation effect + trend".
# * **Not deep learning**, works well on small tabular datasets (≈1,800 rows).
#
# **Hyperparameter design (searched with the forward-chaining validation folds)**
# | Parameter | Range | Reasoning |
# |---|---|---|
# | `learning_rate` | 0.005–0.05 (log) | EBM relies on many small cyclic steps; small rates give smoother, more stable shape functions |
# | `interactions` | 0, 3, 5, 10 | number of pairwise terms: more pairs = more flexibility but less readable; 0 = pure additive model |
# | `min_samples_leaf` | 5–30 | each bin step must be supported by enough district-seasons, so curves don't chase single flood/drought years |
# | `max_leaves` | 2–3 | leaves per per-feature tree; small = smoother curves |
# | `outer_bags` | 8, 14 | number of bagged models averaged; more = lower variance (and confidence bands) |
# | `smoothing_rounds` | 0, 100, 200 | initial rounds with random splits to smooth the shape functions |
# | `max_bins` | 128, 256 | resolution of continuous features |

# %%
ebm_space = {"learning_rate": loguniform(0.005, 0.05), "interactions": [0, 3, 5, 10], "min_samples_leaf": randint(5, 31),
             "max_leaves": [2, 3], "outer_bags": [8, 14], "smoothing_rounds": [0, 100, 200], "max_bins": [128, 256]}
ebm_search = RandomizedSearchCV(ExplainableBoostingRegressor(random_state=RANDOM_STATE, n_jobs=2), ebm_space, n_iter=NI(8),
                                cv=tscv, scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, n_jobs=1, refit=True)
t0 = time.time(); ebm_search.fit(X_train, y_train); train_time["EBM (new)"] = time.time() - t0
cv_scores["EBM (new)"] = -ebm_search.best_score_
fitted["EBM (new)"] = ebm_search.best_estimator_
best_params = {"EBM (new)": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, np.integer) else v)
                             for k, v in ebm_search.best_params_.items()}}
results.append(evaluate("EBM (new)", y_test.values, ebm_search.best_estimator_.predict(X_test)))
print(f"EBM: validation RMSE = {cv_scores['EBM (new)']:.1f} kg/ha  ({train_time['EBM (new)']:.0f}s)")
print("best params:", best_params["EBM (new)"])
pd.DataFrame(ebm_search.cv_results_)[["params", "mean_test_score", "std_test_score"]].assign(
    validation_RMSE=lambda d: -d.mean_test_score).sort_values("validation_RMSE")[["params", "validation_RMSE", "std_test_score"]].round(1)

# %% [markdown]
# ## 8. Benchmarks taught in lectures (RF, XGBoost, LightGBM, CatBoost, Stacking)
# To judge whether EBM is competitive, it is compared with the ensemble methods covered in the module, each tuned with the same validation folds (no default parameters).
#
# | Model | Key search ranges | Reasoning |
# |---|---|---|
# | Random Forest | 300–800 trees, depth 6–20, `min_samples_leaf` 1–10, `max_features` 0.3–0.8 | limit depth / leaf size for a small dataset; feature subsampling de-correlates trees (lag features are correlated) |
# | XGBoost | `learning_rate` 0.01–0.15, depth 2–6, subsample 0.6–1, L1/L2 | shallow trees and regularisation for ~1.8k rows |
# | LightGBM | `num_leaves` 8–40, `min_child_samples` 10–50 | leaf-wise growth over-fits easily on small data |
# | CatBoost | depth 4–8, `l2_leaf_reg` 1–15 | native categorical handling via ordered target statistics |
# | Stacking | RF + XGB + LGBM + CatBoost → Ridge | combines bagging and boosting errors |

# %%
def tune(name, estimator, space, n_iter, prep):
    pipe = Pipeline([("prep", prep), ("model", estimator)])
    search = RandomizedSearchCV(pipe, {f"model__{k}": v for k, v in space.items()}, n_iter=n_iter, cv=tscv,
                                scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, n_jobs=1, refit=True)
    t0 = time.time(); search.fit(X_train, y_train); train_time[name] = time.time() - t0
    cv_scores[name] = -search.best_score_
    fitted[name] = search.best_estimator_
    best = {k.replace("model__", ""): (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, np.integer) else v)
            for k, v in search.best_params_.items()}
    print(f"{name}: validation RMSE = {-search.best_score_:.1f} kg/ha  ({train_time[name]:.0f}s)\n  best params: {best}")
    results.append(evaluate(name, y_test.values, search.best_estimator_.predict(X_test)))
    return best

best_params["Random Forest"] = tune("Random Forest",
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=2),
    {"n_estimators": randint(300, 801), "max_depth": [6, 8, 10, 14, 20, None], "min_samples_leaf": randint(1, 11),
     "max_features": uniform(0.3, 0.5), "max_samples": uniform(0.6, 0.4)}, n_iter=NI(20), prep=onehot_preprocessor())
best_params["XGBoost"] = tune("XGBoost",
    XGBRegressor(random_state=RANDOM_STATE, n_jobs=2, tree_method="hist", objective="reg:squarederror"),
    {"n_estimators": randint(300, 1501), "learning_rate": loguniform(0.01, 0.15), "max_depth": randint(2, 7),
     "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4), "min_child_weight": randint(1, 11),
     "reg_lambda": loguniform(0.1, 20), "reg_alpha": uniform(0, 2)}, n_iter=NI(30), prep=onehot_preprocessor())
best_params["LightGBM"] = tune("LightGBM",
    LGBMRegressor(random_state=RANDOM_STATE, n_jobs=2, verbose=-1),
    {"n_estimators": randint(300, 1501), "learning_rate": loguniform(0.01, 0.15), "num_leaves": randint(8, 41),
     "min_child_samples": randint(10, 51), "subsample": uniform(0.6, 0.4), "subsample_freq": [1],
     "colsample_bytree": uniform(0.6, 0.4), "reg_lambda": loguniform(0.1, 20)}, n_iter=NI(30), prep=onehot_preprocessor())
best_params["CatBoost"] = tune("CatBoost",
    CatBoostSK(random_state=RANDOM_STATE, thread_count=2),
    {"iterations": randint(400, 1201), "learning_rate": loguniform(0.02, 0.15), "depth": randint(4, 9),
     "l2_leaf_reg": loguniform(1, 15), "bagging_temperature": uniform(0, 1)}, n_iter=NI(12), prep=catboost_preprocessor())

stack = StackingRegressor(
    estimators=[(n.replace(" ", "_"), fitted[n]) for n in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]],
    final_estimator=RidgeCV(alphas=np.logspace(-2, 3, 20)), cv=KFold(5, shuffle=False), n_jobs=1)
cv_scores["Stacking"] = -cross_val_score(stack, X_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error").mean()
t0 = time.time(); stack.fit(X_train, y_train); train_time["Stacking"] = time.time() - t0
fitted["Stacking"] = stack
results.append(evaluate("Stacking", y_test.values, stack.predict(X_test)))
print(f"Stacking: validation RMSE = {cv_scores['Stacking']:.1f}")

# %% [markdown]
# ## 9. Evaluation and comparison
# **Metrics (regression task):** RMSE (penalises large misses – important for food-security planning), MAE (typical error in kg/ha), MAPE (relative error), R² (variance explained), and the share of forecasts within ±10 % of the actual yield (an operational "good enough" rate). RMSE is also reported **excluding 2022** to separate normal-year accuracy from the fertilizer-ban shock.
# *Validation RMSE* comes from the forward-chaining folds (used for tuning); *test* metrics come from the untouched 2016–2023 seasons.

# %%
res = pd.DataFrame(results).set_index("model")
res.insert(0, "Validation RMSE", pd.Series(cv_scores))
res["fit time (s)"] = pd.Series(train_time)
res = res.sort_values("RMSE")
res.round(2)

# %%
BASELINES = ["Persistence (same season LY)", "Moving average (last 4)", "Linear regression"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
r = res.sort_values("RMSE")
cols = ["#d4a017" if n == "EBM (new)" else "#c0c0c0" if n in BASELINES else "#2e8b57" for n in r.index]
axes[0].barh(r.index, r["RMSE"], color=cols); axes[0].invert_yaxis()
axes[0].barh(r.index, r["RMSE excl. 2022"], color="none", edgecolor="black", hatch="//", label="excl. 2022")
for y, v in enumerate(r["RMSE"]): axes[0].text(v + 8, y, f"{v:.0f}", va="center", fontsize=8)
axes[0].set_xlim(0, 980); axes[0].set_xlabel("Test RMSE (kg/ha) – lower is better"); axes[0].legend(loc="upper right")
axes[1].barh(r.index, r["Validation RMSE"], color=cols); axes[1].invert_yaxis(); axes[1].set_xlabel("Validation RMSE (kg/ha)")
axes[1].set_yticklabels([])
fig.suptitle("Gold = EBM (new algorithm), green = ensembles taught in lectures, grey = existing approaches")
savefig("05_model_comparison")

# %%
BEST = "EBM (new)"
best_model = fitted[BEST]
test_pred = best_model.predict(X_test)
by_year = test.assign(pred=test_pred, err=test_pred - y_test).groupby(["Year", "Season"]).agg(
    actual=(TARGET, "mean"), predicted=("pred", "mean"), MAE=("err", lambda e: e.abs().mean())).round(0)
by_year

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
axes[0].scatter(y_test, test_pred, s=10, alpha=0.6, c=(test["Year"] == 2022).map({True: "red", False: "#2e8b57"}))
lims = [y_test.min() - 200, y_test.max() + 200]; axes[0].plot(lims, lims, "k--", lw=1)
axes[0].set_xlabel("Actual yield (kg/ha)"); axes[0].set_ylabel("Predicted"); axes[0].set_title("EBM: actual vs predicted on test (red = 2022)")
by_year[["actual", "predicted"]].plot(ax=axes[1], marker="o")
axes[1].set_xticks(range(len(by_year))); axes[1].set_xticklabels([f"{y} {s[0]}" for y, s in by_year.index], rotation=60)
axes[1].set_ylabel("Mean district yield (kg/ha)"); axes[1].set_title("Mean by season (M = Maha, Y = Yala)")
savefig("06_best_model_fit")

# %% [markdown]
# ### What the results indicate
# * **Validation (forward-chaining folds, 1979–2015):** the tree ensembles score 425–437 kg/ha and EBM 441 kg/ha, about 4 % above the best (LightGBM). EBM is 14 % better than linear regression (513) and 8 % better than persistence (477), but only level with the 4-season moving average (442) on these older years.
# * **Test (unseen 2016–2023):** EBM's RMSE is **623 kg/ha**: better than LightGBM (626), XGBoost (633) and linear regression (639), **9 % better than the moving average (685)** and **16 % better than persistence (742)**, and within 2.5 % of the best model (Stacking, 608). Its MAE (449 kg/ha) is second only to Stacking, and **60 % of its district-season forecasts are within ±10 %** of the actual yield (MAPE 12.9 %).
# * **Excluding the 2022 shock**, EBM's RMSE is 457 kg/ha against 583 for persistence (−22 %).
# * **What this indicates:** the new glass-box model reaches the accuracy of the black-box boosted ensembles on future seasons. Interpretability costs at most ~4 % on validation and nothing on test, so EBM is the model deployed in the app.
# * **2022 (fertilizer ban):** EBM's mean error jumps to about 1,400 kg/ha in Maha and 900 kg/ha in Yala, against 200–540 kg/ha in the other test seasons (table above). Every model, including the baselines, over-predicts that year, because the shock is not in any sowing-time feature.
# * **Unseen hold-out case:** for Polonnaruwa Yala 2023 the final EBM predicts 4,731 kg/ha against an actual 4,918 kg/ha (−3.8 %; section 12).
#
# **Strengths and limitations of each model**
# | Model | Strengths | Limitations |
# |---|---|---|
# | Persistence / moving average | No training, transparent | Ignores irrigation mix, sown area and trend; lags every change |
# | Linear regression | Fast, readable, extrapolates the trend | No curves or interactions; unstable with collinear shares; worst validation score |
# | Random Forest | Robust, low tuning sensitivity | Black box; cannot extrapolate; large model file |
# | XGBoost / LightGBM | Best validation scores; strong regularisation | Black box; many interacting hyperparameters; need one-hot districts |
# | CatBoost | Native categorical handling; good test score | Slow; black box |
# | Stacking | Best test RMSE | Most complex to deploy and explain; small gain |
# | **EBM (new)** | **Glass box with exact explanations; boosted-tree-level test accuracy; native categorical and missing-value handling** | **Slowest to train (bagging × cyclic boosting); like trees, its piecewise-constant curves cannot extrapolate beyond the training range; validation ~4 % behind LightGBM** |

# %% [markdown]
# ## 10. Explainability and interpretation
# At least one XAI method is required; four complementary methods are used:
# 1. **EBM global explanation (intrinsic)** – term importances and the learned shape functions: *what the model has learned*.
# 2. **Partial Dependence Plots (PDP)** – model-agnostic average effect of a feature, as a cross-check of the shape functions.
# 3. **SHAP on the best tree benchmark (post-hoc)** – do two very different models learn the same drivers?
# 4. **Local explanations for one unseen case** – EBM's exact additive breakdown, cross-checked with **LIME**.
#
# ### 10.1 Which features are most influential (EBM term importance)

# %%
ebm = best_model
imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.barh(imp.index, imp.values, color=["#d4a017" if "&" in n else "#2e8b57" for n in imp.index])
ax.set_xlabel("mean |contribution| to the forecast (kg/ha)")
ax.set_title("EBM global term importance (gold = pairwise interactions)")
savefig("07_ebm_importance")
imp.sort_values(ascending=False).round(1).to_frame("importance (kg/ha)")

# %% [markdown]
# ### 10.2 What the model has learned: shape functions
# Each panel below **is** part of the model: the forecast is the intercept plus the value read from every curve. Shaded bands show the variation across the bagged models (uncertainty).

# %%
g = ebm.explain_global()
def shape(term):
    d = g.data(ebm.term_names_.index(term))
    return d
fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
for ax, term in zip(axes.ravel()[:5], ["yield_same_season_ly", "yield_avg_last4", "share_rainfed", "share_major", "Year"]):
    d = shape(term); edges = np.asarray(d["names"], dtype=float); sc = np.asarray(d["scores"])
    lo, hi = np.asarray(d["lower_bounds"]), np.asarray(d["upper_bounds"])
    x = np.repeat(edges, 2)[1:-1]; y = np.repeat(sc, 2)
    ax.plot(x, y, color="#1e4d2b"); ax.fill_between(x, np.repeat(lo, 2), np.repeat(hi, 2), color="#97bc62", alpha=0.35)
    ax.axhline(0, color="k", lw=0.5); ax.set_title(term); ax.set_ylabel("contribution (kg/ha)")
d = shape("District"); dd = pd.Series(d["scores"], index=d["names"]).sort_values()
axes[1, 2].barh(dd.index, dd.values, color=["#c0392b" if v < 0 else "#2e8b57" for v in dd.values]); axes[1, 2].tick_params(axis="y", labelsize=6.5)
axes[1, 2].set_title("District"); axes[1, 2].set_xlabel("contribution (kg/ha)")
fig.suptitle("EBM shape functions – the model's learned effect of each feature")
savefig("08_ebm_shapes")

# %%
inter = [t for t in ebm.term_names_ if "&" in t]
if inter:
    # show the strongest *interpretable numeric* interaction (time x irrigation if the model found one)
    numeric = [t for t in inter if not any(c in t for c in CAT_FEATURES)]
    pick = [t for t in numeric if t.startswith("Year &")] or numeric or inter
    top = max(pick, key=lambda t: ebm.term_importances()[ebm.term_names_.index(t)])
    d = shape(top)
    print("Interactions found:", inter, "| plotted:", top)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    left, right = d["left_names"], d["right_names"]
    im = ax.imshow(np.asarray(d["scores"]).T, origin="lower", aspect="auto", cmap="RdBu_r",
                   extent=[0, len(left) - 1, 0, len(right) - 1] if not isinstance(left[0], str) else None)
    a, b = top.split(" & ")
    ax.set_xlabel(a); ax.set_ylabel(b); plt.colorbar(im, label="contribution (kg/ha)")
    if not isinstance(left[0], str):
        ax.set_xticks(np.linspace(0, len(left) - 1, 5)); ax.set_xticklabels([f"{float(v):.0f}" if float(v) > 10 else f"{float(v):.2f}" for v in np.asarray(left, float)[np.linspace(0, len(left) - 1, 5).astype(int)]])
    if not isinstance(right[0], str):
        ax.set_yticks(np.linspace(0, len(right) - 1, 5)); ax.set_yticklabels([f"{float(v):.2f}" if float(v) <= 10 else f"{float(v):.0f}" for v in np.asarray(right, float)[np.linspace(0, len(right) - 1, 5).astype(int)]])
    ax.set_title(f"Interaction: {top}")
    savefig("08b_ebm_interaction")

# %% [markdown]
# ### 10.3 Partial Dependence Plots (model-agnostic cross-check)

# %%
from sklearn.inspection import PartialDependenceDisplay
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
PartialDependenceDisplay.from_estimator(ebm, data[FEATURES].reset_index(drop=True).astype({"Year": float}), ["yield_avg_last4", "share_rainfed", "Year"],
                                        kind="both", subsample=150, random_state=RANDOM_STATE, ax=ax,
                                        ice_lines_kw={"color": "#97bc62", "alpha": 0.25}, pd_line_kw={"color": "#1e4d2b", "lw": 2.5})
fig.suptitle("Partial dependence (dark line) and individual conditional expectation (light lines) – EBM tuned on 1979–2015, all seasons")
savefig("09_pdp")

# %% [markdown]
# ### 10.4 Post-hoc cross-check: SHAP on the best tree benchmark
# If a completely different model (gradient-boosted trees) explained with a different method (SHAP) finds the same drivers, the conclusions are more robust.

# %%
tree_models = [m for m in res.index if m in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]]
TREE = min(tree_models, key=lambda m: cv_scores[m])
prep_t, est_t = fitted[TREE].named_steps["prep"], fitted[TREE].named_steps["model"]
feat_names = list(prep_t.get_feature_names_out())
Xt_test = pd.DataFrame(np.asarray(prep_t.transform(X_test)), columns=feat_names)
if TREE == "CatBoost":
    Xt_test[NUM_FEATURES] = Xt_test[NUM_FEATURES].astype(float)
sv = shap.TreeExplainer(est_t.model_ if TREE == "CatBoost" else est_t)(Xt_test)
num_idx = [feat_names.index(f) for f in NUM_FEATURES]
plt.figure()
shap.summary_plot(sv.values[:, num_idx], X_test[NUM_FEATURES].reset_index(drop=True), feature_names=NUM_FEATURES, show=False, plot_size=(8, 5))
plt.title(f"SHAP beeswarm – {TREE} (benchmark), test set")
savefig("12_shap_benchmark")
def grouped(values, names):
    df = pd.DataFrame(values, columns=names)
    grp = {n: ("District" if n.startswith("District_") else "Season" if n.startswith("Season_") else n) for n in names}
    return df.T.groupby(pd.Series(grp)).sum().T
shap_rank = grouped(sv.values, feat_names).abs().mean().rank(ascending=False)
ebm_main = pd.Series(ebm.term_importances(), index=ebm.term_names_)[FEATURES].rank(ascending=False)
pd.DataFrame({"EBM rank": ebm_main, f"SHAP rank ({TREE})": shap_rank}).sort_values("EBM rank").astype(int)

# %% [markdown]
# ### 10.5 Local explanation of one unseen case: Polonnaruwa, Yala 2023
# For deployment the EBM is **retrained on all available seasons (1979–2023, except the held-out sample)** with the tuned hyperparameters. EBM's local explanation is **exact**: intercept + the listed contributions = the prediction.

# %%
from sklearn.base import clone
final_model = clone(best_model)
final_model.fit(data[FEATURES], data[TARGET])
loc = final_model.explain_local(holdout[FEATURES]).data(0)
contrib = pd.Series(loc["scores"], index=loc["names"])
intercept = float(final_model.intercept_)
pred_h = float(final_model.predict(holdout[FEATURES])[0])
print(f"intercept {intercept:,.0f} + contributions {contrib.sum():+,.0f} = {intercept + contrib.sum():,.0f}  (model prediction {pred_h:,.0f})")
top = contrib.reindex(contrib.abs().sort_values(ascending=True).index).tail(12)
vals = {f: holdout[f].iloc[0] for f in FEATURES}
lab = [f"{n} = {int(vals[n])}" if n == "Year" else f"{n} = {vals[n]:.2f}" if n in vals and not isinstance(vals[n], str) and abs(vals[n]) < 10 else
       f"{n} = {vals[n]:,.0f}" if n in vals and not isinstance(vals[n], str) else f"{n} = {vals[n]}" if n in vals else n for n in top.index]
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.barh(lab, top.values, color=["#2e8b57" if v > 0 else "#c0392b" for v in top.values]); ax.axvline(0, color="k", lw=0.6)
ax.set_xlim(min(top.min(), 0) - 70, max(top.max(), 0) + 70)
for y, v in enumerate(top.values): ax.text(v + (8 if v >= 0 else -8), y, f"{v:+.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
ax.set_xlabel("contribution to the forecast (kg/ha)")
ax.set_title(f"EBM local explanation – Polonnaruwa Yala 2023\nintercept {intercept:,.0f} + contributions = {pred_h:,.0f} kg/ha (actual {holdout[TARGET].iloc[0]:,.0f})")
savefig("10_ebm_local")

# %%
cat_maps = {c: sorted(train[c].unique()) for c in CAT_FEATURES}
def encode(df):
    out = df[FEATURES].copy()
    for c in CAT_FEATURES:
        out[c] = out[c].map({v: i for i, v in enumerate(cat_maps[c])})
    return out.astype(float)
def predict_from_encoded(arr):
    df = pd.DataFrame(arr, columns=FEATURES)
    for c in CAT_FEATURES:
        df[c] = df[c].round().astype(int).clip(0, len(cat_maps[c]) - 1).map(dict(enumerate(cat_maps[c])))
    return final_model.predict(df[FEATURES])
enc_train = encode(train)
lime_exp = LimeTabularExplainer(
    enc_train.fillna(enc_train.median()).values, feature_names=FEATURES, mode="regression",
    categorical_features=[FEATURES.index(c) for c in CAT_FEATURES],
    categorical_names={FEATURES.index(c): cat_maps[c] for c in CAT_FEATURES},
    discretize_continuous=True, random_state=RANDOM_STATE)
x_enc = encode(holdout).fillna(enc_train.median()).values[0]
le = lime_exp.explain_instance(x_enc, predict_from_encoded, num_features=8, num_samples=2000)
print(f"LIME local surrogate R² = {le.score:.2f} | surrogate prediction = {le.local_pred[0]:.0f} kg/ha")
fig = le.as_pyplot_figure(); fig.set_size_inches(8, 4.2)
plt.title("LIME – Polonnaruwa Yala 2023 (final EBM)")
savefig("11_lime")
pd.DataFrame(le.as_list(), columns=["condition", "effect (kg/ha)"]).round(1)

# %% [markdown]
# ### 10.6 Interpretation: what the model learned and whether it matches domain knowledge
# **Which features are most influential?** The EBM's largest terms are last year's same-season yield (≈162 kg/ha average contribution), Year (≈158), the rain-fed share of the sown area (≈138), District (≈113), the 4-season average yield (≈104), the major-irrigation share (≈91) and Season (≈73). Pairwise interactions are all small (≤ 20 kg/ha), so district yield is almost an *additive* combination of history, irrigation and trend. That is exactly why a GAM can match boosted trees here.
#
# **What has the model learned? (shape functions)**
# * **Yield history:** contributions rise almost linearly with the previous same-season yield and the 4-season average. Good districts tend to stay good, which is yield persistence.
# * **Rain-fed share:** a clear *downward* curve, from about +200 kg/ha when nothing is rain-fed to about −150 to −250 kg/ha when most of the area depends on rain alone. That is a gap of roughly 400 kg/ha caused purely by water security.
# * **Major-irrigation share:** flat up to about 60 %, then a sharp rise of up to +250 kg/ha for districts dominated by large irrigation schemes (assured water from reservoirs).
# * **Year:** the long-term technology trend appears as steps, with low levels in the 1980s–90s and successive rises through the 2000s and 2010s (newer rice varieties, the expanded fertilizer subsidy, mechanisation).
# * **District:** once history and irrigation are known, the district terms are *adjustments*. They are positive for dry-zone irrigated districts such as Hambantota, Polonnaruwa and Ampara, and negative for Jaffna and some wet-zone districts.
# * **Season:** Yala is predicted lower than Maha, as expected for the drier, tank-dependent season.
#
# **Do the methods agree?** The PDP/ICE curves have the same direction and shape as the EBM curves. SHAP on LightGBM, a completely different model explained post hoc, puts the same features at the top: yield history, Year, rain-fed share and major share. The only notable disagreement is **District**, ranked 4th by EBM but 12th by LightGBM-SHAP. LightGBM spreads district identity over 25 one-hot columns and mostly lets the correlated yield-history features carry it, while EBM gives District its own term. The location effect is real in both models; they assign the credit to different but correlated features.
#
# **Local explanation (Polonnaruwa, Yala 2023):** 3,519 kg/ha (average) + recent Year (+226) + no rain-fed area (+223) + high 4-season average (+207) + 88 % major irrigation (+174) + Year × rain-fed (+171) + last year's same-season yield (+158) + district (+144) − Yala season (−78) − small terms = **4,731 kg/ha** (actual 4,918). LIME's surrogate (local R² ≈ 0.5) lists the same drivers with the same signs. Because the EBM is additive, its breakdown is exact rather than approximate.
#
# **Does it align with domain knowledge?** Yes. Water security (major irrigation up, rain-fed down), persistence of district performance, the long-run technology trend and a lower Yala season are all well-established features of Sri Lankan paddy production. No suspicious feature (for example, a leakage variable) dominates. These are the checks that let an agricultural planner *trust* the forecast (lecture: transparency, trust calibration, debugging).

# %% [markdown]
# ## 11. Bias and fairness check
# Yield forecasts feed decisions (fertilizer, purchasing, relief) that affect farmers by district. A model that is systematically worse – or systematically optimistic – for some districts could disadvantage them.
# We therefore break the EBM's test errors down by **district** and by **district group**, using the data itself to form groups:
# * **Mostly irrigated:** > 50 % of sown area under major irrigation on average
# * **Mostly rain-fed:** > 50 % rain-fed
# * **Mixed:** the rest
# * **Northern, conflict-affected:** Jaffna, Kilinochchi, Mannar, Mullativu, Vavuniya (seasons missing during the civil-war years → shorter, gappier history)

# %%
NORTH = ["Jaffna", "Kilinochchi", "Mannar", "Mullativu", "Vavuniya"]
shares = model_df.groupby("District")[["share_major", "share_rainfed"]].mean()
def group_of(d):
    if d in NORTH: return "Northern, conflict-affected"
    if shares.loc[d, "share_major"] > 0.5: return "Mostly irrigated"
    if shares.loc[d, "share_rainfed"] > 0.5: return "Mostly rain-fed"
    return "Mixed"
ev = test.assign(pred=test_pred, err=test_pred - y_test, ape=np.abs(test_pred - y_test) / y_test * 100)
ev["group"] = ev["District"].map(group_of)
ev_n = ev[ev.Year != 2022]
grp = ev_n.groupby("group").agg(districts=("District", "nunique"), MAPE=("ape", "mean"), bias_kg_ha=("err", "mean"),
                                within_10pct=("ape", lambda a: (a <= 10).mean() * 100)).round(1)
print("Test seasons excluding 2022 (bias > 0 means the model over-predicts):")
grp

# %%
dist = ev_n.groupby("District").agg(MAPE=("ape", "mean"), bias=("err", "mean"), group=("group", "first")).sort_values("MAPE")
gcol = {"Mostly irrigated": "#1e4d2b", "Mixed": "#97bc62", "Mostly rain-fed": "#d4a017", "Northern, conflict-affected": "#c0392b"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
axes[0].barh(dist.index, dist["MAPE"], color=dist["group"].map(gcol)); axes[0].set_xlabel("MAPE % (test, excl. 2022)")
axes[1].barh(dist.index, dist["bias"], color=dist["group"].map(gcol)); axes[1].axvline(0, color="k", lw=0.6)
axes[1].set_xlabel("mean error kg/ha (+ = over-predicts)")
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color=c, label=k) for k, c in gcol.items()], loc="lower right", fontsize=8)
axes[0].tick_params(axis="y", labelsize=8)
fig.suptitle("EBM error and bias by district (test seasons, excluding the 2022 shock)")
savefig("13_fairness")

# %% [markdown]
# **Findings (test seasons, excluding 2022)**
# * **Mostly irrigated districts** get the most reliable forecasts: MAPE 6.6 %, and 82.5 % of forecasts within ±10 %.
# * **Mostly rain-fed districts** have MAPE 8.8 % and are **over-predicted** on average (+61 kg/ha), because their yield depends on rainfall, which is not a feature.
# * **Northern, conflict-affected districts** are the worst served: **MAPE 12.4 %**, only 54 % within ±10 %, and they are **under-predicted** by 131 kg/ha on average. Their history has gaps (no cultivation recorded during the war years) and post-war recovery is a trend the model has seen little of.
#
# **Why it matters:** if forecasts were used to allocate fertilizer, credit or procurement quotas, the Northern districts' production would be systematically underestimated. Rain-fed areas, often the poorer and more climate-exposed farming areas, would be over-estimated, so shortfalls there could be under-prepared. **Mitigations:** report district-specific error bands alongside each forecast (the app shows the typical error range), keep a human planner in the loop, add rainfall and reservoir data, and monitor per-district error after each season.
# The dataset contains no information on individual farmers (no gender, ethnicity, income or personal identifiers), so the model makes no decisions about individuals; the fairness concern is *between regions*.

# %% [markdown]
# ## 12. Save the model and run inference on the unseen sample
# The final (retrained) EBM is saved with `pickle` together with the metadata the front end needs.

# %%
history = feat_all[["District", "Season", "Year", "t", "Major Sown (Ha)", "Minor Sown (Ha)", "Rainfed Sown (Ha)",
                    "Total Extent (Ha)", "Net Harvest (Ha)", TARGET]]
bundle = {
    "model": final_model, "model_name": "Explainable Boosting Machine (EBM)", "features": FEATURES,
    "districts": sorted(model_df["District"].unique()), "seasons": ["MAHA", "YALA"],
    "test_rmse": float(res.loc[BEST, "RMSE"]), "test_mae": float(res.loc[BEST, "MAE"]),
    "trained_on": "1979-2023 (excluding Polonnaruwa Yala 2023 hold-out)",
}
with open(MODELS / "paddy_yield_model.pkl", "wb") as f:
    pickle.dump(bundle, f)
history.to_csv(MODELS / "district_history.csv", index=False)
res.round(3).to_csv(MODELS / "model_comparison.csv")
grp.to_csv(MODELS / "fairness_by_group.csv")
json.dump(best_params, open(MODELS / "best_hyperparameters.json", "w"), indent=2, default=str)
print("Saved:", MODELS / "paddy_yield_model.pkl", f"({(MODELS / 'paddy_yield_model.pkl').stat().st_size/1e6:.1f} MB)")

# %%
# --- Inference: load the pickle as a separate application would, and predict the unseen sample ---
with open(MODELS / "paddy_yield_model.pkl", "rb") as f:
    loaded = pickle.load(f)
row = holdout.iloc[0]
pred = loaded["model"].predict(holdout[loaded["features"]])[0]
print(f"Input: {row['District']} | {row['Season']} {row['Year']} | sown {row['Total Extent (Ha)']:,.0f} ha "
      f"(major {row['share_major']:.0%}, minor {row['share_minor']:.0%}, rain-fed {row['share_rainfed']:.0%})")
print(f"History: last season {row['yield_prev_season']:,.0f} | same season last year {row['yield_same_season_ly']:,.0f} | avg last 4 {row['yield_avg_last4']:,.0f} kg/ha")
print(f"Predicted yield : {pred:,.0f} kg/ha")
print(f"Actual yield    : {row[TARGET]:,.0f} kg/ha   (error {pred-row[TARGET]:+,.0f} kg/ha, {100*(pred-row[TARGET])/row[TARGET]:+.1f} %)")
print(f"Implied production if the sown area is harvested at last season's ratio: "
      f"{pred/1000 * row['Total Extent (Ha)'] * row['harvest_ratio_prev']:,.0f} t  (actual {row['Production (MT)']:,.0f} t)")

# %% [markdown]
# ## 13. Critical discussion
# **Limitations of the model**
# * **No extrapolation:** EBM's shape functions, like tree ensembles, are piecewise constant. Record yields beyond the training range are under-predicted (2019–2021 in the test set), so the model must be retrained every season. The deployed model is retrained on data up to 2023.
# * **Shocks are invisible:** the 2021/22 fertilizer ban, floods and droughts after sowing are not in the features. 2022 errors were 2.5–4 times larger than in normal seasons.
# * **District averages only:** the model cannot say anything about individual fields, varieties or farmers, and it gives a point forecast plus an error band from the test MAE, not a full probabilistic forecast.
# * **Small data:** about 2,200 district-seasons. Validation scores vary between folds (standard deviation ≈ 40–60 kg/ha), so differences of 1–2 % between models are not meaningful.
#
# **Data-quality issues found and handled**
# * 100 duplicated rows, 4 overlapping regional aggregates, a unit label error (kg/ha shown as "MT/Ha"), numbers stored as text and "-" placeholders.
# * Gaps from the civil-war years in Northern districts, handled as missing lags and imputed (median for the benchmark models, EBM's native missing-value handling).
# * The DCS figures are official *estimates*, compiled from field surveys and administrative reporting, so they carry their own measurement error. Some district boundaries also changed over 45 years (for example, Kilinochchi district was created from Jaffna in 1984), so early figures may not be fully comparable.
#
# **Risks of bias or unfairness:** see section 11. Forecasts are least accurate for Northern conflict-affected districts (under-predicted) and rain-fed districts (over-predicted).
#
# **Potential real-world impact and ethical considerations**
# * **Benefit:** an early, explainable district yield forecast can help the Department of Agriculture, the Paddy Marketing Board and the Ministry plan fertilizer distribution, storage, purchasing and imports before harvest, and can flag districts likely to fall short.
# * **Human oversight:** the model must support, not replace, planners (lecture: accountability, human in the loop). Its exact explanations let officers see *why* a district is predicted low and challenge it with local knowledge.
# * **Over-reliance risk:** a confident-looking number could hide a shock that the features cannot see, as in 2022. The app therefore shows an error range and states its limitations.
# * **Data ethics:** the data are aggregated, publicly released official statistics with no personal or sensitive information, so no consent issues arise. The code and data are shared openly for reproducibility.

# %% [markdown]
# ## 14. Conclusions
# * A local Sri Lankan problem, forecasting district paddy yield at sowing time, was solved with DCS statistics from 1979–2023 after careful cleaning (duplicates, overlapping regions, unit error, gaps) and leakage-free feature design.
# * A **new algorithm not covered in the lectures, the Explainable Boosting Machine**, was tuned on forward-chaining validation folds. On unseen 2016–2023 seasons it matched the black-box ensembles (test RMSE 623 kg/ha vs 608–633), beat linear regression and the moving average, and was 16 % better than persistence (22 % outside the 2022 shock).
# * Its **intrinsic explanations**, cross-checked with PDP, SHAP and LIME, show a model driven by yield history, water security (irrigation versus rain-fed) and the technology trend, consistent with domain knowledge.
# * Critical analysis exposed the model's limits: no extrapolation, blindness to shocks, and lower accuracy for Northern conflict-affected and rain-fed districts. Those districts should get extra human scrutiny.
# * The model is deployed in a Streamlit app that shows each prediction with its exact explanation.
#
# **Future work:** add Department of Meteorology rainfall and reservoir storage at sowing time, plus fertilizer issue data; produce prediction intervals; monitor per-district error and data drift every season (MLOps) and retrain automatically.

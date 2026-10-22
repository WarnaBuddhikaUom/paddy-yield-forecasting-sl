# 🌾 Forecasting District-Level Paddy Yield in Sri Lanka at Sowing Time

**Applied Machine Learning: final assignment**

This project predicts the **average paddy yield (kg/ha) of each Sri Lankan district for the coming Maha or Yala
season**, using only information that is available when the season is sown: area sown by irrigation type, the
district's recent yield history and the long-term trend. A forecast like this supports fertilizer allocation,
Paddy Marketing Board purchasing and storage, and import planning, all of which have to be decided before
the harvest.

| | |
|---|---|
| **Data** | District-wise paddy statistics, 1979–2023, Maha & Yala (Department of Census and Statistics, Sri Lanka) |
| **Task** | Supervised regression on tabular data (2,174 district-seasons after cleaning) |
| **Models** | Persistence, moving average, linear regression (existing approaches) vs Random Forest, XGBoost, LightGBM, CatBoost, Stacking |
| **Validation** | Chronological split (train 1979–2015, test 2016–2023) and forward-chaining `TimeSeriesSplit` for tuning |
| **Explainability** | SHAP (global + local waterfall) and LIME (local surrogate) |
| **Front end** | Streamlit app with per-prediction SHAP explanation, generated with GenAI prompts ([docs/GENAI_FRONTEND_PROMPTS.md](docs/GENAI_FRONTEND_PROMPTS.md)) |
| **Deployment (bonus)** | Dockerfile + GitHub Actions CI (tests the pickle, builds the image) |

## Results (test seasons 2016–2023)

| Model | CV RMSE | Test RMSE | Test MAE | MAPE % | R² | Within ±10 % | RMSE excl. 2022 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **LightGBM (selected)** | **425** | 626 | 451 | 12.9 | 0.51 | **60.4** | 455 |
| Stacking (RF+XGB+LGBM+CatBoost → Ridge) | 430 | **608** | **443** | **12.5** | **0.54** | 58.1 | **451** |
| Random Forest | 430 | 622 | 463 | 12.9 | 0.52 | 55.8 | 485 |
| XGBoost | 431 | 633 | 457 | 13.0 | 0.50 | 59.1 | 465 |
| CatBoost | 437 | 608 | 450 | 12.7 | 0.54 | 58.3 | 465 |
| *Moving average (last 4 seasons)* | 442 | 685 | 496 | 14.2 | 0.42 | 54.7 | 517 |
| *Persistence (same season last year)* | 477 | 742 | 554 | 15.5 | 0.32 | 50.4 | 583 |
| *Linear regression* | 513 | 639 | 467 | 13.6 | 0.49 | 57.8 | 461 |

RMSE/MAE are in kg/ha. The model was **selected on CV RMSE** (training period only); test results are reported, not tuned on.

**Key findings**

* All tuned ensembles beat the existing approaches in cross-validation. LightGBM has **17 % lower RMSE than linear regression** and 11 % lower than persistence.
* On 2016–2023, LightGBM is **16 % better than persistence** (22 % better outside the 2022 shock).
* Tree ensembles **cannot extrapolate** the record yields of 2019–2021, which is why linear regression gets close on the test period. The deployed model is retrained on data up to 2023 for this reason.
* The **2021/22 chemical fertilizer ban** caused yields to collapse in 2022. No model could foresee it from the available features, and its errors that year are 2.5–4 times larger than in normal seasons (MAE about 900–1,400 vs about 350 kg/ha); the baselines fail too.
* SHAP shows the model's drivers: recent yield history, the rain-fed share of the sown area (which lowers yield), the major irrigation share (which raises it), and the long-term trend. These match Sri Lankan agronomy.

## Repository structure

```
├── data/Agriculture_Statistics_Paddy.csv     raw DCS data
├── notebooks/paddy_yield_forecasting.ipynb   full analysis (executed, with outputs)
├── src/features.py                           cleaning + leakage-free feature engineering
├── src/cat_wrapper.py                        scikit-learn wrapper for CatBoost
├── models/paddy_yield_model.pkl              final pickled pipeline + metadata
├── models/district_history.csv               history used to pre-fill the app
├── models/model_comparison.csv               metrics table
├── models/best_hyperparameters.json          tuned hyperparameters
├── app/app.py                                Streamlit front end
├── figures/                                  all plots used in the report and slides
├── tests/test_model.py                       smoke tests (CI)
├── docs/GENAI_FRONTEND_PROMPTS.md            prompts used to generate the front end
├── Dockerfile, requirements*.txt, .github/workflows/ci.yml
```

## How to run

```bash
pip install -r requirements.txt

# 1. Reproduce the analysis (about 6–8 minutes on 2 CPU cores)
jupyter notebook notebooks/paddy_yield_forecasting.ipynb

# 2. Launch the web app
streamlit run app/app.py          # opens http://localhost:8501

# 3. Tests
pytest -q tests
```

**Google Colab:** upload the repository folder (or `git clone` it), run `!pip install -r requirements.txt`, then open the notebook.

> The pickle file depends on library versions. If you use different scikit-learn or LightGBM versions,
> re-run the notebook once to regenerate `models/paddy_yield_model.pkl`.

## Deployment (bonus)

* **Docker:** `docker build -t paddy-yield-app . && docker run -p 8501:8501 paddy-yield-app`
* **Streamlit Community Cloud (free):** push this repo to GitHub, go to share.streamlit.io, choose the repo and set the main file to `app/app.py`.
* **Azure Container Apps (as in the MLOps workshop):** push the image to Azure Container Registry and create a Container App with target port 8501.
* **CI:** every push to `main` runs the smoke tests on the pickled model and builds the Docker image.

## Limitations and future work

* Only sowing-time agronomic history is used. Adding **rainfall (Department of Meteorology)**, **reservoir/tank storage** and **fertilizer issue data** should capture droughts and policy shocks.
* Tree models cannot extrapolate trends, so the model needs regular retraining and **data-drift monitoring** (the MLOps workshop's monitoring and retraining loop) for each new season.
* Some Northern districts have gaps in the war years. Those gaps are treated as missing lags and imputed with the median.

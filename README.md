# 🌾 Forecasting District-Level Paddy Yield in Sri Lanka with an Explainable Boosting Machine

**MSc in AI – Machine Learning assignment** · Buddhika K H W (269086V) · [GitHub repository](https://github.com/WarnaBuddhikaUom/paddy-yield-forecasting-sl)

This project predicts the **average paddy yield (kg/ha) of each Sri Lankan district for the coming Maha or Yala
season**, using only information available when the season is sown: area sown by irrigation type, the district's
recent yield history and the long-term trend. A forecast like this supports fertilizer allocation, Paddy Marketing
Board purchasing and storage, and import planning, all of which are decided before the harvest.

| | |
|---|---|
| **Data** | District-wise paddy statistics, 1979–2023, Maha & Yala (Department of Census and Statistics, Sri Lanka). Aggregated public data, with no personal information |
| **Task** | Supervised regression on tabular data (2,174 district-seasons after cleaning) |
| **New algorithm** | **Explainable Boosting Machine (EBM)**, a glass-box generalised additive model with pairwise interactions (InterpretML), not covered in the lectures |
| **Compared with** | Persistence, moving average, linear regression (existing approaches), plus Random Forest, XGBoost, LightGBM, CatBoost and Stacking (taught in the lectures) |
| **Evaluation** | Train 1979–2015 with 4 forward-chaining validation folds; test on unseen 2016–2023 |
| **Explainability** | EBM global and local explanations (intrinsic), Partial Dependence / ICE, SHAP on LightGBM (cross-check), LIME |
| **Critical analysis** | Limitations, data quality, bias/fairness by district group, ethics |
| **Front end (bonus)** | Streamlit app with the EBM's exact per-prediction explanation, generated with GenAI prompts ([docs/GENAI_FRONTEND_PROMPTS.md](docs/GENAI_FRONTEND_PROMPTS.md)); Dockerfile + GitHub Actions CI |

## Results (test seasons 2016–2023)

| Model | Validation RMSE | Test RMSE | Test MAE | MAPE % | R² | Within ±10 % | RMSE excl. 2022 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Stacking | 430 | 608 | 443 | 12.5 | 0.54 | 58 | 451 |
| CatBoost | 437 | 608 | 450 | 12.7 | 0.54 | 58 | 465 |
| Random Forest | 430 | 622 | 463 | 12.9 | 0.52 | 56 | 485 |
| **EBM – new algorithm (deployed)** | **441** | **623** | **449** | **12.9** | **0.52** | **60** | **457** |
| LightGBM | 425 | 626 | 451 | 12.9 | 0.51 | 60 | 455 |
| XGBoost | 431 | 633 | 457 | 13.0 | 0.50 | 59 | 465 |
| *Linear regression* | 513 | 639 | 467 | 13.6 | 0.49 | 58 | 461 |
| *Moving average (last 4)* | 442 | 685 | 496 | 14.2 | 0.42 | 55 | 517 |
| *Persistence (same season LY)* | 477 | 742 | 554 | 15.5 | 0.32 | 50 | 583 |

RMSE and MAE are in kg/ha. Validation scores come from the forward-chaining folds (1979–2015); test seasons were never used for tuning.

**Key findings**

* The **EBM matches black-box accuracy on unseen years**: test RMSE 623 kg/ha against 608–633 for the tree ensembles. It is **16 % better than persistence** (22 % excluding the 2022 shock), and 60 % of its forecasts are within ±10 %.
* On validation it is about 4 % behind the best ensemble (441 vs 425), so its full transparency costs very little.
* Its shape functions show the drivers: past yield (higher raises the forecast), water security (a higher rain-fed share costs up to ~400 kg/ha, a major-irrigation share adds up to +250), the technology trend, and a lower Yala season. PDP and SHAP agree, and this matches Sri Lankan agronomy.
* **Fairness:** forecasts are most accurate for irrigated districts (MAPE 6.6 %). Northern, conflict-affected districts have about twice the error (12.4 %) and are under-predicted.
* No model could foresee the **2021/22 fertilizer ban**: errors in 2022 are 2.5–4 times normal.

## Repository structure

```
├── data/Agriculture_Statistics_Paddy.csv     raw DCS data
├── notebooks/paddy_yield_forecasting.ipynb   full analysis: preprocessing, training, evaluation, XAI (executed)
├── src/features.py                           cleaning + leakage-free feature engineering
├── src/cat_wrapper.py                        scikit-learn wrapper for CatBoost (benchmark)
├── models/paddy_yield_model.pkl              final EBM + metadata (used by the app)
├── models/district_history.csv               history used to pre-fill the app
├── models/model_comparison.csv               metrics table
├── models/fairness_by_group.csv              error by district group
├── models/best_hyperparameters.json          tuned hyperparameters
├── app/app.py                                Streamlit front end
├── reports/                                  report (PDF + Word) and slides
├── figures/                                  all plots
├── tests/test_model.py                       smoke tests (CI)
├── docs/                                     GenAI prompts, video script
├── Dockerfile, requirements*.txt, .github/workflows/ci.yml
```

## How to run

```bash
pip install -r requirements.txt

# 1. Reproduce the analysis (about 15–20 minutes on 2 CPU cores; EBM tuning is the slow part)
jupyter notebook notebooks/paddy_yield_forecasting.ipynb

# 2. Launch the web app
streamlit run app/app.py          # opens http://localhost:8501

# 3. Tests
pytest -q tests
```

**Google Colab:** upload the repository folder (or `git clone` it), run `!pip install -r requirements.txt`, then open the notebook.

> The pickle file depends on library versions. If you use different scikit-learn or interpret versions,
> re-run the notebook once to regenerate `models/paddy_yield_model.pkl`.

## Dataset

`data/Agriculture_Statistics_Paddy.csv` contains the Department of Census and Statistics (Sri Lanka) district-wise paddy
statistics (sown and harvested extent by irrigation type, yields, production) for Maha and Yala seasons, 1979–2023.
These are public, aggregated official statistics with no personal data.

## Deployment (bonus)

* **Docker:** `docker build -t paddy-yield-app . && docker run -p 8501:8501 paddy-yield-app`
* **Streamlit Community Cloud (free):** push this repo to GitHub, go to share.streamlit.io, choose the repo and set the main file to `app/app.py`.
* **CI:** every push to `main` runs the smoke tests on the pickled model and builds the Docker image.

## Limitations and future work

* Shocks after sowing (policy, floods, droughts) are not in the features. Adding **rainfall**, **reservoir/tank storage** and **fertilizer issue data** is the next step.
* EBM's curves, like tree ensembles, cannot extrapolate beyond the training range, so the model needs seasonal retraining and **drift monitoring**.
* Northern districts have gaps from the war years, which leads to lower accuracy there. Their forecasts need extra human review.

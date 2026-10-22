# GenAI prompts used to build the front end

The assignment asks for the front end to be generated with a GenAI tool (Gemini / ChatGPT / Claude).
These are the prompts used, in order. The generated code was then reviewed, tested against the
notebook's predictions (same 4,225 kg/ha for the held-out sample), and adjusted.

---

## Prompt 1: the base app

> I have trained a machine learning regression model in Python that predicts the **average paddy
> yield (kg/ha) of a Sri Lankan district for a season** (Maha or Yala). The model is a scikit-learn
> `Pipeline` (ColumnTransformer + LightGBM) saved with `pickle` in `models/paddy_yield_model.pkl`.
> The pickle holds a dictionary with keys `model`, `model_name`, `features`, `districts`, `seasons`,
> `test_rmse`, `test_mae`.
>
> The model expects a pandas DataFrame with these columns:
> `District` (one of 25 district names), `Season` ("MAHA"/"YALA"), `Year`, `log_total_sown`,
> `share_major`, `share_minor`, `share_rainfed`, `sown_change_pct`, `yield_prev_season`,
> `yield_same_season_ly`, `yield_avg_last4`, `harvest_ratio_prev`.
> A helper function `make_input_row(...)` in `src/features.py` already builds this row from raw inputs.
>
> Write a **Streamlit** app (`app/app.py`) that:
> 1. loads the pickle once (cached),
> 2. has dropdowns for District and Season and a number box for Year,
> 3. has number boxes for the hectares sown under major irrigation, minor irrigation and rain-fed,
> 4. has number boxes for previous-season yield, same-season-last-year yield, average of the last
>    4 seasons, last season's harvested/sown ratio, and area sown in the same season last year,
> 5. on a "Predict yield" button shows the predicted yield in kg/ha and t/ha, an error range
>    (± test MAE) and an indicative production (yield × sown area × harvest ratio).
> Use clear labels a district agriculture officer would understand.

## Prompt 2: pre-fill from history

> I also have `models/district_history.csv` with columns District, Season, Year, t (= 2·Year + 1 for
> Yala), Major/Minor/Rainfed Sown (Ha), Total Extent (Ha), Net Harvest (Ha), yield_kg_ha.
> When the user selects a district, season and year, pre-fill all the history boxes from this file
> (previous season = t-1, same season last year = t-2, average of t-1..t-4, harvest ratio of t-1),
> but let the user overwrite them. If the selected season exists in the file, also show the recorded
> yield next to the prediction so the user can compare.

## Prompt 3: explain each prediction

> Add a "Why this forecast?" horizontal bar chart under the prediction using SHAP `TreeExplainer`
> on the LightGBM step of the pipeline. Transform the input with the pipeline's `prep` step first,
> sum the one-hot columns that start with `District_` and `Season_` back into single "District" and
> "Season" bars, colour positive contributions green and negative red, and never let an explanation
> error stop the prediction from showing.

## Prompt 4: about tab and deployment

> Add a second tab "About the model" that shows the model comparison CSV and two saved figures, and
> lists the model's limitations. Then write a Dockerfile (python:3.13-slim, needs libgomp1 for
> LightGBM) that runs the app on port 8501, and a GitHub Actions workflow that installs requirements,
> runs pytest smoke tests on the pickle, and builds the Docker image.

---

### What was checked or changed by hand after generation

- Verified that the app's prediction for Polonnaruwa Yala 2023 matches the notebook (4,225 kg/ha).
- Input widgets get keys that include district/season/year, so the pre-filled values refresh when the selection changes.
- Added `sown_change_pct` clipping through `make_input_row` so app inputs are processed exactly like training data.
- Added the CatBoost wrapper import path (`src/`) so the pickle loads outside the notebook.

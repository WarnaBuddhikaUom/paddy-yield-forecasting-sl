# Re-recording script (EBM version) – Buddhika K H W

The model changed from LightGBM to the **Explainable Boosting Machine**, so **8 short parts** need new narration.
The other 6 parts reuse your existing recording, so you don't need to say them again.

Play **Rerecord_teleprompter.mp4** (2:36). It contains only the new parts, with the words on screen.
Record yourself reading along, at 0.5× speed like last time if you prefer. Then send me the recording, and I'll splice it into the full video (about 4:30).

| # | On screen | Status | Say |
|---|---|---|---|
| 1 | Title | ✅ already recorded | _(reuse your existing take)_ |
| 2 | Problem | ✅ already recorded | _(reuse your existing take)_ |
| 3 | Data cleaning | ✅ already recorded | _(reuse your existing take)_ |
| 4 | Notebook: data audit & trend | ✅ already recorded | _(reuse your existing take)_ |
| 5 | EDA & features | ✅ already recorded | _(reuse your existing take)_ |
| 6 | New algorithm: EBM | 🎙 **record** | For the main model I chose an algorithm not covered in the lectures: the Explainable Boosting Machine. It learns one curve per feature, plus a few pairwise interactions, using boosting. So the curves are the model, and the explanations are exact, unlike random forests or XGBoost, which are black boxes. |
| 7 | Notebook: EBM tuning & comparison | 🎙 **record** | In the notebook, the EBM was tuned on the time-series validation folds only, and then compared with the ensembles taught in the lectures. |
| 8 | Results | 🎙 **record** | On the unseen years the EBM's error is 623 kilograms per hectare. That matches the black-box models and is 16 percent better than persistence. Sixty percent of forecasts are within ten percent, so transparency costs very little. |
| 9 | Limitations & fairness | 🎙 **record** | No model could foresee the 2022 fertilizer ban, and the curves cannot extrapolate record yields, so the model is retrained every season. Accuracy is uneven: Northern, conflict-affected districts have double the error and are under-predicted, so they need extra human review. The data are public and contain no personal information. |
| 10 | What the EBM learned | 🎙 **record** | These curves are the model itself. Higher past yields raise the forecast, the rain-fed share lowers it by up to about 400 kilograms, and major irrigation raises it. Partial dependence and SHAP on LightGBM agree, and this matches agronomy. |
| 11 | Held-out sample | 🎙 **record** | For the held-out case, Polonnaruwa Yala 2023, the pickled model predicts 4,731 kilograms per hectare against an actual 4,918, under four percent error. The average plus each contribution equals the prediction exactly. |
| 12 | Front end | ✅ already recorded | _(reuse your existing take)_ |
| 13 | Live app demo | 🎙 **record** | Here is the live app. I click predict: 4,731 kilograms per hectare, with an error range, and the chart shows the EBM's exact explanation. Now Kalutara in Maha, a mostly rain-fed district. The forecast drops to about 2,850, close to the actual 2,735. |
| 14 | Conclusion | 🎙 **record** | To conclude: the Explainable Boosting Machine is as accurate as black-box models, sixteen to twenty-two percent better than persistence, with exact explanations that match agronomy. Thank you. |
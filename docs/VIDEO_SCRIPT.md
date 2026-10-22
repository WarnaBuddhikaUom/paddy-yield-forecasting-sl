# Narration script for the 5-minute video (Buddhika K H W)

Total length **4:37**. Read each block while the matching part of the video plays.
About 140 words per minute is a calm pace. Every block has a second or two of spare time.

| Start | End | On screen | Say |
|---|---|---|---|
| 00:00 | 00:11 | Title | Hello, I am Buddhika K H W. In this video I present my Applied Machine Learning project: forecasting district-level paddy yield in Sri Lanka at sowing time. |
| 00:11 | 00:35 | Problem | Rice is our staple food. Fertilizer, Paddy Marketing Board purchasing, storage and imports are all planned months before the harvest, and the 2022 fertilizer ban showed how costly a surprise can be. So my goal is to predict each district's average yield, in kilograms per hectare, for the coming Maha or Yala season, using only information available at sowing. |
| 00:35 | 00:54 | Data cleaning | I used Department of Census and Statistics paddy data from 1979 to 2023. The audit found four problems: one hundred duplicated rows, four regions that overlap real districts, a unit error where yields labelled tonnes are really kilograms per hectare, and numbers stored as text. |
| 00:54 | 01:16 | Notebook running: data audit & trend | Here is the notebook running. The audit cell counts each problem, and after cleaning we have twenty-five districts over forty-five years. The trend chart shows yields roughly doubled since 1980, then collapsed in 2022 when chemical fertilizer was banned. |
| 01:16 | 01:34 | EDA & features | Irrigation matters: the rain-fed share is strongly negatively correlated with yield. I only used features known at sowing time: district, season, year, sown area by irrigation type, and past yields. Harvested area and production are excluded, because they would leak the answer. |
| 01:34 | 01:56 | Method | I split the data by time, training on 1979 to 2015 and testing on 2016 to 2023, so the model never learns from the future. Hyperparameters were designed, not left at defaults. For example, LightGBM uses few leaves and large minimum leaf sizes, because there are fewer than two thousand training rows. |
| 01:56 | 02:12 | Notebook running: tuning & comparison table | In the notebook you can see the tuning results and the full comparison table. LightGBM had the lowest cross-validation error, so it was selected, using validation scores only, never the test set. |
| 02:12 | 02:34 | Results | Grey bars are existing approaches, green bars are the ensembles. LightGBM's cross-validation error is seventeen percent lower than linear regression. On unseen seasons it is sixteen percent better than persistence, and sixty percent of forecasts are within ten percent of the actual yield. |
| 02:34 | 02:53 | Limitations | There are two honest limitations. Tree models cannot predict above the yields they have seen, so record years are under-predicted, which is why the final model is retrained up to 2023. And no model could foresee the 2022 fertilizer shock. |
| 02:53 | 03:12 | SHAP (global) | SHAP explains the model. Recent yield history is the main driver. A higher rain-fed share lowers the forecast, while major irrigation raises it, and Year captures the technology trend. The district name adds almost nothing, so the model learns real agronomic drivers. |
| 03:12 | 03:34 | Held-out sample: SHAP & LIME | For inference I held out Polonnaruwa, Yala 2023. The pickled model predicts 4,225 kilograms per hectare against an actual 4,918. The SHAP waterfall shows why: the year 2023 pulls the forecast down, because the model learned the 2022 collapse. LIME agrees on the main drivers. |
| 03:34 | 03:42 | Front end | The front end was generated with GenAI prompts, and for the bonus there is a Dockerfile and a CI pipeline. |
| 03:42 | 04:22 | Live app demo | Here is the live app. District history is pre-filled from the statistics. I click predict: 4,225 kilograms per hectare, with an error range and production estimate, and the SHAP chart explains it. Now Kalutara in Maha, a mostly rain-fed district. The forecast drops to about 2,900, close to the actual 2,735, and the rain-fed share now pulls it down. |
| 04:22 | 04:37 | Conclusion | To conclude: careful cleaning, leakage-free features and tuned ensembles give forecasts sixteen to twenty-two percent better than the persistence baseline, with explanations that make sense. Thank you. |

## How to record

1. Play **Paddy_Yield_Presentation_teleprompter.mp4**. The captions show exactly what to say and when.
2. Record your voice with your phone or PC at the same time, reading the captions aloud. One continuous take is easiest.
3. Send the recording back, and I will merge it with the clean video (no captions), fix the timing, and level the volume. Or merge it yourself in any editor (CapCut, Clipchamp, iMovie): put the clean video on the track and your audio underneath.

Tip: start talking when the title slide appears, about half a second in.
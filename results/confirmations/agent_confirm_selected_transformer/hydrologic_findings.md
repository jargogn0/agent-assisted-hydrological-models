# Hydrologic Diagnostic Findings

## Hydrograph Skill
- `transformer`: median test KGE=0.719, NSE=0.615, logNSE=0.682, PBIAS=-2.6%.
- `random_forest`: median test KGE=0.607, NSE=0.555, logNSE=0.611, PBIAS=1.9%.
- `xgboost`: median test KGE=0.485, NSE=0.469, logNSE=0.527, PBIAS=-5.0%.

## Diagnostic Model Contrasts
- Median XGBoost gain over COSERO: nan KGE.
- Median xLSTM memory gain over RF: nan KGE.
- Median Transformer gain over RF: 0.091 KGE.
- Median best-sequence gain over best-tree model: 0.089 KGE.

## Signature Winners
- `baseflow_index` is best preserved by `transformer` (median abs. error 0.0419).
- `fdc_slope` is best preserved by `transformer` (median abs. error 0.221).
- `hfd_mean` is best preserved by `random_forest` (median abs. error 6.5).
- `high_q_dur` is best preserved by `transformer` (median abs. error 0.282).
- `high_q_freq` is best preserved by `random_forest` (median abs. error 0).
- `low_q_dur` is best preserved by `transformer` (median abs. error 1.06).
- `low_q_freq` is best preserved by `random_forest` (median abs. error 0).
- `q05` is best preserved by `transformer` (median abs. error 0.0493).
- `q95` is best preserved by `transformer` (median abs. error 0.263).
- `q_mean` is best preserved by `transformer` (median abs. error 0.0771).
- `runoff_ratio` is best preserved by `transformer` (median abs. error 0.023).
- `zero_q_freq` is best preserved by `random_forest` (median abs. error 0).

## Process-Zone Tables
See `process_zone_summary.csv` for model skill stratified by snow, aridity, storage, geology, human impact, and catchment size.

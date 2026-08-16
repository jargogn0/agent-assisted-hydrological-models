# Hydrologic Diagnostic Findings

## Hydrograph Skill
- `xlstm`: median test KGE=0.684, NSE=0.585, logNSE=0.667, PBIAS=-4.2%.
- `random_forest`: median test KGE=0.607, NSE=0.555, logNSE=0.611, PBIAS=1.9%.
- `xgboost`: median test KGE=0.485, NSE=0.469, logNSE=0.527, PBIAS=-5.0%.

## Diagnostic Model Contrasts
- Median XGBoost gain over COSERO: nan KGE.
- Median xLSTM memory gain over RF: 0.045 KGE.
- Median Transformer gain over RF: nan KGE.
- Median best-sequence gain over best-tree model: 0.036 KGE.

## Signature Winners
- `baseflow_index` is best preserved by `xlstm` (median abs. error 0.0431).
- `fdc_slope` is best preserved by `xlstm` (median abs. error 0.171).
- `hfd_mean` is best preserved by `random_forest` (median abs. error 6.5).
- `high_q_dur` is best preserved by `xlstm` (median abs. error 0.357).
- `high_q_freq` is best preserved by `random_forest` (median abs. error 0).
- `low_q_dur` is best preserved by `xlstm` (median abs. error 1.01).
- `low_q_freq` is best preserved by `random_forest` (median abs. error 0).
- `q05` is best preserved by `xlstm` (median abs. error 0.0526).
- `q95` is best preserved by `random_forest` (median abs. error 0.302).
- `q_mean` is best preserved by `random_forest` (median abs. error 0.0973).
- `runoff_ratio` is best preserved by `random_forest` (median abs. error 0.0306).
- `zero_q_freq` is best preserved by `random_forest` (median abs. error 0).

## Process-Zone Tables
See `process_zone_summary.csv` for model skill stratified by snow, aridity, storage, geology, human impact, and catchment size.

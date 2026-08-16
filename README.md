# Reproducibility archive

**Agent-assisted development of machine-learning hydrological models: testing a reproducible and controlled workflow**

Doudou Ba and Jakub Langhammer
Department of Physical Geography and Geoecology, Faculty of Science, Charles University, Prague

This archive contains every artefact the manuscript's Code and data availability
statement promises: the analysis pipeline, the locked configuration of every
reported experiment, the complete agent log with the verbatim prompt and
response of every iteration, the per-epoch training histories, the
catchment-level metric tables underlying all figures and tables, and the
analysis scripts. The LamaH-CE dataset itself is distributed separately
(Klingler et al., 2021; https://doi.org/10.5281/zenodo.4525244) and is not
redistributed here.

## Layout

| Path | Contents |
|---|---|
| `pipeline/src/` | The hydrological modelling pipeline (feature engineering, model training, evaluation, scorecard). |
| `pipeline/autoresearch/` | The experiment drivers: `agent_loop.py` (agent arm), `random_search_loop.py` (matched-budget random-proposal control), `runner.py` (execution, scoring, promotion rule), `confirm_selected.py` (tier-3 confirmations), and `agent_brief.md` (the research brief shown to the agent). |
| `pipeline/configs/` | Base configurations, including the two anchoring baselines (`configs/generated/agent_dev_base.yaml`, dated 7 June 2026, and `agent_dev_base_seq_fast.yaml`). |
| `pipeline/generated_configs/` | The locked YAML of every executed experiment: the 71 agent iterations (`agent_*`), the 5 anchoring baselines (`*_baseline0_*`), the 8 confirmation runs (`agent_confirm_*`), the 55 random-proposal control runs (`rand_*`), and the earlier pilot configurations. |
| `logs/agent_log.jsonl` | One record per reported-campaign iteration: verbatim prompt, verbatim model response, declared hypothesis and category, decision, scores, wall-clock cost, and model identifier with token usage. |
| `logs/random_search_log.jsonl` | The matched-budget random-proposal control: sampled mutation, decision, scores, seed. |
| `logs/results.tsv` | Every scored run, including the pilot, with the splits computed by each run (`splits_seen`, `selection_splits`). |
| `logs/pilot/` | The methodological pilot's controller log and its 3 July 2026 final evaluation (`final_eval_temporal_spatial.csv`), disclosed in Sect. 2.4 of the manuscript. |
| `results/confirmations/` | `metrics_by_basin.csv` for each of the 8 frozen confirmation runs: per-catchment KGE, NSE, log-NSE, PBIAS per evaluation split. These underlie Tables 4 and the confirmation figures. |
| `results/training_histories/` | Per-epoch training histories of the sequence models. |
| `results/*.json` | Computed outputs of the statistical analyses: paired basin-bootstrap intervals (`bootstrap_ci.json`, `bootstrap_components.json`), scorecard sensitivity (`scorecard_sensitivity.json`), and the agent-versus-control comparison (`random_vs_agent.json`). |
| `analysis/` | The scripts that produce every table and figure in the manuscript from the files above. |

## Verifying the manuscript's claims

The four properties stated in Sect. 2.4 of the manuscript can be checked directly:

1. **No reported configuration is a pilot configuration** — compare the `agent_*` tags in `logs/agent_log.jsonl` against the pilot tags in `logs/pilot/controller_log.jsonl`.
2. **All 76 reported runs computed development-validation metrics only** — filter `logs/results.tsv` to the reported-campaign tags and inspect `splits_seen`.
3. **No reported-campaign prompt contains confirmation quantities** — search the `prompt_verbatim` fields of `logs/agent_log.jsonl`.
4. **The scorecard predates all confirmation results** — the composite score is defined in `pipeline/autoresearch/runner.py`; the pilot's confirmation evaluation of 3 July 2026 is in `logs/pilot/`.

Statistical results are regenerated with, for example:

```
python analysis/bootstrap_ci.py          # Table 4 with paired bootstrap intervals
python analysis/scorecard_sensitivity.py # Table 5
python analysis/trajectory_stats.py      # campaign trajectories (Table 3)
```

(paths inside the scripts assume the original project layout; adjust the ROOT
constant, or place this archive at the pipeline root, to re-run them).

## A note on repeatability

The archived log is the authoritative record of what was proposed and decided.
Re-executing `agent_loop.py` is not expected to reproduce the same trajectory,
because the language model (Anthropic Claude Sonnet 5, identifier
`claude-sonnet-5`) is stochastic and its hosted version may change. All
random seeds for data sampling, model training, the basin bootstrap, and the
random-proposal control are fixed and recorded in the configurations and logs.

## Licences

- Code: BSD 3-Clause (see `LICENSE`)
- Data, logs, and derived tables: CC BY 4.0 (see `LICENSE-DATA`)

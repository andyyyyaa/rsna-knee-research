# Data access and reproducibility

## Public materials

The repository includes original evaluation code, aggregate tables/figures, provenance hashes and an executed aggregate notebook. It excludes medical images, radiology reports, stable study identifiers, individual predictions, checkpoint weights, API keys and local credential files. The notebook can run from the public summaries without Kaggle authentication.

## Recompute from authorized predictions

The main source is [Xinyi Lu's V16 Gold58 weight screen](https://www.kaggle.com/code/andyyyyaa/rsna-knee-v16-gold58-weight-screen). The notebook was private at collection. The account owner can access its outputs; a public repository link does not grant access to those outputs or to competition data.

Use an already authorized Kaggle session and download only the relevant root-level outputs:

```bash
kaggle kernels output andyyyyaa/rsna-knee-v16-gold58-weight-screen \
  -p private/v16-output --page-size 200 \
  --file-pattern '^(v16_[^/]+\.(csv|json)|raptor_raw\.csv|raptor224_raw\.csv|ours_sub\.csv|_own\.csv|_own\.csv\.quality\.csv|optimization_receipt\.json)$'
python scripts/evaluate.py --data-dir private/v16-output --output-dir results
```

Kaggle's latest output can change after a notebook is rerun. Before using it, match every required file against `config/input_manifest.json`. The evaluator enforces those hashes. Keep the authorized snapshot locally if later versions supersede it.

Required inputs:

- `v16_gold_targets.csv`: UID followed by the 12 binary labels in official order.
- `raptor_raw.csv`, `raptor224_raw.csv`, `ours_sub.csv`, `_own.csv`: aligned baseline branches.
- `_own.csv.quality.csv`: Reader quality status; all 58 studies must be valid.
- `v16_royalacecat_raw.csv`, `v16_syouyatobita_g233_raw.csv`, `v16_kawaharataishi_swa_raw.csv`, `v16_tbongc_raw.csv`: all four screened extras, including failed quality candidates.
- `v16_baseline_gold_predictions.csv`, `v16_group_gold_predictions.csv`: actual production outputs used for parity checks.
- `v16_spec.json`, `v16_validation_decision.json`: frozen coefficients, checkpoint identity, screen and runtime evidence.

Main source code was downloaded separately for private inspection; its SHA is recorded in `results/source_snapshot.json`. The public repository implements the required mathematics and does not redistribute the upstream notebook or its embedded third-party sources.

## Official labels and split audit

Obtain `train.csv` through your authorized access to the [competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data). For the optional historical split check, retain the original local adapter `split.json`; its expected hash is in `results/leakage_audit.json`.

```bash
python scripts/audit_split.py --official-train-csv private/train.csv \
  --gold-targets-csv private/v16-output/v16_gold_targets.csv \
  --local-split-json private/historical-adapter-split.json \
  --output results/leakage_audit.json
```

This audits the supplied study split and exact normalized report hashes. It does not infer patient identity from reports, Study UID prefixes or anatomy. A future patient-level audit needs a verified patient-to-study mapping and actual training membership for every checkpoint.

## Historical sensitivity appendix

Source: [Apex Parameter Validation, version 356976389](https://www.kaggle.com/code/andyyyyaa/rsna-apex-parameter-validation?scriptVersionId=356976389). The authorized archive needs `candidate_predictions.npz`, `reader_views.npz`, `reader_view_keys.json`, `labels.npy`, `study_ids.json` and `experiment_results.csv`.

```bash
python scripts/legacy_view_sensitivity.py --archive-dir private/apex/results \
  --gold-targets-csv private/v16-output/v16_gold_targets.csv
```

Only aggregate AUCs and intervals leave that script. The archive remains private and is listed by hash in `results/legacy_provenance.json`.

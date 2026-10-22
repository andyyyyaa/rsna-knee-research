# Methods

## Design and provenance

This is a **retrospective cached-prediction study**, conducted after model development and candidate screening. It is not a new preregistered prospective experiment. Primary evidence is the account's actual `rsna-knee-v16-gold58-weight-screen` Kaggle run. Its V13/V15 baseline recipe is pinned to upstream script version **360646466**, which in turn preserves V13 version **360244791**. File hashes and checkpoint hashes are recorded in `results/input_provenance.json` and `results/frozen_model_spec.json`.

The 58 fully labelled studies are aligned by `StudyInstanceUID`, never by assumed row order. All 12 labels are binary and complete. The full official training table available for audit contains 4,407 studies. All Gold58 labels were checked against it. No missing labels are filled as negatives.

## Frozen inference/fusion recipe

Let R(x) denote average-tie per-label percentile ranking across the same cohort.

1. Baseline Raptor slot: `0.65 R(CoAtNet384) + 0.35 R(CoAtNet224)`.
2. Combine that slot with the four-fold, flip-TTA R6 output. The Raptor weights are 0.5 by default, 0.2 for Lateral OA, 0.6 for PF OA, 0.8 for Synovitis, and 0.35 for Baker's.
3. Preserve the production float32 CSV serialization/reload boundary before the Reader blend. This matters for near ties.
4. Rank the combined output and Reader output, blend with the original per-finding Reader weights scaled to 1/10, then rank again.
5. The SWA variant changes the Raptor slot to `0.50 R(CoAtNet384) + 0.25 R(CoAtNet224) + 0.25 R(SWA)`; R6 and Reader weights stay fixed.

Reader weights in official label order: `[.025,.020,.020,.008,.025,.008,.008,.004,.003,.035,.030,.008]`. R6 stride=1, four folds, and flip TTA are unchanged. The raw R6 output is already a trained-stack/fold-ensemble output, not a single checkpoint. The Reader output is a three-fold ensemble. The two individual CoAtNet arms are single checkpoints with the original inference TTA; SWA is an averaged checkpoint, not a single training epoch.

For removal experiments, a removed Raptor resolution gives its entire slot to the remaining resolution; removal of R6 gives the slot to Raptor. Reader removal bypasses only the last Reader fusion. Equal-class-weight ablation uses a fixed 0.5 Raptor/R6 mixture for all findings. None of these comparisons retrains models or searches coefficients.

## Metrics and uncertainty

- Primary metric: macro average of 12 empirical ROC AUC values, matching the competition's metric definition. AUC is calculated from ranks with half credit for ties and independently compared with [scikit-learn ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
- Primary contrast: fixed SWA variant minus the V13/V15 baseline. Other contrasts are exploratory and are reported together, not cherry-picked.
- Paired ordinary bootstrap: 2,000 accepted study-level resamples of 58 studies with replacement; seed 20261021. One draw is shared across every label and model. A complete draw would be rejected if any finding lacked either class. In the observed run all 2,000 attempts were accepted.
- Fixed-prediction bootstrap uses exact positive/negative pair comparisons with study multiplicities. Tests compare this to explicitly resampling rows, including ties.
- Secondary primary-contrast analysis recomputes all ranks and the float32 CSV boundary in every resample. Both methods produce intervals crossing zero.
- Intervals are pointwise 95% percentile intervals. They are conditional on the available studies, already trained checkpoints, fixed coefficients and prior choices. They are not corrected for repeated checkpoint selection, Gold58 reuse, or multiple testing. Study resampling also cannot account for unknown within-patient clustering.
- The fraction of bootstrap differences above zero is a descriptive resampling fraction, **not a probability of generalization or a clinical benefit**.
- Recall, specificity, precision, TP, FN, FP and TN use a fixed per-label percentile-rank cutoff ≥0.5. The cutoff is not optimized on labels. It selects roughly half the cohort by construction, and its results depend on prevalence and cohort ranks; it is not a calibrated 50% disease probability. Definitions follow [scikit-learn Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html).

## Split and duplication audit

We check unique study IDs, Gold58 label parity, and membership in a historical local adapter train/dev split. Exact normalized report text is lowercased, whitespace-collapsed and hashed locally; only overlap counts are exported. That proxy does not identify patients. A shared report hash may reflect a template or duplicate report, and does not prove the same patient or leakage.

The current public checkpoints do not come with a verifiable patient-level training membership manifest in the retrieved evidence. Neither an architecture name, a fold count, nor an unusually high AUC establishes independence or proves leakage. The correct status is **patient disjointness not verifiable**.

## Robustness and efficiency scope

The supplementary view analysis re-evaluates archived Apex scores from version **356976389**. It compares 11 view recipes at fixed published per-finding fusion weights; a 256px/k16/FOV0.92/no-flip recipe is the reference. Resolution, FOV, slice count, phase/offset and flip changes were applied in that historical GPU inference. They were not re-run on images here. This appendix is not evidence of Gaussian-noise, intensity-shift or scanner/site robustness for V13/V15.

Candidate wall times are from the same 58-study T4 ×2 screen, divided by 58. They include branch setup, preprocessing/cache effects and different checkpoint precision policies; royalacecat used an FP32 numerical fallback. They do not isolate architecture latency. Peak GPU memory was not instrumented in this screen and is left unmeasured. The 58-study runtime projection, with margins, is about **40,372 seconds** for the SWA group; it failed the original 7.5h projection gate and is not a measurement of hidden-test runtime.

## Validation and sharing

Production baseline and saved SWA prediction scores are reconstructed within 1e-12. Every AUC from all 18 main variants matches scikit-learn within 1e-12. Legacy AUCs match their archived per-class values. Unit tests cover ties, one-class rejection, nonfinite input, rank permutation invariance and exact paired bootstrap multiplicities.

Only aggregate tables, figures, code, provenance hashes and the aggregate notebook are published. Recomputing metrics requires the authorized cached outputs; re-rendering the notebook requires only the public summaries. These are distinct levels of reproducibility, made explicit in the data-access documentation.

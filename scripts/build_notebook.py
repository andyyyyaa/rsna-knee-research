"""Build the public aggregate research notebook with nbformat."""
import json
from pathlib import Path
import nbformat as nbf
import pandas as pd

def build(root):
    root=Path(root);ab=pd.read_csv(root/'results/ablation.csv').set_index('variant')
    base=ab.loc['baseline_v13_v15'];swa=ab.loc['extra_kawaharataishi_swa']
    nb=nbf.v4.new_notebook()
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md(f'''# RSNA Knee: Beyond the Leaderboard

**Xinyi Lu · retrospective research audit · 58 complete-label studies · 12 findings**

## tl;dr

The account has a verified **0.958 public leaderboard score**. In the cached Gold58 development run, the V13/V15 baseline has macro ROC AUC **{base.macro_auc:.6f}**. Adding the fixed SWA checkpoint gives **{swa.macro_auc:.6f}**, a change of **{swa.delta_vs_baseline:+.6f}**, with conditional paired 95% interval **[{swa.delta_ci_low:+.6f}, {swa.delta_ci_high:+.6f}]**.

Removing R6 lowers AUC by **0.014553**. Removing the low-weight Reader changes ranks but no positive-negative ordering, leaving AUC unchanged in these 58 cases. Synovitis and Lateral OA remain the weakest classes.

**This is not independent validation.** Gold58 was repeatedly reused, including upstream checkpoint selection and a four-candidate screen. Intervals do not correct selection bias. V16's competition score was pending at collection. These are research results, not a validated clinical system.
''')
    md('''## Context & Methods

This notebook regenerates the research presentation from **public aggregate summaries**. Study-level calculations were executed separately by `scripts/evaluate.py` against the authorized cached predictions; it checks study/label alignment, exact production-fusion parity and every AUC against scikit-learn. See `docs/METHODS.md` for details and `docs/DATA_ACCESS.md` for authorized metric recomputation.

### Key Assumptions

- Freeze existing checkpoints and coefficients; do not retrain or optimize on this analysis.
- Compare all 18 variants. R6 is a four-fold ensemble; Reader is a three-fold ensemble. These are not single checkpoints.
- Pair 2,000 ordinary study bootstrap draws across labels/variants (seed 20261021).
- Report conditional, pointwise 95% intervals without multiple-comparison or selection-bias adjustment.
- Also rerank every branch within bootstrap draws for a primary-contrast sensitivity check.
- Define descriptive Recall at percentile rank ≥0.5. It is not a calibrated probability cutoff.
- Patient identity and public-checkpoint training membership are unavailable; study bootstrap does not establish patient independence.
''')
    code('''from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import display, Image

ROOT = Path.cwd()
if not (ROOT / 'results' / 'summary.json').is_file():
    ROOT = ROOT.parent
assert (ROOT / 'results' / 'summary.json').is_file(), 'Run from the repository or notebooks directory'
sys.path.insert(0, str(ROOT / 'scripts'))
from render_figures import generate
_figures = generate(ROOT)
summary = json.loads((ROOT / 'results' / 'summary.json').read_text())
ablation = pd.read_csv(ROOT / 'results' / 'ablation.csv')
per_class = pd.read_csv(ROOT / 'results' / 'per_class.csv')
print(f"Studies: {summary['cohort_n']} | findings: {len(summary['labels'])} | paired draws: {summary['valid_resamples']}")
print('Analysis: cached-prediction replay; no new training or GPU image inference')
''')
    md('''## Data

The main source is the account's actual [V16 Gold58 screen](https://www.kaggle.com/code/andyyyyaa/rsna-knee-v16-gold58-weight-screen), using the V13/V15 production recipe as baseline. Raw medical data, reports, study identifiers and per-study predictions are excluded here. Hashes preserve input identity without republishing them. `competition_history.csv` comes from a read-only authenticated Kaggle API query.
''')
    code('''display(per_class.loc[per_class.variant.eq('baseline_v13_v15'),
    ['label','positives','negatives','auc','auc_ci_low','auc_ci_high']].round(6).reset_index(drop=True))
''')
    md('''## Results

### 1. Frozen ensemble ablation

Differences are relative to the complete V13/V15 recipe. Their uncertainty is paired, not inferred from overlap of independent model intervals. All extras, including poor-transfer checkpoints, remain in the table.
''')
    code('''display(ablation[['variant','macro_auc','delta_vs_baseline','delta_ci_low','delta_ci_high']].round(6))
display(Image(filename=str(ROOT / 'figures' / 'ablation_forest.png')))
''')
    md('''R6 removal shows the largest observed loss among branch removals. The SWA contrast and 224-removal contrast both have intervals crossing zero. The modest SWA point estimate is a development observation after prior screening, not a verified generalization gain.

### 2. Per-class performance and operating-point errors
''')
    code('''display(Image(filename=str(ROOT / 'figures' / 'per_class_auc.png')))
display(per_class.loc[per_class.variant.eq('baseline_v13_v15'),
    ['label','recall','specificity','precision','tp','fn','fp','tn']].round(4).reset_index(drop=True))
display(Image(filename=str(ROOT / 'figures' / 'recall_and_errors.png')))
''')
    md('''Synovitis, Lateral OA and PF OA have the lowest baseline AUCs. SWA improves six AUCs, ties four and worsens Effusion and Contusion. The cutoff selects roughly half the cohort by rank, so the Recall table needs its specificity and class prevalence for interpretation. No individual case or radiological error diagnosis is published.

### 3. Complementarity and the Reader's unchanged AUC
''')
    code('''display(Image(filename=str(ROOT / 'figures' / 'complementarity.png')))
checks = json.loads((ROOT / 'results' / 'rank_structural_checks.json').read_text())
display(pd.DataFrame(checks).T)
''')
    md('''The Reader moves 30 rank cells but no positive-negative pair ordering in this cohort. Its identical AUC therefore does not imply byte-identical predictions or prove that the branch has no value on a larger test set. SWA combines reasonable single-model discrimination and lower correlation; very low correlation alone does not ensure useful predictions.

### 4. Split and leakage audit
''')
    code('''audit = json.loads((ROOT / 'results' / 'leakage_audit.json').read_text())
display(pd.DataFrame([
    ['Official Gold58 labels match', audit['official_gold_labels_match']],
    ['Historical adapter train/dev study overlap', audit['historical_local_adapter_split']['train_dev_study_overlap']],
    ['Historical adapter Gold58 overlap', audit['historical_local_adapter_split']['gold_train_overlap'] + audit['historical_local_adapter_split']['gold_dev_overlap']],
    ['Historical adapter train/dev normalized report overlap', audit['historical_local_adapter_split']['normalized_report_hash_train_dev_overlap']],
    ['Gold58 / other official normalized report hash overlap', audit['normalized_report_hash_overlap_gold_vs_other_studies']],
    ['Stable patient identifier in official table', audit['official_metadata_has_patient_identifier']],
], columns=['Check','Observed result']))
print(audit['patient_disjointness'])
print(audit['public_checkpoint_training_membership'])
''')
    md('''The historical local adapter split passes the stated study/report checks, but it is a different experiment from the public checkpoints used by V13/V15. One shared report hash is a duplication signal that may arise from templates; it is not proof of the same patient. Patient-level isolation remains unverified.

### 5. Supplementary preprocessing sensitivity and efficiency

The following sensitivity figure uses the **older Apex family** and archived image-inference outputs. It is kept separate from the main current-recipe conclusions. Noise and intensity corruption were not measured.
''')
    code('''display(Image(filename=str(ROOT / 'figures' / 'legacy_view_sensitivity.png')))
display(Image(filename=str(ROOT / 'figures' / 'candidate_runtime.png')))
''')
    md('''All displayed historical view-difference intervals cross zero. Candidate costs are matched 58-study wall times; different precision and cache behavior remain confounders. Peak GPU memory is unmeasured. The archived conservative SWA group projection is about 40,372 seconds for 1,300 studies, exceeding the original 7.5h gate; that projection is not actual hidden-test runtime.

## Takeaways

1. Identify a useful research question beyond the leaderboard: which components change discriminative ordering and where do they help or hurt?
2. Preserve uncertainty and negative findings. R6 contributes meaningfully in this fixed replay; evidence for the SWA addition remains conditional and inconclusive for future generalization.
3. Prioritize Synovitis/Lateral OA error review and an untouched patient/site-disjoint validation cohort with audited checkpoint training membership.
4. Treat preprocessing sensitivity, runtime and peak memory as separate measured questions. Do not replace missing image perturbation tests or memory measurements with assumptions.
5. Credit pretrained checkpoint authors separately from this repository's evaluation and interpretation work.

For a prospective follow-up, see `docs/NEXT_EXPERIMENT.md`. For source attribution and data-access boundaries, see `docs/ATTRIBUTION.md` and `docs/DATA_ACCESS.md`.
''')
    nb.cells=cells
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                 'language_info':{'name':'python','version':'3.12'},
                 'analysis_scope':'Public aggregate replay; private raw inputs required for metric recomputation'}
    nbf.validate(nb)
    path=root/'notebooks/research_audit.ipynb';path.parent.mkdir(exist_ok=True)
    nbf.write(nb,path)
    return path

if __name__=='__main__':build(Path(__file__).resolve().parents[1])

"""Run actual cached-prediction experiments; write aggregate outputs only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rsna_research.metrics import LABELS, UID, per_class_auc, percentile_rank, draw_counts, bootstrap_auc, operating_point
from rsna_research.fusion import fusion, variants

FILES = {'384': 'raptor_raw.csv', '224': 'raptor224_raw.csv', 'r6': 'ours_sub.csv', 'reader': '_own.csv',
         **{k: f'v16_{k}_raw.csv' for k in ['royalacecat', 'syouyatobita_g233', 'kawaharataishi_swa', 'tbongc']}}

def load(path, ids=None, truth=False):
    df = pd.read_csv(path, dtype={UID: str})
    if list(df.columns) != [UID, *LABELS] or not df[UID].is_unique:
        raise ValueError(f'Unexpected columns or duplicate studies: {path.name}')
    if ids is not None:
        if set(df[UID]) != set(ids):
            raise ValueError(f'Study coverage mismatch: {path.name}')
        df = df.set_index(UID).loc[ids].reset_index()
    arr = df[LABELS].to_numpy(dtype=float)
    if not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
        raise ValueError(f'Invalid scores/labels: {path.name}')
    if truth and not np.isin(arr, [0, 1]).all():
        raise ValueError('Nonbinary complete labels')
    return df, arr

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    ap.add_argument('--resamples', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    args = ap.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    manifest=ROOT/'config/input_manifest.json'
    if manifest.exists():
        for item in json.loads(manifest.read_text())['inputs']:
            source=args.data_dir/item['filename']
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError(f'Frozen input fingerprint mismatch: {item["filename"]}')
    df, y = load(args.data_dir / 'v16_gold_targets.csv', truth=True)
    ids = df[UID].tolist()
    if len(y) != 58:
        raise ValueError('This frozen study expects exactly 58 studies')
    branches = {name: load(args.data_dir / file, ids)[1] for name, file in FILES.items()}
    quality = pd.read_csv(args.data_dir / '_own.csv.quality.csv', dtype={UID: str})
    if not quality[UID].is_unique or set(quality[UID]) != set(ids) or not quality.reader_valid.eq(True).all():
        raise ValueError('Reader coverage/quality check failed')
    models = variants(branches)
    for key, filename in [('baseline_v13_v15','v16_baseline_gold_predictions.csv'),
                          ('extra_kawaharataishi_swa','v16_group_gold_predictions.csv')]:
        saved = load(args.data_dir / filename, ids)[1]
        if not np.allclose(models[key], saved, rtol=0, atol=1e-12):
            raise ValueError(f'Production score reconstruction failed: {key}')
    counts, attempted = draw_counts(y, args.resamples, args.seed)
    boot = {name: bootstrap_auc(y, p, counts) for name, p in models.items()}
    reference = boot['baseline_v13_v15']
    ref_auc = per_class_auc(y, models['baseline_v13_v15'])
    rows, perclass = [], []
    for name, p in models.items():
        auc = per_class_auc(y, p)
        if not np.allclose(auc, roc_auc_score(y, p, average=None), rtol=0, atol=1e-12):
            raise ValueError('Independent scikit-learn AUC check failed')
        delta = boot[name].mean(1) - reference.mean(1)
        interval = np.quantile(boot[name].mean(1), [.025,.975])
        dci = np.quantile(delta, [.025,.975])
        rows.append(dict(variant=name, macro_auc=float(auc.mean()), auc_ci_low=interval[0], auc_ci_high=interval[1],
                         delta_vs_baseline=float((auc-ref_auc).mean()), delta_ci_low=dci[0], delta_ci_high=dci[1],
                         bootstrap_positive_fraction=float((delta>0).mean()), cohort_n=len(y), valid_resamples=len(counts)))
        op = operating_point(y, p)
        for j, label in enumerate(LABELS):
            ci = np.quantile(boot[name][:,j], [.025,.975])
            dc = np.quantile(boot[name][:,j] - reference[:,j], [.025,.975])
            perclass.append(dict(variant=name, label=label, positives=int(y[:,j].sum()), negatives=int((1-y[:,j]).sum()),
                                 auc=auc[j], auc_ci_low=ci[0], auc_ci_high=ci[1], delta_vs_baseline=auc[j]-ref_auc[j],
                                 delta_ci_low=dc[0],delta_ci_high=dc[1], threshold_rank=.5,
                                 **{k: v[j].item() for k,v in op.items()}))
    pd.DataFrame(rows).to_csv(out/'ablation.csv',index=False)
    pd.DataFrame(perclass).to_csv(out/'per_class.csv',index=False)
    # Aggregate error co-occurrence, never export per-study errors or labels.
    p = models['baseline_v13_v15']; pred = percentile_rank(p) >= .5
    fn = ((y == 1) & ~pred).astype(int)
    pd.DataFrame(fn.T @ fn,index=LABELS,columns=LABELS).rename_axis('label').to_csv(out/'false_negative_cooccurrence.csv')
    correlations = []
    for name in FILES:
        if name == '384': continue
        for j,label in enumerate(LABELS):
            correlations.append(dict(branch=name,label=label,spearman_with_384=float(spearmanr(branches[name][:,j],branches['384'][:,j]).statistic)))
    pd.DataFrame(correlations).to_csv(out/'branch_correlations.csv',index=False)
    write_json(out/'rank_structural_checks.json', {
        name: dict(changed_rank_cells=int(np.count_nonzero(percentile_rank(p)!=percentile_rank(models['baseline_v13_v15']))),
                   maximum_rank_difference=float(np.max(np.abs(percentile_rank(p)-percentile_rank(models['baseline_v13_v15'])))),
                   changed_positive_negative_pair_orderings=sum(int(np.count_nonzero(
                       np.sign(p[y[:,j]==1,j,None]-p[y[:,j]==0,j][None,:])!=
                       np.sign(models['baseline_v13_v15'][y[:,j]==1,j,None]-models['baseline_v13_v15'][y[:,j]==0,j][None,:]))) for j in range(12)))
        for name,p in models.items() if name in ['without_reader','without_224','without_r6','extra_kawaharataishi_swa']})
    # Secondary sensitivity: recompute all cohort ranks on each identical draw.
    reranked = []
    for c in counts:
        ix = np.repeat(np.arange(len(y)),c)
        b = {k: v[ix] for k,v in branches.items()}
        reranked.append(float(np.mean(per_class_auc(y[ix],fusion(b,extra='kawaharataishi_swa')) - per_class_auc(y[ix],fusion(b)))))
    decision=json.loads((args.data_dir/'v16_validation_decision.json').read_text())
    pd.DataFrame([dict(branch=k, single_macro_auc=v.get('single_macro_auc'),
                      mean_spearman_with_384=v.get('mean_spearman_with_384'),
                      upstream_screen_eligible=v.get('eligible'), seconds_per_study=v.get('seconds_per_study'))
                  for k,v in decision['candidate_results'].items()]).to_csv(out/'candidate_quality_cost.csv',index=False)
    spec=json.loads((args.data_dir/'v16_spec.json').read_text())
    write_json(out/'frozen_model_spec.json',{k:spec[k] for k in ['revision','base_notebook','base_script_version','base_public_score_displayed','evaluation_population','candidates','fusion','screen','runtime']})
    write_json(out/'summary.json',dict(cohort_n=len(y),labels=LABELS,primary_comparison='extra_kawaharataishi_swa vs baseline_v13_v15',
        analysis_type='Retrospective analysis of cached GPU predictions; no new model training or GPU inference',
        seed=args.seed,valid_resamples=len(counts),attempted_resamples=attempted,unit='study',independent_validation=False,
        confidence_intervals='Pointwise 95% percentile bootstrap; conditional on fixed weights, cohort, checkpoints and prior selection; not multiplicity adjusted',
        bootstrap_positive_fraction_is_not_generalization_probability=True,
        recall='Recall/precision/specificity at fixed average-percentile-rank >=0.5; uncalibrated descriptive operating point',
        production_baseline_max_abs_error=float(np.max(np.abs(models['baseline_v13_v15']-load(args.data_dir/'v16_baseline_gold_predictions.csv',ids)[1]))),
        reranked_primary_delta=dict(mean=float(np.mean(reranked)),ci95=np.quantile(reranked,[.025,.975]).tolist(),positive_fraction=float((np.asarray(reranked)>0).mean())),
        upstream_group_runtime=decision.get('group',{}),
        validations=['Exact production baseline and SWA score parity within 1e-12','All 18 variants cross-checked against scikit-learn','Exact 58-study coverage and binary complete labels','All baseline Reader studies marked valid']))
    records=[]
    for f in ['v16_gold_targets.csv',*FILES.values(),'v16_baseline_gold_predictions.csv','v16_group_gold_predictions.csv','v16_spec.json','v16_validation_decision.json','_own.csv.quality.csv']:
        p=args.data_dir/f
        records.append(dict(filename=f,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
                            source='https://www.kaggle.com/code/andyyyyaa/rsna-knee-v16-gold58-weight-screen',redistributed=False))
    write_json(out/'input_provenance.json',dict(inputs=records,raw_data_published=False,patient_identifiers_published=False))
    write_json(out/'environment.json',dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','matplotlib','nbformat','nbclient']}))
    print(pd.DataFrame(rows)[['variant','macro_auc','delta_vs_baseline','delta_ci_low','delta_ci_high']].to_string(index=False))

if __name__=='__main__': main()

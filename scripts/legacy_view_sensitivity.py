"""Supplementary cached Apex view perturbations; not the V13/V15 model."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rsna_research.metrics import LABELS,UID,per_class_auc,draw_counts,bootstrap_auc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive-dir',type=Path,required=True)
    ap.add_argument('--gold-targets-csv',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results');a=ap.parse_args()
    ids=json.loads((a.archive_dir/'study_ids.json').read_text())
    truth=np.load(a.archive_dir/'labels.npy',allow_pickle=False)
    gold=pd.read_csv(a.gold_targets_csv,dtype={UID:str}).set_index(UID)
    if len(set(ids))!=58 or set(ids)!=set(gold.index) or not np.array_equal(truth,gold.loc[ids,LABELS].to_numpy()):
        raise ValueError('Legacy archive UID/truth alignment failed')
    scores=np.load(a.archive_dir/'candidate_predictions.npz',allow_pickle=False)
    historical=pd.read_csv(a.archive_dir/'experiment_results.csv').set_index('name')
    counts,_=draw_counts(truth,2000,20261021)
    base=bootstrap_auc(truth,scores['view_r256'],counts).mean(1)
    rows=[]
    for name in scores.files:
        auc=per_class_auc(truth,scores[name]);expected=historical.loc[name,LABELS].to_numpy(dtype=float)
        if not np.allclose(auc,expected,rtol=0,atol=1e-12):raise ValueError('Legacy saved AUC mismatch')
        row=dict(variant=name,macro_auc=float(auc.mean()),cohort_n=len(truth),model_family='Legacy Apex, not V13/V15',
                 delta_vs_r256=float(auc.mean()-per_class_auc(truth,scores['view_r256']).mean()))
        if name.startswith('view_'):
            d=bootstrap_auc(truth,scores[name],counts).mean(1)-base
            ci=np.quantile(d,[.025,.975]);row.update(delta_ci_low=ci[0],delta_ci_high=ci[1])
        rows.append(row)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    all_rows=pd.DataFrame(rows);all_rows.to_csv(a.output_dir/'legacy_all21.csv',index=False)
    all_rows[all_rows.variant.str.startswith('view_')].to_csv(a.output_dir/'legacy_view_sensitivity.csv',index=False)
    provenance=dict(source='https://www.kaggle.com/code/andyyyyaa/rsna-apex-parameter-validation?scriptVersionId=356976389',
        archived_inference_only=True,images_reprocessed_in_this_analysis=False,model_family='Legacy Apex',
        scope='Resolution, FOV, slice count, offset and flip sensitivity. No new noise/intensity corruption run.',
        raw_redistributed=False,input_hashes={f:hashlib.sha256((a.archive_dir/f).read_bytes()).hexdigest()
            for f in ['candidate_predictions.npz','reader_views.npz','reader_view_keys.json','labels.npy','study_ids.json','experiment_results.csv']})
    (a.output_dir/'legacy_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf8')
    print(all_rows[all_rows.variant.str.startswith('view_')].to_string(index=False))

if __name__=='__main__':main()

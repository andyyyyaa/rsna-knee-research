"""Audit available study/report splits without publishing row-level records."""
import argparse, hashlib, json, re
from pathlib import Path
import pandas as pd

def report_hash(x):
    # Proxy only: identical normalized report text is not patient identity.
    normalized=re.sub(r'\s+',' ',str(x)).strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest() if normalized else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--official-train-csv',type=Path,required=True)
    ap.add_argument('--gold-targets-csv',type=Path,required=True)
    ap.add_argument('--local-split-json',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); uid='StudyInstanceUID'
    official=pd.read_csv(a.official_train_csv,dtype={uid:str})
    gold=pd.read_csv(a.gold_targets_csv,dtype={uid:str})
    labels=list(gold.columns[1:]); ids=set(gold[uid])
    if not official[uid].is_unique or not gold[uid].is_unique:
        raise ValueError('Duplicate study identifiers')
    aligned=official.set_index(uid).loc[gold[uid],labels].to_numpy()
    if not (aligned==gold[labels].to_numpy()).all():
        raise ValueError('Gold labels differ from official labels')
    complete=official[labels].notna().all(axis=1)
    if set(official.loc[complete,uid])!=ids:
        raise ValueError('Gold58 is not exactly the complete-label official subset')
    hashes=official.set_index(uid)['Report'].map(report_hash)
    patient_columns=[c for c in official.columns if re.sub('[^a-z0-9]','',c.lower()) in
                     {'patientid','patientidentifier','patientuid','subjectid','subjectuid','patient'}]
    gh=set(hashes.loc[list(ids)].dropna())
    nh=set(hashes.loc[~hashes.index.isin(ids)].dropna())
    result=dict(official_studies=len(official),complete_label_studies=int(complete.sum()),gold_studies=len(gold),
        study_ids_unique=True,official_gold_labels_match=True,
        gold_is_subset_of_official_training=True,
        official_metadata_has_patient_identifier=bool(patient_columns),
        metadata_columns=list(official.columns),
        normalized_report_hash_overlap_gold_vs_other_studies=len(gh&nh),
        patient_disjointness=('NOT_VERIFIABLE: patient fields are present but checkpoint membership still requires audit' if patient_columns else
                              'NOT_VERIFIABLE: official train table has StudyInstanceUID and reports, no stable patient identifier'),
        public_checkpoint_training_membership='UNKNOWN: fold names and checkpoint filenames do not prove Gold58 exclusion',
        upstream_selection='Gold58 reused for checkpoint/candidate selection; all intervals are conditional development estimates',
        report_group_is_patient_id=False,
        input_hashes={a.official_train_csv.name:hashlib.sha256(a.official_train_csv.read_bytes()).hexdigest(),
                      a.gold_targets_csv.name:hashlib.sha256(a.gold_targets_csv.read_bytes()).hexdigest()})
    if a.local_split_json:
        split=json.loads(a.local_split_json.read_text(encoding='utf8'))
        train=set(split['train_ids']); dev=set(split['dev_ids'])
        tr=set(hashes.reindex(list(train)).dropna()); dv=set(hashes.reindex(list(dev)).dropna())
        result['historical_local_adapter_split']=dict(train_studies=len(train),dev_studies=len(dev),
            train_dev_study_overlap=len(train&dev),gold_train_overlap=len(ids&train),gold_dev_overlap=len(ids&dev),
            missing_train_ids=len(train-set(hashes.index)),missing_dev_ids=len(dev-set(hashes.index)),
            normalized_report_hash_train_dev_overlap=len(tr&dv),declared_group_column=split.get('group_column'),
            scope='Historical local adapter split only; does not validate current public checkpoints or patient separation',
            split_sha256=hashlib.sha256(a.local_split_json.read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

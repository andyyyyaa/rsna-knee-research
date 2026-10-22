"""Validate the aggregate-only release, links, tables and saved notebook."""
from pathlib import Path
import json,re
import nbformat
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    forbidden_suffix={'.npz','.npy','.parquet','.dcm','.pt','.pth','.pkl','.pckl'}
    forbidden=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and
               '.git' not in p.parts and p.suffix.lower() in forbidden_suffix]
    if forbidden:raise ValueError(f'Nonaggregate data in release: {forbidden}')
    rules={
        'private Study UID':r'1\.2\.826\.0\.1\.3680043\.8\.498\.\d{12,}',
        'GitHub credential':r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})',
        'private key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'private local path':r'(?:C:[/\\]Users[/\\]|D:[/\\]RSNA_V9)'}
    checked=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts or p.suffix=='.png':continue
        if p.name=='check_release.py':continue # Patterns above intentionally name prohibited forms.
        text=p.read_text(encoding='utf8')
        for name,pattern in rules.items():
            if re.search(pattern,text):raise ValueError(f'{name} found in {p.relative_to(ROOT)}')
        checked.append(p.relative_to(ROOT).as_posix())
    for p in ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#')[0]
            if target and not (p.parent/target).exists():raise ValueError(f'Broken local link {p.name}: {target}')
    ab=pd.read_csv(ROOT/'results/ablation.csv')
    pc=pd.read_csv(ROOT/'results/per_class.csv')
    assert len(ab)==18 and ab.variant.is_unique and len(pc)==18*12
    assert (pc.positives+pc.negatives).eq(58).all()
    assert np.allclose(pc.tp+pc.fn,pc.positives) and np.allclose(pc.fp+pc.tn,pc.negatives)
    assert np.allclose(pc.recall,pc.tp/pc.positives) and np.allclose(pc.specificity,pc.tn/pc.negatives)
    for name,row in ab.set_index('variant').iterrows():
        assert np.isclose(pc[pc.variant==name].auc.mean(),row.macro_auc,atol=1e-12)
    checks=json.loads((ROOT/'results/rank_structural_checks.json').read_text())
    assert checks['without_reader']['changed_rank_cells']==30
    assert checks['without_reader']['changed_positive_negative_pair_orderings']==0
    nb=nbformat.read(ROOT/'notebooks/research_audit.ipynb',as_version=4);nbformat.validate(nb)
    codes=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in codes)
    assert not any(o.output_type=='error' for c in codes for o in c.get('outputs',[]))
    imgs=sum('image/png' in o.get('data',{}) for c in codes for o in c.get('outputs',[]))
    assert imgs==6 and len(list((ROOT/'figures').glob('*.png')))==6
    result=dict(status='PASS',public_aggregate_only=True,text_files_checked=len(checked),
                raw_prediction_or_weight_files=0,private_uid_or_credential_matches=0,
                all_local_markdown_links_resolve=True,main_variants=18,per_class_rows=216,
                executed_code_cells=len(codes),saved_figure_outputs=imgs,
                numeric_table_reconciliation='PASS',
                limitation='Pattern scan supplements an explicit file review; it is not a guarantee about all possible secret formats')
    (ROOT/'results/release_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

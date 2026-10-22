"""Regenerate every research figure using public aggregate tables only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#27699C'; ORANGE='#C17B25'; GRAY='#66717A'
NAMES={'raptor384_single':'CoAtNet 384 single checkpoint', 'raptor224_single':'CoAtNet 224 single checkpoint',
       'r6_fourfold':'R6 four-fold ensemble', 'reader_threefold':'Reader three-fold ensemble',
       'raptor384_224':'384 + 224 rank ensemble', 'baseline_v13_v15':'V13/V15 baseline',
       'without_224':'Remove 224 branch','without_r6':'Remove R6 branch',
       'without_reader':'Remove Reader branch','equal_class_weights':'Equal Raptor/R6 class weights',
       'extra_kawaharataishi_swa':'Add SWA (fixed weights)'}

def finish(fig,path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=170,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)

def generate(root):
    root=Path(root);results=root/'results';out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#26333B','text.color':'#26333B',
                         'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'})
    ab=pd.read_csv(results/'ablation.csv').set_index('variant')
    names=list(NAMES);frame=ab.loc[names]
    fig,ax=plt.subplots(figsize=(10,6.3));y=np.arange(len(frame))
    for i,(key,row) in enumerate(frame.iterrows()):
        color=ORANGE if key=='extra_kawaharataishi_swa' else BLUE
        ax.plot([row.delta_ci_low,row.delta_ci_high],[i,i],color=color,lw=2)
        ax.scatter(row.delta_vs_baseline,i,color=color,s=45,zorder=3)
    ax.axvline(0,color=GRAY,lw=1,ls='--');ax.set_yticks(y,[NAMES[k] for k in names]);ax.invert_yaxis()
    ax.set_xlabel('Macro AUC difference vs V13/V15 baseline (focused delta axis)')
    ax.set_title('Frozen-ensemble ablation | Gold58 reused development cohort',loc='left',pad=18)
    fig.text(.02,.005,'Points: observed differences. Lines: conditional 95% paired study-bootstrap intervals (2,000 draws).',fontsize=9)
    finish(fig,out/'ablation_forest')

    pc=pd.read_csv(results/'per_class.csv');base=pc[pc.variant=='baseline_v13_v15'].set_index('label')
    swa=pc[pc.variant=='extra_kawaharataishi_swa'].set_index('label')
    order=base.sort_values('auc').index;y=np.arange(len(order));b=base.loc[order];s=swa.loc[order]
    fig,ax=plt.subplots(figsize=(10,6.6))
    ax.errorbar(b.auc,y-.12,xerr=np.vstack([b.auc-b.auc_ci_low,b.auc_ci_high-b.auc]),fmt='o',color=BLUE,
                label='V13/V15',capsize=2,markersize=5)
    ax.scatter(s.auc,y+.12,color=ORANGE,marker='s',label='+ SWA',s=30,zorder=3)
    ax.set_yticks(y,[f'{x}  (+{int(b.loc[x,"positives"])}/−{int(b.loc[x,"negatives"])})' for x in order])
    ax.set_xlim(0,1.02);ax.set_xlabel('ROC AUC');ax.set_title('Per-class discrimination | 58 studies, 12 findings',loc='left',pad=15)
    ax.legend(loc='lower left');ax.grid(axis='x',alpha=.15)
    fig.text(.02,.005,'Labels show positive/negative counts. Baseline bars: pointwise conditional 95% bootstrap intervals.',fontsize=9)
    finish(fig,out/'per_class_auc')

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6),sharey=True)
    labels=base.index;y=np.arange(len(labels));w=.35
    ax1.barh(y,base.recall,color=BLUE,height=.6)
    ax1.set_yticks(y,labels);ax1.set_xlim(0,1.03);ax1.invert_yaxis();ax1.set_xlabel('Recall = TP / (TP + FN)')
    ax1.set_title('Baseline recall',loc='left')
    ax2.barh(y-w/2,base.fp,height=w,label='False positives',color=ORANGE)
    ax2.barh(y+w/2,base.fn,height=w,label='False negatives',color=BLUE)
    ax2.set_xlabel('Number of study-label errors');ax2.set_title('Error counts',loc='left');ax2.legend(loc='lower right')
    fig.suptitle('Fixed percentile-rank cutoff ≥ 0.5 | descriptive, uncalibrated operating point',fontsize=13,y=1.02)
    finish(fig,out/'recall_and_errors')

    corr=pd.read_csv(results/'branch_correlations.csv').groupby('branch').spearman_with_384.mean()
    quality=pd.read_csv(results/'candidate_quality_cost.csv').set_index('branch')
    keys=['syouyatobita_g233','kawaharataishi_swa','tbongc']
    fig,ax=plt.subplots(figsize=(9,5))
    for k in keys:
        row=ab.loc['extra_'+k]
        ax.errorbar(corr[k],row.delta_vs_baseline,yerr=[[row.delta_vs_baseline-row.delta_ci_low],
                    [row.delta_ci_high-row.delta_vs_baseline]],fmt='o',markersize=7,
                    color=ORANGE if 'swa' in k else BLUE,capsize=3)
        ax.annotate(k.replace('kawaharataishi_',''),(corr[k],row.delta_vs_baseline),
                    xytext=(8,7),textcoords='offset points',fontsize=10)
    ax.axhline(0,color=GRAY,ls='--',lw=1);ax.set_xlim(.87,1)
    ax.set_xlabel('Mean per-label Spearman correlation with CoAtNet 384')
    ax.set_ylabel('Fixed ensemble macro AUC change');ax.set_title('Candidate complementarity | conditional development evidence',loc='left',pad=15)
    fig.text(.02,.005,'Bars: conditional 95% paired intervals. royalacecat omitted: correlation 0.052, single AUC 0.544; full data in CSV.',fontsize=8.6)
    finish(fig,out/'complementarity')

    view=pd.read_csv(results/'legacy_view_sensitivity.csv')
    view=view[~view.variant.isin(['view_native','view_r256'])]
    labels={'view_r320':'Resolution 320','view_dual_res':'256 + 320 average','view_flip256':'Flip TTA',
            'view_offset256':'Slice offset blend','view_k8':'8 slice positions','view_k24':'24 slice positions',
            'view_fov088':'FOV 0.88','view_fov096':'FOV 0.96','view_full_v5':'Full V5 view recipe'}
    fig,ax=plt.subplots(figsize=(9,5.7));y=np.arange(len(view))
    ax.errorbar(view.delta_vs_r256,y,xerr=np.vstack([view.delta_vs_r256-view.delta_ci_low,view.delta_ci_high-view.delta_vs_r256]),
                fmt='o',color=BLUE,capsize=3)
    ax.axvline(0,color=GRAY,lw=1,ls='--');ax.set_yticks(y,[labels[v] for v in view.variant]);ax.invert_yaxis()
    ax.set_xlabel('Macro AUC difference vs 256px / k16 / FOV 0.92 / no flip')
    ax.set_title('Supplement: archived Apex view sensitivity | a different model family',loc='left',pad=15)
    fig.text(.02,.005,'Cached image-inference outputs, not new corruption tests; conditional 95% paired intervals, n=58.',fontsize=9)
    finish(fig,out/'legacy_view_sensitivity')

    fig,ax=plt.subplots(figsize=(9,4.5));q=quality.sort_values('seconds_per_study')
    bars=ax.barh(q.index,q.seconds_per_study,color=[ORANGE if 'swa' in x else BLUE for x in q.index])
    for bar,v in zip(bars,q.seconds_per_study):ax.text(v+.05,bar.get_y()+bar.get_height()/2,f'{v:.2f}',va='center')
    ax.set_xlim(0,float(q.seconds_per_study.max())*1.18)
    ax.set_xlabel('Archived branch wall time / 58 studies (seconds per study, T4 ×2)')
    ax.set_title('Candidate inference cost | same 58-study screen',loc='left',pad=15)
    fig.text(.02,.005,'Different precision/configuration per checkpoint. Startup/caching included; not isolated latency or an end-to-end speedup.',fontsize=9)
    finish(fig,out/'candidate_runtime')
    return sorted(out.glob('*.png'))

if __name__=='__main__':
    generate(Path(__file__).resolve().parents[1])

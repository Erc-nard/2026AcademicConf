"""Rebuild the report figure from immutable public numerical results."""
import os
from pathlib import Path

from .meld_experiment import read


def main():
    os.environ.setdefault('MPLCONFIGDIR',str(Path('results/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='2026AcademicConf-report'
    scores=read(Path('experiments/meld_heldout_evaluation_2026-10-04.json'))['scores']
    stats=read(Path('experiments/report_final_analysis_2026-10-17.json'))
    masks=['0','25','50'];labels=['Clean','25% mask','50% mask']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    values=[scores[m]['C0']['wer']*100 for m in masks]
    bars=axes[0].bar(labels,values,color=['#35679a','#7c94b8','#a6b9d1'],width=.58)
    axes[0].bar_label(bars,fmt='%.2f%%',padding=4,fontsize=10)
    axes[0].set(ylim=(0,130),ylabel='Corpus WER (%)',title='A. Visual-only baseline (20 utterances)')
    axes[0].axhline(100,color='#999999',lw=.8,ls=':')
    for i,m in enumerate(masks):
        v=stats['vs_raw'][m]['C1'];point=v['delta_wer_pp'];lo,hi=v['percentile_95_interval_pp']
        axes[1].errorbar(i,point,yerr=[[point-lo],[hi-point]],fmt='o',color='#b05241',capsize=6,lw=2)
        axes[1].annotate(f'{point:+.2f}',(i,point),xytext=(0,10),textcoords='offset points',ha='center')
    axes[1].axhline(0,color='#777777',lw=1,ls='--')
    axes[1].set(xticks=range(3),xticklabels=labels,ylim=(-2.6,.6),ylabel='WER difference (percentage points)',
                title='B. Previous VSR context minus baseline')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.supxlabel('B: paired episode bootstrap, 10,000 draws; exploratory 95% intervals. Negative favors context.',fontsize=9)
    folder=Path('docs/figures');folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'report_main_results.png',dpi=180)
    svg=folder/'report_main_results.svg'
    fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)


if __name__=='__main__':main()

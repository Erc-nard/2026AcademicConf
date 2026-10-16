"""Final paired uncertainty and visual-availability sensitivity analysis; no model calls."""
import argparse
from collections import Counter
from pathlib import Path
import random

from .meld_experiment import read,save,sha

MAIN=Path('experiments/meld_heldout_evaluation_2026-10-04.json')
PLAN=Path('experiments/meld_heldout_final_plan_2026-10-03.json')


def indexed(metric):
    result={x['id']:x for x in metric['details']}
    if len(result)!=len(metric['details']):raise ValueError('Duplicate metric ID')
    if not result or any(x['reference_words']<=0 for x in result.values()):raise ValueError('Invalid reference denominator')
    for total in ['word_errors','reference_words','char_errors','reference_chars']:
        if sum(x[total] for x in result.values())!=metric[total]:raise ValueError('Inconsistent corpus total')
    return result


def percentile(values,q):
    values=sorted(values);position=(len(values)-1)*q;low=int(position);fraction=position-low
    return values[low]*(1-fraction)+values[min(low+1,len(values)-1)]*fraction


def paired(before,after,groups,draws=10000,seed=20261017):
    """Resample episode clusters, retaining paired conditions and word denominators."""
    a=indexed(before);b=indexed(after)
    if set(a)!=set(b) or set(a)!=set(groups):raise ValueError('Unpaired IDs')
    if any(a[k]['reference_words']!=b[k]['reference_words'] for k in a):raise ValueError('Changed paired denominator')
    totals={}
    for key in sorted(a):
        delta,words=totals.get(groups[key],(0,0))
        totals[groups[key]]=(delta+b[key]['word_errors']-a[key]['word_errors'],words+a[key]['reference_words'])
    units=list(totals.values());rng=random.Random(seed);values=[]
    for _ in range(draws):
        sample=[units[rng.randrange(len(units))] for _ in units]
        values.append(100*sum(x[0] for x in sample)/sum(x[1] for x in sample))
    return dict(delta_wer_pp=100*sum(x[0] for x in units)/sum(x[1] for x in units),
        percentile_95_interval_pp=[percentile(values,.025),percentile(values,.975)],clusters=len(units),samples=len(a),
        improved=sum(b[k]['word_errors']<a[k]['word_errors'] for k in a),
        same=sum(b[k]['word_errors']==a[k]['word_errors'] for k in a),
        worsened=sum(b[k]['word_errors']>a[k]['word_errors'] for k in a))


def subset(metric,keys):
    rows=indexed(metric);selected=[rows[k] for k in sorted(keys)]
    if not selected:raise ValueError('Empty subgroup')
    result={field:sum(x[field] for x in selected) for field in ['word_errors','reference_words','char_errors','reference_chars']}
    return dict(result,samples=len(selected),wer=result['word_errors']/result['reference_words'],
                cer=result['char_errors']/result['reference_chars'],details=selected)


def run(output):
    report=read(MAIN);plan=read(PLAN);scores=report['scores'];keys=set(plan['pairs'])
    groups={r['id']:f"S{r['season']}E{r['episode']}" for r in plan['selected']}
    sources=[MAIN,PLAN,Path('experiments/meld_heldout_analysis_2026-10-04.json'),
             Path('experiments/meld_heldout_provenance_2026-10-04.json')]
    # Local completion and frozen-output verification without reading target transcript strings.
    root=Path('results/context_heldout_2026-10-03');rows=read(root/'comparison/outputs.json')
    from .meld_heldout_evaluation import validate_trials
    validate_trials(rows,plan['pairs'])
    if sha(root/'comparison/outputs.json')!=report['output_sha256']:raise ValueError('Changed scored outputs')
    if sha(Path(report['plan']['config']['vsr']))!=report['plan']['vsr_sha256']:raise ValueError('Changed VSR')
    if sha(Path(report['plan']['config']['visual_contexts']))!=report['plan']['visual_sha256']:raise ValueError('Changed scenes')
    uncertainty={m:{c:paired(table['C0'],metric,groups) for c,metric in table.items() if c!='C0'} for m,table in scores.items()}
    controls={m:dict(real_vs_wrong=paired(t['C4'],t['C1'],groups),
                     combined_vs_dialogue=paired(t['C1'],t['C3'],groups)) for m,t in scores.items()}
    failed=set(report['visual_failed_ids']);subgroups={}
    for name,ids in [('visual_success',keys-failed),('visual_failed',failed)]:
        subgroups[name]={m:{c:subset(metric,ids) for c,metric in table.items()} for m,table in scores.items()}
    result=dict(scope='Post-hoc final report analysis of frozen twenty targets; no model calls or retuning',
        source_hashes={str(p):sha(p) for p in sources},source_sha256=sha(Path(__file__)),
        method=dict(unit='season+episode',clusters=len(set(groups.values())),targets=len(keys),draws=10000,seed=20261017,
                    interval='2.5/97.5 linear percentiles of paired cluster bootstrap corpus-WER differences in percentage points',
                    multiplicity='Exploratory intervals, no simultaneous coverage or confirmatory p-values',
                    caution='Zero-width interval for identical observed errors does not establish population equivalence'),
        audit=dict(output_rows=len(rows),model_calls=sum(r['condition']!='C0' for r in rows),
                   execution_failures=sum(r['status']=='failed' for r in rows),groups=groups,
                   status_counts=dict(Counter(r['status'] for r in rows)),outputs_sha256=report['output_sha256']),
        vs_raw=uncertainty,context_controls=controls,
        masking=dict(mask25_vs_clean=paired(scores['0']['C0'],scores['25']['C0'],groups),
                     mask50_vs_clean=paired(scores['0']['C0'],scores['50']['C0'],groups)),
        visual_availability_subgroups=subgroups,
        limitations=['Small fixed selection from MELD train, not official test',
                     'Visual success is parser completion, not verified semantic accuracy',
                     'Subgroups were analyzed after outcomes; descriptive sensitivity only',
                     'Same show; clusters may retain cross-episode dependence'])
    if output.exists():raise ValueError('Preserve previous final analysis')
    save(output,result)
    print('clusters',result['method']['clusters'])
    print('C1 vs raw', {m:r['C1'] for m,r in uncertainty.items()})
    print('masking',result['masking'])
    print('visual success WER',{m:{c:round(t[c]['wer']*100,2) for c in ['C0','C1','C2','C3']} for m,t in subgroups['visual_success'].items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('experiments/report_final_analysis_2026-10-17.json'))
    run(parser.parse_args().output)

"""Reference-free edit-budget rejection on frozen, previously generated subtitles."""
import argparse
from pathlib import Path

from . import meld_experiment as base
from .evaluation import distance, evaluate

SOURCE = Path('results/meld_prompt_ablation_2026-09-28')
BUDGETS = {'zero':0, 'one':1, 'two':2, 'four':4, 'unlimited':None}


def gate(current, candidate, status, budget):
    """Count insertions + deletions + substitutions against current VSR, not truth."""
    if budget is not None and (type(budget) is not int or budget<0):
        raise ValueError('Budget must be a nonnegative integer or None')
    edits=distance(base.normalize_english(current).split(),base.normalize_english(candidate).split())
    accepted=status=='ok' and edits>0 and (budget is None or edits<=budget)
    reason='accepted' if accepted else ('generation_failure' if status!='ok' else 'no_word_change' if edits==0 else 'over_budget')
    return dict(hypothesis=candidate if accepted else current,edits=edits,accepted=accepted,reason=reason)


def metric(refs, rows):
    result=evaluate([dict(id=r['id'],reference=base.normalize_english(r['reference'])) for r in refs],
                    [dict(id=r['id'],hypothesis=base.normalize_english(r['hypothesis'])) for r in rows])
    result['normalization']=base.normalize_english.__doc__+'; CER excludes spaces'
    return result


def decision_counts(rows):
    return {reason:sum(r['reason']==reason for r in rows) for reason in ['accepted','no_word_change','over_budget','generation_failure']}


def run(output, public_output=None):
    output.mkdir(parents=True,exist_ok=False)
    prior=base.read(SOURCE/'summary.json')
    if base.sha(SOURCE/'meld/corrections.json')!=prior['meld']['correction_output_sha256']:
        raise ValueError('Frozen correction outputs changed')
    plan=dict(source_summary_sha256=base.sha(SOURCE/'summary.json'),source_corrections_sha256=base.sha(SOURCE/'meld/corrections.json'),
              source_synthetic_sha256=base.sha(SOURCE/'synthetic_outputs.json'),budgets=BUDGETS,primary_budget='one',
              distance='Normalized word Levenshtein distance; insertion/deletion/substitution cost 1',
              acceptance='Successful generation and 0 < distance <= budget; unlimited has no upper bound',
              fallback='Keep original VSR verbatim; cosmetic-only edits also keep original',
              scope='Post-hoc development diagnostic on the same known five MELD targets, not held-out evaluation',
              selection='All fixed budgets reported; one edit is the primary condition, not chosen by resulting WER',
              model_calls=0,reference_used_by_gate=False,conditions=base.CONDITIONS,masks=[0,25,50])
    base.save(output/'plan.json',plan)
    source=base.read(SOURCE/'meld/corrections.json')
    if len(source)!=75 or len({(r['id'],r['mask'],r['condition']) for r in source})!=75:
        raise ValueError('Expected 75 unique fixed trials')
    decisions=[]
    for row in source:
        for name,budget in BUDGETS.items():
            decisions.append(dict(id=row['id'],mask=row['mask'],condition=row['condition'],budget=name,
                                  **gate(row['input']['current_subtitle'],row['hypothesis'],row['status'],budget)))
    base.save(output/'decisions.json',decisions)
    lines=['# Local edit-budget decisions','', 'Only proposals with normalized word changes; generated text remains local.','',
           '| ID | Mask | Context | Original VSR | Proposed correction | Word edits | Accepted budgets |',
           '|---|---:|---|---|---|---:|---|']
    for row in source:
        matched=[d for d in decisions if (d['id'],d['mask'],d['condition'])==(row['id'],row['mask'],row['condition'])]
        if matched[0]['edits']==0:continue
        values=[row['id'],str(row['mask']),row['condition'],row['input']['current_subtitle'],row['hypothesis'],
                str(matched[0]['edits']),', '.join(d['budget'] for d in matched if d['accepted'])]
        lines.append('| '+' | '.join(v.replace('|','\\|').replace('\n',' ') for v in values)+' |')
    (output/'sample_decisions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # The MELD target reference is first loaded only after all decisions are saved.
    refs=[r for r in base.read(base.SOURCE/'target_references.json') if r['id'] in base.PAIRS]
    scores={}; comparisons={}; acceptance={}
    for name in BUDGETS:
        scores[name]={};comparisons[name]={};acceptance[name]={}
        for m in [0,25,50]:
            scores[name][str(m)]={};comparisons[name][str(m)]={};acceptance[name][str(m)]={}
            for c in base.CONDITIONS:
                rows=[r for r in decisions if r['budget']==name and r['mask']==m and r['condition']==c]
                score=metric(refs,rows);scores[name][str(m)][c]=score
                comparisons[name][str(m)][c]=dict(vs_raw=base.comparison(prior['meld']['scores'][str(m)]['raw_vsr'],score),
                                                 vs_unlimited=base.comparison(prior['meld']['scores'][str(m)][c],score))
                acceptance[name][str(m)][c]=decision_counts(rows)
    for m in ['0','25','50']:
        for c in base.CONDITIONS:
            for name,expected in [('zero',prior['meld']['scores'][m]['raw_vsr']),('unlimited',prior['meld']['scores'][m][c])]:
                if scores[name][m][c]!=expected:raise ValueError('Zero or unlimited control does not reproduce source scores')
    synth=base.read(SOURCE/'synthetic_outputs.json')
    synth=[r for r in synth if r['prompt']=='active']
    if len(synth)!=12:raise ValueError('Expected 12 active synthetic trials')
    synth_refs=[dict(id=r['id'],reference=r['reference']) for r in synth]
    raw=metric(synth_refs,[dict(id=r['id'],hypothesis=r['current']) for r in synth])
    synthetic={};synth_decisions=[]
    for name,budget in BUDGETS.items():
        rows=[dict(id=r['id'],budget=name,**gate(r['current'],r['hypothesis'],r['status'],budget)) for r in synth]
        score=metric(synth_refs,rows)
        synthetic[name]=dict(scores=score,vs_raw=base.comparison(raw,score),decisions=decision_counts(rows))
        synth_decisions.extend(rows)
    base.save(output/'synthetic_decisions.json',synth_decisions)
    public=dict(plan=plan,plan_sha256=base.sha(output/'plan.json'),scores=scores,comparisons=comparisons,acceptance=acceptance,
                synthetic=synthetic,meld_decisions=len(decisions),synthetic_decisions=len(synth_decisions),
                source_generation_failures=prior['generation_failures'],zero_and_unlimited_controls_reproduced=True,
                decision_sha256=base.sha(output/'decisions.json'),synthetic_decision_sha256=base.sha(output/'synthetic_decisions.json'))
    base.save(output/'summary.json',public)
    if public_output is not None:base.save(public_output,public)
    for name in BUDGETS:
        print(name,{c:round(v['wer'],4) for c,v in scores[name]['0'].items()},flush=True)
        print('synthetic',name,synthetic[name]['scores']['wer'],synthetic[name]['decisions'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--public-output',type=Path)
    args=parser.parse_args();run(args.output,args.public_output)

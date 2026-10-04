"""Post-evaluation case counts without publishing subtitle/reference strings."""
import argparse
from collections import Counter
from pathlib import Path

from .meld_experiment import read,save,sha,normalize_english,comparison


def analyze(folder,public_output):
    summary=read(folder/'summary.json');rows=read(folder/'outputs.json')
    if sha(folder/'outputs.json')!=summary['output_sha256']:raise ValueError('Scored outputs changed')
    scores=summary['scores'];ungated=summary['ungated_scores'];cases=[]
    for row in rows:
        before=normalize_english(row['input']['current_vsr'])
        proposed=normalize_english(row['candidate']);after=normalize_english(row['hypothesis'])
        if before==proposed:continue
        mask=str(row['mask']);condition=row['condition'];key=row['id']
        def errors(table,c):
            return next(x for x in table[mask][c]['details'] if x['id']==key)
        baseline=errors(scores,'C0');raw=errors(ungated,condition);final=errors(scores,condition)
        visual=row['input']['visual_context']
        cases.append(dict(id=key,mask=row['mask'],condition=condition,
            proposed_word_edits=row['decision']['edits'],accepted=row['decision']['accepted'],
            decision=row['decision']['reason'],final_word_changed=after!=before,
            baseline_word_errors=baseline['word_errors'],proposed_word_errors=raw['word_errors'],
            final_word_errors=final['word_errors'],reference_words=baseline['reference_words'],
            final_char_error_delta=final['char_errors']-baseline['char_errors'],
            visual_status=row['visual_status'],visual_fields_present=[] if visual is None else [k for k,v in visual.items() if v]))
    contexts=read(Path(summary['plan']['config']['visual_contexts']))
    report=dict(scope=summary['plan']['config']['scope'],outputs_sha256=summary['output_sha256'],
                changed_proposals=len(cases),accepted_word_changes=sum(c['final_word_changed'] for c in cases),
                rejected_changed_proposals=sum(not c['accepted'] for c in cases),cases=cases,
                ungated_comparisons={m:{c:comparison(scores[m]['C0'],v) for c,v in table.items()} for m,table in ungated.items()},
                visual_success=sum(r['status']=='ok' for r in contexts),visual_failures=sum(r['status']!='ok' for r in contexts),
                visual_errors=dict(Counter(r['error'] for r in contexts if r['status']!='ok')),
                visual_information_present=sum(r['status']=='ok' and bool(r['parsed_json']['scene'] or r['parsed_json']['objects'] or r['parsed_json']['actions']) for r in contexts),
                limitations=['Twenty project-held-out MELD train dialogues; not official test split',
                             'Same base show and model pretraining overlap not ruled out',
                             'No context-field ablation; cases are associations, not causal attribution',
                             'Masking and VLM failures retained; no post-score sample replacement'])
    save(folder/'case_analysis.json',report);save(public_output,report)
    print({k:report[k] for k in ['changed_proposals','accepted_word_changes','rejected_changed_proposals','visual_success','visual_failures']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--public-output',type=Path,required=True)
    args=parser.parse_args();analyze(args.output,args.public_output)

"""Held-out generation inputs and reference access only after complete generation."""
import argparse
import csv
from pathlib import Path

from . import meld_experiment as base
from . import context_comparison as comparison

METADATA=Path('external/meld/data/MELD/train_sent_emo.csv')


def validate_trials(rows,pairs):
    expected={(k,m,c) for k in pairs for m in [0,25,50] for c in comparison.CONDITIONS}
    actual={(r['id'],r['mask'],r['condition']) for r in rows}
    if len(rows)!=len(expected) or actual!=expected:
        raise ValueError('All held-out generation must finish before target-reference access')


def project_texts(path,keys):
    """Dataset ingestion projection; expose text only for the requested IDs."""
    result={}
    with path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            key=f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            if key in keys:
                if key in result:raise ValueError('Duplicate requested metadata ID')
                result[key]=row['Utterance']
    if set(result)!=set(keys):raise ValueError('Missing requested metadata ID')
    return result


def previous(output):
    plan=base.read(output/'final_plan.json');pairs=plan['pairs']
    if set(pairs)&set(pairs.values()):raise ValueError('Held-out previous texts must be disjoint from all targets')
    path=output/'oracle_previous_context.json'
    if path.exists():raise ValueError('Do not overwrite previous-only projection')
    texts=project_texts(METADATA,set(pairs.values()))
    base.save(path,[dict(id=k,previous_id=p,previous_text=texts[p]) for k,p in pairs.items()])


def config(output,scene_output=None):
    final=base.read(output/'final_plan.json');settings=base.read(output/'settings_plan.json')
    if base.sha(output/'settings_plan.json')!=final['settings_sha256']:raise ValueError('Frozen settings changed')
    for name,digest in settings['source_hashes'].items():
        if base.sha(Path(name))!=digest:raise ValueError('Frozen generation source changed')
    vsr=output/'final_vsr/vsr.json';visual=(scene_output or output/'scenes_final')/'contexts.json'
    pairs=final['pairs'];rows=base.read(vsr)
    expected={(k,m) for k in pairs for m in [0,25,50]}|{(k,0) for k in pairs.values()}
    if len(rows)!=len(expected) or {(r['id'],r['mask']) for r in rows}!=expected:raise ValueError('VSR incomplete')
    contexts=base.read(visual)
    if len(contexts)!=len(pairs) or {r['id'] for r in contexts}!=set(pairs):raise ValueError('Scene generation incomplete')
    oracle=base.read(output/'oracle_previous_context.json')
    if {r['previous_id'] for r in oracle}!=set(pairs.values()):raise ValueError('Previous-only oracle IDs changed')
    path=output/'comparison_config.json'
    if path.exists():raise ValueError('Do not overwrite generation configuration')
    base.save(path,dict(scope='MELD train project-held-out dialogues and episodes; frozen twenty pairs',
                       pairs=pairs,wrong=final['wrong'],vsr=str(vsr),expected_vsr_sha256=base.sha(vsr),
                       visual_contexts=str(visual),oracle_previous=str(output/'oracle_previous_context.json'),
                       final_plan_sha256=base.sha(output/'final_plan.json'),settings_sha256=base.sha(output/'settings_plan.json')))


def evaluate(output,public_output=None,metadata=METADATA):
    pairs=base.read(output/'final_plan.json')['pairs'];folder=output/'comparison'
    rows=base.read(folder/'outputs.json')
    validate_trials(rows,pairs)  # Must precede opening metadata/target texts.
    plan=base.read(folder/'plan.json')
    if base.sha(Path(plan['config']['vsr']))!=plan['vsr_sha256']:raise ValueError('VSR evidence changed')
    if base.sha(Path(plan['config']['visual_contexts']))!=plan['visual_sha256']:raise ValueError('Visual evidence changed')
    references=output/'target_references.json'
    if references.exists():raise ValueError('Evaluation already started; preserve original reference-access record')
    texts=project_texts(metadata,set(pairs))
    base.save(references,[dict(id=k,reference=texts[k]) for k in pairs])
    base.save(output/'reference_access.json',dict(generation_rows=len(rows),outputs_sha256=base.sha(folder/'outputs.json'),
              metadata_sha256=base.sha(metadata),references_sha256=base.sha(references),
              access_order='All target/mask/context outputs validated before target text projection'))
    comparison.evaluate_run(folder,references,public_output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['previous','config','evaluate'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--public-output',type=Path)
    parser.add_argument('--scene-output',type=Path)
    args=parser.parse_args()
    if args.stage=='previous':previous(args.output)
    elif args.stage=='config':config(args.output,args.scene_output)
    else:evaluate(args.output,args.public_output)

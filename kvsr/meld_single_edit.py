"""Generate at most one-word edit, then enforce the existing reference-free gate."""
import argparse
from pathlib import Path

from . import meld_experiment as base
from . import meld_prompt_ablation as generation
from . import meld_edit_budget as filtering

LIMIT = (' Make at most ONE word-level insertion, deletion, or substitution in the current subtitle. '
         'All other words must remain in the same order. If a useful correction would require '
         'more than one word edit, return the current subtitle unchanged. '
         'Do not append words from the previous utterance merely to make a longer sentence.')


def oracle_limit(metric):
    """Ideal one-edit bounds from reference distance, not achieved model performance."""
    errors=[r['word_errors'] for r in metric['details']]
    minimum=sum(max(e-1,0) for e in errors)
    return dict(minimum_possible_word_errors=minimum,minimum_possible_wer=minimum/metric['reference_words'],
                maximum_possible_exact_sentences=sum(e<=1 for e in errors),samples=len(errors),
                interpretation='Perfect reference-aware one-word edits; unavailable to the real gate/model')


def run(output, port, public_output=None):
    output.mkdir(parents=True,exist_ok=False)
    prior=base.read(Path('results/meld_prompt_ablation_2026-09-28/summary.json'))
    previous_gate=base.read(Path('results/meld_edit_budget_2026-09-28/summary.json'))
    prompt=generation.ACTIVE_PROMPT+LIMIT
    plan=dict(prompt=prompt,change='Append one-word edit instruction to the fixed active system prompt',
              primary='Generate once then apply existing one-word edit gate; no retries or candidate selection',
              previous_generation_summary_sha256=base.sha(Path('results/meld_prompt_ablation_2026-09-28/summary.json')),
              previous_gate_summary_sha256=base.sha(Path('results/meld_edit_budget_2026-09-28/summary.json')),
              calls=dict(synthetic=24,meld=75,baseline_replay=3,vsr=0),
              scope='Known five-target development diagnostic; no held-out claim',
              controls='Report ungated generation and gated output, including all masks and five contexts')
    base.save(output/'plan.json',plan)
    generation.ACTIVE_PROMPT=prompt
    generation.run(output/'generation',port)
    generation.validate_one_factor(base.read(Path('results/meld_prompt_ablation_2026-09-28/meld/corrections.json')),
                                   base.read(output/'generation/meld/corrections.json'))
    filtering.SOURCE=output/'generation'
    filtering.run(output/'gate')
    new=base.read(output/'generation/summary.json')
    gated=base.read(output/'gate/summary.json')
    comparisons={m:{c:dict(
        ungated_vs_previous_active=base.comparison(prior['meld']['scores'][m][c],new['meld']['scores'][m][c]),
        gated_vs_previous_gate=base.comparison(previous_gate['scores']['one'][m][c],gated['scores']['one'][m][c]),
        gated_vs_raw=base.comparison(prior['meld']['scores'][m]['raw_vsr'],gated['scores']['one'][m][c]))
        for c in base.CONDITIONS} for m in ['0','25','50']}
    public=dict(plan=plan,plan_sha256=base.sha(output/'plan.json'),generation=new,active_request_fields_verified=True,
                one_edit_scores=gated['scores']['one'],one_edit_acceptance=gated['acceptance']['one'],
                synthetic_one_edit=gated['synthetic']['one'],comparisons=comparisons,
                theoretical_oracle_limits={m:oracle_limit(prior['meld']['scores'][m]['raw_vsr']) for m in ['0','25','50']},
                generation_summary_sha256=base.sha(output/'generation/summary.json'),gate_summary_sha256=base.sha(output/'gate/summary.json'))
    base.save(output/'summary.json',public)
    if public_output is not None:base.save(public_output,public)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--public-output',type=Path)
    args=parser.parse_args();run(args.output,args.port,args.public_output)

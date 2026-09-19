"""Frozen synthetic occlusion suite; no thresholds fitted on evaluation references."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .corruption import CONDITIONS
from .preprocessing import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'repo', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(conditions=list(CONDITIONS), decoder='ctc_beam', beam_size=25,
                source_plan_sha256=sha256(args.source / 'plan.json'),
                operation='mask uint8 resized input with 127 before per-clip mean/std normalization',
                spatial='central full-width horizontal band, 25%/50% of height, every frame',
                temporal='middle 25% of frames fully masked, nearest integer frame count',
                purpose='synthetic robustness and uncalibrated diagnostic probe; not real occlusion accuracy')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    summaries, failures = {}, {}
    for condition in CONDITIONS:
        output = args.output / condition
        command = [sys.executable, '-m', 'kvsr.nbest_experiment', '--source', str(args.source),
                   '--checkpoint', str(args.checkpoint), '--repo', str(args.repo),
                   '--output', str(output), '--corruption', condition]
        try:
            subprocess.run(command, timeout=1500, check=True)
            summaries[condition] = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        except subprocess.SubprocessError as error:
            failures[condition] = str(error)
        report = dict(plan=plan, conditions=summaries, failed_conditions=failures,
                      complete=len(summaries) == len(CONDITIONS))
        (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f'Condition {condition}: {"OK" if condition in summaries else "FAILED"}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

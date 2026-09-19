"""Replay a frozen crop experiment with reference-free N-best decoding."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .data import read_jsonl
from .evaluation import evaluate
from .nbest import evaluate_nbest
from .preprocessing import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=25)
    parser.add_argument("--decoder", choices=["ctc_beam", "hybrid_rescore"], default="ctc_beam")
    parser.add_argument("--corruption", choices=["none", "spatial25", "spatial50", "temporal25"], default="none")
    args = parser.parse_args()
    if args.beam_size < 1:
        parser.error("beam size must be positive")
    source = json.loads((args.source / "plan.json").read_text(encoding="utf-8"))
    if source["split"] != "train":
        parser.error("This development runner accepts train experiments only")
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    inputs = [dict(id=r["id"], mouth=str((args.source / "crops" / "scale1.7_none" /
                  r["utterance"] / "mouth.npz").resolve())) for r in source["selected"]]
    for row in inputs:
        row["sha256"] = sha256(row["mouth"])
    save("plan.json", dict(scope=source["scope"], source_plan_sha256=sha256(args.source / "plan.json"),
                           inputs=inputs, beam_size=args.beam_size, ks=[1, 5, 10, 25, 100],
                           decoder=args.decoder, corruption=args.corruption, size=96, target_fps=None,
                           hybrid_settings=dict(attention_beam=5, max_tokens=128, ctc_weight=.6)
                               if args.decoder == "hybrid_rescore" else None,
                           checkpoint_sha256=sha256(args.checkpoint),
                           reference_used_in_decoding=False, language_model_used=False))
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", MPLCONFIGDIR=str((args.output / "matplotlib").resolve()))
    predictions, failures = [], []
    for row in inputs:
        stem = row["id"].split("/")[-1]
        output = args.output / f"{stem}.jsonl"
        command = [sys.executable, "-m", "kvsr", "infer-olkavs", "--mouth", row["mouth"],
                   "--checkpoint", str(args.checkpoint), "--repo", str(args.repo), "--id", row["id"],
                   "--output", str(output), "--decoder", args.decoder, "--beam-size", str(args.beam_size),
                   "--corruption", args.corruption]
        try:
            with (args.output / f"{stem}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                               timeout=300, check=True)
            decoded = read_jsonl(output)
            if len(decoded) != 1 or decoded[0]["id"] != row["id"]:
                raise ValueError("Decoder returned mismatched IDs")
            predictions.extend(decoded)
        except (subprocess.SubprocessError, OSError) as error:
            failures.append(dict(id=row["id"], error=str(error)))
        save("progress.json", dict(completed=len(predictions), failures=failures, total=len(inputs)))
        print(f"Completed {len(predictions)} / {len(inputs)}; failures {len(failures)}", flush=True)
    # Read references only after all decoder subprocesses have exited.
    references = read_jsonl(args.source / "references.jsonl")
    if failures:
        save("summary.json", dict(status="incomplete; no partial accuracy claim", failures=failures))
        raise SystemExit(1)
    summary = evaluate_nbest(references, predictions, ks=(1, 5, 10, 25, 100))
    if args.decoder == "hybrid_rescore":
        summary["hybrid_search"] = {r["id"]: r["hybrid_settings"] for r in predictions}
        summary["ctc_comparison"] = evaluate_nbest(references,
            [dict(id=r["id"], candidates=r["ctc_candidates"]) for r in predictions])
    greedy = [dict(id=r["id"], hypothesis=r["greedy_hypothesis"]) for r in predictions]
    summary.update(greedy=evaluate(references, greedy), scope=source["scope"],
                   diagnostics={r["id"]: r.get("ctc_diagnostics") for r in predictions},
                   corruption={r["id"]: r.get("corruption") for r in predictions},
                   plan_sha256=sha256(args.output / "plan.json"),
                   beam_size=args.beam_size, coverage=dict(selected=len(inputs), succeeded=len(predictions), failed=0),
                   provenance=[{k: r[k] for k in ("id", "input_sha256", "checkpoint_sha256",
                       "upstream_commit", "vocabulary_sha256", "torch_version", "source_fps", "input_size",
                       "crop_scale", "alignment", "reference_used", "audio_used")} for r in predictions])
    save("summary.json", summary)


if __name__ == "__main__":
    main()

"""Reference-free CTC prefix beam search and separate oracle evaluation."""

import math

from .evaluation import distance, evaluate, indexed, normalize


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def prefix_beam(log_probs, beam_size=25, blank=3):
    """Return token prefixes and approximate log mass, without a language model.

    All vocabulary tokens participate. EOS/special-token rendering happens later.
    Beam pruning loses mass; scores are not calibrated confidence values.
    """
    if not isinstance(beam_size, int) or beam_size < 1:
        raise ValueError("beam_size must be a positive integer")
    beam = {(): (0., -math.inf)}
    width = None
    for frame in log_probs:
        if width is None:
            width = len(frame)
        if len(frame) != width or not 0 <= blank < width:
            raise ValueError("Invalid vocabulary dimensions or blank")
        if any(math.isnan(p) or p > 0 for p in frame) or all(p == -math.inf for p in frame):
            raise ValueError("Expected finite or negative-infinity log probabilities")
        next_beam = {}

        def add(prefix, slot, value):
            if value == -math.inf:
                return
            pair = next_beam.setdefault(prefix, [-math.inf, -math.inf])
            pair[slot] = logadd(pair[slot], value)

        for prefix, (pb, pn) in beam.items():
            total = logadd(pb, pn)
            add(prefix, 0, total + frame[blank])
            for token, prob in enumerate(frame):
                if token == blank:
                    continue
                if prefix and token == prefix[-1]:
                    add(prefix, 1, pn + prob)
                    add(prefix + (token,), 1, pb + prob)
                else:
                    add(prefix + (token,), 1, total + prob)
        beam = dict(sorted(next_beam.items(), key=lambda item: (-logadd(*item[1]), item[0]))[:beam_size])
    return [(list(p), logadd(*scores)) for p, scores in
            sorted(beam.items(), key=lambda item: (-logadd(*item[1]), item[0]))]


def render_candidates(prefixes, vocabulary, compose):
    """Deduplicate rendered text, preserving the highest-scoring prefix order."""
    seen, candidates = set(), []
    for prefix, score in prefixes:
        tokens = []
        for token in prefix:
            if token == 2:
                break
            if token not in {0, 1, 3}:
                tokens.append(token)
        graphemes = "".join(vocabulary[t] for t in tokens)
        hypothesis = normalize(compose(graphemes))
        if hypothesis in seen:
            continue
        seen.add(hypothesis)
        candidates.append(dict(hypothesis=hypothesis, token_ids=tokens,
                               ctc_prefix=prefix, raw_graphemes=graphemes,
                               log_mass=score))
    return candidates


def evaluate_nbest(references, predictions, ks=(1, 5, 10, 25)):
    """Oracle uses references only here; selects by CER, ties by model rank."""
    refs = indexed(references, "reference")
    rows = {}
    for row in predictions:
        if row["id"] in rows or not row["candidates"]:
            raise ValueError("Duplicate prediction ID or empty candidate list")
        for candidate in row["candidates"]:
            normalize(candidate["hypothesis"])
        rows[row["id"]] = row
    if not refs or refs.keys() != rows.keys():
        raise ValueError("Reference and prediction ID sets must match exactly")
    if any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("K must be positive")
    reports = {}
    for k in ks:
        selected, ranks = [], {}
        for sample_id, reference in refs.items():
            candidates = rows[sample_id]["candidates"][:k]
            rank = min(range(len(candidates)), key=lambda i: distance(
                reference.replace(" ", ""), normalize(candidates[i]["hypothesis"]).replace(" ", "")))
            selected.append(dict(id=sample_id, hypothesis=candidates[rank]["hypothesis"]))
            ranks[sample_id] = rank + 1
        reports[str(k)] = dict(evaluate(references, selected), selected_ranks=ranks)
    return dict(oracle_by_k=reports,
                candidate_counts={key: len(row["candidates"]) for key, row in rows.items()},
                interpretation="Reference-assisted CER oracle; not deployable accuracy. WER uses CER-selected candidate.")

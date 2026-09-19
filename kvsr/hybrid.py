"""Two-pass candidate union and CTC/attention rescoring, without references."""

import math


def attention_beam(next_scores, beam_size=5, nbest=25, max_tokens=128, eos=2,
                   excluded=(0, 1, 3)):
    """Bounded autoregressive search. Scores include EOS, without length bonus."""
    if min(beam_size, nbest, max_tokens) < 1:
        raise ValueError("Search limits must be positive")
    active, completed = [((), 0.)], []
    stopped = False
    for step in range(max_tokens + 1):
        scores = next_scores([p for p, _ in active])
        if len(scores) != len(active):
            raise ValueError("Score batch size mismatch")
        expanded = []
        for (prefix, previous), frame in zip(active, scores):
            if not 0 <= eos < len(frame) or any(math.isnan(s) or s > 1e-6 for s in frame):
                raise ValueError("Invalid attention log probabilities")
            if math.isfinite(frame[eos]):
                completed.append((prefix, previous + frame[eos]))
            if step < max_tokens:
                choices = sorted((t for t in range(len(frame)) if t not in excluded and t != eos
                                  and math.isfinite(frame[t])), key=lambda t: (-frame[t], t))[:beam_size]
                expanded.extend((prefix + (t,), previous + frame[t]) for t in choices)
        completed = sorted(completed, key=lambda x: (-x[1], x[0]))[:nbest]
        active = sorted(expanded, key=lambda x: (-x[1], x[0]))[:beam_size]
        if not active or (len(completed) == nbest and active[0][1] <= completed[-1][1]):
            stopped = step < max_tokens
            break
    return completed, dict(steps=step + 1, max_tokens=max_tokens,
                          reached_length_limit=not stopped)


def exact_ctc_scores(log_probs, sequences, blank=3):
    """Log P(sequence|video); caller explicitly supplies EOS where required."""
    import torch
    if not sequences:
        return []
    if any(blank in seq for seq in sequences):
        raise ValueError("CTC target cannot contain blank")
    targets = torch.tensor([t for seq in sequences for t in seq], dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    probs = log_probs[:, None, :].expand(-1, len(sequences), -1).contiguous()
    losses = torch.nn.functional.ctc_loss(probs, targets,
        torch.full((len(sequences),), len(log_probs), dtype=torch.long), lengths,
        blank=blank, reduction="none", zero_infinity=False)
    return (-losses).tolist()


def hybrid_candidates(model, features, log_probs, ctc_candidates, vocabulary, compose,
                      beam_size=5, max_tokens=128, ctc_weight=.6):
    import torch
    from .evaluation import normalize
    if not 0 < ctc_weight < 1:
        raise ValueError("Hybrid CTC weight must be between zero and one")
    vocab_size = len(vocabulary)

    def next_scores(prefixes):
        ids = torch.tensor([[1, *p] for p in prefixes], dtype=torch.long)
        embedded = model.embedder(torch.nn.functional.one_hot(ids, vocab_size).float())
        output = model.decoder.attdecoder(inputs=features.expand(len(prefixes), -1, -1),
                                          labels=embedded, pad_id=0)
        return output[:, -1].log_softmax(-1).tolist()

    with torch.inference_mode():
        attention, search = attention_beam(next_scores, beam_size=beam_size, max_tokens=max_tokens)
        pool = {}
        for candidate in ctc_candidates:
            pool.setdefault(tuple(candidate["token_ids"]), set()).add("ctc")
        for tokens, _ in attention:
            pool.setdefault(tokens, set()).add("attention")
        sequences = list(pool)
        # Training drops SOS but keeps EOS in both CTC and attention targets.
        ctc_scores = exact_ctc_scores(log_probs, [list(s) + [2] for s in sequences])
        att_scores = []
        for start in range(0, len(sequences), 8):
            batch = sequences[start:start + 8]
            width = max(len(s) for s in batch) + 1
            ids = torch.zeros((len(batch), width), dtype=torch.long)
            target = torch.zeros_like(ids)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for i, seq in enumerate(batch):
                ids[i, :len(seq)+1] = torch.tensor([1, *seq])
                target[i, :len(seq)+1] = torch.tensor([*seq, 2])
                mask[i, :len(seq)+1] = True
            embedded = model.embedder(torch.nn.functional.one_hot(ids, vocab_size).float())
            output = model.decoder.attdecoder(inputs=features.expand(len(batch), -1, -1),
                                              labels=embedded, pad_id=0).log_softmax(-1)
            scores = output.gather(-1, target[..., None]).squeeze(-1)
            att_scores.extend(scores.masked_fill(~mask, 0).sum(-1).tolist())
    candidates = []
    for seq, cs, ats in zip(sequences, ctc_scores, att_scores):
        if not math.isfinite(cs) or not math.isfinite(ats):
            continue
        raw = "".join(vocabulary[t] for t in seq)
        candidates.append(dict(token_ids=list(seq), hypothesis=normalize(compose(raw)),
            raw_graphemes=raw, ctc_log_probability=cs, attention_log_probability=ats,
            hybrid_score=ctc_weight * cs + (1-ctc_weight) * ats,
            sources=sorted(pool[seq])))
    candidates.sort(key=lambda c: (-c["hybrid_score"], c["token_ids"]))
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate["hypothesis"], candidate)
    if not unique:
        raise ValueError("No finite hybrid candidates")
    return list(unique.values()), dict(search, attention_beam=beam_size, ctc_weight=ctc_weight,
        attention_completed=len(attention), token_candidates=len(sequences),
        nonfinite_candidates=len(sequences)-len(candidates),
        method="two-pass CTC/attention candidate union with exact sequence rescoring; not joint prefix search",
        eos_scored=True, length_normalized=False)

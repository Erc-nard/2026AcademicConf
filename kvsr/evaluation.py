"""Corpus CER/WER with explicit Korean normalization and strict ID matching."""

import unicodedata


def normalize(text):
    if not isinstance(text, str):
        raise ValueError("reference and hypothesis must be strings")
    # Preserve punctuation and case. NFC makes decomposed Hangul comparable.
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def indexed(rows, field):
    result = {}
    for row in rows:
        sample_id = row["id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise ValueError(f"Invalid or duplicate ID: {sample_id!r}")
        result[sample_id] = normalize(row[field])
    return result


def evaluate(references, predictions):
    refs, hyps = indexed(references, "reference"), indexed(predictions, "hypothesis")
    if not refs or refs.keys() != hyps.keys():
        raise ValueError("Reference and prediction ID sets must match exactly and be nonempty")
    char_errors = word_errors = chars = words = 0
    details = []
    for sample_id, reference in refs.items():
        hypothesis = hyps[sample_id]
        rchar, hchar = reference.replace(" ", ""), hypothesis.replace(" ", "")
        rword, hword = reference.split(), hypothesis.split()
        ce, we = distance(rchar, hchar), distance(rword, hword)
        char_errors += ce
        word_errors += we
        chars += len(rchar)
        words += len(rword)
        details.append(dict(id=sample_id, char_errors=ce, reference_chars=len(rchar),
                            word_errors=we, reference_words=len(rword)))
    if not chars or not words:
        raise ValueError("Corpus reference denominator is zero")
    return dict(samples=len(refs), cer=char_errors / chars, wer=word_errors / words,
                char_errors=char_errors, reference_chars=chars,
                word_errors=word_errors, reference_words=words,
                normalization="NFC; collapse whitespace; CER excludes whitespace; punctuation preserved",
                details=details)


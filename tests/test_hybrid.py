import importlib.util
import itertools
import math
import unittest

from kvsr.hybrid import attention_beam, exact_ctc_scores


class AttentionSearchTests(unittest.TestCase):
    def test_matches_exhaustive_bounded_tree(self):
        probabilities = [0., 0., .4, 0., .35, .25]
        scores = [math.log(p) if p else -math.inf for p in probabilities]
        result, info = attention_beam(lambda prefixes: [scores for _ in prefixes],
                                     beam_size=16, nbest=20, max_tokens=3)
        expected = []
        for length in range(4):
            for seq in itertools.product((4, 5), repeat=length):
                expected.append((seq, sum(scores[t] for t in seq) + scores[2]))
        expected.sort(key=lambda x: (-x[1], x[0]))
        self.assertEqual(dict(result).keys(), dict(expected).keys())
        for seq, actual in result:
            self.assertAlmostEqual(actual, dict(expected)[seq])
        self.assertTrue(all(a[1] >= b[1] for a, b in zip(result, result[1:])))
        self.assertTrue(info['reached_length_limit'])

    def test_early_stop_and_no_special_token_expansion(self):
        scores = [-math.inf, -math.inf, math.log(.9), -math.inf, math.log(.1)]
        result, info = attention_beam(lambda prefixes: [scores for _ in prefixes], nbest=1)
        self.assertEqual(result[0][0], ())
        self.assertFalse(info['reached_length_limit'])

    def test_bad_scores_rejected(self):
        with self.assertRaises(ValueError):
            attention_beam(lambda prefixes: [[0., 0., float('nan')]])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch dependency')
class CTCRescoreTests(unittest.TestCase):
    def test_exact_scores_match_path_enumeration_and_impossible_is_negative_inf(self):
        import torch
        probs = [[.2, .5, .3], [.4, .4, .2], [.3, .2, .5]]
        expected = {}
        for path in itertools.product(range(3), repeat=3):
            seq = tuple(t for i, t in enumerate(path) if t != 0 and (i == 0 or path[i-1] != t))
            expected[seq] = expected.get(seq, 0) + math.prod(probs[i][t] for i, t in enumerate(path))
        sequences = list(expected) + [(1, 1, 1)]
        actual = exact_ctc_scores(torch.tensor(probs, dtype=torch.float64).log(), sequences, blank=0)
        for seq, score in zip(sequences[:-1], actual[:-1]):
            self.assertAlmostEqual(math.exp(score), expected[seq], places=12)
        self.assertEqual(actual[-1], -math.inf)
        with self.assertRaises(ValueError):
            exact_ctc_scores(torch.tensor(probs).log(), [[0]], blank=0)

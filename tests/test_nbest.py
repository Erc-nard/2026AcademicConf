import itertools
import math
import unittest

from kvsr.nbest import evaluate_nbest, prefix_beam, render_candidates


class NBestTests(unittest.TestCase):
    def test_matches_exhaustive_ctc_path_probabilities(self):
        probs = [[.2, .5, .3], [.4, .4, .2], [.3, .2, .5], [.1, .6, .3]]
        expected = {}
        for path in itertools.product(range(3), repeat=4):
            collapsed = tuple(token for i, token in enumerate(path)
                              if token != 0 and (i == 0 or path[i-1] != token))
            mass = math.prod(probs[i][token] for i, token in enumerate(path))
            expected[collapsed] = expected.get(collapsed, 0) + mass
        actual = dict((tuple(p), math.exp(s)) for p, s in
                      prefix_beam([[math.log(p) for p in f] for f in probs], 100, 0))
        self.assertEqual(actual.keys(), expected.keys())
        for prefix in expected:
            self.assertAlmostEqual(actual[prefix], expected[prefix], places=12)

    def test_render_deduplicates_and_preserves_repeats(self):
        rows = render_candidates([([4, 4, 2, 5], -1), ([4, 4], -2), ([5], -3)],
                                 {4: "가", 5: "나"}, lambda x: x)
        self.assertEqual([r["hypothesis"] for r in rows], ["가가", "나"])
        self.assertEqual(rows[0]["log_mass"], -1)

    def test_oracle_uses_corpus_counts_and_model_order_ties(self):
        refs = [dict(id="a", reference="가"), dict(id="b", reference="나다라")]
        rows = [dict(id="a", candidates=[dict(hypothesis="나"), dict(hypothesis="가")]),
                dict(id="b", candidates=[dict(hypothesis="나마라"), dict(hypothesis="나다라")])]
        scores = evaluate_nbest(refs, rows, (1, 2, 5))["oracle_by_k"]
        self.assertEqual(scores["1"]["cer"], .5)
        self.assertEqual(scores["2"]["cer"], 0)
        self.assertEqual(scores["5"]["selected_ranks"], {"a": 2, "b": 2})
        tied = [dict(id="a", candidates=[dict(hypothesis="나"), dict(hypothesis="다")])]
        self.assertEqual(evaluate_nbest(refs[:1], tied)["oracle_by_k"]["5"]["selected_ranks"], {"a": 1})

    def test_rejects_missing_duplicate_empty_candidates(self):
        refs = [dict(id="a", reference="가")]
        good = dict(id="a", candidates=[dict(hypothesis="")])
        for rows in ([], [good, good], [dict(id="a", candidates=[])]):
            with self.assertRaises(ValueError):
                evaluate_nbest(refs, rows)
        self.assertEqual(evaluate_nbest(refs, [good])["oracle_by_k"]["1"]["cer"], 1)

    def test_rejects_bad_beam_inputs(self):
        for matrix in ([[float("nan"), -1]], [[-math.inf, -math.inf]], [[0], [0, 0]]):
            with self.assertRaises(ValueError):
                prefix_beam(matrix, blank=0)
        with self.assertRaises(ValueError):
            prefix_beam([[0]], beam_size=0, blank=0)

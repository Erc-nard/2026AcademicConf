import unittest

from kvsr.meld_single_edit import oracle_limit


class SingleEditBoundTests(unittest.TestCase):
    def test_exact_sentences_cannot_gain_and_large_errors_cannot_be_fully_repaired(self):
        result=oracle_limit(dict(reference_words=10,details=[dict(word_errors=e) for e in [0,1,3]]))
        self.assertEqual(result['minimum_possible_word_errors'],2)
        self.assertEqual(result['minimum_possible_wer'],.2)
        self.assertEqual(result['maximum_possible_exact_sentences'],2)

import unittest
from kvsr.development_rerank import selected_text


class SelectionTests(unittest.TestCase):
    def test_only_existing_exact_candidate_can_be_returned(self):
        pool=[dict(text='baseline'),dict(text='different words')]
        self.assertEqual(selected_text(pool,dict(candidate_index=1)),('different words',1))
        for bad in [dict(candidate_index=-1),dict(candidate_index=2),dict(candidate_index=True),
                    dict(candidate_index='1'),dict(candidate_index=1,text='invented')]:
            with self.assertRaises(ValueError):selected_text(pool,bad)

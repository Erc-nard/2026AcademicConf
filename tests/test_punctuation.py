import copy
import unittest

from kvsr.punctuation_audit import audit, strip_punctuation


class PunctuationTests(unittest.TestCase):
    def test_unicode_categories_preserve_nonpunctuation_and_nfc(self):
        self.assertEqual(strip_punctuation(' “가”, 나… 3+2=5! '), '가 나 3+2=5')

    def test_diagnostic_reports_denominator_change_without_mutating_inputs(self):
        refs = [dict(id='a',reference='가, 나!')]
        hyps = [dict(id='a',hypothesis='가 나')]
        original = copy.deepcopy((refs,hyps))
        report = audit(refs,hyps)
        self.assertEqual(report['primary']['char_errors'],2)
        self.assertEqual(report['primary']['reference_chars'],4)
        self.assertEqual(report['punctuation_removed_auxiliary']['char_errors'],0)
        self.assertEqual(report['punctuation_removed_auxiliary']['reference_chars'],2)
        self.assertEqual(report['removed_reference_characters'],2)
        self.assertEqual((refs,hyps), original)

    def test_different_or_duplicate_ids_are_not_silently_scored(self):
        for hyps in [[dict(id='b',hypothesis='가')], [dict(id='a',hypothesis='가')]*2]:
            with self.assertRaises(ValueError):
                audit([dict(id='a',reference='가!')], hyps)

    def test_punctuation_in_hypothesis_disables_no_punctuation_lower_bound(self):
        report = audit([dict(id='a',reference='가!')],[dict(id='a',hypothesis='가?')])
        self.assertIsNone(report['reference_characters_requiring_at_least_one_edit_if_hypotheses_have_no_punctuation'])
        self.assertEqual(report['removed_hypothesis_characters'],1)

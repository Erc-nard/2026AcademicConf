import unittest

from kvsr.adaptation_comparison import comparison_groups


class ComparisonGroupsTests(unittest.TestCase):
    def fixture(self):
        original=[dict(id='old',split='train'),dict(id='v',split='valid')]
        expanded=original+[dict(id='new',split='train')]
        return original,expanded

    def test_added_group_excludes_original_and_validation(self):
        original,expanded=self.fixture()
        groups=comparison_groups(original,expanded)
        self.assertEqual(groups,dict(train_original={'old'},train_added={'new'},train_all={'old','new'},validation={'v'}))

    def test_changed_validation_missing_original_and_duplicates_fail(self):
        original,expanded=self.fixture()
        for changed in [expanded+[expanded[0]],expanded[1:],
            [dict(id='other',split='valid')]+[r for r in expanded if r['split']=='train'],original]:
            with self.assertRaises(ValueError):
                comparison_groups(original,changed)

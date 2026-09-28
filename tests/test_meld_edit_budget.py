import unittest

from kvsr.meld_edit_budget import gate


class EditBudgetTests(unittest.TestCase):
    def test_insert_delete_substitute_and_boundary(self):
        for current,candidate in [('I go','I will go'),('I will go','I go'),('I go','I went')]:
            self.assertEqual(gate(current,candidate,'ok',1)['edits'],1)
            self.assertTrue(gate(current,candidate,'ok',1)['accepted'])
            self.assertEqual(gate(current,candidate,'ok',0)['hypothesis'],current)
        self.assertEqual(gate('I go','I will not go','ok',1)['reason'],'over_budget')
        self.assertTrue(gate('I go','I will not go','ok',2)['accepted'])

    def test_failures_cosmetic_changes_and_invalid_budgets(self):
        self.assertEqual(gate('Hello.','HELLO!','ok',1)['hypothesis'],'Hello.')
        self.assertEqual(gate('I go','I went','failed_keep_vsr',None)['reason'],'generation_failure')
        for value in [-1,True,1.5]:
            with self.assertRaises(ValueError):gate('a','b','ok',value)

    def test_small_distance_does_not_guarantee_semantic_safety(self):
        # Explicitly document the limitation: a single accepted edit can reverse meaning.
        self.assertTrue(gate('I can go','I cannot go','ok',1)['accepted'])

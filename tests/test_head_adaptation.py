import unittest

from kvsr.head_adaptation import require_ctc_alignment,build_targets


class CTCTargetTests(unittest.TestCase):
    def test_repeated_labels_require_intervening_blank_frame(self):
        self.assertEqual(require_ctc_alignment([5,5,2],4),4)
        with self.assertRaises(ValueError):
            require_ctc_alignment([5,5,2],3)

    def test_empty_or_special_training_targets_are_rejected(self):
        for tokens in [[],[0,2],[1,2],[3,2]]:
            with self.assertRaises(ValueError):
                require_ctc_alignment(tokens,20)

    def test_normalization_precedes_tokenization_and_eos_is_retained(self):
        target=build_targets('가!',lambda text:text.replace('!',''),lambda text:list(text),{'가':5},2)
        self.assertEqual(target,[5,2])
        with self.assertRaises(ValueError):
            build_targets('!',lambda text:'',list,{},20)

    def test_unknown_labels_are_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            build_targets('가?',lambda text:text,list,{'가':5},20)

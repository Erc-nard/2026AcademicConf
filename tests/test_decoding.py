import unittest

from kvsr.olkavs import collapse_ctc


class DecodingTests(unittest.TestCase):
    def test_blank_separates_repeated_tokens(self):
        self.assertEqual(collapse_ctc([3, 5, 5, 3, 5, 6, 6, 3]), [5, 5, 6])

    def test_special_tokens_and_eos(self):
        self.assertEqual(collapse_ctc([0, 1, 5, 3, 2, 6]), [5])
        self.assertEqual(collapse_ctc([3, 3]), [])


if __name__ == "__main__":
    unittest.main()

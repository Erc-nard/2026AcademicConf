import unittest
from kvsr.report_analysis import paired,percentile,subset


def metric(errors,words):
    rows=[dict(id=str(i),word_errors=e,reference_words=w,char_errors=e,reference_chars=w) for i,(e,w) in enumerate(zip(errors,words))]
    return dict(word_errors=sum(errors),reference_words=sum(words),char_errors=sum(errors),reference_chars=sum(words),details=rows)


class ReportAnalysisTests(unittest.TestCase):
    def test_resamples_clusters_and_corpus_denominator_not_sentence_average(self):
        a=metric([1,9],[1,9]);b=metric([0,9],[1,9])
        result=paired(a,b,{'0':'same','1':'same'},draws=20)
        self.assertEqual(result['delta_wer_pp'],-10)
        self.assertEqual(result['percentile_95_interval_pp'],[-10,-10])
        self.assertEqual(result['clusters'],1)

    def test_mismatched_pairs_or_denominators_rejected(self):
        a=metric([1,2],[2,4])
        with self.assertRaises(ValueError):paired(a,metric([1],[2]),{'0':'a','1':'b'},draws=2)
        with self.assertRaises(ValueError):paired(a,metric([1,2],[2,5]),{'0':'a','1':'b'},draws=2)

    def test_percentiles_and_subgroups_use_exact_counts(self):
        self.assertEqual(percentile([10,0],.25),2.5)
        value=subset(metric([1,2],[2,8]),{'1'})
        self.assertEqual(value['wer'],.25)
        self.assertEqual(value['samples'],1)

import unittest
from kvsr.development_diagnostics import apply_operation
from kvsr.evaluation import distance
from kvsr.meld_experiment import normalize_english


class OperationTests(unittest.TestCase):
    def test_edits_use_exact_index_and_one_word(self):
        source='ONE two TWO.'
        for op,index,word,expected in [('replace',2,'three','one two three'),('delete',1,'','one two'),
                                       ('insert',3,'four','one two two four'),('insert',0,'zero','zero one two two')]:
            result=apply_operation(source,dict(op=op,index=index,word=word))
            self.assertEqual(result,expected)
            self.assertEqual(distance(normalize_english(source).split(),result.split()),1)
        self.assertEqual(apply_operation(source,dict(op='keep',index=0,word='')),source)

    def test_invalid_indices_and_sentence_rewrite_rejected(self):
        for op,index,word in [('replace',2,'a'),('delete',-1,''),('insert',True,'a'),('insert',3,'a'),
                              ('replace',0,'whole new sentence'),('keep',0,'extra'),('delete',0,'extra'),
                              ('replace',0,'a!'),('replace',0,'')]:
            with self.assertRaises(ValueError):apply_operation('a b',dict(op=op,index=index,word=word))
        with self.assertRaises(ValueError):apply_operation('a',{'op':'keep'})

    def test_empty_input_only_allows_keep_or_insert(self):
        self.assertEqual(apply_operation('',dict(op='insert',index=0,word='hello')),'hello')
        with self.assertRaises(ValueError):apply_operation('',dict(op='delete',index=0,word=''))

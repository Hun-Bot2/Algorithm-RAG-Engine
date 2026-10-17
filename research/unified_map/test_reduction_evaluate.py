import unittest
import numpy as np
from research.unified_map.reduction_evaluate import metrics

class ReductionQualityTests(unittest.TestCase):
    def test_unknown_labels_are_not_negatives(self):
        nodes={k:{'normalized_taxonomy':{'mapped_concepts':[]}} for k in ('a','b','c')}
        neighbors=np.array([[1,2],[0,2],[0,1]])
        r=metrics(['a','b','c'],neighbors,nodes,{('a','b'):2,('b','a'):None})
        self.assertEqual(r['slots'],6)
        self.assertEqual(r['judged_slots'],1)
        self.assertEqual(r['observed_precision_judged'],1)
        self.assertIsNone(r['mean_tag_jaccard'])

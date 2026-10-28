import unittest
import numpy as np
from research.recommendation_inputs.inputs import section_inputs, curriculum_inputs, token_chunks
from research.recommendation_inputs.experiments import retrieve, unit


class InputTests(unittest.TestCase):
    def test_sections_do_not_treat_sample_io_as_task_io(self):
        view=section_inputs('Find a value.\nInput\nAn integer n.\nOutput\nThe value.\nExamples\nInput\n2\nOutput\n4\nNote\nExplanation.')
        self.assertEqual(view['issues'],[])
        self.assertEqual(view['sections']['input'],'An integer n.')
        self.assertNotIn('\n2',view['task_text'])
        self.assertNotIn('Explanation',view['task_text'])
        self.assertFalse(view['present']['constraints'])

    def test_unavailable_and_ambiguous_sections_stay_explicit(self):
        self.assertIn('missing_input',section_inputs('Task only.')['issues'])
        self.assertIn('duplicate_input_heading',section_inputs('Task\nInput\na\nInput\nb\nOutput\nc')['issues'])
        v=section_inputs('Problem Statement\nTask\nConstraints\n1<=n\nInput Format\nn\nOutput Format\nanswer\nSample Input 1\n1\nSample Output 1\n2')
        self.assertEqual(v['issues'],[])
        self.assertEqual(v['sections']['constraints'],'1<=n')
        self.assertNotIn('SAMPLE',v['task_text'])

    def test_curriculum_view_copies_sources_and_records_missing_proof(self):
        p={'concepts':['hash-map'],'review':{'status':'approved','reviewer':'AI'}}
        a={'title':'Two Sum','concept_rationale':'Look up a complement.'}
        view=curriculum_inputs(p,a)
        self.assertEqual(view['idea_rationale'],a['concept_rationale'])
        self.assertIn('hash-map',view['idea_structured'])
        self.assertIn('correctness argument',view['missing_strategy_fields'])
        self.assertEqual(p['review']['reviewer'],'AI')
        with self.assertRaises(ValueError):curriculum_inputs(p,{'title':'Missing'})

    def test_original_tokens_are_covered_exactly(self):
        ids=[list(range(600)),[7,8]]
        chunks=token_chunks(ids)
        for owner,tokens in enumerate(ids):
            self.assertEqual([t for i,part in chunks if i==owner for t in part],tokens)
        self.assertTrue(all(len(part)<=254 for i,part in chunks))
        with self.assertRaises(ValueError):token_chunks([[]])

    def test_scope_and_duplicate_exclusion_share_candidate_rules(self):
        p={key:{'platform':platform} for key,platform in [('a','leetcode'),('b','leetcode'),('c','atcoder'),('d','atcoder')]}
        v=np.array([[1.,0.],[.8,.6],[1.,0.],[0.,1.]],dtype=np.float32)
        cross=retrieve(p,v,mode='cross',k=2,hashes=['same','b','same','d'])
        self.assertEqual(cross['a'][0][0],'d')
        self.assertNotIn('c',[key for key,score in cross['a']])
        same=retrieve(p,v,mode='same',k=2)
        self.assertEqual(same['a'][0][0],'b')
        with self.assertRaises(ValueError):unit(np.zeros((2,3)))


if __name__=='__main__':unittest.main()

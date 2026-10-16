import tempfile
from pathlib import Path
import unittest
from src.learning.embedding_backend import LocalEmbeddingBackend
from src.learning.research import ResearchError, build_profiles, sha256_file


class EncoderTest(unittest.TestCase):
    def test_manifest_verified_before_loading_and_e5_prefixes_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.safetensors';path.write_bytes(b'not-real-weights-test-only')
            manifest=dict(schema_version=1,namespace='e5-fixture',model_id='intfloat/multilingual-e5-small',revision='a'*40,
                license_reference='fixture-only',dimensions=384,max_tokens=128,files_sha256={'model.safetensors':sha256_file(path)})
            calls=[]
            class FakeEncoder:
                def encode(self,texts,**options):
                    calls.append((texts,options));return [[1.0]+[0.0]*383 for _ in texts]
            def loader(directory,**options):
                self.assertTrue(options['local_files_only']);self.assertFalse(options['trust_remote_code'])
                return FakeEncoder()
            backend=LocalEmbeddingBackend(tmp,manifest,loader=loader)
            profiles=build_profiles({'schema_version':1,'problems':[{'problem_key':'leetcode:a','concepts':['array']},{'problem_key':'leetcode:b','concepts':['hash']}]},{})
            result=backend.neighbors(profiles)
            self.assertTrue(calls[0][0][0].startswith('query: '));self.assertTrue(calls[1][0][0].startswith('passage: '))
            self.assertEqual(result['manifest']['dimensions'],384)
            path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ResearchError,'hash'):
                LocalEmbeddingBackend(tmp,manifest,loader=lambda *a,**kw:self.fail('should not load'))

    def test_absent_model_never_attempts_load(self):
        with self.assertRaisesRegex(ResearchError,'unavailable'):
            LocalEmbeddingBackend('/definitely/missing/model',{},loader=lambda *a,**k:self.fail('no load'))

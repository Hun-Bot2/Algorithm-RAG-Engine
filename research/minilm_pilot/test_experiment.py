import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research.minilm_pilot.experiment import DenseProvider, ranks

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('download_minilm',ROOT/'scripts/learning/download_minilm.py')
fetcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(fetcher)


class PilotChecks(unittest.TestCase):
    def test_cross_platform_rank_excludes_self_and_same_platform(self):
        keys=['leetcode:a','leetcode:b','atcoder:c','codeforces:d']
        matrix=np.array([[1,.95,.8,.8],[.95,1,.2,.3],[.8,.2,1,.4],[.8,.3,.4,1]])
        self.assertEqual(ranks(matrix,keys,0,cross_platform=True),[2,3])
        self.assertEqual(ranks(matrix,keys,0),[1,2,3])
        self.assertEqual(DenseProvider(keys,matrix).score('leetcode:a','atcoder:c'),.8)

    def test_model_download_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'asset';fetcher.write_new(path,b'original')
            with self.assertRaises(FileExistsError):fetcher.write_new(path,b'replacement')
            self.assertEqual(path.read_bytes(),b'original')
            self.assertEqual(path.stat().st_mode & 0o777,0o600)

    def test_download_allowlist_has_only_safetensor_weights(self):
        self.assertIn('model.safetensors',fetcher.ALLOWED)
        self.assertFalse(any(name.endswith(('.py','.bin','.pkl')) for name in fetcher.ALLOWED))
        self.assertEqual(len(fetcher.REVISION),40)
        self.assertEqual(len(fetcher.ALLOWED),len(set(fetcher.ALLOWED)))


if __name__=='__main__': unittest.main()

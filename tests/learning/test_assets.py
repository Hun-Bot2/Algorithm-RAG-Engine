import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('asset_tool', ROOT / 'scripts/learning/assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetTest(unittest.TestCase):
    def test_frozen_private_assets(self):
        registry = assets.verify()
        self.assertFalse(any(e['hosted_release'] for e in registry['assets']))
        manifest = json.loads((ROOT / 'assets/legacy_numpy.manifest.v1.json').read_text())
        self.assertEqual(manifest['count'], 3053)
        self.assertEqual(manifest['dimensions'], 1536)
        self.assertIsNone(manifest['preprocessing_hash'])
        self.assertFalse(manifest['inference_compatible'])

    def test_bundle_excludes_all_unapproved_assets(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'release'
            self.assertEqual(assets.bundle(dest), 0)
            self.assertEqual(list(dest.iterdir()), [])

    def test_tamper_missing_and_path_escape(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data.txt'
            source.write_text('original')
            entry = dict(path='data.txt', status='present', sha256=assets.sha256(source),
                         hosted_release=True, rights_status='unreviewed', permitted_uses=['hosted_bundle'])
            registry = dict(schema_version=1, assets=[entry])
            self.assertEqual(assets.bundle(root / 'bundle', root, registry), 0)
            source.write_text('changed')
            with self.assertRaises(ValueError):
                assets.verify(root, registry)
            entry.update(path='../escape')
            with self.assertRaises(ValueError):
                assets.verify(root, registry)
            entry.update(path='new.txt', status='missing')
            assets.verify(root, registry)
            (root / 'new.txt').touch()
            with self.assertRaises(ValueError):
                assets.verify(root, registry)

    def test_approved_allowlist_and_destination_guards(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'allowed.txt').write_text('ok')
            registry = dict(schema_version=1, assets=[dict(
                path='allowed.txt', status='present', sha256=assets.sha256(root / 'allowed.txt'),
                hosted_release=True, rights_status='approved', permitted_uses=['hosted_bundle'])])
            self.assertEqual(assets.bundle(root / 'release', root, registry), 1)
            self.assertEqual((root / 'release/allowed.txt').read_text(), 'ok')
            with self.assertRaises(ValueError):
                assets.bundle(root / 'release', root, registry)
            with self.assertRaises(ValueError):
                assets.bundle(root / 'allowed.txt/child', root, registry)

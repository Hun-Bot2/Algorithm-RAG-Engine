import json
import tempfile
import unittest
from pathlib import Path

from research.language_bias import models as MD

SPEC = {"repo": "org/x", "revision": "a" * 40, "license": "mit", "kind": "embedder",
        "files": {"a.bin": (5, None), "b.json": (3, MD.sha256_bytes(b"{}\n")[:8])}}


def fake_fetch(url, limit):
    return {"a.bin": b"12345", "b.json": b"{}\n"}[url.rsplit("/", 1)[1]]


class Fetch(unittest.TestCase):
    def test_downloads_into_new_dir_and_returns_full_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "m"
            digests = MD.fetch_pinned(SPEC, "x", dest, fake_fetch)
            self.assertEqual(digests["a.bin"], MD.sha256_bytes(b"12345"))
            self.assertEqual((dest / "a.bin").stat().st_mode & 0o777, 0o600)
            with self.assertRaises(MD.ModelError):
                MD.fetch_pinned(SPEC, "x", dest, fake_fetch)       # never overwrite

    def test_wrong_size_or_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MD.ModelError):
                MD.fetch_pinned(SPEC, "x", Path(tmp) / "m1", lambda u, l: b"1234")
            bad = dict(SPEC, files={"b.json": (3, "deadbeef")})
            with self.assertRaises(MD.ModelError):
                MD.fetch_pinned(bad, "x", Path(tmp) / "m2", lambda u, l: b"{}\n")

    def test_http_fetch_enforces_the_size_limit(self):
        class Resp:
            def __init__(self): self.n = 0
            def read(self, k): self.n += 1; return b"x" * k if self.n < 5 else b""
            def __enter__(self): return self
            def __exit__(self, *a): return False
        import urllib.request as ur
        original = ur.urlopen
        ur.urlopen = lambda *a, **k: Resp()
        try:
            with self.assertRaises(MD.ModelError):
                MD.http_fetch("https://example.invalid/x", 10)
        finally:
            ur.urlopen = original


class Approval(unittest.TestCase):
    def test_requires_flag_and_approved_name(self):
        with self.assertRaises(MD.ModelError):
            MD.download("e5-small", approved=False)
        with self.assertRaises(MD.ModelError):
            MD.download("LaBSE", approved=True)

    def test_downloader_list_excludes_user_downloaded_and_declined_models(self):
        self.assertEqual(sorted(MD.APPROVED), ["e5-small", "opus-mt-ja-en", "paraphrase-multilingual"])
        self.assertTrue(set(MD.APPROVED) <= set(MD.PINNED))
        self.assertEqual(MD.PINNED["labse"]["downloaded_by"], "user")      # pinned, but never fetched by this code
        self.assertFalse(any("bge" in s["repo"].lower() for s in MD.PINNED.values()))
        with self.assertRaises(MD.ModelError):
            MD.download("labse", approved=True)


class Register(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name)
        self.saved = (MD.MODELS_DIR, MD.ROOT)
        MD.MODELS_DIR, MD.ROOT = self.models, self.models
        self.spec = {"repo": "org/u", "revision": "b" * 40, "license": "mit", "kind": "embedder", "downloaded_by": "user",
                     "files": {"w.bin": (4, MD.sha256_bytes(b"data")[:8]), "c.json": (2, None)}}
        MD.PINNED["unit-test-user-model"] = self.spec
        self.dir = MD.model_dir("unit-test-user-model")
        self.dir.mkdir(parents=True)
        (self.dir / "w.bin").write_bytes(b"data")
        (self.dir / "c.json").write_bytes(b"{}")

    def tearDown(self):
        MD.MODELS_DIR, MD.ROOT = self.saved
        MD.PINNED.pop("unit-test-user-model", None)
        self.tmp.cleanup()

    def test_registers_exact_files_and_writes_manifest_outside_dir(self):
        manifest = MD.register_existing("unit-test-user-model")
        self.assertEqual(manifest["files_sha256"]["w.bin"], MD.sha256_bytes(b"data"))
        self.assertTrue(MD.manifest_path("unit-test-user-model").exists())
        MD.verify_directory(self.dir, json.loads(MD.manifest_path("unit-test-user-model").read_text()))
        with self.assertRaises(MD.ModelError):
            MD.register_existing("unit-test-user-model")          # never overwrite

    def test_rejects_extra_files_wrong_size_and_wrong_digest(self):
        (self.dir / ".cache").mkdir()
        (self.dir / ".cache/x").write_text("meta")
        with self.assertRaises(MD.ModelError):
            MD.register_existing("unit-test-user-model")
        (self.dir / ".cache/x").unlink()
        (self.dir / ".cache").rmdir()
        (self.dir / "w.bin").write_bytes(b"datum!")
        with self.assertRaises(MD.ModelError):
            MD.register_existing("unit-test-user-model")
        (self.dir / "w.bin").write_bytes(b"dat0")
        with self.assertRaises(MD.ModelError):
            MD.register_existing("unit-test-user-model")

    def test_refuses_models_the_user_did_not_download(self):
        with self.assertRaises(MD.ModelError):
            MD.register_existing("e5-small")


class Verify(unittest.TestCase):
    def test_exact_membership_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.bin").write_bytes(b"12345")
            (d / "sub").mkdir()
            (d / "sub/c.json").write_bytes(b"{}")
            manifest = {"files_sha256": {"a.bin": MD.sha256_bytes(b"12345"), "sub/c.json": MD.sha256_bytes(b"{}")}}
            MD.verify_directory(d, manifest)
            (d / "extra.txt").write_text("x")
            with self.assertRaises(MD.ModelError):
                MD.verify_directory(d, manifest)
            (d / "extra.txt").unlink()
            (d / "a.bin").write_bytes(b"tampered")
            with self.assertRaises(MD.ModelError):
                MD.verify_directory(d, manifest)

    def test_pinned_paths_stay_under_private_models_and_manifest_is_outside_dir(self):
        for name in MD.APPROVED:
            self.assertEqual(MD.model_dir(name).parent, MD.MODELS_DIR)
            self.assertNotIn(MD.model_dir(name), MD.manifest_path(name).parents)

    def test_pinned_data_is_complete(self):
        for name, spec in MD.PINNED.items():
            self.assertEqual(len(spec["revision"]), 40)
            weights = [f for f in spec["files"] if f.endswith((".safetensors", ".bin"))]
            self.assertGreaterEqual(len(weights), 1, name)
            for weight in weights:
                self.assertTrue(spec["files"][weight][1], f"{name}/{weight} must carry a published LFS digest")


if __name__ == "__main__":
    unittest.main()

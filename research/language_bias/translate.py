"""English-translated titles with per-title provenance.

Order of evidence: the platform's own English wording where the title is "Japanese (English)", the
pinned local MT model otherwise (Japanese only: the model cannot read Russian or Korean, so those titles
stay as written and are marked untranslated). No hosted service and no hand translation.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from research.language_bias import languages as L
from research.language_bias import models as MD

SOURCES = ("identity", "parenthetical", "machine", "machine_empty_fallback", "untranslated")


def translate_title(title: str, cls: str, mt: Callable[[list[str]], list[str]] | None) -> tuple[str, str]:
    """Return (english text, source) for a single title; `mt` is only consulted for Japanese-only text."""
    if cls == "en":
        return title, "identity"
    if cls == "ja_parallel":
        return L.parse_parallel(title)[1], "parenthetical"
    if cls in ("ja_pure", "ja_other"):
        if mt is None:
            raise ValueError("machine translation required but not provided")
        out = mt([title])[0].strip()
        return (out, "machine") if out else (title, "machine_empty_fallback")
    return title, "untranslated"


def translate_all(titles: list[str], classes: list[str], mt: Callable[[list[str]], list[str]] | None,
                  batch_size: int = 32) -> list[dict]:
    """Translate every title, running MT once per distinct Japanese-only string in fixed order."""
    need = sorted({t for t, c in zip(titles, classes) if c in ("ja_pure", "ja_other")})
    cache: dict[str, str] = {}
    if need:
        if mt is None:
            raise ValueError("machine translation required but not provided")
        for i in range(0, len(need), batch_size):
            chunk = need[i:i + batch_size]
            out = mt(chunk)
            if len(out) != len(chunk):
                raise ValueError("mt_batch_length_mismatch")
            cache.update(zip(chunk, out))
    result = []
    for title, cls in zip(titles, classes):
        if cls in ("ja_pure", "ja_other"):
            text = cache[title].strip()
            result.append({"text": text, "source": "machine"} if text else {"text": title, "source": "machine_empty_fallback"})
        else:
            text, source = translate_title(title, cls, None)
            result.append({"text": text, "source": source})
    return result


def provenance_counts(rows: list[dict], classes: list[str]) -> dict:
    return {"by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
            "by_class_and_source": {f"{c}/{s}": n for (c, s), n in sorted(Counter(
                (c, r["source"]) for c, r in zip(classes, rows)).items())}}


class CachedMT:
    """Wrap a translator so every distinct text is machine-translated at most once."""

    def __init__(self, mt: Callable[[list[str]], list[str]], batch_size: int = 32):
        self.mt, self.batch_size, self.cache = mt, batch_size, {}

    def __call__(self, texts: list[str]) -> list[str]:
        need = sorted({t for t in texts if t not in self.cache})
        for i in range(0, len(need), self.batch_size):
            chunk = need[i:i + self.batch_size]
            self.cache.update(zip(chunk, self.mt(chunk)))
        return [self.cache[t] for t in texts]


def cached_translations(path: Path, titles: list[str], classes: list[str], mt) -> list[dict]:
    """Translate the catalog once; later runs reload and check the file matches the same titles."""
    import hashlib
    digest = hashlib.sha256(json.dumps(titles, ensure_ascii=False).encode()).hexdigest()
    if path.exists():
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        if lines[0]["titles_sha256"] != digest:
            raise ValueError("cached translations were built from different titles")
        return lines[1:]
    rows = translate_all(titles, classes, mt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps({"titles_sha256": digest, "model": "Helsinki-NLP/opus-mt-ja-en@0770961a",
                                 "decoding": "beam 4, max_length 64, no sampling"}) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


class MarianTranslator:
    """Pinned opus-mt-ja-en loaded from the verified safetensors copy; deterministic beam search."""

    def __init__(self, num_beams: int = 4, max_length: int = 64):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        torch.set_num_threads(1)
        name = "opus-mt-ja-en"
        directory = MD.model_dir(name)
        MD.verify_directory(directory, json.loads(MD.manifest_path(name).read_text(encoding="utf-8")))
        self.tokenizer = MarianTokenizer.from_pretrained(str(directory), local_files_only=True)
        self.model = MarianMTModel.from_pretrained(str(directory), local_files_only=True, use_safetensors=True).eval()
        self.num_beams, self.max_length, self.torch = num_beams, max_length, torch

    def __call__(self, texts: list[str]) -> list[str]:
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        with self.torch.no_grad():
            out = self.model.generate(**batch, num_beams=self.num_beams, max_length=self.max_length, do_sample=False)
        # Titles carry no sentence-final period; the model adds one, so it is stripped (and nothing else).
        return [t.strip().rstrip(".").strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]


def convert_pickle_to_safetensors(name: str = "opus-mt-ja-en") -> dict:
    """Read the quarantined pickle weights with torch's weights-only unpickler; save a safetensors copy.

    The pickle file is read exactly once, here, and never used afterwards. The copy is checked tensor by tensor
    against what was read, and a manifest (outside the model directory) records both digests.
    """
    import socket
    import torch
    from transformers import MarianConfig, MarianMTModel
    spec = MD.PINNED[name]
    src = MD.MODELS_DIR / "_pickle-quarantine" / f"{name}-{spec['revision'][:8]}"
    dst = MD.model_dir(name)
    if dst.exists() or MD.manifest_path(name).exists():
        raise MD.ModelError("refusing to overwrite an existing converted model")
    source_manifest = json.loads((MD.MODELS_DIR / f"{name}-{spec['revision'][:8]}.pickle-source.manifest.json").read_text())
    MD.verify_directory(src, source_manifest)
    blocked = RuntimeError("network disabled during pickle conversion")
    real_connect, real_create = socket.socket.connect, socket.create_connection
    socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(blocked)
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(blocked)
    try:
        state = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model = MarianMTModel(MarianConfig.from_pretrained(str(src)))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise MD.ModelError(f"unexpected tensors in pickle: {sorted(unexpected)[:5]}")
        model.save_pretrained(str(dst), safe_serialization=True)
        reloaded = MarianMTModel.from_pretrained(str(dst), use_safetensors=True, local_files_only=True)
        new_state = reloaded.state_dict()
        compared = 0
        for key, tensor in state.items():
            if key in new_state:
                if not torch.equal(tensor, new_state[key]):
                    raise MD.ModelError(f"tensor differs after conversion: {key}")
                compared += 1
    finally:
        socket.socket.connect, socket.create_connection = real_connect, real_create
    for file in ("source.spm", "target.spm", "vocab.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copyfile(src / file, dst / file)
    digests = {str(p.relative_to(dst)): MD.sha256_file(p) for p in sorted(dst.rglob("*")) if p.is_file()}
    manifest = MD.manifest_for(name, spec, digests, {
        "converted_from_pickle_sha256": source_manifest["files_sha256"]["pytorch_model.bin"],
        "conversion": "torch.load(weights_only=True) then save_pretrained(safe_serialization=True)",
        "tensors_compared": compared, "missing_after_load": sorted(missing)})
    manifest["sizes"] = {f: (dst / f).stat().st_size for f in digests}
    MD.write_new(MD.manifest_path(name), (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
    return {"tensors_compared": compared, "files": sorted(digests), "missing_after_load": sorted(missing)}


if __name__ == "__main__":
    print(json.dumps(convert_pickle_to_safetensors(), indent=1))

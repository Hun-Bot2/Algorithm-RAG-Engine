"""Offline, bounded importer for publisher-supplied Project CodeNet descriptions.

This module never fetches data and never executes extracted files. It processes
only the description archive and problem metadata supplied by the caller.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import re
import sys
import tarfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
PARSER_VERSION = "codenet-description-html.v1"
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_EXPANDED_BYTES = 50 * 1024 * 1024
MAX_MEMBER_BYTES = 5 * 1024 * 1024
MAX_METADATA_BYTES = 25 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_ARCHIVE_MEMBERS = 200_000
PINNED_SOURCE_COMMIT = "55c323b527ab3d6510d55edff188bc0a0d7bc5e5"
PINNED_ARCHIVE_GIT_BLOB_SHA1 = "22a8342ab304d69edd537be86e38743584296ef3"
PROBLEM_ID = re.compile(r"p[0-9]{5}", re.IGNORECASE)
ATCODER_TASK_URL = re.compile(
    r"^https://atcoder\.jp/contests/([A-Za-z0-9_-]+)/tasks/([A-Za-z0-9_-]+)$", re.IGNORECASE
)
BLOCK_TAGS = {"address", "article", "blockquote", "dd", "div", "dl", "dt", "h1", "h2", "h3", "h4", "h5", "li", "ol", "p", "pre", "section", "table", "tbody", "td", "th", "thead", "tr", "ul"}
DROP_TAGS = {"script", "style", "noscript"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member_name(name: str) -> PurePosixPath:
    if "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe_archive_path:{name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("..", "") for part in path.parts):
        raise ValueError(f"unsafe_archive_path:{name!r}")
    return path


def _archive_description_members(archive_path: Path, max_expanded_bytes=MAX_EXPANDED_BYTES):
    """Yield (member path, problem id, bytes); reject unsafe or oversized tar entries."""
    archive_path = Path(archive_path)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("archive_compressed_size_limit")
    total_expanded = 0
    count = 0
    member_count = 0
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("invalid_tar_archive") from exc
    with archive:
        for member in archive:
            member_count += 1
            if member_count > MAX_ARCHIVE_MEMBERS:
                raise ValueError("archive_member_count_limit")
            path = _safe_member_name(member.name)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"unsupported_archive_entry:{member.name}")
            if not member.isdir() and not member.isfile():
                raise ValueError(f"unsupported_archive_entry:{member.name}")
            if member.isdir():
                continue
            if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                raise ValueError(f"archive_member_size_limit:{member.name}")
            total_expanded += member.size
            if total_expanded > max_expanded_bytes:
                raise ValueError("archive_expanded_size_limit")
            if path.suffix.casefold() != ".html":
                continue
            match = PROBLEM_ID.fullmatch(path.stem)
            if not match:
                continue
            count += 1
            if count > MAX_RECORDS:
                raise ValueError("archive_record_count_limit")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"archive_member_unreadable:{member.name}")
            data = stream.read(MAX_MEMBER_BYTES + 1)
            if len(data) != member.size or len(data) > MAX_MEMBER_BYTES:
                raise ValueError(f"archive_member_read_limit:{member.name}")
            yield member.name, match.group(0).lower(), data


class _TextFromHTML(HTMLParser):
    """Preserve text-node whitespace and mathematical markup's visible content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.drop_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.casefold()
        if tag in DROP_TAGS:
            self.drop_depth += 1
        elif self.drop_depth == 0:
            if tag in BLOCK_TAGS:
                self.parts.append("\n")
            elif tag == "br":
                self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.casefold()
        if tag in DROP_TAGS and self.drop_depth:
            self.drop_depth -= 1
        elif self.drop_depth == 0 and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.drop_depth == 0:
            self.parts.append(data)

    def handle_entityref(self, name):
        if self.drop_depth == 0:
            self.parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name):
        if self.drop_depth == 0:
            self.parts.append(html.unescape(f"&#{name};"))


def extract_text(raw_html: bytes) -> str:
    """Decode UTF-8 with replacement, remove markup, and retain internal whitespace."""
    parser = _TextFromHTML()
    parser.feed(raw_html.decode("utf-8", errors="replace"))
    parser.close()
    text = "".join(parser.parts).replace("\r\n", "\n").replace("\r", "\n")
    # Remove only excess blank lines and outer document whitespace. In particular,
    # do not collapse spaces or newlines inside preformatted sample blocks.
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
    return text.strip()


def language_script_heuristic(text: str) -> dict:
    kana = sum("\u3040" <= ch <= "\u30ff" for ch in text)
    hangul = sum("\uac00" <= ch <= "\ud7af" for ch in text)
    cyrillic = sum("\u0400" <= ch <= "\u052f" for ch in text)
    han = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    if kana:
        label = "ja_script_present"
    elif hangul:
        label = "hangul_script_present"
    elif cyrillic:
        label = "cyrillic_script_present"
    elif han:
        label = "han_script_present_language_ambiguous"
    else:
        label = "latin_or_other_script_only"
    return {"label": label, "basis": "Unicode script counts; heuristic, not language identification",
            "counts": {"kana": kana, "han": han, "hangul": hangul, "cyrillic": cyrillic}}


def field_availability(text: str) -> dict:
    folded = text.casefold()
    constraints = bool(re.search(r"\bconstraints?\b|制約|제한", folded))
    sample_input = bool(re.search(r"sample\s*input|入力例|예제\s*입력", folded))
    sample_output = bool(re.search(r"sample\s*output|出力例|예제\s*출력", folded))
    return {"constraints": "heuristic_present" if constraints else "not_detected",
            "samples": "heuristic_present" if sample_input or sample_output else "not_detected",
            "sample_input_label_detected": sample_input,
            "sample_output_label_detected": sample_output}


def _normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(url.strip())
    if parts.scheme.casefold() not in {"http", "https"} or not parts.netloc:
        return None
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), path, "", ""))


def _catalog_url_index(catalog_rows):
    index = {}
    for row in catalog_rows or []:
        normalized = _normalize_url(row.get("url"))
        if normalized:
            if normalized in index:
                raise ValueError(f"catalog_url_not_unique:{normalized}")
            index[normalized] = row["problem_key"]
    return index


def _source_identity(metadata: dict):
    source = (metadata.get("dataset") or metadata.get("source") or "").strip()
    folded = source.casefold()
    if "atcoder" in folded:
        platform = "atcoder"
    elif "aizu" in folded or "aoj" in folded or "code judg" in folded:
        platform = "aizu"
    else:
        platform = None
    return source or None, platform


class _CanonicalURLFinder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        attrs = {key.casefold(): value for key, value in attrs if key and value}
        candidates = []
        if tag.casefold() == "a" and attrs.get("href"):
            candidates.append(attrs["href"])
        if tag.casefold() == "link" and "canonical" in attrs.get("rel", "").casefold().split():
            if attrs.get("href"):
                candidates.append(attrs["href"])
        for value in candidates:
            # Only absolute, HTTPS publisher task URLs count as explicit evidence.
            match = ATCODER_TASK_URL.fullmatch(value.strip())
            if match and re.fullmatch(r"[A-Za-z0-9_-]+", match.group(1)) and re.fullmatch(r"[A-Za-z0-9_-]+", match.group(2)):
                self.urls.add(_normalize_url(value))


def canonical_task_urls(raw_html: bytes) -> list[str]:
    finder = _CanonicalURLFinder()
    finder.feed(raw_html.decode("utf-8", errors="replace"))
    finder.close()
    return sorted(finder.urls)


def make_record(problem_id: str, raw_html: bytes, metadata: dict,
                catalog_url_index: dict | None = None) -> dict:
    text = extract_text(raw_html)
    source, source_platform = _source_identity(metadata)
    urls = canonical_task_urls(raw_html)
    canonical_url = urls[0] if len(urls) == 1 else None
    platform_basis = "problem_list.csv dataset field" if source_platform else "unavailable"
    if source_platform is None and canonical_url is not None:
        source_platform = "atcoder"
        platform_basis = "unique canonical AtCoder task URL in description HTML"
    # A cross-reference in an AIZU description is not proof that the AIZU item is
    # the linked AtCoder task. Require both the source metadata and explicit URL.
    is_atcoder = source_platform == "atcoder" and canonical_url is not None
    match = ATCODER_TASK_URL.fullmatch(canonical_url) if is_atcoder else None
    native_key = f"atcoder:{match.group(2)}" if match else None
    catalog_key = (catalog_url_index or {}).get(canonical_url) if native_key else None
    return {
        "dataset_record_key": f"codenet:{problem_id}",
        "problem_id": problem_id,
        "problem_key": catalog_key,
        "native_key": native_key,
        "source_platform": source_platform,
        "source_platform_basis": platform_basis,
        "source_dataset": source,
        "title": metadata.get("name") or metadata.get("title") or None,
        "source_url": canonical_url,
        "canonical_task_url_status": "valid_unique" if len(urls) == 1 else ("ambiguous_multiple" if len(urls) > 1 else "not_found"),
        "raw_html": raw_html.decode("utf-8", errors="replace"),
        "raw_html_base64": base64.b64encode(raw_html).decode("ascii"),
        "raw_html_sha256": sha256_bytes(raw_html),
        "full_text": text,
        "full_text_sha256": sha256_bytes(text.encode("utf-8")),
        "parser_version": PARSER_VERSION,
        "language_heuristic": language_script_heuristic(text),
        "field_availability": field_availability(text),
        "metadata": metadata,
        "metadata_status": "present" if metadata else "not_available_or_missing_row",
    }


def read_metadata_csv(path: Path) -> dict[str, dict]:
    path = Path(path)
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("metadata_size_limit")
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("metadata_missing_header")
        id_column = "problem_id" if "problem_id" in reader.fieldnames else ("id" if "id" in reader.fieldnames else None)
        if id_column is None:
            raise ValueError("metadata_missing_problem_id_or_id_column")
        for row in reader:
            problem_id = (row.get(id_column) or "").strip().lower()
            if not PROBLEM_ID.fullmatch(problem_id):
                continue
            if problem_id in rows:
                raise ValueError(f"duplicate_metadata_problem_id:{problem_id}")
            rows[problem_id] = {k: (v or "").strip() for k, v in row.items() if k}
            if len(rows) > MAX_RECORDS:
                raise ValueError("metadata_record_count_limit")
    return rows


def build_records(archive_path: Path, metadata_rows: dict[str, dict], catalog_rows=None,
                  max_expanded_bytes=MAX_EXPANDED_BYTES) -> tuple[list[dict], dict]:
    url_index = _catalog_url_index(catalog_rows)
    records, missing_metadata = [], 0
    seen = set()
    for member_name, problem_id, raw_html in _archive_description_members(
            archive_path, max_expanded_bytes=max_expanded_bytes):
        if problem_id in seen:
            raise ValueError(f"duplicate_archive_problem_id:{problem_id}")
        seen.add(problem_id)
        metadata = metadata_rows.get(problem_id)
        if metadata is None:
            missing_metadata += 1
            metadata = {}
        records.append(make_record(problem_id, raw_html, metadata, url_index))
    records.sort(key=lambda row: row["problem_id"])
    summary = {
        "description_members": len(records),
        "metadata_rows": len(metadata_rows),
        "descriptions_without_metadata": missing_metadata,
        "joined_current_catalog_keys": sum(row["problem_key"] is not None for row in records),
        "native_atcoder_keys_from_explicit_urls": sum(row["native_key"] is not None for row in records),
        "source_platform_counts": dict(sorted(Counter(row["source_platform"] or "unknown" for row in records).items())),
        "language_heuristic_counts": dict(sorted(Counter(row["language_heuristic"]["label"] for row in records).items())),
        "constraints_heuristic_present": sum(row["field_availability"]["constraints"] == "heuristic_present" for row in records),
        "samples_heuristic_present": sum(row["field_availability"]["samples"] == "heuristic_present" for row in records),
    }
    return records, summary


def write_private_import(output_dir: Path, archive_path: Path, metadata_csv: Path | None,
                         source_commit: str, catalog_path: Path | None = None) -> dict:
    """Write private records to a new directory; never overwrite existing output."""
    output_dir = Path(output_dir).resolve()
    private_root = (ROOT / "private-learning").resolve()
    if not output_dir.is_relative_to(private_root) or output_dir == private_root:
        raise ValueError("output_must_be_new_directory_under_private_learning")
    if output_dir.exists():
        raise FileExistsError(f"output_exists:{output_dir}")
    if source_commit != PINNED_SOURCE_COMMIT:
        raise ValueError("source_commit_must_match_pinned_archive_revision")
    archive_path = Path(archive_path)
    metadata_csv = Path(metadata_csv) if metadata_csv else None
    archive_blob_sha1 = git_blob_sha1_file(archive_path)
    if archive_blob_sha1 != PINNED_ARCHIVE_GIT_BLOB_SHA1:
        raise ValueError("archive_blob_sha1_mismatch_for_pinned_source")
    metadata = read_metadata_csv(metadata_csv) if metadata_csv else {}
    catalog_rows = None
    if catalog_path:
        catalog_rows = [json.loads(line) for line in Path(catalog_path).open(encoding="utf-8") if line.strip()]
    records, summary = build_records(archive_path, metadata, catalog_rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    receipt = {
        "schema": "codenet-import-receipt.v1",
        "status": "private_import_complete",
        "source_repository": "https://github.com/IBM/Project_CodeNet",
        "source_commit": source_commit,
        "description_archive": {"path_name": archive_path.name, "sha256": sha256_file(archive_path),
                                "git_blob_sha1": archive_blob_sha1,
                                "expected_git_blob_sha1": PINNED_ARCHIVE_GIT_BLOB_SHA1,
                                "pinned_blob_match": git_blob_sha1_file(archive_path) == PINNED_ARCHIVE_GIT_BLOB_SHA1,
                                "bytes": archive_path.stat().st_size},
        "metadata_csv": ({"path_name": metadata_csv.name, "sha256": sha256_file(metadata_csv),
                          "bytes": metadata_csv.stat().st_size} if metadata_csv else None),
        "license": {"dataset_license": "CDLA-Permissive-2.0",
                    "license_evidence_url": "https://www.linuxfoundation.org/press/press-release/enabling-easier-collaboration-on-open-data-for-ai-and-ml-with-cdla-permissive-2-0",
                    "publisher_repository_license": "Apache-2.0 (tools repository; not used as the dataset license basis)",
                    "attribution": "IBM Project CodeNet",
                    "scope_caveat": "Dataset license evidence does not erase separate rights in original problem descriptions; retain upstream copyright and terms status."},
        "limits": {"max_archive_bytes": MAX_ARCHIVE_BYTES, "max_expanded_bytes": MAX_EXPANDED_BYTES,
                   "max_member_bytes": MAX_MEMBER_BYTES, "max_archive_members": MAX_ARCHIVE_MEMBERS,
                   "raw_content_private": True},
        "parser_version": PARSER_VERSION,
        "records_sha256": sha256_file(records_path),
        "summary": summary,
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="optional problem_list.csv; omit when only the description archive is available")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--catalog", type=Path, help="optional current metadata-only catalog.v1.jsonl")
    args = parser.parse_args(argv)
    receipt = write_private_import(args.output, args.archive, args.metadata, args.source_commit, args.catalog)
    print(json.dumps(receipt["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, tarfile.TarError) as exc:
        sys.exit(str(exc))

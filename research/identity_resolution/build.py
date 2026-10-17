"""Validate bounded AtCoder identity evidence and create AI-only drafts.

This module never fetches source pages or writes to the verification workbench.
It only consumes evidence already recorded by a reviewer and returns records for
separate review/import.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import date
from urllib.parse import parse_qsl, urlparse


REVIEWER = "codex-atcoder-identity-v1"
_STATUSES = {"match", "conflict", "unavailable"}


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_task_url(value: str | None, native_key: str) -> bool:
    if not isinstance(value, str) or not isinstance(native_key, str):
        return False
    if not re.fullmatch(r"atcoder:[A-Za-z0-9_-]+", native_key):
        return False
    try:
        parsed = urlparse(value)
        port = parsed.port
        query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True) if parsed.query else []
    except ValueError:
        return False
    expected = native_key.removeprefix("atcoder:")
    parts = [part for part in parsed.path.split("/") if part]
    return (
        parsed.scheme == "https"
        and parsed.hostname in {"atcoder.jp", "www.atcoder.jp"}
        and parsed.netloc in {"atcoder.jp", "www.atcoder.jp"}
        and parsed.username is None
        and parsed.password is None
        and port is None
        and (not query or query in [[("lang", "en")], [("lang", "ja")]])
        and not parsed.fragment
        and ("?" not in value or bool(parsed.query))
        and "#" not in value
        and len(parts) == 4
        and parts[0] == "contests"
        and re.fullmatch(r"[A-Za-z0-9_-]+", parts[1]) is not None
        and parts[2] == "tasks"
        and parts[3] == expected
        and parsed.path == f"/contests/{parts[1]}/tasks/{expected}"
        and ";" not in parsed.path
    )


def _comparison_statuses(comparison: dict) -> dict[str, str]:
    statuses = {}
    for dimension in ("contest", "semantics", "constraints", "samples"):
        value = comparison.get(dimension)
        # Accept the compact authored form {same: "match"} as well as a direct
        # status string, while keeping emitted evidence packets unchanged.
        if isinstance(value, dict):
            value = value.get("same")
        if value not in _STATUSES:
            raise ValueError(f"invalid comparison status: {dimension}")
        statuses[dimension] = value
    return statuses


def validate_evidence(rows, corpus, catalog, required_keys):
    """Validate exact source-key coverage, provenance, URLs and decision gates.

    ``corpus`` and ``catalog`` may be mappings keyed by problem_key or iterables
    of their JSONL rows. ``required_keys`` is explicit so fixture subsets and the
    full 37-key investigation use the same checks.
    """
    corpus = corpus if hasattr(corpus, "get") else {r["problem_key"]: r for r in corpus}
    catalog = catalog if hasattr(catalog, "get") else {r["problem_key"]: r for r in catalog}
    required = list(required_keys)
    if len(required) != len(set(required)):
        raise ValueError("required source keys must be unique")
    if len(rows) != len(required):
        raise ValueError("evidence must cover every required source key exactly once")
    keys = [r.get("source_key") for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(required):
        raise ValueError("evidence source-key coverage mismatch or duplicate")
    positions = [r.get("identity_position") for r in rows]
    if any(type(p) is not int for p in positions) or len(set(positions)) != len(positions):
        raise ValueError("identity positions must be unique integers")
    expected_positions = {key: index for index, key in enumerate(required, start=1)}
    if any(row.get("identity_position") != expected_positions.get(row.get("source_key"))
           for row in rows):
        raise ValueError("identity positions must follow required-key order")

    for row in rows:
        key = row["source_key"]
        source = corpus.get(key)
        if source is None:
            raise ValueError(f"source key absent from corpus: {key}")
        actual_text = source.get("statement_text")
        if not isinstance(actual_text, str):
            raise ValueError(f"source content unavailable for hash validation: {key}")
        actual_hash = _sha_text(actual_text)
        if row.get("source_content_sha256") != actual_hash or source.get("content_sha256") != actual_hash:
            raise ValueError(f"source content hash mismatch: {key}")
        if source.get("platform") != "atcoder":
            raise ValueError(f"identity source is not AtCoder: {key}")
        access = row.get("official_access")
        if not isinstance(access, dict) or access.get("status") not in {
            "read_via_web_tool", "unavailable", "not_read"
        }:
            raise ValueError(f"invalid official access observation: {key}")
        observed_date = access.get("observed_date")
        try:
            if not isinstance(observed_date, str) or date.fromisoformat(observed_date).isoformat() != observed_date:
                raise ValueError
        except ValueError:
            raise ValueError(f"invalid official access observed_date: {key}") from None
        for name in ("canonical_url", "official_url"):
            value = row.get(name)
            selected = row.get("selected_native_key")
            if value is not None and (not isinstance(selected, str) or not _valid_task_url(value, selected)):
                raise ValueError(f"{name} host or task ID mismatch: {key}")

        selected = row.get("selected_native_key")
        if selected is not None and (not isinstance(selected, str) or not selected.startswith("atcoder:")):
            raise ValueError(f"selected native key must be AtCoder-qualified: {key}")
        member = bool(selected and selected in catalog)
        if row.get("catalog_member") is not member:
            raise ValueError(f"catalog_member does not match frozen catalog: {key}")
        comparison = row.get("comparison")
        if not isinstance(comparison, dict):
            raise ValueError(f"comparison object required: {key}")
        statuses = _comparison_statuses(comparison)
        if row.get("status") == "ai_proposed":
            if not selected or not member:
                raise ValueError(f"AI proposal requires an existing catalog identity: {key}")
            if access["status"] != "read_via_web_tool":
                raise ValueError(f"AI proposal requires observed official source access: {key}")
            if not _valid_task_url(row.get("official_url"), selected):
                raise ValueError(f"AI proposal requires a valid official task URL: {key}")
            if row.get("canonical_url") != catalog[selected].get("url"):
                raise ValueError(f"canonical URL must equal the frozen catalog URL: {key}")
            if row.get("publisher_title") != source.get("title"):
                raise ValueError(f"publisher title does not match source corpus: {key}")
            if (not isinstance(row.get("notes"), list) or not row["notes"]
                    or any(not isinstance(note, str) or len(note.strip()) < 10
                           for note in row["notes"])):
                raise ValueError(f"AI proposal requires substantive authored notes: {key}")
            if statuses["contest"] != "match" or statuses["semantics"] != "match":
                raise ValueError(f"AI proposal requires contest and semantic matches: {key}")
            if statuses["constraints"] != "match" and statuses["samples"] != "match":
                raise ValueError(f"AI proposal requires constraint or sample match: {key}")
            if "conflict" in statuses.values():
                raise ValueError(f"AI proposal cannot contain conflicting evidence: {key}")
        elif row.get("status") == "source_identified_missing_catalog":
            if not selected or member or access["status"] != "read_via_web_tool":
                raise ValueError(f"missing-catalog finding requires an observed absent native identity: {key}")
            if not _valid_task_url(row.get("official_url"), selected):
                raise ValueError(f"missing-catalog finding requires explicit official task ID: {key}")
        elif row.get("status") != "unresolved":
            raise ValueError(f"invalid evidence status: {key}")
        if not isinstance(row.get("notes"), list) or any(not isinstance(n, str) for n in row["notes"]):
            raise ValueError(f"authored notes must be strings: {key}")
        if not isinstance(row.get("rejected_candidates"), list):
            raise ValueError(f"rejected_candidates must be a list: {key}")
    return {"records": len(rows), "source_keys": len(set(keys)), "positions": len(set(positions)),
            "status_counts": dict(sorted(Counter(r["status"] for r in rows).items()))}


def proposal_records(rows, workbench):
    """Return validated AI identity drafts, without appending them to the DB."""
    required_keys = sorted({
        key
        for pair in workbench.pilot["items"]
        for key in (pair["query"], pair["target"])
        if key in workbench.corpus
        and workbench.corpus[key].get("platform") == "atcoder"
        and workbench.corpus[key].get("native_key") not in workbench.catalog
    })
    validate_evidence(rows, workbench.corpus, workbench.catalog, required_keys)
    occurrences = []
    for pair in workbench.pilot["items"]:
        for side in ("query", "target"):
            key = pair[side]
            occurrences.append((pair["pair_id"], key))
    by_key = {row["source_key"]: row for row in rows}
    output = []
    for pair_id, key in occurrences:
        row = by_key.get(key)
        if row is None or row.get("status") != "ai_proposed":
            continue
        native_key = row["selected_native_key"]
        if native_key not in workbench.catalog:
            continue
        canonical_url = workbench.catalog[native_key]["url"]
        notes = [note.strip() for note in row["notes"] if note.strip()]
        rationale = " ".join(notes) or "Official task identity compared against the publisher record."
        evidence = [canonical_url]
        if row.get("official_url") and row["official_url"] not in evidence:
            evidence.append(row["official_url"])
        evidence.append(f"publisher source key: {key}; SHA-256: {row['source_content_sha256']}")
        record = {
            "pair_id": pair_id,
            "reviewer": REVIEWER,
            "role": "ai",
            "kind": "identity",
            "rationale": rationale[:4000],
            "evidence": evidence,
            "subject_key": key,
            "native_key": native_key,
            "status": "proposed",
            "independent": False,
        }
        workbench.validate(record, workbench.records())
        output.append(record)
    return output


def report_markdown(rows):
    """Render private metadata and authored notes; never include source text."""
    def cell(value):
        if value is None:
            value = "—"
        if isinstance(value, (list, dict)):
            value = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
        return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")

    def official_link(value):
        if not isinstance(value, str) or not value:
            return "unavailable"
        destination = value.replace("<", "%3C").replace(">", "%3E").replace(" ", "%20")
        return f"[official page](<{destination}>)"

    def dimension(row, name):
        value = row.get("comparison", {}).get(name, "unavailable")
        if isinstance(value, dict):
            value = value.get("same", "unavailable")
        return value

    def urls_in(value):
        if isinstance(value, str):
            return [url.rstrip(".,;:!?)]}\"'") for url in re.findall(r"https://[^\s<>]+", value)]
        if isinstance(value, list):
            return [url for item in value for url in urls_in(item)]
        if isinstance(value, dict):
            result = []
            for key, item in value.items():
                if key.endswith("url") or key.endswith("_urls") or key == "url":
                    result.extend(urls_in(item))
            return result
        return []

    lines = ["# AtCoder identity evidence review", "",
             "AI proposals are pending review. This report contains authored notes and metadata only.", "",
             "| Position | Publisher title | Source key | Status | Selected native key | Catalog member | Contest | Semantics | Constraints | Samples | Quality flags | Official page |",
             "|---:|---|---|---|---|:---:|---|---|---|---|---|---|"]
    ordered = sorted(rows, key=lambda r: r["identity_position"])
    for row in ordered:
        flags = row.get("quality_flags", [])
        lines.append("| " + " | ".join((
            cell(row["identity_position"]),
            cell(row.get("publisher_title")),
            cell(row["source_key"]),
            cell(row["status"]),
            cell(row.get("selected_native_key")),
            "yes" if row.get("catalog_member") else "no",
            cell(dimension(row, "contest")),
            cell(dimension(row, "semantics")),
            cell(dimension(row, "constraints")),
            cell(dimension(row, "samples")),
            cell(flags),
            official_link(row.get("official_url")),
        )) + " |")
    lines.append("")
    for row in ordered:
        title = cell(row.get("publisher_title") or "Untitled source").replace("<br>", " ")
        lines.append("")
        lines.append(f"### {row['identity_position']}. {title}")
        lines.append("")
        lines.append(f"Source key: {cell(row['source_key'])}")
        lines.append(f"Status: {cell(row['status'])}; catalog member: {'yes' if row.get('catalog_member') else 'no'}")
        lines.append(f"Official URL: {official_link(row.get('official_url'))}")
        lines.append("Comparison: " + "; ".join(
            f"{name}={cell(dimension(row, name))}"
            for name in ("contest", "semantics", "constraints", "samples")
        ))
        if row.get("quality_flags"):
            lines.append(f"Quality flags: {cell(row['quality_flags'])}")
        for note in row.get("notes", []):
            lines.append(f"- {cell(note)}")
        for candidate in row.get("rejected_candidates", []):
            lines.append(f"- Rejected {cell(candidate.get('native_key'))}: {cell(candidate.get('reason', ''))}")
        if row.get("status") == "unresolved":
            attempted = []
            attempted.extend(urls_in(row.get("attempted_urls", [])))
            attempted.extend(urls_in(row.get("attempted_official_urls", [])))
            attempted.extend(urls_in(row.get("rejected_candidates", [])))
            attempted.extend(urls_in(row.get("notes", [])))
            attempted = list(dict.fromkeys(attempted))
            lines.append("Attempted candidate URLs:")
            if attempted:
                lines.extend(f"- [{cell(url)}](<{url}>)" for url in attempted)
            else:
                lines.append("- None recorded.")
        lines.append("")
    return "\n".join(lines)

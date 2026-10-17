# Independent identity evidence review (2026-10-17)

Reviewed the frozen 37-key pilot, `evidence.final.json` at SHA-256
`f6f67e83f41240a235484885be07810687292f78034e1a885726f801f54c2426`,
and the proposal builder at SHA-256
`09485cd131ca61e86e8bf6efbe58beea5b354fc49259683221a6b93708804aab`.
This review concerns AI identity proposals only. It does not approve native IDs,
verify statement fidelity, assert source reuse rights, or create human reviews.

## Findings

- **Coverage and provenance:** The merged file has 37 unique source keys in
  positions 1–37, bound to the required pilot order. Every publisher description
  hash matches its frozen packet and each selected canonical URL matches the
  frozen native catalog. There are 36 `ai_proposed` records and one `unresolved`
  record. No evidence record contains a full official statement or an approved
  or human-verified identity field.
- **Ambiguous names:** [AGC021 C Tiling](https://atcoder.jp/contests/agc021/tasks/agc021_c)
  matches the publisher's fixed-count horizontal/vertical domino construction;
  the same-title `abc345_d` candidate has a different contest and tiling rules.
  [AGC036 A Triangle](https://atcoder.jp/contests/agc036/tasks/agc036_a)
  constructs coordinates for a specified area, unlike the count-oriented
  same-title alternatives. [Educational DP Q Flowers](https://atcoder.jp/contests/dp/tasks/dp_q)
  matches the publisher's increasing-height beauty maximization; other
  Flowers candidates were rejected on contest metadata, with no claim that
  their statements were read. The [ABC053 task list](https://atcoder.jp/contests/abc053/tasks)
  links Card Eater D to native task `arc068_b`, also shown by the
  [ARC068 task list](https://atcoder.jp/contests/arc068/tasks). The frozen
  catalog's ABC053 appearance URL is therefore consistent with that native key.
- **Other native-ID edges:** The [CODE FESTIVAL 2017 qual B task list](https://atcoder.jp/contests/code-festival-2017-qualb/tasks)
  identifies 3 Steps as `code_festival_2017_qualb_c` despite the original
  title-candidate packet being empty. ARC103 displays Tr/ee as task E while
  its [official URL](https://atcoder.jp/contests/arc103/tasks/arc103_c)
  carries native ID `arc103_c`; the evidence preserves that ID.
- **Special Trains:** The [official ABC084 C page](https://atcoder.jp/contests/abc084/tasks/abc084_c?lang=en)
  gives `F_i <= 10` in English, matching the publisher's English description.
  Its Japanese section gives `F_i <= 100`, and its third public sample uses
  `F_i = 64`. Position 11 correctly records an English-to-English constraint
  match plus separate `official_en_ja_constraint_disagreement` and
  `sample_exceeds_english_bound` flags. This supports task identity, while the
  cross-language constraint inconsistency remains open for statement fidelity.
- **Capture:** Position 16 has no selected native key, an unavailable official
  task-page observation, and no semantic/constraint/sample match. It remains
  unresolved and contributes no proposal. The catalog hint `relay2_f` is not
  promoted from title or editorial metadata alone.

## Builder and validation

The builder requires exact 37-key coverage, source hashes, title and canonical
URL consistency, official task access, contest and semantic matches, and a
constraint or sample match before emitting an AI draft. It accepts an optional
official `lang=en` or `lang=ja` URL and hyphenated contest slugs; earlier
integration failures on those real URLs were fixed before this review.
`proposal_records` emits `role=ai`, `status=proposed`, and `independent=false`;
it does not append to the workbench database. `run.py` defaults to preview;
`--append` is required to write reviews. Before writing, it validates the
evidence and builds all proposals; afterward, it checks that every prior
record ID and payload remains present. The existing workbench requires
two independent human identity reviews and a separate human adjudicator for an
accepted overlay.

Validation run from the repository root:

- `python3 -m unittest discover -s research/identity_resolution -v`: 20 passed,
  including private-output, append, idempotency, and human-record preservation
  fixtures.
- `python3 -m unittest discover -s research/verification_workbench -p test_core.py -v`: 11 passed.
- `validate_evidence(evidence.final.json, pilot corpus, pilot catalog, 37 required keys)`:
  passed with 36 proposed and one unresolved.
- `proposal_records` preview against the frozen pilot using a disposable SQLite
  database: 39 proposed records for 36 distinct source keys; zero records were
  written to that database, and Capture was absent. The generated private
  Markdown report did not contain any complete publisher statement text.
- `preview-v2/receipt.json` was checked against its three output hashes
  (receipt SHA-256 `8028c5ec46bccfe67b52a1bee3159e4154cd2cf068256c0d1c01fadb3139bc05`).
  Its before/after audit has the same 80 records and chain head; a read-only
  SQLite query found 40 human and 40 AI records in the real pilot DB. The
  preview did not append. Its Capture section lists all four attempted
  official URLs, and its 39 draft records are AI, proposed, and
  non-independent. Capture is absent from those drafts. `preview-v1` is
  superseded by this corrected preview.

The official pages above were inspected through the web tool's cached page
presentation; this is not a live HTTP-status receipt. Evidence remains AI
authored and pending human review. No real workbench DB, frozen catalog,
publisher description, or statement corpus was modified by this review.

**Disposition:** The evidence and implementation gates pass for appending the
39 AI-only proposal records represented by `preview-v2`. This would add
review records, not approve native IDs or change the frozen catalog. The
append loop commits one record at a time; if interrupted, a fresh-output
idempotent rerun is needed to reconcile a partial prefix. Human identity
verification and accepted overlays remain separate gates.

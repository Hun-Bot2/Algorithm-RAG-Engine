# AI-first inspection amendment — 2026-10-17

The user requested an AI review first and will judge the drafts afterward.

- The existing 40-pair pilot SHALL receive exactly one AI relationship draft per
  pair, with separate similarity, prerequisite, transfer and difficulty judgments.
  Full provided statement hashes SHALL bind the inputs. Missing images and
  incomplete solution derivations SHALL remain explicit uncertainty.
- Drafts SHALL be imported only with role=ai and independent=false. No human
  judgment, native-ID verification, accepted overlay or historical label is created.
- A human MAY explicitly select AI-assisted viewing. This SHALL expose AI records
  and that reviewer's own records only, retain hidden retrieval ranks, record
  persistent exposure, and block later independent claims for that pair.
- Assisted human judgments SHALL remain recordable with independent=false, without
  relaxing the two-independent-human-plus-third-adjudicator verification gate.
- Review explanations SHALL be readable as prose, with raw records and source
  references available. A private report SHALL let the user inspect all 40 drafts.
- Draft corrections before import SHALL preserve provenance. If an initial snapshot
  is reconstructed afterward, that limitation SHALL be stated rather than claiming
  contemporaneous capture. No model-quality improvement may be inferred from this
  AI-only pilot or its nonrepresentative sampled counts.

Validation: AI-only import completeness/idempotency and prevalidation; scoped AI
visibility; exposure lock across reload; assisted submission; other-human label
hiding; role and source hashes; full existing workbench regression tests and a
synthetic browser workflow. Real-pilot browser checks are read-only.

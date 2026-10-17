# AtCoder pilot identity proposals

This bounded manual investigation covers 37 unresolved publisher identities in
the frozen 40-pair verification pilot. It produces 36 AI identity proposals and
one unresolved case (Capture), represented by 39 pair-occurrence review records.
These are proposals pending human assessment, not verified native identifiers.

The private report and receipt are in
`private-learning/verification-workbench/identity-proposals-v1/import-v1/`.
The frozen catalog and statement corpus are unchanged. No official full
statements were collected, no model was trained, and production was untouched.

## Evidence and limits

Each proposal binds the publisher statement hash, frozen catalog URL, inspected
primary task URL, and authored contest/semantic/constraint/sample comparisons.
Same-title candidates are insufficient. Official native IDs can differ from
visible task letters (for example ABC053 D uses `arc068_b`). Manual web-tool
observations may be cached; live HTTP statuses and official response hashes
were not measured. Public access is not blanket statement reuse permission.
See `source-policy.v1.json` and the independent `REVIEW.md`.

Capture's candidate could not be inspected and remains unresolved. Special
Trains has an official English/Japanese constraint discrepancy, retained as
quality flags rather than rewritten publisher text. Identity agreement does
not establish statement fidelity.

A title-candidate bug was also corrected: normalization now preserves numeric
problem titles such as “3 Steps” and “01 Unbalanced” after removing the publisher
and contest wrappers. Existing research runs were not regenerated.

## Reproduce

The manually authored evidence and frozen inputs stay private. Preview does not
append records; use a fresh private output directory each time:

```sh
python3 research/identity_resolution/run.py \
  --evidence private-learning/verification-workbench/identity-proposals-v1/evidence.final.json \
  --output private-learning/verification-workbench/identity-proposals-v1/new-preview
python3 -m unittest discover -s research/identity_resolution -v
python3 -m unittest discover -s research/verification_workbench -v
```

`--append` imports only AI/proposed/nonindependent records through the existing
idempotent workbench API. It preserves prior records and never promotes native
IDs. Human assessment and independent adjudication remain separate gates.

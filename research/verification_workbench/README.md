# Local identity and relationship verification workbench

This workbench prepares a private pilot and records evidence. It does not resolve
identities from titles, manufacture human labels, modify source catalogs, train a
model, or change recommendations. Reviewer roles and independence are explicitly
self-attested, not authenticated proof of identity or expertise.

## Open the prepared pilot

From repository root:

```sh
python3 research/verification_workbench/server.py serve
```

Open **http://127.0.0.1:8765/**. The server binds only loopback. Stop with Ctrl-C.
There is no cloud deployment, API credential, model server, or new dependency.
The existing pilot is private at
`private-learning/verification-workbench/pilot-v1/pilot.json`; its SQLite store is
beside it. `prepare` refuses overwrites; use a new `--output` for another pilot.
Source hash drift blocks startup rather than silently changing annotation inputs.

The pilot contains **40 pairs**, four from each of ten existing method/platform
strata, selected deterministically with seed 17. It contains **74 distinct problem
keys**, of which **37 have unresolved native IDs**. Title-wrapper normalization
finds one or more unverified catalog candidates for **36/37** unresolved problems;
one has none. These are suggestions only. Missing topic and difficulty strata
remain unavailable. [The receipt](pilot-receipt.v1.json) records actual counts and
source hashes; statements remain private.

## Run the annotation pilot

1. Agree on the [annotation guide](ANNOTATION-GUIDE.md) using separate practice
   examples. Record reviewer identities consistently; use distinct real reviewers.
2. Reviewer A enters a name and explicitly selects Human. Read both descriptions,
   constraints and source limitations. Complete all four relationship dimensions,
   evidence and rationale. Claim independence only before seeing others' judgments.
3. Reviewer B repeats independently under a different name. Others' labels and
   retrieval ranks are hidden until an independent human review is submitted.
   New reviewer/problem selection clears previous form values.
4. For an unresolved native identity, inspect suggested original problem URLs
   yourself using legitimate access. Record an evidence-backed proposal with the
   exact catalog key and canonical URL. Do not infer equivalence from title alone.
   Two human reviewers must agree on the same subject/key before identity acceptance.
5. A **third human adjudicator** chooses Reveal reviews, reads both records,
   and submits their IDs, explicit decision, evidence and rationale. Revealing
   labels permanently blocks new independent claims by that reviewer for the pair.
   The UI currently accepts decision details as an explicit JSON object.
6. Export the record history and accepted overlay. Inspect disagreement, unknown
   rates, missing statements, and guide ambiguities before expanding beyond 40.
   Do not use this pilot to tune and then claim final-test performance.

AI assistance is stored under AI role and never satisfies independent human gates.
A direct assertion marked verified is still only a review assertion; only accepted
adjudication appears in the separate overlay. The software checks evidence fields
and canonical URL membership, not the truth of the semantic evidence. A second
submission is a non-independent correction; corrections suspend old accepted
results that referenced the earlier review. Fresh independent reviewers and new
adjudication are required to restore acceptance. Names are exact local identifiers;
this tool does not prevent a person impersonating another name.

## Backup, audit, and recovery

```sh
python3 research/verification_workbench/server.py export --output /tmp/review-backup.json
python3 research/verification_workbench/server.py restore --input /tmp/review-backup.json --db /tmp/restored-reviews.sqlite3
```

Exports contain private research evidence and should stay private. Restore only
accepts a fresh database and the exact same frozen pilot. It validates the record
hash chain, timestamps, review gates, exposure history, and derived overlay.
SQLite writes are serialized and records/exposures reject update/delete. Hash
chains detect accidental corruption; they are not authenticated external seals.
No restored overlay is automatically imported into production or historical labels.

## Validation

```sh
python3 -m unittest discover -s research/verification_workbench -v
node --check research/verification_workbench/static/app.js
```

The HTTP tests bind a temporary loopback port and use authored fixtures. The
browser smoke script requires Chrome CDP on port 9223, real pilot server on 8765,
and a separate synthetic-fixture server on 8766. It reads the real pilot only;
all test reviews use the synthetic store. Screenshot receipts stay private because
real-pilot screenshots can contain licensed descriptions. See `validation.v1.json`
and [independent review](REVIEW.md) for the measured evidence and remaining gates.

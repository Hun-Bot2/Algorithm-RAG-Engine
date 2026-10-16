# Private asset inventory

`source_registry.v1.json` is a frozen byte-level inventory, not permission to
redistribute. Every historical asset is restricted to preservation/integrity
checks pending a source/use review. Missing slug manifest is explicitly recorded.
The model manifest's ordered keys identify private historical rows; do not copy
this directory wholesale to a public/static bundle. Exact embedding revision,
license and preprocessing/input hashes are unknown, not reconstructed facts.

Verify with `python3 scripts/learning/assets.py verify`. Build a new directory with
`python3 scripts/learning/assets.py bundle --destination /tmp/learning-release`.
The initial bundle is intentionally empty. Publication requires an explicit
approved rights status, permitted hosted use and hosted_release=true. A checksum
mismatch or newly appeared missing file requires a reviewed new registry version;
the verifier never rewrites the baseline. No model weights are downloaded.

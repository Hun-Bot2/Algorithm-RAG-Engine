"""Apply the pre-registered selection rule to a saved evaluation (no new judgment is made here).

    .venv/bin/python -m research.language_bias.select_representation \
        --evaluation research/language_bias/receipts/evaluation.v1.json \
        --output research/language_bias/receipts/selection.v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research.language_bias import metrics as M

CONTROLS = ("random", "shuffled")
BASELINE = "C0"


def build_table(evaluation: dict) -> tuple[dict, dict]:
    table, sizes = {}, {}
    for name, c in evaluation["candidates"].items():
        cons = c["cross_lingual_neighbor_consistency"]["overlap"]
        coherence = c["pure_japanese_concept_coherence"]
        diff = c.get("consistency_diff_vs_C0")
        table[name] = {
            "english_retention": c["english_quality_retention"],
            "mrr": c["translation_retrieval"]["mrr"]["mean"],
            "consistency": cons["mean"],
            "consistency_halfwidth": (cons["hi"] - cons["lo"]) / 2,
            "consistency_diff_vs_baseline_lo": diff["lo"] if diff else None,
            "coherence_lift": coherence["lift"],
            "coherence_coverage": coherence["tagged_slot_coverage"],
        }
        sizes[name] = c["parameters_millions"]
    return table, sizes


def decide(evaluation: dict) -> dict:
    table, sizes = build_table(evaluation)
    decision = M.apply_selection_rule(table, baseline=BASELINE, controls=CONTROLS, sizes=sizes)
    decision["table"] = table
    decision["homophily_and_mixing_used_in_selection"] = False
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    decision = decide(evaluation)
    decision["evaluation_sha256"] = hashlib.sha256(args.evaluation.read_bytes()).hexdigest()
    decision["rule"] = "specs/unified-similarity-map/language-bias.md#pre-registered-selection-rule-fixed-before-any-new-result"
    with open(args.output, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(decision, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: decision[k] for k in ("eligible", "ineligible_reasons", "winner", "replaced", "gain_vs_baseline", "tied", "note")
                      if k in decision}, indent=1))


if __name__ == "__main__":
    main()

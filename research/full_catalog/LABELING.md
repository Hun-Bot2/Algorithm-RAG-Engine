# Judged-pool labeling guideline v1

Applies to `pool.v1.json` tier-1 pairs. Labels are **AI-reviewed algorithm
judgments** (`label_source: "ai_reviewed"`). They are not human gold and not learning
outcomes. Pass A (draft) and pass B (review) label each pair blind and independently.
Disagreements are resolved by rule or by an adjudication pass, described below.

## Evidence allowed

- Only the fields in the batch file: platform key, title, tags, native difficulty
  and URL (as an identifier).
- The labeler's own recollection of a problem identified by platform, ID and title.
  This is allowed only when the labeler recognizes that exact problem. Set
  `basis: "recognized_problem"` in that case, otherwise `"metadata_only"`.
- **Not allowed:**
  - fetching URLs, web search, or any network access;
  - reading the quarantined LeetCode statement fields;
  - guessing a statement from the title and presenting it as recognized.

## Question

Would a learner who studied the **query** problem benefit from practising the
**target** problem *because they share the same core algorithmic idea or technique*?
Grade the algorithm, not the story, wording or difficulty.

| label | meaning |
|---|---|
| 3 `strongly_relevant` | Same core technique and problem structure (the same pattern applied again). |
| 2 `relevant` | Shares the main technique; structure or setting differs. |
| 1 `partially_related` | Shares a secondary technique or a broad family only (for example, both are DP of unrelated kinds). |
| 0 `unrelated` | No meaningful shared algorithmic idea. |
| `null` + `abstain_reason` | The evidence is insufficient to judge (typically an unrecognized AtCoder title with no tags). Never guess. |

- `deceptive: true` marks a pair whose titles or narratives look alike but whose
  algorithms differ. It is only possible with label 0 or 1. Otherwise it is `false`.
  With `null` it is `null`.
- `confidence`: `high`, `medium` or `low`.
- `rationale`: at most 20 words, naming techniques only, for example "both prefix-sum
  over 2D grid". Never quote or paraphrase statement text.

## Output (one JSON object per line, every pair in the batch, same order)

```json
{"pair_id": "…", "label": 0, "deceptive": false, "basis": "metadata_only",
 "confidence": "low", "rationale": "…", "abstain_reason": null}
```

## Resolution

- Both passes give the same label (abstentions included) and the same deceptive
  flag: the label is final.
- Both passes give non-null labels that differ by 1, with the same deceptive flag:
  the lower label is final (`resolution: "conservative_min"`).
- Any other disagreement goes to an adjudication pass. The adjudicator sees both
  labels and rationales, under the same evidence rules.
- A final `null` stays unjudged. It is never counted as a negative.

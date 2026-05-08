# Ralph Setup

This directory contains the repo-local Ralph-style scaffold for Algorithm RAG
Engine.

This repository is Codex-first. Ralph is used here as a manual iterative
execution workflow after SDD planning. It should not replace requirements,
design review, validation, or repository memory.

## Files

- `ralph.sh` - local validator/status helper.
- `prd.json` - current Ralph story state.
- `progress.txt` - append-only learnings and iteration history.

Reusable Codex instructions live in:

```text
.agents/skills/ralph-sdd/SKILL.md
```

## Prerequisites

- Git.
- `jq`.

## Manual Codex-First Flow

1. Ask Codex to use `$ralph-sdd`, or read
   `.agents/skills/ralph-sdd/SKILL.md` manually if the skill has not been picked
   up yet.
2. Read `scripts/ralph/prd.json`.
3. Run:

```bash
./scripts/ralph/ralph.sh next
```

4. Complete exactly one story in Codex.
5. Validate the story.
6. Update `prd.json` and append to `progress.txt`.

Automatic Codex CLI execution is intentionally not implemented yet. Add it only
after confirming the local `codex` command and its stable non-interactive
invocation.

## Local Validation

```bash
bash -n scripts/ralph/ralph.sh
./scripts/ralph/ralph.sh validate
python3 -m json.tool scripts/ralph/prd.json
```

## Safety Rules

- One story per iteration.
- No live Slack/OpenAI/S3/Docker push side effects unless explicitly approved.
- No online judge execution or database migrations without specs and validation.
- Do not mark stories passing without validation.

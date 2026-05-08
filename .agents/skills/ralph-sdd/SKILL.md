---
name: ralph-sdd
description: Run Codex-first Ralph-style Spec-Driven Development iterations in this repository. Use when the user asks to use Ralph, execute the next prd.json story, update progress.txt, run an SDD iteration, or manage small validated stories through scripts/ralph.
---

# Ralph SDD

Use this skill for manual Codex-first Ralph-style execution. The root
`AGENTS.md` remains the repository-level operating memory and takes precedence.

Automatic Codex CLI execution is not assumed. Until the local `codex` CLI and
its stable non-interactive invocation are confirmed, use this skill manually
inside Codex.

## Required Files

- `AGENTS.md` - repository-level operating memory.
- `scripts/ralph/prd.json` - active story state.
- `scripts/ralph/progress.txt` - active iteration memory.
- `scripts/ralph/ralph.sh` - local validation/status helper.

## Manual Iteration Loop

1. Read root `AGENTS.md`.
2. Read `scripts/ralph/prd.json`.
3. Read `scripts/ralph/progress.txt`, especially `## Codebase Patterns`.
4. Run `./scripts/ralph/ralph.sh next` to identify the next story.
5. Pick exactly one highest-priority user story where `passes` is `false`.
6. Before modifying files, state the exact task, affected files, and validation
   strategy.
7. Implement only that one story.
8. Run the story's validation commands.
9. Update docs or `AGENTS.md` if reusable repository memory was discovered.
10. Update `scripts/ralph/prd.json` only after validation passes.
11. Append a progress entry to `scripts/ralph/progress.txt`.
12. Suggest a focused commit message.

If all stories already pass, respond with:

```text
COMPLETE
```

## Validation Commands

Scaffold validation:

```bash
bash -n scripts/ralph/ralph.sh
./scripts/ralph/ralph.sh validate
python3 -m json.tool scripts/ralph/prd.json
```

Default Python syntax validation for production-code stories:

```bash
python3 -m compileall src scripts
```

Use smaller story-specific checks when possible.

Do not run live Slack delivery, OpenAI calls, S3 sync, Docker pushes, or
untrusted code execution unless the story explicitly requires it and the user has
approved the necessary credentials and side effects.

## Progress Entry Format

Append to `scripts/ralph/progress.txt`; never replace the file.

```text
## YYYY-MM-DD HH:MM - STORY-ID - Story title
- What was implemented:
- Files changed:
- Validation:
- Learnings for future iterations:
---
```

## Repository Memory

Add reusable learnings to `## Codebase Patterns` near the top of
`scripts/ralph/progress.txt`. If a learning should affect all future agents,
also update root `AGENTS.md`.

Do not add story-specific implementation notes to `AGENTS.md`.

## Story Rules

- Work on one story per iteration.
- Keep changes small and deterministic.
- Do not commit or suggest committing broken code.
- Do not include unrelated user changes.
- Do not mark a story passing without validation.

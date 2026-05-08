# Ralph Instructions for Codex

Use this file for manual Codex-first Ralph-style execution. The root
`AGENTS.md` remains the repository-level operating memory and takes precedence.

Automatic Codex CLI execution is not assumed yet. Until the local `codex` CLI
and its stable non-interactive invocation are confirmed, use this file manually
inside Codex.

## Manual Iteration Loop

1. Read root `AGENTS.md`.
2. Read `scripts/ralph/prd.json`.
3. Read `scripts/ralph/progress.txt`, especially `## Codebase Patterns`.
4. Pick exactly one highest-priority user story where `passes` is `false`.
5. Before modifying files, state:
   - Exact task.
   - Affected files.
   - Validation strategy.
6. Implement only that one story.
7. Run the story's validation commands.
8. Update docs or `AGENTS.md` if reusable repository memory was discovered.
9. Update `scripts/ralph/prd.json` only after validation passes.
10. Append a progress entry to `scripts/ralph/progress.txt`.
11. Suggest a focused commit message.

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

---
name: log-and-ship
description: Persist a chunk of work into the project's second brain and version control — update the right docs/*.md and the cross-session memory, then commit (and push only if asked). Use when Leon says "log", "log and commit", "log doc commit", "update the docs", "commit this", "push it", or otherwise wants the just-finished experiment/code change recorded and shipped. Encodes which docs to touch, the commit conventions, and the commit-vs-push and stage-what rules.
---

# Log and ship

Leon treats `docs/*.md` and the auto-memory as a **second brain** — a set of lazy-loaded
Markdown files that carry context across sessions. A chunk of work isn't "done" until it's
written there AND committed. This skill is the checklist for doing that consistently.

Run it whenever a unit of work concludes (a sweep finished + analysed, a code change made, a
tool built) and Leon asks to log / commit / push. **Do the doc/memory update even when he only
says "commit"** — an undocumented commit is a half-logged change.

## Golden rules (do not skip)

- **Commit only when asked. Push ONLY when explicitly asked.** "log and commit" ≠ push. He
  pushes deliberately; wait for "push it".
- **Every commit message ends with the trailer:**
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  ```
- **Write docs directly.** The `ask-kimi`/cheap-worker delegation for docs is BROKEN
  (OpenAI client init error) — do not try to route doc writing through it. Edit the `.md` yourself.
- **Verify git state before asserting "everything is committed"** — run `git status`/`git log`,
  don't trust memory. Report honestly if something is still dirty.
- **Don't commit others' in-flight edits blind.** If a source file has changes you didn't make
  (or a corrupted artifact), flag it and ask before staging — see the 2026-07-20 tasks.py episode.

## Step 1 — update the right doc(s)

Pick by what the work was (full map in `CLAUDE.md`):

| Work | Doc to update |
|---|---|
| A sweep ran (config, result, status) | `docs/experiment_log.md` — the sweep inventory (append a dated block) |
| Active ring→lower-plane / go-vs-nogo / attractor thread | `docs/ring_lowerplane_log.md` (most-current narrative detail) |
| New model/task/loss/nonlinearity | `docs/architecture.md` and/or `docs/nonlinearities.md` |
| New run/sweep/GPU procedure | `docs/running.md` |
| New plot flag / figure type | `docs/analysis.md` |
| New math (potential, bifurcation) | `docs/theory_landscape.md` (render with `./make_pdf.sh <doc>.md`) |

Rules for the entry: **date it** (absolute date), state the config/levers, the result numbers,
and the status/next-step. If a change **supersedes or contradicts** an older entry, say so
explicitly at the old spot (don't silently leave stale claims — the next session will trust them).
Convert "today/last week" to absolute dates.

## Step 2 — update the cross-session memory (the OTHER second brain)

Location: `/home/leon/.claude/projects/-home-leon-rnn/memory/`

- **`project_state.md`** — the live status memory (current sweeps, what's built, open issues,
  latest result). Add/refresh the relevant section; keep it compact and point to
  `docs/experiment_log.md` (etc.) for the long form. Mark superseded claims.
- **`MEMORY.md`** — one-line index. Add a pointer line ONLY if you created a *new* memory file;
  never put memory content in the index. Existing pointers rarely need touching.

Skip this step only for a trivial mechanical commit (typo, comment fix) that changes no state
worth recalling.

## Step 3 — stage, commit, (push)

**Stage source + docs, not bulk result dumps.** Leon versions `results/` and `results/figures/`
separately and usually leaves `CLAUDE.md` and `.claude/settings.local.json` uncommitted. So:

```bash
git add <the source files> <the docs you edited>      # e.g. src/tasks.py docs/experiment_log.md
git status --short                                     # confirm nothing unintended staged
```

Do NOT `git add -A` / `git add .` — it sweeps in results, logs, and settings. Add named paths.
(The memory files under `~/.claude/...` are outside the repo and are never staged.)

Commit with a concise subject + a body explaining WHAT changed and WHY, plus the trailer:

```bash
git commit -m "$(cat <<'EOF'
<subject: what changed, imperative, ~65 chars>

<body: the why, the key result/mechanism, any "not yet re-swept" caveat>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
git log --oneline -1        # confirm
```

**Push only if asked:**
```bash
git push        # origin main — only after an explicit "push it"
```

## Step 4 — report

Tell Leon, in the final message: the commit hash + one-line subject, which docs/memory you
updated, and what (if anything) is still dirty and why (e.g. "results/ left unstaged as usual").
If you pushed, say so; if you didn't, note it's local-only pending "push it".

# Web App — Corpus, Tagging & Generation UI

## Goal

One local web UI over the existing Python pipeline covering the full loop:
**ingest WADs → browse/tag maps → run training → generate WADs → download/
preview results.** Everything the CLIs do today, but visual — tagging with
previews side by side is the workflow that actually needs a UI (readline
autocomplete is a poor substitute for clicking a map thumbnail).

## Non-goals (v1)

- Multi-user/auth/remote hosting. This binds to localhost for one user.
  (If exposed later: auth + CSRF + upload limits become Phase W6.)
- In-browser 3D preview. Top-down SVGs already exist and are good enough
  to tag by; a WebGL viewer is a stretch goal.
- Editing map geometry. This is a corpus/tagging/generation console, not
  an editor.

## Stack

- **Backend: FastAPI** — async, wraps the existing modules directly
  (wad_parser, tagging, data, models, generation, wad_writer are all
  importable in-process; no RPC layer needed).
- **Frontend: server-rendered Jinja templates + HTMX** (+ a sprinkle of
  Alpine.js for client-only state like tag-input chips). Rationale: no
  Node toolchain in the repo, the interactions are CRUD + polling, and
  SVG previews render natively. Revisit React only if the UI outgrows
  this (e.g. the WebGL viewer).
- **No database.** The existing on-disk stores ARE the state:
  `corpus/wads/` (+ provenance JSON from the scraper), `dataset/maps/`
  (extracted JSON), `tags/` (registry + assignments + auto layer),
  `checkpoints/`, `generated/`. The app reads/writes through the same
  code paths as the CLIs, so CLI and web usage stay interchangeable. A
  tiny `webapp_state.json` holds job history only.

## Long-running work: job model

Training (hours) and generation (seconds-minutes) must not block requests.

- A single in-process **job queue** (one worker thread; GPU jobs are
  serialized anyway on one card). Job = {id, kind, args, status, log tail,
  artifacts}. Persisted to `webapp_state.json` so a restart shows history.
- Training runs as a **subprocess** (`train-layout ...`) with its stdout
  tailed into the job log — crash isolation from the web process, and the
  CLI stays the single source of truth for training behavior.
- Generation/parse/auto-tag run in-process on the worker thread (fast,
  and they share parsed caches).
- UI polls job status via HTMX (`hx-trigger="every 2s"`); no websockets
  needed at this scale.

## Pages & endpoints

| Page | What it does | Backing calls |
|---|---|---|
| **Corpus** | drag-drop WAD upload; scraper launcher (dir/search + dry-run preview of license verdicts); list of WADs with provenance/license badges | `WadFile`, `extract_all_maps` → JSON into `dataset/maps/`, `tools/idgames_scraper` |
| **Maps** | grid of map cards (SVG preview thumbnails, map name, WAD, effective tags with auto/user provenance markers) with filter-by-tag | `tagging.store.effective_tags_detailed`, `tagging.preview` |
| **Map detail** | large preview, raw stats (sectors/things/format tier), tag editor: add/remove with registry autocomplete, weights, exclude-inherited, veto-auto | `TagStore` mutations; `registry.complete` for typeahead |
| **Tags** | registry table (usage counts, descriptions, auto-badge), rename/merge with corpus-wide preview of what changes | `rename_tag`/`merge_tags`, `tag_usage` |
| **Train** | start/monitor training jobs (corpus size, resolution, steps, CFG dropout), live loss log tail, checkpoint list | job queue → `train-layout` subprocess |
| **Generate** | pick checkpoint, compose tag set (chips with weights), n/steps/guidance/seed; result panel shows per-map SVG previews + repairs summary + download link | `generation.pipeline.generate`, `tagging.preview` |

API is JSON-first (`/api/...`) with the HTML pages calling the same
endpoints — keeps a future React/CLI/script consumer free.

## Upload & ingest flow

1. POST WAD → hash → skip if `dataset/maps/<hash>` already present.
2. Parse to per-map JSON + texture-set doc (same as `parse-wad`).
3. Run `tag-maps auto` derivation automatically (facts layer only).
4. Render preview SVGs once, cache next to the JSON.
5. Card appears in Maps grid; user tags at leisure.

Scraped WADs enter the same flow, carrying their provenance JSON; the
license verdict is displayed on the WAD card permanently (reuse-allowed
corpus vs. training-only shelf must stay visually distinct).

## Module layout

```
webapp/
├── main.py            # FastAPI app factory, routes -> handlers
├── jobs.py            # queue, worker thread, subprocess runner, state file
├── ingest.py           # upload/scrape -> parse -> auto-tag -> previews
├── api/                # JSON endpoints (corpus, maps, tags, train, generate)
├── templates/          # Jinja + HTMX pages
└── static/             # htmx.min.js, alpine.min.js, one stylesheet
cli/
└── serve.py            # `doom-gen-serve --port 8080` entry point
```

## Phases

- **W1 — read-only console**: corpus list, map grid with previews,
  tag display. (Proves ingest + rendering; no mutations yet.)
- **W2 — tagging UI**: full tag editor incl. autocomplete, weights,
  exclusions, vetoes, rename/merge. Checkpoint: tag a megawad end-to-end
  faster than the CLI.
- **W3 — ingest**: upload + scraper launcher with license badges.
- **W4 — jobs**: train + generate pages with polling logs, checkpoint
  management, WAD download.
- **W5 — polish/stretch**: corpus stats dashboards (tag distributions,
  feature-tier counts), diff view for rename/merge preview, WebGL viewer.

W1-W3 need nothing beyond FastAPI+Jinja+HTMX; W4 reuses the job model
above. Each phase ships usable.

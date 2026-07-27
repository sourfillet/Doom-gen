"""Scrape the /idgames archive for WADs whose upload permissions allow
modification/reuse, and download them with provenance records.

Uses the official Doomworld idgames API (JSON), not HTML scraping:
    https://www.doomworld.com/idgames/api/

License filtering: every idgames upload carries a standard text-file
template with a "Copyright / Permissions" section. The canonical phrasing
is one of:

    Authors MAY use the contents of this file as a base for modification
    or reuse. ...
    Authors may NOT use the contents of this file as a base for ...

Only entries with an affirmative MAY (and no NOT) are downloaded. The
matched permission text is stored alongside each download in a
provenance JSON so the "this one allows reuse" judgment stays auditable -
if a WAD's tag data or geometry ends up in a future project, the evidence
of permission travels with it.

Usage:
    python tools/idgames_scraper.py --dir levels/doom2/a-c --limit 10 --dry-run
    python tools/idgames_scraper.py --dir levels/doom --out corpus/wads
    python tools/idgames_scraper.py --search "boom" --search-type title --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = "https://www.doomworld.com/idgames/api/api.php"
# verified working 2026-07: ftp.fu-berlin.de, youfailit.net, gamers.org
DEFAULT_MIRROR = "https://ftp.fu-berlin.de/pc/games/idgames/"
USER_AGENT = "doom-gen corpus scraper (respectful; 1 req/sec)"
RATE_LIMIT_S = 1.0

# Canonical permission grammar from the idgames text-file template.
_MAY_RE = re.compile(
    r"authors?\s+(?:\(?s\)?\s+)?may\s+use\s+(?:the\s+)?contents", re.IGNORECASE)
_MAY_NOT_RE = re.compile(
    r"authors?\s+(?:\(?s\)?\s+)?may\s+not\s+use\s+(?:the\s+)?contents", re.IGNORECASE)

_last_request = [0.0]


def _api(params: dict) -> dict:
    elapsed = time.time() - _last_request[0]
    if elapsed < RATE_LIMIT_S:
        time.sleep(RATE_LIMIT_S - elapsed)
    _last_request[0] = time.time()
    url = f"{API}?{urllib.parse.urlencode({**params, 'out': 'json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def classify_permissions(textfile: str | None) -> tuple[str, str]:
    """Returns (verdict, evidence). Verdicts: "reuse-allowed",
    "reuse-forbidden", "unclear" (no recognizable statement - treated as
    NOT downloadable; absence of permission is not permission)."""
    if not textfile:
        return "unclear", ""
    if _MAY_NOT_RE.search(textfile):
        m = _MAY_NOT_RE.search(textfile)
        return "reuse-forbidden", textfile[m.start(): m.start() + 200]
    m = _MAY_RE.search(textfile)
    if m:
        return "reuse-allowed", textfile[m.start(): m.start() + 200]
    return "unclear", ""


def iter_directory(dir_path: str):
    """Yields file metadata dicts for every entry under an idgames
    directory (recursing into subdirectories)."""
    content = _api({"action": "getcontents", "name": dir_path}).get("content") or {}
    files = content.get("file") or []
    if isinstance(files, dict):  # single-entry dirs come back un-listed
        files = [files]
    yield from files
    subdirs = content.get("dir") or []
    if isinstance(subdirs, dict):
        subdirs = [subdirs]
    for sub in subdirs:
        yield from iter_directory(sub["name"])


def search(query: str, search_type: str = "title"):
    content = _api({"action": "search", "query": query, "type": search_type,
                    "sort": "date", "dir": "desc"}).get("content") or {}
    files = content.get("file") or []
    return [files] if isinstance(files, dict) else files


def fetch_details(entry: dict) -> dict:
    """getcontents/search omit the textfile; action=get includes it."""
    if entry.get("textfile"):
        return entry
    return (_api({"action": "get", "id": entry["id"]}).get("content")) or entry


def download_and_extract(entry: dict, out_dir: Path, mirror: str) -> list[str]:
    url = mirror.rstrip("/") + "/" + entry["dir"] + entry["filename"]
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    extracted = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".wad") and not name.endswith("/"):
                target = out_dir / f"{entry['id']}_{Path(name).name}"
                target.write_bytes(zf.read(name))
                extracted.append(str(target))
    return extracted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="idgames directory, e.g. levels/doom2/a-c")
    src.add_argument("--search", help="search query instead of a directory walk")
    ap.add_argument("--search-type", default="title",
                    choices=["title", "filename", "author", "description"])
    ap.add_argument("--out", type=Path, default=Path("corpus/wads"))
    ap.add_argument("--mirror", default=DEFAULT_MIRROR)
    ap.add_argument("--limit", type=int, default=None, help="max downloads this run")
    ap.add_argument("--dry-run", action="store_true",
                    help="classify permissions but download nothing")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "scrape_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": {}}

    entries = search(args.search, args.search_type) if args.search else iter_directory(args.dir)

    stats = {"reuse-allowed": 0, "reuse-forbidden": 0, "unclear": 0, "downloaded": 0}
    for entry in entries:
        eid = str(entry["id"])
        if eid in state["done"]:
            continue
        if args.limit is not None and stats["downloaded"] >= args.limit:
            break
        entry = fetch_details(entry)
        verdict, evidence = classify_permissions(entry.get("textfile"))
        stats[verdict] += 1
        label = f"[{verdict:15s}] {entry['dir']}{entry['filename']} - {entry.get('title') or '?'}"
        print(label)

        record = {
            "id": entry["id"], "dir": entry["dir"], "filename": entry["filename"],
            "title": entry.get("title"), "author": entry.get("author"),
            "date": entry.get("date"), "url": entry.get("url"),
            "license_verdict": verdict, "license_evidence": evidence,
            "wads": [],
        }
        if verdict == "reuse-allowed" and not args.dry_run:
            try:
                record["wads"] = download_and_extract(entry, args.out, args.mirror)
                stats["downloaded"] += 1
                print(f"    -> {len(record['wads'])} wad(s)")
            except Exception as exc:  # noqa: BLE001 - keep scraping, record failure
                record["download_error"] = str(exc)
                print(f"    -> download failed: {exc}")
        (args.out / f"{eid}.provenance.json").write_text(json.dumps(record, indent=2))
        state["done"][eid] = verdict
        state_path.write_text(json.dumps(state, indent=2))

    print(f"\nsummary: {stats}")
    print("NOTE: 'unclear' entries are skipped by design - absence of a "
          "permission statement is not permission.")


if __name__ == "__main__":
    main()

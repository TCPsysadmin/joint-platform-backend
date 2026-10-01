"""Knowledge-base tool: index the user library and check the LLM wiki against it.

Layout (per workspace):
    knowledge/workspaces/<slug>/library/**.md   user-organized sources, each with a stable `id`
    knowledge/workspaces/<slug>/wiki/**.md      LLM-maintained pages that cite sources by id
    knowledge/workspaces/<slug>/index.json      generated: id -> current library path

Users may move or rename library files freely; the wiki cites ids, never paths,
so `index` re-resolves every id and `check` catches anything that broke.

Usage:
    python tools/kb/kb.py new-id
    python tools/kb/kb.py index [--root knowledge]
    python tools/kb/kb.py check [--root knowledge]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ID_RE = re.compile(r"^src_[0-9]{8}_[0-9a-z]{6}$")
CITE_RE = re.compile(r"\[\[(src_[0-9]{8}_[0-9a-z]{6})\]\]")
SOURCE_TYPES = {"transcript", "summary", "document", "note"}


def new_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    return f"src_{now:%Y%m%d}_" + "".join(secrets.choice(alphabet) for _ in range(6))


def parse_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Parse a small YAML subset: `key: value` and `key: [a, b]` between --- fences."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return {}, text
    meta: dict[str, object] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        value = raw.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key.strip()] = [_unquote(v.strip()) for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key.strip()] = _unquote(value)
    return meta, "\n".join(lines[end + 1 :])


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def scan_library(workspace: Path, report: Report) -> dict[str, dict[str, object]]:
    library = workspace / "library"
    entries: dict[str, dict[str, object]] = {}
    for path in sorted(library.rglob("*.md")):
        rel = path.relative_to(workspace).as_posix()
        text = path.read_text(encoding="utf-8")
        meta, _ = parse_front_matter(text)
        sid = meta.get("id")
        if not isinstance(sid, str) or not ID_RE.match(sid):
            report.errors.append(f"{rel}: missing or malformed `id` (run `kb.py new-id`)")
            continue
        if sid in entries:
            report.errors.append(f"{rel}: duplicate id {sid} (also {entries[sid]['path']})")
            continue
        stype = meta.get("type")
        if stype not in SOURCE_TYPES:
            report.errors.append(f"{rel}: `type` must be one of {sorted(SOURCE_TYPES)}")
        entries[sid] = {
            "path": rel,
            "title": meta.get("title") or path.stem,
            "type": stype,
            "media": meta.get("media"),
            "summary_of": meta.get("summary_of"),
            "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        }
    for sid, entry in entries.items():
        target = entry["summary_of"]
        if target and target not in entries:
            report.errors.append(f"{entry['path']}: summary_of {target} is not in the library")
    return entries


def check_wiki(workspace: Path, entries: dict[str, dict[str, object]], report: Report) -> None:
    wiki = workspace / "wiki"
    cited: set[str] = set()
    for path in sorted(wiki.rglob("*.md")):
        rel = path.relative_to(workspace).as_posix()
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        listed = meta.get("sources") or []
        listed = listed if isinstance(listed, list) else [listed]
        inline = set(CITE_RE.findall(body))
        for sid in set(listed) | inline:
            if sid not in entries:
                report.errors.append(f"{rel}: cites unknown source {sid}")
        for sid in inline - set(listed):
            report.warnings.append(f"{rel}: cites {sid} inline but not in `sources`")
        cited |= set(listed) | inline
    for sid, entry in entries.items():
        if sid not in cited and entry["type"] != "summary":
            report.warnings.append(f"{entry['path']}: not cited by any wiki page yet")


def workspaces(root: Path) -> list[Path]:
    base = root / "workspaces"
    return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


def run(root: Path, write_index: bool) -> Report:
    report = Report()
    for ws in workspaces(root):
        entries = scan_library(ws, report)
        check_wiki(ws, entries, report)
        index_path = ws / "index.json"
        rendered = json.dumps(entries, indent=2, sort_keys=True) + "\n"
        if write_index:
            index_path.write_text(rendered, encoding="utf-8", newline="\n")
        elif not index_path.exists() or index_path.read_text(encoding="utf-8") != rendered:
            report.errors.append(f"{index_path.relative_to(root).as_posix()}: stale, run `kb.py index`")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["new-id", "index", "check"])
    parser.add_argument("--root", default="knowledge", type=Path)
    args = parser.parse_args(argv)

    if args.command == "new-id":
        print(new_id())
        return 0

    report = run(args.root, write_index=args.command == "index")
    for w in report.warnings:
        print(f"warning: {w}")
    for e in report.errors:
        print(f"error: {e}")
    print(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())

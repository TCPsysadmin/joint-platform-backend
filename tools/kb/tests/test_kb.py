import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kb  # noqa: E402

A = "src_20260101_aaaaaa"
B = "src_20260101_bbbbbb"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_ws(tmp_path: Path) -> Path:
    ws = tmp_path / "workspaces" / "acme"
    write(ws / "library" / "calls" / "a.md", f"---\nid: {A}\ntype: transcript\n---\nbody\n")
    write(ws / "wiki" / "t.md", f"---\nsources: [{A}]\n---\nsee [[{A}]]\n")
    return ws


def test_new_id_matches_pattern():
    assert kb.ID_RE.match(kb.new_id())


def test_front_matter_lists_and_quotes():
    meta, body = kb.parse_front_matter('---\na: [x, "y"]\nb: \'z\'\n---\nhi')
    assert meta == {"a": ["x", "y"], "b": "z"}
    assert body == "hi"


def test_moving_a_library_file_keeps_wiki_valid(tmp_path):
    ws = make_ws(tmp_path)
    assert not kb.run(tmp_path, write_index=True).errors
    (ws / "library" / "calls" / "a.md").rename(ws / "library" / "renamed.md")
    assert not kb.run(tmp_path, write_index=True).errors
    assert '"path": "library/renamed.md"' in (ws / "index.json").read_text()


def test_stale_index_fails_check(tmp_path):
    ws = make_ws(tmp_path)
    kb.run(tmp_path, write_index=True)
    (ws / "library" / "calls" / "a.md").rename(ws / "library" / "moved.md")
    errors = kb.run(tmp_path, write_index=False).errors
    assert any("stale" in e for e in errors)


def test_unknown_citation_and_duplicate_id(tmp_path):
    ws = make_ws(tmp_path)
    write(ws / "library" / "dup.md", f"---\nid: {A}\ntype: note\n---\n")
    write(ws / "wiki" / "bad.md", f"---\nsources: [{B}]\n---\n")
    errors = kb.run(tmp_path, write_index=True).errors
    assert any("duplicate id" in e for e in errors)
    assert any(f"unknown source {B}" in e for e in errors)


def test_missing_id_and_bad_summary_of(tmp_path):
    ws = make_ws(tmp_path)
    write(ws / "library" / "noid.md", "no front matter\n")
    write(ws / "library" / "s.md", f"---\nid: {B}\ntype: summary\nsummary_of: src_20990101_zzzzzz\n---\n")
    errors = kb.run(tmp_path, write_index=True).errors
    assert any("noid.md: missing" in e for e in errors)
    assert any("summary_of" in e for e in errors)

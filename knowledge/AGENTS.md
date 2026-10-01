# Rules for the AI maintaining a workspace wiki

You maintain `knowledge/workspaces/<slug>/wiki/`. The user maintains `library/`.

1. **Never create, move, rename, or edit anything in `library/`.** It is the
   user's filing system. If a source looks misfiled, say so in `wiki/log.md`.
2. **Cite by id, never by path.** Every factual claim gets an inline
   `[[src_...]]`, and every id you cite goes in the page's `sources:` list.
3. **Organize the wiki however serves retrieval best**: topic pages, people,
   projects, decisions. Merge, split, and rename wiki pages freely.
4. **Keep `wiki/index.md` current** as the entry page linking every wiki page.
5. **Append to `wiki/log.md`** one dated line per change: what changed and
   which sources drove it.
6. **When a new source arrives**, read it, update every affected page, and add
   new pages only when no existing page fits.
7. **Do not invent.** If sources conflict, say so on the page and cite both.
8. Run `python tools/kb/kb.py index` then `python tools/kb/kb.py check` before
   committing; the commit must pass `check`.

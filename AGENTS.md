# AGENTS.md

Working agreements for agents operating in this repository.

## Tools available on this machine

### ocrc — PDF/paper → readable markdown

`ocrc` (installed at `~/.local/bin/ocrc`, source `~/ocrc-feature/ocrc.py`) is a
CLI client for a local dots.mocr parsing service (default
`http://127.0.0.1:8601`, override with `OCRC_SERVER`). Stdlib-only, always
available on this host.

```bash
ocrc parse https://arxiv.org/pdf/<id> --out /tmp/ocrc_out > result.tsv 2> log.txt
```

- Output: `<out>/<sha12>/document.md` + `images/` + layout JSON. The service
  deduplicates by content hash — re-parsing the same document is a cache hit.
- **Stream discipline**: with exactly one input and piped stdout, the zip
  bundle goes to stdout and `--out` is **ignored**; never combine
  `> file 2>&1` (the tool refuses). Redirect stdout to a `.zip` and unpack
  it (`unzip -d <dir> out.zip`), or pass `--quiet` and keep stdout on a TTY.
- **Failure modes seen**: arXiv downloads occasionally fail on flaky DNS —
  retry with `curl --retry 3 --retry-all-errors` to a local PDF and parse the
  file. `ocrc` exits non-zero on the first failed document, so batch loops
  need per-document retries, not one multi-arg call. (Historical: pages whose
  embedded images exceed 30M px were rejected by a false-positive guard —
  DeepSeek-R1 died at page 16, Llama 3 at page 2; fixed upstream in
  RepnikovPavel/ocr PR #17, all papers now parse as single full-document
  tasks.)
- **`--split N`** fans one document across N page-range tasks and merges the
  bundles. Works as a workaround for per-task engine crashes, but verify the
  merge: a chunk collision once produced a `document.md` with pages 11–21
  replaced by a duplicate of pages 22–32. After any parse, check
  `meta.json.pages_done` against the set of `layout/*page_N.json` files.
- Other commands: `ocrc queue`, `ocrc watch`, `ocrc search "<text>"`
  (full-text over everything ever parsed), `ocrc stats`.
- Long papers take minutes; run parses as background tasks and poll the
  output dir.

## Code and commit style

Observed conventions on branch `deepbench` — follow them:

- **English everywhere** in code, comments, docs and commit messages.
- **One docker image per stage.** Wrapper scripts in `scripts/*.sh` build
  their own image from their own `docker/DockerFile<Stage>` and run it with
  `--rm --init --user $(id -u):$(id -g)`, mounting only what the stage needs.
  Never add dependencies to an existing stage's image for a new purpose —
  create a new Dockerfile instead.
- **Comments explain why, not what**; tricky semantics (overflows, resume
  rules, dtype bounds) get a paragraph with the measured/verified fact.
- **Claims are measured.** Docs and commit messages carry real numbers
  ("93.2M tok/s", "byte-identical", "99.80% vocab overlap"). Never state a
  performance or parity claim you did not verify on this machine.
- **Commit messages**: one-line subject, dense, with the outcome/numbers in
  parentheses; no conventional-commits prefixes, no body unless needed.
  Examples: `count_tokens: 31.5->23.3 min (flat-cache + readahead + 256MiB
  batches), verified same 176.1B tokens`.
- **Docs next to code**: each subsystem has `docs/` with pipeline notes and a
  `gotchas.md`-style file recording known issues, including deliberately
  preserved bug-for-bug parity.
- Generated docs (e.g. `scripts/docs/*.md`) are produced by their generator
  scripts — patch the generator template *and* the generated file.

## Hard rules

- **No `git commit`/`git push` without an explicit go-ahead** in the current
  conversation. Prepare the change, show it, wait.
- **Do not rebuild/replace existing docker images** for a different purpose;
  new stage → new image name.
- Long-running stage commands run in foreground streaming logs (scripts use
  `python -u` / `PYTHONUNBUFFERED=1`); indicatif/tqdm bars are invisible in
  `docker logs`, so stages print explicit progress lines — keep that pattern.

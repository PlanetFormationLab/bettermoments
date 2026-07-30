# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Version bumps

When asked to bump the version number, don't just pick one — propose a new
version following [SemVer](https://semver.org) (`MAJOR.MINOR.PATCH`) and wait
for confirmation before applying it:

- **MAJOR** — incompatible/breaking changes, including silent changes to
  numerical/scientific output that would break reproducibility (e.g. a
  different uncertainty calculation or a re-interpreted CLI flag).
- **MINOR** — new backward-compatible functionality.
- **PATCH** — backward-compatible bug fixes only.

State which category applies and why, propose the resulting version number,
and get a go-ahead before editing `bettermoments/__init__.py` and
`docs/conf.py` (both must stay in sync).

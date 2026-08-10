# Contributing to Falcata

Falcata welcomes contributions from anyone. A pull request is judged by its code,
its tests and its reasoning, not by who or what wrote it — AI-assisted patches are
held to exactly the same bar as any other, and disclosing that assistance is
appreciated but never required.

## How to contribute

- **Bug reports / feature requests:** open an issue with a minimal reproducer.
- **Pull requests:** keep them small and focused. Add at least one regression test where reasonable. Reference any related upstream LightGBM issue or PR.
- **Documentation and examples:** especially welcome — these age fastest in a fork.

## Linting

```shell
pre-commit run --all-files
```

This runs the static analyzers on changed files and auto-formats where possible.

## Relationship to upstream LightGBM

Falcata began as a fork of LightGBM and stays interoperable with it at the data boundaries (model text format, binary datasets, parameter names) -- see docs/design/format-compatibility.md. It is otherwise developed independently, and is not affiliated with or endorsed by the LightGBM maintainers.

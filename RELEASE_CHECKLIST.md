# Release Checklist

## Completed
- Added a Typer-based CLI entry point for init, ingest, embed, search, benchmark, evaluate, and stats.
- Added packaging metadata and a clearer install path.
- Added release documentation under docs/.
- Added benchmark documentation and a concise changelog.
- Added MIT licensing and contribution guidance.
- Added a CLI-focused regression test.

## Remaining
- Confirm the local database and model dependencies work for a fresh clone in the target environment.

## Known limitations
- The retrieval pipeline is functional but not a research-grade ranking system.
- Some benchmark queries remain difficult due to the underlying dataset and ingestion quality.
- The memory engine is still experimental.

## Future work (v2.0)
- Improve retrieval quality with more robust ranking and evaluation.
- Harden the ingestion pipeline for broader ChatGPT export variants.
- Expand the memory engine into a more polished public feature.

## Release readiness score
- 8/10

## Final release verdict
Weft can be tagged as v1.0.0 today if the maintainers are comfortable with a solid first release that is documented, testable, and usable locally rather than a perfect search platform.

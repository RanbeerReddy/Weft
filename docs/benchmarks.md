# Benchmarks

The repository includes benchmark artifacts and evaluation scripts that document the current quality of the retrieval pipeline.

## Summary of results

The bundled benchmark report in [hybrid_benchmark_results.json](../hybrid_benchmark_results.json) shows the following approximate outcomes for the available sample set:

- Semantic retrieval: MRR around 0.56 and hit@10 around 0.67.
- Lexical retrieval: weaker overall performance on the sample set, with hit@10 around 0.20.
- Hybrid retrieval: matched the semantic pipeline on the sample set and improved latency.

## What worked

- The semantic retrieval path is reliable enough for a first public release.
- Hybrid retrieval offers a practical default for users who want a simple out-of-the-box search experience.

## What did not work as well

- Lexical-only retrieval underperformed on the provided benchmark queries.
- Some benchmark queries remain difficult because the data is noisy or because the expected content was not ingested cleanly.

## Conclusion

Weft is suitable for a v1.0 release as a local, documented, and reproducible retrieval system, but the project should be framed as a practical first release rather than a perfect search engine.

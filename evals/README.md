# Discovery evals

Opt-in evaluation of discovery runs on fixed reference sets with real models, built on
[pydantic-evals](https://ai.pydantic.dev/evals/). The runs spend subscription quota or API credit and
reach the network (deadwax, Brave, the LLM), so they are not part of `poe check`. The offline tests in
`tests/unit/evals/` do run in `poe check`; they keep the harness working.

```bash
uv sync --all-groups                                     # the `evals` group: pydantic-evals, opentelemetry-sdk
uv run poe evals run --case slowcore-strict --depth quick --out reports/baseline
uv run poe evals run --mode agentic --model claude-code:opus --judge
uv run poe evals compare reports/baseline/pipeline.json reports/next/pipeline.json
```

`run` researches each case in pipeline and agentic mode (`--mode` picks one), prints a report per mode,
then the agentic report against the pipeline one. Settings come from `.env` like the server's; storage
is always in memory, so evals can run next to a server. Every case starts with no history.
`--model --helper-model --depth --minutes --max-usd --effort --count --parallel` override the case
settings. `--repeat` and `--concurrency` (default 1, because MusicBrainz allows one request per second)
control the runs. `--out` saves `<mode>.json` reports with each case's run and diagnostic, which
`compare` reads back.

Cases live in `datasets/discovery.yaml`. They were converted from mixtable's
`server/tools/discovery_eval_fixtures.json`, with local references turned into manual ones.

## Metrics (from each run, V1 `summarize`)

verified, leads, verified_share, distinct_artist_share, mean_similarity, mean_audio_similarity,
seconds, research_seconds, profile_seconds, cost_usd, input/cached/output tokens, web_searches,
research_searches, candidates, sonic_evidence_share, overlap. Attributes: status, failure, partial,
tracks (`✓` verified, `·` lead, with research/audio similarity), strategies, rejections.

## Evaluators

| Evaluator | Result |
|---|---|
| `RunCompleted` | assertion: the run completed; the reason gives the failure code, detail and partial flag |
| `VerifiedCount` | `verified_share` of `desired_count`; `any_verified` |
| `ExclusionsRespected` | assertions over shown tracks: excluded and reference artists, reference recordings, year range, per-artist cap |
| `KnownReferenceIds` | assertion: answers and tracks cite only the run's reference IDs |
| `Novelty` | `distinct_artist_share`, `new_artist_share` (artists not among the references) |
| `GroundedUrls` | `grounded_url_share`: the share of answer URLs that a tool returned or the agent opened |
| `CatalogCheckedBeforeAnswer` | agentic only: `catalog_checked`, `checked_share` of answered candidates passed to `verify_candidates` |
| `ToolUsage` | `tool_calls` and `calls_<tool>` counts |
| `InstructionFit` (`--judge`) | the helper model's `instruction_fit` score and pass |

The span-based evaluators read the agents' tool calls from OpenTelemetry spans that are recorded in
memory (`tracing.enable()`); nothing is exported. Without spans, these evaluators return no result.

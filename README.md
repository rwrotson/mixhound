# mixhound

Crate-digging agent: finds recordings similar to weighted references, verifies them against catalogs, and returns playable sources.

## Development

```bash
uv sync --all-groups
uv run pre-commit install
uv run poe serve     # http://127.0.0.1:8000/docs
uv run poe check
```

Configuration is read from `MIXHOUND_*` environment variables and `.env`; see `.env.example`.

## Docker

```bash
docker compose up --build
```

## Web search

Research needs a web search: a Brave Search API key (`MIXHOUND_SEARCH__BRAVE_API_KEY`), a SearXNG instance, or both (Brave first, SearXNG when it fails). A local SearXNG with JSON output and Google disabled:

```bash
docker compose --profile search up -d searxng
echo 'MIXHOUND_SEARCH__SEARXNG_URL=http://localhost:8888' >> .env
```

SearXNG sends queries to its upstream engines from your IP; results are cached for a day and queries are spaced by `MIXHOUND_SEARCH__SEARXNG_MIN_INTERVAL`.

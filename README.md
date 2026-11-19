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

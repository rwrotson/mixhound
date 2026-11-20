# Changelog

All notable changes to this project will be documented in this file.

## v0.1.0 (2026-11-20)

### Feat

- add SearXNG web search with Brave fallback and cached results
- add pydantic-evals discovery evals with span-based tool checks and mode comparison
- add session and run services, run queue lifecycle and /api/v1 routes with SSE
- add agentic mode with orchestrator agent, strategy delegation and shared finalize
- add research agents, tools and pipeline-mode research on pydantic-ai
- add LLM model factory and vendored Claude Code adapter on claude-agent-sdk
- port seeding, catalog and page verification, and preview audio matching onto deadwax and librosa
- add infrastructure adapters for deadwax catalog, safe page fetching, cache, search, storage and queue
- port discovery models, evidence rules, context builder and skills from mixtable

### Refactor

- package agent skills inside mixhound so the wheel is self-contained

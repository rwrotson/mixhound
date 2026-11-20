from mixhound.services.ports import SearchHit, SearchUnavailableError, WebSearch


class FallbackSearch:
    """Asks the secondary search whenever the primary one fails a query."""

    def __init__(self, primary: WebSearch, secondary: WebSearch) -> None:
        self.primary = primary
        self.secondary = secondary

    async def search(self, query: str, *, count: int | None = None) -> list[SearchHit]:
        """Return the primary search's results, else the secondary's."""
        try:
            return await self.primary.search(query, count=count)
        except SearchUnavailableError as first:
            try:
                return await self.secondary.search(query, count=count)
            except SearchUnavailableError as second:
                raise SearchUnavailableError(f"{first}; fallback: {second}") from second

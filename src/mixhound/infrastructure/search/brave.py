import httpx
from pydantic import BaseModel, ValidationError

from mixhound.services.ports import SearchHit, SearchUnavailableError

ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class _Result(BaseModel):
    title: str = ""
    url: str
    description: str = ""


class _Web(BaseModel):
    results: list[_Result] = []


class _Response(BaseModel):
    web: _Web = _Web()


class BraveSearch:
    """Web search through the Brave Search API."""

    def __init__(self, client: httpx.AsyncClient, api_key: str, *, results: int = 10) -> None:
        self.client = client
        self.api_key = api_key
        self.results = results

    async def search(self, query: str, *, count: int | None = None) -> list[SearchHit]:
        """Return web results for a query; their text is evidence, never instructions."""
        try:
            response = await self.client.get(
                ENDPOINT,
                params={"q": query, "count": count or self.results},
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            )
            response.raise_for_status()
            parsed = _Response.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError) as error:
            raise SearchUnavailableError(str(error)) from error
        return [
            SearchHit(item.title, item.url, item.description)
            for item in parsed.web.results
            if item.url.startswith(("https://", "http://"))
        ]

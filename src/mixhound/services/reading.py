import json
import re
from html.parser import HTMLParser

PAGE_TEXT_LIMIT = 20_000
_SKIPPED = {"script", "style", "noscript", "svg", "template", "head"}
_BLOCKS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
    "dt", "dd", "td", "th", "table", "ul", "ol", "header", "footer", "blockquote",
}  # fmt: skip
_JSON_LD_KEYS = ("name", "genre", "keywords", "byArtist", "datePublished", "description")
_SPACES = re.compile(r"[ \t\f\v]+")
_LINES = re.compile(r"\n\s*\n+")


class _TextParser(HTMLParser):
    """Collects a page's title, description, JSON-LD and visible text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.description = ""
        self.json_ld: list[str] = []
        self.parts: list[str] = []
        self._skipping: list[str] = []
        self._in_title = False
        self._in_json_ld = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "meta" and (values.get("name") or values.get("property") or "").lower() in {
            "description",
            "og:description",
        }:
            self.description = self.description or (values.get("content") or "")
        elif tag == "script" and values.get("type") == "application/ld+json":
            self._in_json_ld = True
        if tag in _SKIPPED:
            self._skipping.append(tag)
        elif tag in _BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag == "script":
            self._in_json_ld = False
        if self._skipping and self._skipping[-1] == tag:
            self._skipping.pop()
        if tag in _BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._in_json_ld:
            self.json_ld.append(data)
        elif not self._skipping:
            self.parts.append(data)


def _json_ld_facts(raw: str) -> list[str]:
    """Return the descriptive JSON-LD fields of one script block."""
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get("@graph", [data])
    items = data if isinstance(data, list) else []
    facts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in _JSON_LD_KEYS:
            value = item.get(key)
            if isinstance(value, dict):
                value = value.get("name")
            if value:
                facts.append(f"{key}: {json.dumps(value, ensure_ascii=False)[:500]}")
    return facts


def _collapse(text: str) -> str:
    lines = (_SPACES.sub(" ", line).strip() for line in text.splitlines())
    return _LINES.sub("\n\n", "\n".join(lines)).strip()


def page_text(content: bytes, limit: int = PAGE_TEXT_LIMIT) -> str:
    """Return a page's title, description, structured data and visible text, capped."""
    parser = _TextParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    parser.close()
    header = [
        f"Title: {_collapse(parser.title)}" if parser.title.strip() else "",
        f"Description: {_collapse(parser.description)}" if parser.description.strip() else "",
        *(fact for raw in parser.json_ld for fact in _json_ld_facts(raw)),
    ]
    text = "\n".join(line for line in header if line)
    body = _collapse("".join(parser.parts))
    combined = f"{text}\n\n{body}".strip() if text else body
    if len(combined) <= limit:
        return combined
    return combined[:limit].rstrip() + "\n[truncated]"

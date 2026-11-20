import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import yaml

from mixhound.services.errors import MixhoundError, NotFoundError

SKILL_FILE = "SKILL.md"
MAX_NAME = 64
MAX_DESCRIPTION = 1024
MAX_COMPATIBILITY = 500
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONT_MATTER = re.compile(r"\A---\s*\n(?P<meta>.*?)\n---\s*(?:\n|\Z)(?P<body>.*)\Z", re.DOTALL)

RESEARCH_SKILL = "music-similarity-research"
PROFILE_SKILL = "sonic-profile"
LINK_SEARCH_SKILL = "listening-link-search"
ORCHESTRATION_SKILL = "discovery-orchestration"
# Read only by parallel strategy runs; cut from the single-pass instructions.
STRATEGY_SECTION = "## Strategy runs"


class InvalidSkillError(MixhoundError, ValueError):
    """A SKILL.md file does not follow the Agent Skills specification."""


@dataclass(frozen=True)
class Skill:
    """An agent skill: its metadata, instructions, and bundled resource files."""

    name: str
    description: str
    body: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    resources: Mapping[str, str] = field(default_factory=dict)

    def instructions(self, *, without_section: str | None = None) -> str:
        """Return the body, optionally cut at a heading and everything after it."""
        text = self.body
        if without_section is not None:
            text = text.split(without_section, 1)[0]
        return text.strip() + "\n"

    def resource(self, path: str) -> str:
        """Return a bundled file by its path relative to the skill directory."""
        try:
            return self.resources[path]
        except KeyError:
            raise NotFoundError(f"Skill {self.name} has no resource {path}") from None


def _text(meta: dict[str, object], key: str, limit: int, *, required: bool) -> str | None:
    value = meta.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise InvalidSkillError(f"Skill `{key}` must be 1-{limit} characters")
    return value.strip()


def parse_skill(text: str, directory: str, resources: Mapping[str, str] | None = None) -> Skill:
    """Parse a SKILL.md file found in `directory` and validate its front matter."""
    found = _FRONT_MATTER.match(text)
    if found is None:
        raise InvalidSkillError(f"{directory}/{SKILL_FILE} lacks YAML front matter")
    try:
        meta = yaml.safe_load(found["meta"])
    except yaml.YAMLError as error:
        raise InvalidSkillError(f"{directory}/{SKILL_FILE} has invalid front matter") from error
    if not isinstance(meta, dict):
        raise InvalidSkillError(f"{directory}/{SKILL_FILE} front matter must be a mapping")
    name = _text(meta, "name", MAX_NAME, required=True)
    if name is None or not _NAME.match(name):
        raise InvalidSkillError("Skill name must be lowercase letters, digits and single hyphens")
    if name != directory:
        raise InvalidSkillError(f"Skill name {name} must match its directory {directory}")
    description = _text(meta, "description", MAX_DESCRIPTION, required=True)
    _text(meta, "compatibility", MAX_COMPATIBILITY, required=False)
    metadata = meta.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise InvalidSkillError("Skill metadata must be a mapping")
    return Skill(
        name=name,
        description=description or "",
        body=found["body"],
        metadata={str(key): str(value) for key, value in metadata.items()},
        resources=dict(resources or {}),
    )


@dataclass(frozen=True)
class SkillCatalog:
    """Every skill available to the agents, by name."""

    skills: Mapping[str, Skill]

    def get(self, name: str) -> Skill:
        """Return a skill by name."""
        try:
            return self.skills[name]
        except KeyError:
            raise NotFoundError(f"Unknown skill {name}") from None

    def summary(self, names: list[str] | None = None) -> str:
        """List skill names and descriptions, the metadata an agent sees before loading one."""
        chosen = names if names is not None else sorted(self.skills)
        return "\n".join(f"- {name}: {self.get(name).description}" for name in chosen)

    def research_instructions(self, *, strategy: bool = False) -> str:
        """The research skill body, without the strategy section unless asked."""
        skill = self.get(RESEARCH_SKILL)
        return skill.instructions(without_section=None if strategy else STRATEGY_SECTION)

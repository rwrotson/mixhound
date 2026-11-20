from pathlib import Path

import pytest

from mixhound.infrastructure.skills import load_catalog
from mixhound.services.errors import NotFoundError
from mixhound.services.skills import (
    LINK_SEARCH_SKILL,
    ORCHESTRATION_SKILL,
    PROFILE_SKILL,
    RESEARCH_SKILL,
    InvalidSkillError,
    parse_skill,
)

SKILLS = Path(__file__).parents[3] / ".agents" / "skills"


def skill_text(front_matter: str, body: str = "# Body\n") -> str:
    return f"---\n{front_matter}\n---\n{body}"


def test_packaged_skills_load_and_follow_the_specification() -> None:
    catalog = load_catalog(SKILLS)

    assert {RESEARCH_SKILL, PROFILE_SKILL, LINK_SEARCH_SKILL, ORCHESTRATION_SKILL} <= set(
        catalog.skills
    )
    assert f"- {PROFILE_SKILL}: " in catalog.summary()


def test_research_instructions_strip_front_matter_and_strategy_section() -> None:
    catalog = load_catalog(SKILLS)

    single = catalog.research_instructions()
    strategy = catalog.research_instructions(strategy=True)

    assert not single.startswith("---")
    assert "name: music-similarity-research" not in single
    assert "Interim notes" in single
    assert "max_per_artist" in single
    assert "## Strategy runs" not in single
    assert "## Strategy runs" in strategy
    assert strategy.startswith(single.strip())


def test_resources_are_bundled_and_read_by_path(tmp_path: Path) -> None:
    directory = tmp_path / "web-research"
    (directory / "references").mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        skill_text("name: web-research\ndescription: Search.\nmetadata:\n  version: 1")
    )
    (directory / "references" / "method.md").write_text("Method")
    (directory / ".hidden").write_text("x")
    (directory / "references" / "big.bin").write_bytes(b"\xff\xfe")
    (tmp_path / "not-a-skill").mkdir()

    skill = load_catalog(tmp_path).get("web-research")

    assert skill.resource("references/method.md") == "Method"
    assert skill.metadata == {"version": "1"}
    assert set(skill.resources) == {"references/method.md"}
    with pytest.raises(NotFoundError, match="no resource"):
        skill.resource("missing.md")


def test_unknown_skill_and_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError, match="Unknown skill"):
        load_catalog(tmp_path).get("nope")
    with pytest.raises(InvalidSkillError, match="does not exist"):
        load_catalog(tmp_path / "missing")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("# No front matter\n", "lacks YAML front matter"),
        (skill_text("name: [unclosed"), "invalid front matter"),
        (skill_text("- a list"), "must be a mapping"),
        (skill_text("name: Bad-Name\ndescription: x"), "lowercase"),
        (skill_text("name: bad--name\ndescription: x"), "lowercase"),
        (skill_text("name: other\ndescription: x"), "must match its directory"),
        (skill_text("name: skill"), "description"),
        (skill_text(f"name: skill\ndescription: {'x' * 1025}"), "description"),
        (skill_text("name: skill\ndescription: x\ncompatibility: ''"), "compatibility"),
        (skill_text("name: skill\ndescription: x\nmetadata: [1]"), "metadata"),
    ],
)
def test_invalid_skills_are_rejected(text: str, message: str) -> None:
    with pytest.raises(InvalidSkillError, match=message):
        parse_skill(text, "skill")

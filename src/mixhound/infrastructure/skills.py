from pathlib import Path

from mixhound.services.skills import SKILL_FILE, InvalidSkillError, Skill, SkillCatalog, parse_skill

MAX_RESOURCE_BYTES = 256_000


def _resources(directory: Path) -> dict[str, str]:
    """Read every bundled text file of a skill except SKILL.md."""
    found: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == SKILL_FILE or path.name.startswith("."):
            continue
        if path.stat().st_size > MAX_RESOURCE_BYTES:
            continue
        try:
            found[path.relative_to(directory).as_posix()] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return found


def load_skill(directory: Path) -> Skill:
    """Read and validate one skill directory."""
    text = (directory / SKILL_FILE).read_text(encoding="utf-8")
    return parse_skill(text, directory.name, _resources(directory))


def load_catalog(root: Path) -> SkillCatalog:
    """Read every skill directory under `root`."""
    if not root.is_dir():
        raise InvalidSkillError(f"Skills directory {root} does not exist")
    skills = [
        load_skill(directory)
        for directory in sorted(root.iterdir())
        if (directory / SKILL_FILE).is_file()
    ]
    return SkillCatalog({skill.name: skill for skill in skills})

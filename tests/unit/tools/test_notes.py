from mixhound.models.research import ResearchCandidate
from mixhound.tools.notes import record_candidate
from tests.support.agents import call, context, tool_deps
from tests.support.factories import candidate


def test_candidates_are_noted_up_to_the_cap() -> None:
    deps = tool_deps(agent_call=call(candidates=2))

    replies = [record_candidate(context(deps), candidate(title=title)) for title in "ABC"]

    assert replies[:2] == ["Noted (1).", "Noted (2)."]
    assert "limit reached" in replies[2]
    assert [ResearchCandidate.model_validate_json(note).title for note in deps.notes] == ["A", "B"]
    assert deps.state.counts["research"] == (0, 2)

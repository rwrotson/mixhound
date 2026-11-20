import pytest

from evals.cases import load_dataset


def test_bundled_dataset_loads_every_case() -> None:
    dataset = load_dataset()
    assert [case.name for case in dataset.cases] == ["psych", "noch", "slowcore-strict"]
    for case in dataset.cases:
        assert case.inputs.instructions
        assert case.inputs.references
        assert {ref.kind for ref in case.inputs.references} == {"manual"}
        assert case.inputs.settings.exclude_reference_artists


def test_load_dataset_keeps_the_named_cases() -> None:
    assert [case.name for case in load_dataset(names=["noch"]).cases] == ["noch"]


def test_load_dataset_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown cases: nope"):
        load_dataset(names=["noch", "nope"])

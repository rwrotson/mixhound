from decimal import Decimal

import pytest

from mixhound.agents.clock import BudgetExceededError, check_budget, clock_line


def test_clock_line_shows_real_time_and_the_wrap_point() -> None:
    assert clock_line(75.4, 600, 3) == (
        "Clock: 1:15 of 10:00 used (real time, not your estimate); "
        "keep researching until about 9:00. Notes written: 3."
    )


@pytest.mark.parametrize(("cost", "cap"), [(None, 1.0), (Decimal("0.99"), 1.0), (Decimal(5), None)])
def test_budget_allows_unpriced_uncapped_and_cheaper_calls(
    cost: Decimal | None, cap: float | None
) -> None:
    check_budget(cost, cap)


def test_budget_stops_at_the_cap() -> None:
    with pytest.raises(BudgetExceededError, match=r"reached the cap 0\.5 USD"):
        check_budget(Decimal("0.5"), 0.5)

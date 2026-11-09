from __future__ import annotations

import pytest

from rivwq.gee.years import parse_years

FIRST, LAST = 2018, 2026


def test_single_year_range_and_list():
    assert parse_years("2024", FIRST, LAST) == [2024]
    assert parse_years("2025-2026", FIRST, LAST) == [2025, 2026]
    assert parse_years("2018-2020,2023", FIRST, LAST) == [2018, 2019, 2020, 2023]


def test_order_is_kept_and_duplicates_removed():
    assert parse_years("2025,2024,2025", FIRST, LAST) == [2025, 2024]
    assert parse_years("2024-2025,2025", FIRST, LAST) == [2024, 2025]


def test_spaces_are_allowed():
    assert parse_years(" 2023 , 2025 - 2026 ".replace(" - ", "-"), FIRST, LAST) == [
        2023,
        2025,
        2026,
    ]


@pytest.mark.parametrize("text", ["2017", "2027", "2020-2030", "2026-2025", "", "2024,", "abc"])
def test_bad_input_is_rejected(text):
    with pytest.raises(ValueError):
        parse_years(text, FIRST, LAST)

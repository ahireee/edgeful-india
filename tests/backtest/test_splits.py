"""Hand-checked tests for the 60/20/20 chronological split."""

from __future__ import annotations

from datetime import date

import pytest

from backtest.splits import split_60_20_20


class TestSixtyTwentyTwentySplit:
    def test_basic_split_proportions(self) -> None:
        # 1000-day span: train 600, validate 200, test 200.
        train, validate, test = split_60_20_20(date(2020, 1, 1), date(2022, 9, 27))
        assert (train.end - train.start).days == int(1000 * 0.6)
        assert (validate.end - validate.start).days == (int(1000 * 0.8) - int(1000 * 0.6))
        assert test.end == date(2022, 9, 27)

    def test_splits_are_contiguous_and_non_overlapping(self) -> None:
        train, validate, test = split_60_20_20(date(2024, 1, 1), date(2025, 1, 1))
        assert train.end == validate.start
        assert validate.end == test.start
        # No gaps, no overlap.

    def test_contains_method(self) -> None:
        train, validate, test = split_60_20_20(date(2024, 1, 1), date(2025, 1, 1))
        assert train.contains(train.start)
        assert not train.contains(train.end)  # end is exclusive
        assert validate.contains(validate.start)
        assert test.contains(date(2024, 12, 31))

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError):
            split_60_20_20(date(2024, 1, 10), date(2024, 1, 5))
        with pytest.raises(ValueError):
            split_60_20_20(date(2024, 1, 10), date(2024, 1, 10))  # zero span

    def test_names_are_correct(self) -> None:
        train, validate, test = split_60_20_20(date(2024, 1, 1), date(2025, 1, 1))
        assert train.name == "train"
        assert validate.name == "validate"
        assert test.name == "test"

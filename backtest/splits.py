"""60/20/20 chronological train/validate/test split for backtests.

The split is by calendar day, not by trade index.  Trades / bars whose
date falls inside [start, end) of a slice belong to that slice.  A trade
that straddles a boundary belongs to whichever slice contains its entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Split:
    name: str  # "train" | "validate" | "test"
    start: date  # inclusive
    end: date  # exclusive

    def contains(self, d: date) -> bool:
        return self.start <= d < self.end


def split_60_20_20(start: date, end: date) -> tuple[Split, Split, Split]:
    """Return (train, validate, test) covering [start, end).

    The returned Split objects are non-overlapping and contiguous.
    The final split's ``end`` is set to ``end`` (the original argument)
    so the test slice always extends to the last day of the period.
    """
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")

    span_days = (end - start).days
    train_end = start + timedelta(days=int(span_days * 0.60))
    validate_end = start + timedelta(days=int(span_days * 0.80))

    return (
        Split(name="train", start=start, end=train_end),
        Split(name="validate", start=train_end, end=validate_end),
        Split(name="test", start=validate_end, end=end),
    )

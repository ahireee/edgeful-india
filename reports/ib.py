"""Initial Balance Breakout (IB) report.

The Initial Balance is the high and low of the first 60 minutes of trading
(09:15-10:14 IST).  Breakout / continuation / false-break logic is identical
to ORB — IB is just ORB with a 60-minute window — so this module is a thin
wrapper around :func:`reports.orb.compute`.

See SPEC.md section 3 (Initial Balance Breakout).
"""

from __future__ import annotations

import polars as pl

from reports.base import ReportParams, ReportResult
from reports.orb import compute as _orb_compute

IB_MINUTES: int = 60


def compute(bars: pl.DataFrame, params: ReportParams) -> ReportResult:
    """Compute the Initial Balance Breakout probability table.

    Forces ``or_minutes = 60`` regardless of what the caller passed in;
    that is the definition of the Initial Balance.  All other behaviour
    (breakout detection, continuation/false-break logic, recency window,
    short-session exclusion) comes from :func:`reports.orb.compute`.
    """
    ib_params = params.model_copy(update={"or_minutes": IB_MINUTES})
    result = _orb_compute(bars, ib_params)
    result["methodology"] = _METHODOLOGY
    return result


_METHODOLOGY = f"""\
Initial Balance Breakout (IB) Report
=====================================
Initial Balance = high and low of the first {IB_MINUTES} minutes of trading
(09:15-10:14 IST, {IB_MINUTES} bars).

A breakout occurs when, after the IB window closes, price trades above the
IB high (upside) or below the IB low (downside).  Whichever side breaks
first determines the direction.  If both sides break in the same 1-min bar,
the day is excluded (ambiguous).

Continuation: EOD close is on the breakout side of the breakout level.
False break: breakout occurred but EOD close is on the other side.

Compared to a 15-minute ORB, the longer 60-minute window is intended to
filter out morning noise — the hypothesis being that breakouts after the
first hour are more "decided" and therefore continue more often.
"""

"""Smoke tests for data.universe — ensures the universe definition is valid."""

from data.universe import UNIVERSE, by_instrument_key, by_symbol


def test_universe_has_twelve_instruments() -> None:
    assert len(UNIVERSE) == 12


def test_by_symbol_roundtrip() -> None:
    for inst in UNIVERSE:
        assert by_symbol(inst.symbol) is inst


def test_by_instrument_key_roundtrip() -> None:
    for inst in UNIVERSE:
        assert by_instrument_key(inst.instrument_key) is inst

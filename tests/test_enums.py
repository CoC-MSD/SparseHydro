"""Tests for the ModelState enumeration."""

import pytest

from sparsehydro.enums import ModelState


def test_all_states_present():
    names = {s.name for s in ModelState}
    assert names == {"CREATED", "INITIALIZED", "VALIDATED", "PREPARED", "PREDICTED", "FINALIZED"}


def test_state_values():
    assert ModelState.CREATED.value == "created"
    assert ModelState.INITIALIZED.value == "initialized"
    assert ModelState.VALIDATED.value == "validated"
    assert ModelState.PREPARED.value == "prepared"
    assert ModelState.PREDICTED.value == "predicted"
    assert ModelState.FINALIZED.value == "finalized"


def test_state_count():
    assert len(list(ModelState)) == 6


def test_state_identity():
    assert ModelState("created") is ModelState.CREATED
    assert ModelState("finalized") is ModelState.FINALIZED


def test_invalid_state_raises():
    with pytest.raises(ValueError):
        ModelState("unknown")

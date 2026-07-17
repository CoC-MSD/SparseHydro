"""Tests for ScalarParameter and VectorParameter."""

import numpy as np
import pytest

from sparsehydro.parameters import ScalarParameter, VectorParameter


class TestScalarParameter:
    def test_basic(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        assert p.name == "k"
        assert p.value == pytest.approx(0.5)

    def test_is_valid_within_bounds(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        assert p.is_valid()

    def test_is_valid_at_lower_bound(self):
        p = ScalarParameter("k", value=0.0, lower_bound=0.0, upper_bound=1.0)
        assert p.is_valid()

    def test_is_valid_at_upper_bound(self):
        p = ScalarParameter("k", value=1.0, lower_bound=0.0, upper_bound=1.0)
        assert p.is_valid()

    def test_is_invalid_below_lower(self):
        p = ScalarParameter("k", value=-0.1, lower_bound=0.0, upper_bound=1.0)
        assert not p.is_valid()

    def test_is_invalid_above_upper(self):
        p = ScalarParameter("k", value=1.1, lower_bound=0.0, upper_bound=1.0)
        assert not p.is_valid()

    def test_inverted_bounds_raises(self):
        with pytest.raises(ValueError, match="lower_bound"):
            ScalarParameter("k", value=0.5, lower_bound=1.0, upper_bound=0.0)

    def test_normalize_midpoint(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        assert p.normalize() == pytest.approx(0.5)

    def test_normalize_full_range(self):
        p = ScalarParameter("k", value=75.0, lower_bound=50.0, upper_bound=100.0)
        assert p.normalize() == pytest.approx(0.5)

    def test_normalize_zero_span(self):
        p = ScalarParameter("k", value=5.0, lower_bound=5.0, upper_bound=5.0)
        assert p.normalize() == pytest.approx(0.0)

    def test_clamp_above(self):
        p = ScalarParameter("k", value=2.0, lower_bound=0.0, upper_bound=1.0)
        clamped = p.clamp()
        assert clamped.value == pytest.approx(1.0)
        assert clamped.is_valid()
        assert clamped.name == "k"

    def test_clamp_below(self):
        p = ScalarParameter("k", value=-1.0, lower_bound=0.0, upper_bound=1.0)
        clamped = p.clamp()
        assert clamped.value == pytest.approx(0.0)

    def test_clamp_preserves_bounds(self):
        p = ScalarParameter("k", value=2.0, lower_bound=0.0, upper_bound=1.0)
        clamped = p.clamp()
        assert clamped.lower_bound == pytest.approx(0.0)
        assert clamped.upper_bound == pytest.approx(1.0)

    def test_optional_metadata(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0,
                            units="1/day", description="recession coefficient")
        assert p.units == "1/day"
        assert p.description == "recession coefficient"


class TestVectorParameter:
    def test_basic(self):
        p = VectorParameter("beta", values=[0.2, 0.8], lower_bounds=0.0, upper_bounds=1.0)
        assert p.name == "beta"
        assert p.size == 2

    def test_is_valid_within_bounds(self):
        p = VectorParameter("beta", values=[0.2, 0.8], lower_bounds=0.0, upper_bounds=1.0)
        assert p.is_valid()

    def test_is_invalid_one_element_out(self):
        p = VectorParameter("beta", values=[0.5, 1.5], lower_bounds=0.0, upper_bounds=1.0)
        assert not p.is_valid()

    def test_scalar_bounds_broadcast(self):
        p = VectorParameter("beta", values=[0.1, 0.5, 0.9], lower_bounds=0.0, upper_bounds=1.0)
        assert p.lower_bounds.shape == (3,)
        assert p.upper_bounds.shape == (3,)
        np.testing.assert_array_equal(p.lower_bounds, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(p.upper_bounds, [1.0, 1.0, 1.0])

    def test_vector_bounds(self):
        p = VectorParameter("beta", values=[0.5, 5.0],
                            lower_bounds=[0.0, 0.0], upper_bounds=[1.0, 10.0])
        assert p.is_valid()

    def test_mismatched_lower_bounds_raises(self):
        with pytest.raises(ValueError, match="lower_bounds length"):
            VectorParameter("beta", values=[0.5, 0.5],
                            lower_bounds=[0.0, 0.0, 0.0], upper_bounds=[1.0, 1.0])

    def test_mismatched_upper_bounds_raises(self):
        with pytest.raises(ValueError, match="upper_bounds length"):
            VectorParameter("beta", values=[0.5, 0.5],
                            lower_bounds=[0.0, 0.0], upper_bounds=[1.0, 1.0, 1.0])

    def test_inverted_bounds_raises(self):
        with pytest.raises(ValueError, match="lower_bound > upper_bound"):
            VectorParameter("beta", values=[0.5, 0.5],
                            lower_bounds=[0.0, 1.0], upper_bounds=[1.0, 0.0])

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            VectorParameter("beta", values=[[0.5, 0.5]], lower_bounds=0.0, upper_bounds=1.0)

    def test_normalize(self):
        p = VectorParameter("beta", values=[0.5, 1.0],
                            lower_bounds=[0.0, 0.0], upper_bounds=[1.0, 2.0])
        np.testing.assert_allclose(p.normalize(), [0.5, 0.5])

    def test_normalize_zero_span(self):
        p = VectorParameter("beta", values=[3.0, 0.5],
                            lower_bounds=[3.0, 0.0], upper_bounds=[3.0, 1.0])
        norm = p.normalize()
        assert norm[0] == pytest.approx(0.0)
        assert norm[1] == pytest.approx(0.5)

    def test_clamp(self):
        p = VectorParameter("beta", values=[0.5, 3.0],
                            lower_bounds=[0.0, 0.0], upper_bounds=[1.0, 2.0])
        clamped = p.clamp()
        np.testing.assert_allclose(clamped.values, [0.5, 2.0])
        assert clamped.is_valid()

    def test_clamp_does_not_mutate_original(self):
        p = VectorParameter("beta", values=[0.5, 3.0],
                            lower_bounds=[0.0, 0.0], upper_bounds=[1.0, 2.0])
        _ = p.clamp()
        assert p.values[1] == pytest.approx(3.0)


class TestScalarParameterUpdate:
    def test_update_value(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        p.update(value=0.8)
        assert p.value == pytest.approx(0.8)

    def test_update_units(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        p.update(units="mm")
        assert p.units == "mm"

    def test_update_description(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        p.update(description="recession rate")
        assert p.description == "recession rate"

    def test_update_calibrate(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        p.update(calibrate=False)
        assert p.calibrate is False

    def test_update_bounds(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        p.update(lower_bound=0.1, upper_bound=0.9)
        assert p.lower_bound == pytest.approx(0.1)
        assert p.upper_bound == pytest.approx(0.9)

    def test_update_multiple_at_once(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        p.update(value=0.3, units="in", description="depth")
        assert p.value == pytest.approx(0.3)
        assert p.units == "in"
        assert p.description == "depth"

    def test_update_invalid_bounds_raises(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        with pytest.raises(ValueError, match="lower_bound"):
            p.update(lower_bound=2.0)

    def test_update_leaves_other_fields_unchanged(self):
        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0,
                            units="1/hr", description="original")
        p.update(value=0.6)
        assert p.units == "1/hr"
        assert p.description == "original"
        assert p.lower_bound == pytest.approx(0.0)


class TestVectorParameterUpdate:
    def test_update_values(self):
        p = VectorParameter("beta", values=[0.2, 0.8], lower_bounds=0.0, upper_bounds=1.0)
        p.update(values=[0.3, 0.7])
        np.testing.assert_allclose(p.values, [0.3, 0.7])

    def test_update_units(self):
        p = VectorParameter("beta", values=[0.2, 0.8], lower_bounds=0.0, upper_bounds=1.0)
        p.update(units="m/s")
        assert p.units == "m/s"

    def test_update_description(self):
        p = VectorParameter("beta", values=[0.2, 0.8], lower_bounds=0.0, upper_bounds=1.0)
        p.update(description="shape params")
        assert p.description == "shape params"

    def test_update_bounds(self):
        p = VectorParameter("beta", values=[0.2, 0.8], lower_bounds=0.0, upper_bounds=1.0)
        p.update(lower_bounds=0.1, upper_bounds=0.9)
        np.testing.assert_allclose(p.lower_bounds, [0.1, 0.1])
        np.testing.assert_allclose(p.upper_bounds, [0.9, 0.9])

    def test_update_invalid_bounds_raises(self):
        p = VectorParameter("beta", values=[0.2, 0.8], lower_bounds=0.0, upper_bounds=1.0)
        with pytest.raises(ValueError):
            p.update(lower_bounds=[2.0, 2.0])

    def test_update_leaves_other_fields_unchanged(self):
        p = VectorParameter("beta", values=[0.2, 0.8], lower_bounds=0.0, upper_bounds=1.0,
                            units="cm", description="original")
        p.update(units="m")
        assert p.description == "original"

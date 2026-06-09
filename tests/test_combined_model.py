"""Tests for CombinedHydroModel parameter rename, update, and unit system."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.rdii import IAModel, RDIIModel
from sparsehydro.rdii.combined_model import CombinedHydroModel

_MM_TO_IN = 1.0 / 25.4


@pytest.fixture()
def minimal_data():
    """Minimal hourly DataFrame with a short storm — imperial (rainfall_in)."""
    dates = pd.date_range("2020-01-01", periods=48, freq="h")
    rain = np.zeros(48)
    rain[5:10] = 5.0 * _MM_TO_IN   # ~0.197 in per hour
    return pd.DataFrame({"datetime": dates, "rainfall_in": rain})


@pytest.fixture()
def minimal_data_metric():
    """Minimal hourly DataFrame with a short storm — metric (rainfall_mm)."""
    dates = pd.date_range("2020-01-01", periods=48, freq="h")
    rain = np.zeros(48)
    rain[5:10] = 5.0
    return pd.DataFrame({"datetime": dates, "rainfall_mm": rain})


@pytest.fixture()
def initialized_model():
    """Default CombinedHydroModel (imperial)."""
    m = CombinedHydroModel()
    m.initialize()
    return m


@pytest.fixture()
def initialized_model_metric():
    """Metric CombinedHydroModel."""
    m = CombinedHydroModel(units="metric")
    m.initialize()
    return m


class TestCombinedHydroModelRename:
    def test_rename_uh_param_updates_registry(self, initialized_model):
        m = initialized_model
        m.rename_scalar_parameter("uh1_T", "time_to_peak_fast")
        assert "time_to_peak_fast" in m.scalar_parameter_names
        assert "uh1_T" not in m.scalar_parameter_names

    def test_rename_uh_param_updates_name_field(self, initialized_model):
        m = initialized_model
        m.rename_scalar_parameter("uh1_T", "time_to_peak_fast")
        assert m.get_scalar_parameter("time_to_peak_fast").name == "time_to_peak_fast"

    def test_rename_uh_param_preserves_value(self, initialized_model):
        m = initialized_model
        original_value = m.get_scalar_parameter("uh1_T").value
        m.rename_scalar_parameter("uh1_T", "time_to_peak_fast")
        assert m.get_scalar_parameter("time_to_peak_fast").value == pytest.approx(original_value)

    def test_rename_uh_param_updates_uh_param_map(self, initialized_model):
        m = initialized_model
        m.rename_scalar_parameter("uh1_T", "time_to_peak_fast")
        assert m._uh_param_maps[0]["T"] == "time_to_peak_fast"

    def test_rename_ia_param_updates_registry(self, initialized_model):
        m = initialized_model
        m.rename_scalar_parameter("ia_max", "max_abstraction")
        assert "max_abstraction" in m.scalar_parameter_names
        assert "ia_max" not in m.scalar_parameter_names

    def test_rename_ia_param_updates_ia_param_name_map(self, initialized_model):
        m = initialized_model
        m.rename_scalar_parameter("ia_max", "max_abstraction")
        assert m._ia_param_name_map["ia_max"] == "max_abstraction"

    def test_predict_after_uh_rename(self, initialized_model, minimal_data):
        m = initialized_model
        m.rename_scalar_parameter("uh1_T", "time_to_peak_fast")
        m.rename_scalar_parameter("uh1_K", "recession_fast")
        m.validate()
        m.prepare(minimal_data)
        result = m.predict()
        assert "rdii_cfs" in result.columns
        assert not result["rdii_cfs"].isnull().any()

    def test_predict_after_ia_rename(self, initialized_model, minimal_data):
        m = initialized_model
        m.rename_scalar_parameter("ia_max", "max_abstraction")
        m.validate()
        m.prepare(minimal_data)
        result = m.predict()
        assert "rdii_cfs" in result.columns
        assert not result["rdii_cfs"].isnull().any()

    def test_predict_unchanged_by_rename(self, minimal_data):
        """Renaming a parameter must not change prediction values."""
        m_ref = CombinedHydroModel()
        m_ref.initialize()
        m_ref.validate()
        m_ref.prepare(minimal_data)
        ref = m_ref.predict()

        m_renamed = CombinedHydroModel()
        m_renamed.initialize()
        m_renamed.rename_scalar_parameter("uh1_T", "time_to_peak_fast")
        m_renamed.validate()
        m_renamed.prepare(minimal_data)
        renamed = m_renamed.predict()

        pd.testing.assert_series_equal(ref["rdii_cfs"], renamed["rdii_cfs"])


class TestImperialUnits:
    """Verify default imperial behaviour and metric backward-compat."""

    def test_default_model_expects_rainfall_in(self, minimal_data):
        m = CombinedHydroModel()
        m.initialize()
        m.validate()
        m.prepare(minimal_data)
        result = m.predict()
        assert "rdii_in" in result.columns
        assert "p_excess_in" in result.columns
        assert "rdii_cfs" in result.columns

    def test_metric_model_expects_rainfall_mm(self, minimal_data_metric):
        m = CombinedHydroModel(units="metric")
        m.initialize()
        m.validate()
        m.prepare(minimal_data_metric)
        result = m.predict()
        assert "rdii_mm" in result.columns
        assert "p_excess_mm" in result.columns

    def test_imperial_ia_max_default_is_inches(self):
        m = CombinedHydroModel()
        m.initialize()
        p = m.get_scalar_parameter("ia_max")
        assert p.units == "in"
        assert p.value == pytest.approx(0.2)

    def test_metric_ia_max_default_is_mm(self):
        m = CombinedHydroModel(units="metric")
        m.initialize()
        p = m.get_scalar_parameter("ia_max")
        assert p.units == "mm"
        assert p.value == pytest.approx(5.0)

    def test_imperial_kdep_default(self):
        m = CombinedHydroModel()
        m.initialize()
        p = m.get_scalar_parameter("ia_k_dep")
        assert p.units == "1/in"
        assert p.value == pytest.approx(7.62)

    def test_rdii_model_imperial_default(self):
        m = RDIIModel()
        m.initialize()
        assert m.get_scalar_parameter("ia_max").units == "in"
        assert m.get_scalar_parameter("ia_k_dep").units == "1/in"

    def test_rdii_model_imperial_predict(self):
        dates = pd.date_range("2020-01-01", periods=48, freq="h")
        rain = np.zeros(48)
        rain[5:10] = 0.2
        df = pd.DataFrame({"datetime": dates, "rainfall_in": rain})
        m = RDIIModel()
        m.initialize()
        m.validate()
        m.prepare(df)
        result = m.predict()
        assert "rdii_in" in result.columns
        assert "p_excess_in" in result.columns

    def test_imperial_and_metric_produce_same_cfs(self):
        """Imperial and metric inputs for the same storm should yield equal CFS."""
        n = 48
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        rain_mm = np.zeros(n)
        rain_mm[5:10] = 5.0
        rain_in = rain_mm * _MM_TO_IN

        m_metric = RDIIModel(units="metric")
        m_metric.initialize()
        m_metric.validate()
        m_metric.prepare(pd.DataFrame({"datetime": dates, "rainfall_mm": rain_mm}))
        result_metric = m_metric.predict()

        m_imp = RDIIModel(units="imperial")
        m_imp.initialize()
        # Match ia_max and k_dep to metric equivalents (converted)
        m_imp.get_scalar_parameter("ia_max").update(value=5.0 * _MM_TO_IN)
        m_imp.get_scalar_parameter("ia_k_dep").update(value=0.3 / _MM_TO_IN)
        m_imp.validate()
        m_imp.prepare(pd.DataFrame({"datetime": dates, "rainfall_in": rain_in}))
        result_imp = m_imp.predict()

        np.testing.assert_allclose(
            result_metric["rdii_cfs"].to_numpy(),
            result_imp["rdii_cfs"].to_numpy(),
            rtol=1e-6,
        )

    def test_invalid_units_raises(self):
        with pytest.raises(ValueError, match="units must be"):
            CombinedHydroModel(units="furlongs")

    def test_prepare_wrong_column_raises(self, minimal_data_metric):
        """Imperial model rejects rainfall_mm input with clear error."""
        m = CombinedHydroModel()   # imperial default
        m.initialize()
        m.validate()
        with pytest.raises(ValueError, match="missing required columns"):
            m.prepare(minimal_data_metric)  # has rainfall_mm, not rainfall_in

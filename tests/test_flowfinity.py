"""Tests for sparsehydro.data.services.flowfinity — unittest.TestCase.

All HTTP interaction is mocked via an injected fake session; no network access
occurs.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from sparsehydro.data.services.flowfinity import DownloadResult, FlowFinityClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STORM_TSF = (
    "IDs:\tTC-NB-001 Flow Edited Revised\n"
    "Date/Time\tStormFlow\n"
    "M/d/yyyy\tMGD\n"
    "1/25/2022 12:40:00 PM\t0.10\n"
    "1/25/2022 12:45:00 PM\t0.20\n"
    "1/26/2022 12:40:00 PM\t0.30\n"
    "1/27/2022 12:40:00 PM\t0.40\n"
)

_RAIN_TSF = (
    "IDs:\tTC-NB-001 Flow Edited Revised\n"
    "Date/Time\tRain\n"
    "M/d/yyyy\tin\n"
    "1/25/2022 12:40:00 PM\t0.00\n"
    "1/25/2022 12:45:00 PM\t0.05\n"
)

_RECORDS = [
    {
        "DataTitle": "TC-NB-001 Flow Edited Revised",
        "StormFlow": {"URL": "?token=storm"},
        "Rain": {"URL": "?token=rain"},
    },
    {
        "DataTitle": "MC-EB-122 Flow",
        "Flow": {"URL": "?token=flow"},
    },
]


class _FakeResponse:
    def __init__(self, *, json_data=None, content=b""):
        self._json = json_data
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


class _FakeSession:
    """Routes GETs by URL to canned responses and records calls."""

    def __init__(self):
        self.calls = []

    def get(self, url, auth=None, headers=None, timeout=None):
        self.calls.append({"url": url, "auth": auth, "headers": headers, "timeout": timeout})
        if url.endswith(FlowFinityClient._LIST_PATH):
            return _FakeResponse(json_data=_RECORDS)
        if "token=storm" in url:
            return _FakeResponse(content=_STORM_TSF.encode("utf-8"))
        if "token=rain" in url:
            return _FakeResponse(content=_RAIN_TSF.encode("utf-8"))
        if "token=flow" in url:
            return _FakeResponse(content=b"flow-bytes")
        raise AssertionError(f"Unexpected URL requested: {url}")


def _make_client(session=None) -> FlowFinityClient:
    return FlowFinityClient(
        username="u",
        password="p",
        session=session or _FakeSession(),
    )


# ===========================================================================
# Construction / credentials
# ===========================================================================

class TestConstruction(unittest.TestCase):

    def test_explicit_credentials_ok(self):
        client = _make_client()
        self.assertEqual(client.base_url, FlowFinityClient.DEFAULT_BASE_URL)
        self.assertEqual(client.host_header, FlowFinityClient.DEFAULT_HOST_HEADER)

    def test_env_credentials_ok(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"FLOWFINITY_USERNAME": "eu", "FLOWFINITY_PASSWORD": "ep"}):
            client = FlowFinityClient(session=_FakeSession())
        self.assertIsInstance(client, FlowFinityClient)

    def test_missing_username_raises(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                FlowFinityClient(password="p", session=_FakeSession())

    def test_missing_password_raises(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"FLOWFINITY_USERNAME": "u"}, clear=True):
            with self.assertRaises(ValueError):
                FlowFinityClient(session=_FakeSession())

    def test_base_url_override_and_strip(self):
        client = FlowFinityClient(
            username="u", password="p", base_url="https://example.test/api/", session=_FakeSession()
        )
        self.assertEqual(client.base_url, "https://example.test/api")


# ===========================================================================
# Resource discovery
# ===========================================================================

class TestDiscovery(unittest.TestCase):

    def setUp(self):
        self.session = _FakeSession()
        self.client = _make_client(self.session)

    def test_list_resources_unfiltered(self):
        records = self.client.list_resources()
        self.assertEqual(len(records), 2)

    def test_list_resources_substring_filter(self):
        records = self.client.list_resources("tc-nb")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["DataTitle"], "TC-NB-001 Flow Edited Revised")

    def test_list_resources_regex_filter(self):
        records = self.client.list_resources(r"^MC-", regex=True)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["DataTitle"], "MC-EB-122 Flow")

    def test_get_resource_exact(self):
        rec = self.client.get_resource("MC-EB-122 Flow")
        self.assertIn("Flow", rec)

    def test_get_resource_missing_raises(self):
        with self.assertRaises(KeyError):
            self.client.get_resource("does-not-exist")

    def test_available_types(self):
        types = self.client.available_types("TC-NB-001 Flow Edited Revised")
        self.assertEqual(set(types), {"StormFlow", "Rain"})

    def test_get_sends_host_header_and_auth(self):
        self.client.list_resources()
        call = self.session.calls[-1]
        self.assertEqual(call["headers"], {"Host": FlowFinityClient.DEFAULT_HOST_HEADER})
        self.assertIsNotNone(call["auth"])


# ===========================================================================
# Downloading
# ===========================================================================

class TestDownload(unittest.TestCase):

    def setUp(self):
        self.session = _FakeSession()
        self.client = _make_client(self.session)

    def test_download_all_types_writes_files(self):
        with TemporaryDirectory() as tmp:
            results = self.client.download(
                "TC-NB-001 Flow Edited Revised", output_dir=tmp
            )
            self.assertEqual(set(results), {"StormFlow", "Rain"})
            for res in results.values():
                self.assertIsInstance(res, DownloadResult)
                self.assertTrue(res.path.exists())
                self.assertIsNone(res.dataframe)

    def test_download_subset_of_types(self):
        with TemporaryDirectory() as tmp:
            results = self.client.download(
                "TC-NB-001 Flow Edited Revised", types=["StormFlow"], output_dir=tmp
            )
            self.assertEqual(set(results), {"StormFlow"})

    def test_download_unknown_type_raises(self):
        with self.assertRaises(KeyError):
            self.client.download("TC-NB-001 Flow Edited Revised", types=["Bogus"])

    def test_download_parse_returns_dataframe(self):
        with TemporaryDirectory() as tmp:
            results = self.client.download(
                "TC-NB-001 Flow Edited Revised",
                types=["StormFlow"],
                output_dir=tmp,
                parse=True,
            )
            df = results["StormFlow"].dataframe
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 4)
            self.assertIn("StormFlow", df.columns)

    def test_download_parse_with_date_range(self):
        with TemporaryDirectory() as tmp:
            results = self.client.download(
                "TC-NB-001 Flow Edited Revised",
                types=["StormFlow"],
                output_dir=tmp,
                parse=True,
                start="2022-01-26",
                end="2022-01-26 23:59:59",
            )
            df = results["StormFlow"].dataframe
            self.assertEqual(len(df), 1)
            self.assertAlmostEqual(float(df["StormFlow"].iloc[0]), 0.30)

    def test_download_in_memory_only(self):
        results = self.client.download(
            "TC-NB-001 Flow Edited Revised",
            types=["StormFlow"],
            output_dir=None,
            parse=True,
        )
        res = results["StormFlow"]
        self.assertIsNone(res.path)
        self.assertEqual(len(res.dataframe), 4)

    def test_download_no_overwrite_reuses_file(self):
        with TemporaryDirectory() as tmp:
            # First download writes the file.
            self.client.download(
                "TC-NB-001 Flow Edited Revised", types=["StormFlow"], output_dir=tmp
            )
            calls_after_first = len(self.session.calls)
            # Second call with overwrite=False should not re-request the file.
            self.client.download(
                "TC-NB-001 Flow Edited Revised",
                types=["StormFlow"],
                output_dir=tmp,
                overwrite=False,
            )
            # Only the list endpoint should have been hit again (1 extra call).
            self.assertEqual(len(self.session.calls), calls_after_first + 1)


if __name__ == "__main__":
    unittest.main()

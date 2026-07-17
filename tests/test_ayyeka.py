"""Tests for the Ayyeka REST API client.

All HTTP interaction is mocked via a lightweight fake :class:`requests.Session`,
so these tests never touch the network.
"""

from __future__ import annotations

import base64
from typing import Any

import pandas as pd
import pytest

from sparsehydro.data.services.ayyeka import AyyekaClient, Site, Stream, _haversine_km


class FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, json_data: Any, status_code: int = 200) -> None:
        self._json = json_data
        self.status_code = status_code

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(f"status {self.status_code}")


class FakeSession:
    """Records requests and replays queued/scripted responses."""

    def __init__(self) -> None:
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        #: Sequence of FakeResponse returned by successive POSTs.
        self.post_responses: list[FakeResponse] = []
        #: Mapping of URL -> FakeResponse (or list for sequential) for GETs.
        self.get_responses: dict[str, Any] = {}

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.post_calls.append({"url": url, **kwargs})
        if self.post_responses:
            return self.post_responses.pop(0)
        return FakeResponse({"access_token": "tok", "token_type": "JWT", "expires_in": 3600})

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.get_calls.append({"url": url, **kwargs})
        entry = self.get_responses.get(url)
        if isinstance(entry, list):
            return entry.pop(0)
        if entry is not None:
            return entry
        return FakeResponse([])


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def client(session: FakeSession) -> AyyekaClient:
    return AyyekaClient("key", "secret", session=session)


# ----------------------------------------------------------------------
# Construction / credentials
# ----------------------------------------------------------------------


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AYYEKA_CLIENT_KEY", raising=False)
    monkeypatch.delenv("AYYEKA_CLIENT_SECRET", raising=False)
    with pytest.raises(ValueError, match="client key is required"):
        AyyekaClient()


def test_missing_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AYYEKA_CLIENT_KEY", "k")
    monkeypatch.delenv("AYYEKA_CLIENT_SECRET", raising=False)
    with pytest.raises(ValueError, match="client secret is required"):
        AyyekaClient()


def test_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AYYEKA_CLIENT_KEY", "envkey")
    monkeypatch.setenv("AYYEKA_CLIENT_SECRET", "envsecret")
    c = AyyekaClient()
    assert c._client_key == "envkey"
    assert c._client_secret == "envsecret"


# ----------------------------------------------------------------------
# Authentication / token caching
# ----------------------------------------------------------------------


def test_basic_auth_header(client: AyyekaClient) -> None:
    expected = "Basic " + base64.b64encode(b"key:secret").decode("ascii")
    assert client._basic_auth_header() == expected


def test_token_requested_once_and_cached(client: AyyekaClient, session: FakeSession) -> None:
    client.list_sites()
    client.list_sites()
    # Only one auth POST despite two GET requests.
    assert len(session.post_calls) == 1
    assert session.post_calls[0]["url"] == client.auth_url
    assert session.post_calls[0]["data"] == {"grant_type": "client_credentials"}


def test_token_refreshes_when_expired(client: AyyekaClient, session: FakeSession) -> None:
    session.post_responses = [
        FakeResponse({"access_token": "t1", "expires_in": 0}),
        FakeResponse({"access_token": "t2", "expires_in": 3600}),
    ]
    client.list_sites()
    client.list_sites()
    assert len(session.post_calls) == 2


def test_auth_without_access_token_raises(client: AyyekaClient, session: FakeSession) -> None:
    session.post_responses = [FakeResponse({"token_type": "JWT"})]
    with pytest.raises(ValueError, match="access_token"):
        client.list_sites()


def test_403_triggers_single_reauth(client: AyyekaClient, session: FakeSession) -> None:
    url = client.api_url + "/site"
    session.get_responses[url] = [
        FakeResponse(None, status_code=403),
        FakeResponse([{"id": 1, "displayName": "S1", "status": "Active"}]),
    ]
    sites = client.list_sites()
    assert [s.id for s in sites] == [1]
    # Initial auth + re-auth after the 403.
    assert len(session.post_calls) == 2


# ----------------------------------------------------------------------
# Site discovery + filtering
# ----------------------------------------------------------------------


def _site_payload() -> list[dict[str, Any]]:
    return [
        {"id": 1, "displayName": "North Plant", "status": "Active",
         "location": {"latitude": 39.10, "longitude": -84.51}},
        {"id": 2, "displayName": "South Pump", "status": "Inactive",
         "latitude": 39.50, "longitude": -84.20},
        {"id": 3, "displayName": "North Outfall", "status": "Active"},
    ]


def test_list_sites_parses(client: AyyekaClient, session: FakeSession) -> None:
    session.get_responses[client.api_url + "/site"] = FakeResponse(_site_payload())
    sites = client.list_sites()
    assert [s.id for s in sites] == [1, 2, 3]
    assert sites[0].latitude == pytest.approx(39.10)
    assert sites[1].longitude == pytest.approx(-84.20)
    assert sites[2].latitude is None


def test_list_sites_name_filter(client: AyyekaClient, session: FakeSession) -> None:
    session.get_responses[client.api_url + "/site"] = FakeResponse(_site_payload())
    sites = client.list_sites("north")
    assert {s.id for s in sites} == {1, 3}


def test_list_sites_regex_filter(client: AyyekaClient, session: FakeSession) -> None:
    session.get_responses[client.api_url + "/site"] = FakeResponse(_site_payload())
    sites = client.list_sites(r"^North", regex=True)
    assert {s.id for s in sites} == {1, 3}


def test_list_sites_status_filter(client: AyyekaClient, session: FakeSession) -> None:
    session.get_responses[client.api_url + "/site"] = FakeResponse(_site_payload())
    sites = client.list_sites(status="active")
    assert {s.id for s in sites} == {1, 3}


def test_get_site_by_id_and_name(client: AyyekaClient, session: FakeSession) -> None:
    session.get_responses[client.api_url + "/site"] = FakeResponse(_site_payload())
    assert client.get_site(2).display_name == "South Pump"
    assert client.get_site("North Plant").id == 1


def test_get_site_missing_raises(client: AyyekaClient, session: FakeSession) -> None:
    session.get_responses[client.api_url + "/site"] = FakeResponse(_site_payload())
    with pytest.raises(KeyError):
        client.get_site(999)


# ----------------------------------------------------------------------
# Location filters
# ----------------------------------------------------------------------


def test_find_sites_in_bbox(client: AyyekaClient, session: FakeSession) -> None:
    session.get_responses[client.api_url + "/site"] = FakeResponse(_site_payload())
    found = client.find_sites_in_bbox(39.0, -84.6, 39.2, -84.4)
    assert {s.id for s in found} == {1}


def test_find_sites_near_sorted(client: AyyekaClient, session: FakeSession) -> None:
    session.get_responses[client.api_url + "/site"] = FakeResponse(_site_payload())
    found = client.find_sites_near(39.10, -84.51, radius_km=60)
    # Site 1 is at the reference point; site 2 is farther but within radius.
    assert [s.id for s in found] == [1, 2]


def test_haversine_known_distance() -> None:
    # ~111 km per degree of latitude near the equator.
    assert _haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.5)


# ----------------------------------------------------------------------
# Stream discovery
# ----------------------------------------------------------------------


def test_list_streams(client: AyyekaClient, session: FakeSession) -> None:
    url = client.api_url + "/site/7/streams"
    session.get_responses[url] = FakeResponse(
        [
            {"id": 100, "displayName": "Level", "units": "ft", "sensorType": "ultrasonic"},
            {"id": 101, "displayName": "Flow", "unit": "MGD"},
        ]
    )
    streams = client.list_streams(7)
    assert [s.id for s in streams] == [100, 101]
    assert streams[0].units == "ft"
    assert streams[0].site_id == 7
    assert streams[0].attributes["sensorType"] == "ultrasonic"
    assert streams[1].units == "MGD"


# ----------------------------------------------------------------------
# Time series
# ----------------------------------------------------------------------


def test_to_epoch_naive_is_utc() -> None:
    assert AyyekaClient._to_epoch("2021-09-27T06:00:00") == 1632722400


def test_get_stream_timeseries_raw(client: AyyekaClient, session: FakeSession) -> None:
    url = client.api_url + "/stream/100/sample"
    session.get_responses[url] = FakeResponse(
        [
            {"datetime": 1632722400, "value": "1.5"},
            {"datetime": 1632726000, "value": "2.0"},
        ]
    )
    df = client.get_stream_timeseries(100, "2021-09-27", "2021-09-28")
    assert list(df.columns) == ["value"]
    assert len(df) == 2
    assert df.index.name == "datetime"
    assert df["value"].tolist() == [1.5, 2.0]
    # Query params carried epoch bounds.
    params = session.get_calls[-1]["params"]
    assert params["fromDate"] == AyyekaClient._to_epoch("2021-09-27")
    assert params["toDate"] == AyyekaClient._to_epoch("2021-09-28")


def test_get_stream_timeseries_raw_empty(client: AyyekaClient, session: FakeSession) -> None:
    session.get_responses[client.api_url + "/stream/100/sample"] = FakeResponse([])
    df = client.get_stream_timeseries(100, "2021-09-27", "2021-09-28")
    assert df.empty
    assert list(df.columns) == ["value"]


def test_get_stream_timeseries_aggregate(client: AyyekaClient, session: FakeSession) -> None:
    url = client.api_url + "/stream/123456/sample/agg"
    session.get_responses[url] = FakeResponse(
        [
            {"stream_id": 123456, "start_date": "2021-09-27T06:00:00Z",
             "number_of_samples": 53,
             "agg_values": {"min": "0.384", "max": "0.3855", "avg": "0.3843", "sum": "20.369"}},
            {"stream_id": 123456, "start_date": "2021-09-27T07:00:00Z",
             "number_of_samples": 3,
             "agg_values": {"min": "0.3843", "max": "0.3845", "avg": "0.3844", "sum": "1.1532"}},
        ]
    )
    df = client.get_stream_timeseries(
        123456, "2021-09-27T06:00:00Z", "2021-09-27T10:00:00Z",
        aggregate=True, time_frame=1,
    )
    assert list(df.columns) == ["min", "max", "avg", "sum", "number_of_samples"]
    assert len(df) == 2
    assert df.index.name == "start_date"
    assert df["avg"].iloc[0] == pytest.approx(0.3843)
    params = session.get_calls[-1]["params"]
    assert params["timeFrame"] == 1


def test_aggregate_without_time_frame_raises(client: AyyekaClient, session: FakeSession) -> None:
    with pytest.raises(ValueError, match="time_frame"):
        client.get_stream_timeseries(1, "2021-09-27", "2021-09-28", aggregate=True)


def test_aggregate_empty(client: AyyekaClient, session: FakeSession) -> None:
    session.get_responses[client.api_url + "/stream/1/sample/agg"] = FakeResponse([])
    df = client.get_stream_timeseries(1, "2021-09-27", "2021-09-28", aggregate=True, time_frame=1)
    assert df.empty
    assert "avg" in df.columns


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------


def test_site_from_json_nested_location() -> None:
    site = Site.from_json(
        {"id": 5, "displayname": "lower-case", "location": {"lat": 1.0, "lng": 2.0}}
    )
    assert site.id == 5
    assert site.display_name == "lower-case"
    assert site.latitude == 1.0
    assert site.longitude == 2.0


def test_stream_from_json_attributes() -> None:
    stream = Stream.from_json({"id": 9, "name": "Q", "extra": "keep"}, site_id=3)
    assert stream.id == 9
    assert stream.display_name == "Q"
    assert stream.site_id == 3
    assert stream.attributes == {"extra": "keep"}

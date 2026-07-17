"""Ayyeka REST API client for cloud-hosted sensor data.

:class:`AyyekaClient` talks to the Ayyeka REST API (``restapi.ayyeka.com``) to
discover monitoring sites, enumerate their streams, and download stream
time-series for a chosen date range.

Authentication uses the OAuth 2.0 *client-credentials* grant: an API client key
and secret (created in the Ayyeka UI) are exchanged for a short-lived bearer
access token, which is cached and transparently refreshed.  Credentials are
supplied explicitly or via the ``AYYEKA_CLIENT_KEY`` / ``AYYEKA_CLIENT_SECRET``
environment variables.

The Ayyeka data model is hierarchical::

    Account -> Site -> Device -> Channel -> Stream -> Sample

Sites carry location metadata, streams describe a measured quantity (with
units), and samples are the time-stamped scalar readings.

Example
-------
::

    from sparsehydro.data.services.ayyeka import AyyekaClient

    client = AyyekaClient()  # reads creds from the environment

    # Find sites near a coordinate.
    sites = client.find_sites_near(39.10, -84.51, radius_km=10)

    # List the streams available on the first site.
    streams = client.list_streams(sites[0].id)

    # Pull a stream's raw time series as a DataFrame.
    df = client.get_stream_timeseries(
        streams[0].id, start="2024-01-01", end="2024-02-01"
    )

.. note::
    The public *Getting Started* documentation only fully specifies the auth
    flow and the ``/stream/{id}/sample/agg`` endpoint.  The remaining endpoint
    paths (site listing, site streams, raw samples) follow the documented v2.0
    REST conventions and are exposed as overridable class constants so they can
    be adjusted without changing the client logic.

References
----------
- https://www.ayyeka.com/en/knowledge/getting-started-with-rest-api
"""

from __future__ import annotations

import base64
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd
import requests
from requests import Session

__all__ = ["AyyekaClient", "Site", "Stream"]


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-``None`` value among *keys* in *mapping*.

    The Ayyeka API is inconsistent about casing (``displayName`` vs
    ``displayname``) and nesting, so callers pass several candidate keys.

    :param mapping: Source dictionary.
    :param keys: Candidate keys to try, in order.
    :param default: Value returned when no key is present.
    :returns: The first matching value, or *default*.
    """
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _to_float(value: Any) -> float | None:
    """Coerce *value* to ``float``, returning ``None`` when it cannot be parsed.

    The Ayyeka API sometimes reports missing coordinates as an empty string or
    other non-numeric placeholder, so this tolerates those gracefully.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class Site:
    """A monitoring site returned by the Ayyeka API.

    :param id: Numeric site identifier.
    :param display_name: Human-readable site name.
    :param status: Site status string (e.g. ``"Active"``).
    :param latitude: Site latitude in decimal degrees, when available.
    :param longitude: Site longitude in decimal degrees, when available.
    :param raw: The unmodified site dict as returned by the service.
    """

    id: int
    display_name: str | None = None
    status: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Site:
        """Build a :class:`Site` from a raw API dict.

        :param payload: Site object as returned by the service.
        :returns: Parsed :class:`Site`.
        """
        location = payload.get("location")
        if isinstance(location, dict):
            lat = _first(location, "latitude", "lat")
            lon = _first(location, "longitude", "lon", "lng")
        else:
            lat = _first(payload, "latitude", "lat")
            lon = _first(payload, "longitude", "lon", "lng")

        return cls(
            id=int(payload["id"]),
            display_name=_first(payload, "displayName", "displayname", "display_name", "name"),
            status=_first(payload, "status"),
            latitude=_to_float(lat),
            longitude=_to_float(lon),
            raw=payload,
        )


@dataclass
class Stream:
    """A measurement stream belonging to a site/device/channel.

    :param id: Numeric stream identifier.
    :param display_name: Human-readable stream name.
    :param units: Engineering units of the measured value, when available.
    :param site_id: Owning site identifier, when known.
    :param attributes: Remaining stream metadata (everything not promoted to a
        dedicated field).
    :param raw: The unmodified stream dict as returned by the service.
    """

    id: int
    display_name: str | None = None
    units: str | None = None
    site_id: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict, repr=False)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, payload: dict[str, Any], *, site_id: int | None = None) -> Stream:
        """Build a :class:`Stream` from a raw API dict.

        :param payload: Stream object as returned by the service.
        :param site_id: Owning site id to attach, when known.
        :returns: Parsed :class:`Stream`.
        """
        promoted = {
            "id",
            "displayName",
            "displayname",
            "display_name",
            "name",
            "units",
            "unit",
            "siteId",
            "site_id",
        }
        return cls(
            id=int(payload["id"]),
            display_name=_first(payload, "displayName", "displayname", "display_name", "name"),
            units=_first(payload, "units", "unit"),
            site_id=site_id if site_id is not None else _first(payload, "siteId", "site_id"),
            attributes={k: v for k, v in payload.items() if k not in promoted},
            raw=payload,
        )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in kilometres.

    :param lat1: Latitude of the first point (decimal degrees).
    :param lon1: Longitude of the first point (decimal degrees).
    :param lat2: Latitude of the second point (decimal degrees).
    :param lon2: Longitude of the second point (decimal degrees).
    :returns: Distance in kilometres.
    """
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


class AyyekaClient:
    """Client for the Ayyeka cloud REST API.

    :param client_key: API client key.  Falls back to the
        ``AYYEKA_CLIENT_KEY`` environment variable when ``None``.
    :param client_secret: API client secret.  Falls back to the
        ``AYYEKA_CLIENT_SECRET`` environment variable when ``None``.
    :param api_url: Base REST URL.  Defaults to :attr:`DEFAULT_API_URL`.
    :param auth_url: Token endpoint URL.  Defaults to :attr:`DEFAULT_AUTH_URL`.
    :param timeout: Per-request timeout in seconds.
    :param session: Optional pre-configured :class:`requests.Session`.
    :raises ValueError: If a key or secret is not provided and the
        corresponding environment variable is unset.
    """

    #: Default base URL for the Ayyeka cloud REST API.
    DEFAULT_API_URL: str = "https://restapi.ayyeka.com/v2.0"
    #: Default OAuth 2.0 token endpoint.
    DEFAULT_AUTH_URL: str = "https://restapi.ayyeka.com/auth/token"

    #: Endpoint returning the list of sites.
    _SITES_PATH: str = "/site"
    #: Endpoint template returning the streams for a site.
    _SITE_STREAMS_PATH: str = "/site/{site_id}/streams"
    #: Endpoint template returning raw samples for a stream.
    _STREAM_SAMPLES_PATH: str = "/stream/{stream_id}/sample"
    #: Endpoint template returning aggregated samples for a stream.
    _STREAM_SAMPLES_AGG_PATH: str = "/stream/{stream_id}/sample/agg"

    #: Seconds of safety margin subtracted from token expiry before refresh.
    _TOKEN_REFRESH_MARGIN: float = 60.0

    def __init__(
        self,
        client_key: str | None = None,
        client_secret: str | None = None,
        *,
        api_url: str | None = None,
        auth_url: str | None = None,
        timeout: float = 30.0,
        session: Session | None = None,
    ) -> None:
        client_key = client_key if client_key is not None else os.environ.get("AYYEKA_CLIENT_KEY")
        client_secret = (
            client_secret if client_secret is not None else os.environ.get("AYYEKA_CLIENT_SECRET")
        )
        if not client_key:
            raise ValueError(
                "Ayyeka client key is required: pass client_key= or set the "
                "AYYEKA_CLIENT_KEY environment variable."
            )
        if not client_secret:
            raise ValueError(
                "Ayyeka client secret is required: pass client_secret= or set the "
                "AYYEKA_CLIENT_SECRET environment variable."
            )

        self._client_key = client_key
        self._client_secret = client_secret
        self.api_url = (api_url or self.DEFAULT_API_URL).rstrip("/")
        self.auth_url = auth_url or self.DEFAULT_AUTH_URL
        self.timeout = timeout
        self._session = session if session is not None else Session()

        self._token: str | None = None
        self._token_expiry: float = 0.0

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _basic_auth_header(self) -> str:
        """Return the ``Basic`` authorization header value for token requests.

        :returns: ``"Basic <base64(key:secret)>"``.
        """
        raw = f"{self._client_key}:{self._client_secret}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _authenticate(self) -> None:
        """Obtain and cache a fresh access token.

        :raises requests.exceptions.RequestException: On a failed request.
        :raises ValueError: If the response lacks an ``access_token``.
        """
        response = self._session.post(
            self.auth_url,
            headers={
                "Authorization": self._basic_auth_header(),
                "Content-Type": "application/x-www-form-urlencoded",
                "Cache-Control": "no-cache",
            },
            data={"grant_type": "client_credentials"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        token = payload.get("access_token")
        if not token:
            raise ValueError("Ayyeka auth response did not contain an access_token.")

        expires_in = float(payload.get("expires_in", 3600))
        self._token = token
        self._token_expiry = time.monotonic() + expires_in

    def _ensure_token(self) -> str:
        """Return a valid access token, refreshing it when expired.

        :returns: A non-expired bearer token.
        """
        if self._token is None or time.monotonic() >= (
            self._token_expiry - self._TOKEN_REFRESH_MARGIN
        ):
            self._authenticate()
        assert self._token is not None  # for type-checkers; set by _authenticate
        return self._token

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> requests.Response:
        """Issue an authenticated GET, refreshing the token on a 403.

        :param path: Path relative to :attr:`api_url` (leading slash optional).
        :param params: Optional query-string parameters.
        :returns: The successful :class:`requests.Response`.
        :raises requests.exceptions.RequestException: On connection failure,
            timeout, or a non-recoverable 4xx/5xx status code.
        """
        url = self.api_url + "/" + path.lstrip("/")
        token = self._ensure_token()
        response = self._session.get(
            url,
            headers={"Authorization": f"Bearer {token}", "Cache-Control": "no-cache"},
            params=params,
            timeout=self.timeout,
        )

        if response.status_code == 403:
            # Token may have been revoked/expired server-side; re-auth once.
            self._token = None
            token = self._ensure_token()
            response = self._session.get(
                url,
                headers={"Authorization": f"Bearer {token}", "Cache-Control": "no-cache"},
                params=params,
                timeout=self.timeout,
            )

        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Site discovery
    # ------------------------------------------------------------------

    def list_sites(
        self,
        name_filter: str | None = None,
        *,
        status: str | None = None,
        regex: bool = False,
    ) -> list[Site]:
        """Return sites, optionally filtered by name and/or status.

        Filtering is applied client-side on the returned records so it behaves
        consistently regardless of server-side query support.

        :param name_filter: When given, keep only sites whose ``display_name``
            matches.  Case-insensitive substring test by default; set
            ``regex=True`` to interpret it as a regular expression.
        :param status: When given, keep only sites whose ``status`` equals this
            value (case-insensitive).
        :param regex: Treat ``name_filter`` as a regular expression.
        :returns: List of matching :class:`Site` objects.
        :raises requests.exceptions.RequestException: On a failed request.
        """
        response = self._get(self._SITES_PATH)
        records = response.json()
        sites = [Site.from_json(item) for item in records]

        if status is not None:
            wanted = status.casefold()
            sites = [s for s in sites if (s.status or "").casefold() == wanted]

        if name_filter is not None:
            if regex:
                pattern = re.compile(name_filter, re.IGNORECASE)
                sites = [s for s in sites if pattern.search(s.display_name or "")]
            else:
                needle = name_filter.casefold()
                sites = [s for s in sites if needle in (s.display_name or "").casefold()]

        return sites

    def get_site(self, identifier: int | str) -> Site:
        """Return a single site by numeric id or exact display name.

        :param identifier: Numeric site id, or an exact ``display_name``.
        :returns: The matching :class:`Site`.
        :raises KeyError: If no site matches *identifier*.
        :raises requests.exceptions.RequestException: On a failed request.
        """
        for site in self.list_sites():
            if isinstance(identifier, int) and site.id == identifier:
                return site
            if isinstance(identifier, str) and site.display_name == identifier:
                return site
        raise KeyError(f"No Ayyeka site matching {identifier!r}.")

    def find_sites_in_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> list[Site]:
        """Return sites whose coordinates fall within a bounding box.

        Sites without location metadata are skipped.

        :param min_lat: Minimum (south) latitude.
        :param min_lon: Minimum (west) longitude.
        :param max_lat: Maximum (north) latitude.
        :param max_lon: Maximum (east) longitude.
        :returns: List of :class:`Site` objects inside the box.
        :raises requests.exceptions.RequestException: On a failed request.
        """
        result: list[Site] = []
        for site in self.list_sites():
            if site.latitude is None or site.longitude is None:
                continue
            if min_lat <= site.latitude <= max_lat and min_lon <= site.longitude <= max_lon:
                result.append(site)
        return result

    def find_sites_near(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> list[Site]:
        """Return sites within *radius_km* of a coordinate, nearest first.

        Sites without location metadata are skipped.

        :param latitude: Reference latitude in decimal degrees.
        :param longitude: Reference longitude in decimal degrees.
        :param radius_km: Search radius in kilometres.
        :returns: List of :class:`Site` objects within the radius, sorted by
            ascending distance.
        :raises requests.exceptions.RequestException: On a failed request.
        """
        scored: list[tuple[float, Site]] = []
        for site in self.list_sites():
            if site.latitude is None or site.longitude is None:
                continue
            distance = _haversine_km(latitude, longitude, site.latitude, site.longitude)
            if distance <= radius_km:
                scored.append((distance, site))
        scored.sort(key=lambda pair: pair[0])
        return [site for _, site in scored]

    # ------------------------------------------------------------------
    # Stream discovery
    # ------------------------------------------------------------------

    def list_streams(self, site_id: int) -> list[Stream]:
        """Return the streams belonging to a site.

        :param site_id: Numeric site identifier.
        :returns: List of :class:`Stream` objects, including their available
            attributes.
        :raises requests.exceptions.RequestException: On a failed request.
        """
        response = self._get(self._SITE_STREAMS_PATH.format(site_id=site_id))
        records = response.json()
        return [Stream.from_json(item, site_id=site_id) for item in records]

    # ------------------------------------------------------------------
    # Time series
    # ------------------------------------------------------------------

    @staticmethod
    def _to_epoch(value: Any) -> int:
        """Convert a datetime-like value to whole Unix epoch seconds (UTC).

        :param value: Anything :func:`pandas.to_datetime` accepts.
        :returns: Unix timestamp in seconds.
        """
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp())

    def get_stream_timeseries(
        self,
        stream_id: int,
        start: Any,
        end: Any,
        *,
        aggregate: bool = False,
        time_frame: int | None = None,
    ) -> pd.DataFrame:
        """Retrieve a stream's time series for a date range as a DataFrame.

        :param stream_id: Numeric stream identifier.
        :param start: Inclusive lower datetime bound (anything
            :func:`pandas.to_datetime` accepts).
        :param end: Inclusive upper datetime bound.
        :param aggregate: When ``True``, request hourly-style aggregates via the
            ``/sample/agg`` endpoint instead of raw samples.
        :param time_frame: Aggregation window in hours; required when
            ``aggregate=True``.
        :returns: DataFrame indexed by datetime.  For raw samples the column is
            ``value``; for aggregates the columns are ``min``, ``max``, ``avg``,
            ``sum`` (plus ``number_of_samples`` when present).
        :raises ValueError: If ``aggregate=True`` without a ``time_frame``.
        :raises requests.exceptions.RequestException: On a failed request.
        """
        from_epoch = self._to_epoch(start)
        to_epoch = self._to_epoch(end)

        if aggregate:
            if time_frame is None:
                raise ValueError("time_frame (hours) is required when aggregate=True.")
            response = self._get(
                self._STREAM_SAMPLES_AGG_PATH.format(stream_id=stream_id),
                params={"timeFrame": time_frame, "fromDate": from_epoch, "toDate": to_epoch},
            )
            return self._parse_aggregates(response.json())

        response = self._get(
            self._STREAM_SAMPLES_PATH.format(stream_id=stream_id),
            params={"fromDate": from_epoch, "toDate": to_epoch},
        )
        return self._parse_samples(response.json())

    @staticmethod
    def _parse_samples(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
        """Parse raw sample records into a datetime-indexed DataFrame.

        :param records: Iterable of sample dicts. Each is expected to expose a
            timestamp (``datetime``/``timestamp``/``date``) and a ``value``.
        :returns: DataFrame indexed by ``datetime`` with a ``value`` column.
            Empty input yields an empty, correctly-typed frame.
        """
        rows = list(records)
        if not rows:
            return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], name="datetime"))

        frame = pd.DataFrame(rows)
        time_col = next(
            (
                c
                for c in ("datetime", "timestamp", "date", "sampleDate", "sample_date")
                if c in frame.columns
            ),
            None,
        )
        value_col = next((c for c in ("value", "scalarValue") if c in frame.columns), None)
        if time_col is None or value_col is None:
            return frame

        index = pd.to_datetime(frame[time_col], unit="s", errors="coerce")
        if index.isna().all():
            index = pd.to_datetime(frame[time_col], errors="coerce")

        out = pd.DataFrame({"value": pd.to_numeric(frame[value_col], errors="coerce")})
        out.index = pd.DatetimeIndex(index, name="datetime")
        return out.dropna(how="all").sort_index()

    @staticmethod
    def _parse_aggregates(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
        """Parse aggregation records into a datetime-indexed DataFrame.

        :param records: Iterable of aggregate dicts, each with a ``start_date``
            and an ``agg_values`` mapping (``min``/``max``/``avg``/``sum``).
        :returns: DataFrame indexed by ``start_date`` with aggregate columns.
            Empty input yields an empty, correctly-typed frame.
        """
        rows = list(records)
        columns = ["min", "max", "avg", "sum", "number_of_samples"]
        if not rows:
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="start_date"))

        flattened: list[dict[str, Any]] = []
        index_values: list[Any] = []
        for row in rows:
            agg = row.get("agg_values", {}) or {}
            record: dict[str, Any] = {
                "min": agg.get("min"),
                "max": agg.get("max"),
                "avg": agg.get("avg"),
                "sum": agg.get("sum"),
                "number_of_samples": row.get("number_of_samples"),
            }
            flattened.append(record)
            index_values.append(_first(row, "start_date", "startDate", "fromDate"))

        frame = pd.DataFrame(flattened)
        for col in ("min", "max", "avg", "sum"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.index = pd.DatetimeIndex(
            pd.to_datetime(index_values, errors="coerce"), name="start_date"
        )
        return frame.sort_index()

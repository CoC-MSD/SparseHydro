"""FlowFinity REST client for MSDGC flow-monitoring data.

:class:`FlowFinityClient` queries the MSDGC FlowFinity ``ModelOpt`` REST service
for standardized flow-monitoring records and downloads the associated data
files (flow, level, velocity, rain, storm-flow, base-flow, and the recommended
optimization event file).

Authentication uses HTTP Basic Auth together with a host-header SSL adapter
(the service is reached via an IP/alias but presents a certificate for
``msdgc.org``).  Credentials are supplied explicitly or via the
``FLOWFINITY_USERNAME`` / ``FLOWFINITY_PASSWORD`` environment variables.

Example
-------
::

    from sparsehydro.data.services.flowfinity import FlowFinityClient

    client = FlowFinityClient()  # reads creds from the environment

    # List every record whose DataTitle contains "TC-NB-001"
    for record in client.list_resources("TC-NB-001"):
        print(record["DataTitle"])

    # Download storm-flow and rain for one record, parsed to DataFrames,
    # clipped to a date window.
    results = client.download(
        "TC-NB-001 Flow Edited Revised",
        types=["StormFlow", "Rain"],
        output_dir="downloads",
        parse=True,
        start="2022-01-25",
        end="2022-02-01",
    )
    df = results["StormFlow"].dataframe
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests import Session
from requests.auth import HTTPBasicAuth
from requests_toolbelt.adapters.host_header_ssl import HostHeaderSSLAdapter

from sparsehydro.data.utilities import read_tsf

__all__ = ["FlowFinityClient", "DownloadResult"]


@dataclass
class DownloadResult:
    """Outcome of downloading a single data type for a record.

    :param data_type: The FlowFinity data type (e.g. ``"StormFlow"``).
    :param path: Path of the written file, or ``None`` if the file was not
        saved to disk.
    :param dataframe: Parsed time series when ``parse=True`` and the payload is
        a ``.tsf`` file; otherwise ``None``.
    """

    data_type: str
    path: Path | None = None
    dataframe: pd.DataFrame | None = None


class FlowFinityClient:
    """Client for the MSDGC FlowFinity ``ModelOpt`` REST service.

    :param username: FlowFinity username.  Falls back to the
        ``FLOWFINITY_USERNAME`` environment variable when ``None``.
    :param password: FlowFinity password.  Falls back to the
        ``FLOWFINITY_PASSWORD`` environment variable when ``None``.
    :param base_url: Base REST URL.  Defaults to :attr:`DEFAULT_BASE_URL`;
        override to target a different host or service path.
    :param host_header: Value sent as the ``Host`` header (and used for SNI by
        the host-header SSL adapter).  Defaults to :attr:`DEFAULT_HOST_HEADER`.
    :param timeout: Per-request timeout in seconds.
    :param session: Optional pre-configured :class:`requests.Session`.  When
        ``None`` a session with a :class:`HostHeaderSSLAdapter` is created.
    :raises ValueError: If a username or password is not provided and the
        corresponding environment variable is unset.
    """

    #: Default base URL for the FlowFinity ModelOpt REST service.
    DEFAULT_BASE_URL: str = "https://flowfinity.msdgc.org/Flowfinity/rest/ModelOpt"
    #: Default ``Host`` header / SNI name the certificate is issued for.
    DEFAULT_HOST_HEADER: str = "msdgc.org"

    #: Endpoint returning the list of standardized records.
    _LIST_PATH: str = "/DataSets/FStandardizedView"
    #: Endpoint prefix for downloading an individual record's files.
    _RECORD_PATH: str = "/DataSets/ViewStandardRecord"

    #: Mapping of FlowFinity data type to its conventional output filename.
    DATA_TYPES: dict[str, str] = {
        "Flow": "flow_list.tsf",
        "Level": "lstDepthTs.tsf",
        "Velocity": "lstVelocityTs.tsf",
        "Rain": "rain_list.tsf",
        "StormFlow": "lstStormTs.tsf",
        "BaseFlow": "lstBaseflowTs.tsf",
        "RecommendedEventFileForOptimization": "lstEventsImported Events 0.dat",
    }

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        base_url: str | None = None,
        host_header: str | None = None,
        timeout: float = 30.0,
        session: Session | None = None,
    ) -> None:
        username = username if username is not None else os.environ.get("FLOWFINITY_USERNAME")
        password = password if password is not None else os.environ.get("FLOWFINITY_PASSWORD")
        if not username:
            raise ValueError(
                "FlowFinity username is required: pass username= or set the "
                "FLOWFINITY_USERNAME environment variable."
            )
        if not password:
            raise ValueError(
                "FlowFinity password is required: pass password= or set the "
                "FLOWFINITY_PASSWORD environment variable."
            )

        self._auth = HTTPBasicAuth(username, password)
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.host_header = host_header or self.DEFAULT_HOST_HEADER
        self.timeout = timeout

        if session is None:
            session = Session()
            session.mount("https://", HostHeaderSSLAdapter())
        self._session = session

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, url: str) -> requests.Response:
        """Issue an authenticated GET and raise on HTTP errors.

        :param url: Fully-qualified request URL.
        :returns: The successful :class:`requests.Response`.
        :raises requests.exceptions.RequestException: On connection failure,
            timeout, or a 4xx/5xx status code.
        """
        response = self._session.get(
            url,
            auth=self._auth,
            headers={"Host": self.host_header},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Resource discovery
    # ------------------------------------------------------------------

    def list_resources(
        self,
        name_filter: str | None = None,
        *,
        regex: bool = False,
    ) -> list[dict[str, Any]]:
        """Return standardized records, optionally filtered by ``DataTitle``.

        :param name_filter: When given, keep only records whose ``DataTitle``
            matches.  By default this is a case-insensitive substring test; set
            ``regex=True`` to interpret it as a regular expression
            (:func:`re.search`).
        :param regex: Treat ``name_filter`` as a regular expression.
        :returns: List of record dicts as returned by the service.
        :raises requests.exceptions.RequestException: On a failed request.
        """
        response = self._get(self.base_url + self._LIST_PATH)
        records: list[dict[str, Any]] = response.json()

        if name_filter is None:
            return records

        if regex:
            pattern = re.compile(name_filter, re.IGNORECASE)
            return [r for r in records if pattern.search(str(r.get("DataTitle", "")))]

        needle = name_filter.casefold()
        return [r for r in records if needle in str(r.get("DataTitle", "")).casefold()]

    def get_resource(self, data_title: str) -> dict[str, Any]:
        """Return the single record whose ``DataTitle`` equals *data_title*.

        :param data_title: Exact ``DataTitle`` to match.
        :returns: The matching record dict.
        :raises KeyError: If no record matches *data_title*.
        :raises requests.exceptions.RequestException: On a failed request.
        """
        for record in self.list_resources():
            if record.get("DataTitle") == data_title:
                return record
        raise KeyError(f"No FlowFinity record with DataTitle == {data_title!r}.")

    def available_types(self, data_title: str) -> list[str]:
        """List the downloadable data types present on a record.

        :param data_title: Exact ``DataTitle`` to inspect.
        :returns: Known data types (a subset of :attr:`DATA_TYPES`) that the
            record exposes with a ``URL``.
        :raises KeyError: If no record matches *data_title*.
        """
        record = self.get_resource(data_title)
        return [
            data_type
            for data_type in self.DATA_TYPES
            if isinstance(record.get(data_type), dict) and record[data_type].get("URL")
        ]

    # ------------------------------------------------------------------
    # Downloading
    # ------------------------------------------------------------------

    def download(
        self,
        data_title: str,
        *,
        types: Iterable[str] | None = None,
        output_dir: str | os.PathLike[str] | None = ".",
        parse: bool = False,
        start: Any = None,
        end: Any = None,
        overwrite: bool = True,
    ) -> dict[str, DownloadResult]:
        """Download one or more data files for a record.

        :param data_title: Exact ``DataTitle`` identifying the record.
        :param types: Data types to download (keys of :attr:`DATA_TYPES`).
            ``None`` downloads every type the record exposes.
        :param output_dir: Directory to write files into.  Created if needed.
            Pass ``None`` to skip writing files to disk (in-memory only; useful
            with ``parse=True``).
        :param parse: When ``True``, parse ``.tsf`` payloads into pandas
            DataFrames (returned on :attr:`DownloadResult.dataframe`).  Non-tsf
            payloads (e.g. the ``.dat`` event file) are never parsed.
        :param start: Optional inclusive lower datetime bound applied to parsed
            DataFrames (anything :func:`pandas.to_datetime` accepts).
        :param end: Optional inclusive upper datetime bound for parsed
            DataFrames.
        :param overwrite: When ``False``, existing files are left untouched and
            reused (and re-parsed if ``parse=True``) instead of re-downloaded.
        :returns: Mapping of data type to :class:`DownloadResult`.
        :raises KeyError: If *data_title* is unknown or an unrecognized type is
            requested.
        :raises requests.exceptions.RequestException: On a failed download.
        """
        record = self.get_resource(data_title)

        if types is None:
            requested = list(self.available_types(data_title))
        else:
            requested = list(types)
            unknown = [t for t in requested if t not in self.DATA_TYPES]
            if unknown:
                raise KeyError(
                    f"Unknown data type(s) {unknown!r}; valid types are "
                    f"{sorted(self.DATA_TYPES)}."
                )

        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, DownloadResult] = {}
        for data_type in requested:
            type_obj = record.get(data_type)
            if not isinstance(type_obj, dict) or not type_obj.get("URL"):
                # Record does not expose this type; skip silently.
                continue

            filename = self.DATA_TYPES[data_type]
            dest = out_dir / filename if out_dir is not None else None

            if dest is not None and dest.exists() and not overwrite:
                content = dest.read_bytes()
            else:
                content = self._get(self.base_url + self._RECORD_PATH + type_obj["URL"]).content
                if dest is not None:
                    dest.write_bytes(content)

            dataframe = None
            if parse and filename.endswith(".tsf"):
                dataframe = read_tsf(content, start=start, end=end)

            results[data_type] = DownloadResult(
                data_type=data_type, path=dest, dataframe=dataframe
            )

        return results

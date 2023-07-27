from io import StringIO
from typing import Sequence
from urllib.parse import urlencode
from aiohttp_retry import RetryClient,RetryOptionsBase,ExponentialRetry
import pandas as pd
import requests
from warnings import warn
from storms._datasource import _DataSource
from storms._utils import datetime_like


class ASOS(_DataSource):
    def __init__(self, FAA: str):
        """Methods for pulling data from the Iowa  State University
        Environmental Mesonet:

        https://mesonet.agron.iastate.edu/ASOS/

        Parameters
        ----------
        FAA: str
            The FAA code of the station

        Returns
        -------

        """
        self.faa = self.FAA = FAA
        self.URL = "https://mesonet.agron.iastate.edu/request/asos/1min_dl.php?"
        self.latlon = None
        self.utc_offset = None
        self.get_meta()

    @staticmethod
    def map():
        """A folium map to help find asos stations. Using in
        jupyter notebook will display the map inline. Using in terminal
        will only produce a folium map object. The map can be saved to html
        with `map.save('path_to_html_file.html')`

        Returns
        -------
        folium.Map
            interactive map of active asos stations in Iowa mesonet's database

        Raises
        ------
        ModuleNotFoundError
            Error if folium not installed. Since this function is not imperative to the
            the library, folium is not included in major requirements.txt file

        Examples
        --------
        >>> from storms.precip import datasets as pds
        >>> pds.asos.map()
        >>> # if you are not in juptyer, you'll need to save to html
        >>> map.save('ASOS_Stations.html')
        """

        # if "folium" not in sys.modules:
        try:
            # globals()["folium"] = __import__("folium")
            # globals()["folium.plugins"] = __import__("folium.plugins")
            import folium
            import folium.plugins

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "folium is needed to produce this map. Install it with `pip install folium`"
            )

        url = "https://mesonet.agron.iastate.edu/geojson/network/ASOS1MIN.geojson?only_online=1"
        geojson = requests.get(url).json()

        m = folium.Map(
            location=[40.742, -80.956],
            zoom_start=4,
            tiles="http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            attr="Carto",
        )

        popups = []
        locations = []
        for feat in geojson["features"]:
            locations.append(feat["geometry"]["coordinates"][::-1])
            popups.append(
                folium.Popup(
                    html="<br>".join(
                        [
                            "<b>SID:</b> " + feat["properties"]["sid"],
                            "<b>Name:</b> " + feat["properties"]["sname"],
                            "<b>latlon:</b> "
                            + str(tuple(feat["geometry"]["coordinates"][::-1])),
                        ]
                    ),
                    max_width="300",
                ),
            )

        folium.plugins.MarkerCluster(locations=locations, popups=popups).add_to(m)
        return m

    def get_meta(self):
        """ """
        url = f"https://www.ncei.noaa.gov/access/homr/services/station/search?qid=FAA:{self.faa}"
        json = requests.get(url).json()
        self._meta_json = json
        por = self._meta_json["stationCollection"]["stations"][0]["header"]["por"]
        self.por = dict(
            start=pd.Timestamp(por["beginDate"]),
            end=pd.Timestamp(por["endDate"].replace("Present", "now")),
        )
        if len(json["stationCollection"]["stations"]) > 1:
            raise UserWarning(
                "Found multiple stations in HORM meta data. ",
                "Please raise in issue in the package repository regarding this station. ",
                "Meta data cannot be found, lat and lon will not be set.",
            )
        try:
            station = json["stationCollection"]["stations"][0]
            self.latlon = (
                float(station["header"]["latitude_dec"]),
                float(station["header"]["longitude_dec"]),
            )
            self.utc_offset = int(
                station["location"]["geoInfo"]["utcOffsets"][0]["utcOffset"]
            )
        except Exception as e:
            print(
                "Failed collecting meta data for this station. Please raise an issue in the package repository.",
                str(e),
            )

    def request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        aSync=False,
        progress: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Request precipitation DataFrame with annual requests to Iowa Mesonet
        For potential faster data pulls for longer time periods, use with aSync=True

        https://mesonet.agron.iastate.edu/ASOS/

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime

        process_data: bool
            Switch to process mesonet data dropping suspect values or not.
            Set to False to get all the raw data pulled from mesonet.
            Set to True to drop O precip values NP ptype values

        aSync: bool
            Switch to pull data asynchronously (can be much faster for longer data pulls), defaults to False

        progress: bool
            Switch to show tqdm bar for data download progress, defaults to True

        **kwargs: dict
            Keyword arguments to pass onto _async_request_dataframe or _sync_request_dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame of precipitation records.

        Examples
        --------
        >>> from storms.precip import datasets as pds
        >>> bos = pds.asos("BOS")  #station ID for KBOS
        >>> # use aSync to download large datasets fast and process_data=True (default) to post process raw data, removing suspect values.
        >>> df = bos.request_dataframe("1/1/2010","1/1/2021", aSync=True)
        """
        return super().request_dataframe(
            start, end, process_data, aSync, progress, **kwargs
        )

    def _request_url(
        self,
        start: datetime_like,
        end: datetime_like,
        datatype: Sequence[str] = ["precip", "ptype"],
    ) -> str:
        """Get a formatted asos request URL for a given datatype. Default data type is precip
        for minutely precip and ptype for QAQC records of each value.

        See the request php code for more details
        https://github.com/akrherz/iem/blob/main/htdocs/request/asos

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime

        vars: list, optional
            datatype to pull, should be one of:
            * `tmpf`: Air Temperature [F]
            * `dwpf`: Dew Point Temperature [F]
            * `sknt`: Wind Speed [knots]
            * `drct`: Wind Direction
            * `gust_drct`: 5 sec gust Wind Direction
            * `gust_sknt`: 5 sec gust Wind Speed [knots]
            * `vis1_coeff`: Visibility 1 Coefficient
            * `vis1_nd`: Visibility 1 Night/Day
            * `vis2_coeff`: Visibility 2 Coefficient
            * `vis2_nd`: Visibility 2 Night/Day
            * `vis3_coeff`: Visibility 3 Coefficient
            * `vis3_nd`: Visibility 3 Night/Day
            * `ptype`: Precip Type Code
            * `precip`: 1 minute precip [inches]
            * `pres1`: Sensor 1 Station Pressure [inches]
            * `pres2`: Sensor 2 Station Pressure [inches]
            * `pres3`: Sensor 3 Station Pressure [inches]
            , defaults to ['precip','ptype']

        Returns
        -------
        str
            Iowa Mesonet URL for requested asus data

        """
        st = pd.to_datetime(start)  # .tz_localize(self.TZ)
        ed = pd.to_datetime(end)  # .tz_localize(self.TZ)

        if self.utc_offset is not None:
            st -= pd.Timedelta(f"{self.utc_offset}H")
            ed -= pd.Timedelta(f"{self.utc_offset}H")

        params = (
            ("day1", st.strftime("%d")),
            ("day2", ed.strftime("%d")),
            ("month1", st.strftime("%m")),
            ("month2", ed.strftime("%m")),
            ("year1", st.strftime("%Y")),
            ("year2", ed.strftime("%Y")),
            ("hour1", st.strftime("%H")),
            ("hour2", ed.strftime("%H")),
            ("minute1", st.strftime("%M")),
            ("minute2", ed.strftime("%M")),
            ("tz", "UTC"),
            ("station[]", [self.faa]),
            ("vars[]", datatype),
            ("sample", "1min"),
            ("delim", "comma"),
            ("gis", "no"),
            ("what", "download"),
        )
        return self.URL + urlencode(params, doseq=True)

    def _sync_request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        pull_freq: str = "AS",
    ) -> pd.DataFrame:
        """Request precipitation DataFrame with synchronous annual requests to Iowa Mesonet
        For potential faster data pulls for longer time periods, use the
        asynchronous request method `_async_request_dataframe`.

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime

        process_data: bool
            Switch to process mesonet data dropping suspect values or not.
            Set to False to get all the raw data pulled from mesonet.
            Set to True to drop O precip values NP ptype values

        pull_freq: str
            Pandas offset string for the length of time for each async data request, defaults to annual (AS).

        Returns
        -------
        pd.DataFrame
            DataFrame of precipitation records.

        """
        data = self._sync_request_data_series(start, end, pull_freq)

        datastr = "\n".join(data)
        with StringIO(datastr) as s:
            df = pd.read_csv(
                s,
                header=None,
                names=["FAA", "Station", "dtUTC", "precip", "ptype"],
                usecols=range(5),
            )

        if process_data:
            self._update_progress_description("Processing")
            df = self._process_data(df)

        self._close_progress()

        return df

    def _sync_request_data(
        self, start: datetime_like, end: datetime_like, session: requests.Session
    ) -> str:
        """Synchronously Request minutely rainfall CSV data from mesonet

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime

        session: requests.Session
            requests Session to use for API calls

        Returns
        -------
        str
            string of csv data returned from mesonet

        """
        url = self._request_url(start, end, ["precip", "ptype"])
        with session.get(url) as response:
            return "\n".join(response.text.split("\n")[1:])

    async def _async_request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        pull_freq: str = "AS",
        conn_limit: int = 5,
        retry_options: RetryOptionsBase = ExponentialRetry(attempts=5, start_timeout=0.1)
    ) -> pd.DataFrame:
        """
        Request precipitation DataFrame with asynchronous annual requests to Iowa Mesonet

        This is an async function and must be awaited as::

            df = await gage._async_request_dataframe(start = '1/1/2020', end = '1/1/2021')

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime

        process_data: bool
            Switch to process mesonet data dropping suspect values or not.
            Set to False to get all the raw data pulled from mesonet.
            Set to True to drop O precip values NP ptype values

        pull_freq: str
            Pandas offset string for the length of time for each async data request, defaults to annual (AS).

        conn_limit: int
            Connection limit for aiohttp session, defaults to 5

        retry_options: RetryOptionsBase
            Retry options to pass to aiohttp_retry client

        Returns
        -------
        pd.DataFrame
            DataFrame of precipitation records.

        """

        # I think the mesonet limits to 5 connections over 300s or 300ms?
        # setting conn_limit to 5 for now, seems to work okay
        # https://github.com/akrherz/iem/blob/main/include/throttle.php
        data = await self._async_request_data_series(start, end, pull_freq, conn_limit, retry_options)
        datastr = "\n".join(data)
        with StringIO(datastr) as s:
            df = pd.read_csv(
                s,
                header=None,
                names=["FAA", "Station", "dtUTC", "precip", "ptype"],
                usecols=range(5),
            )
        if process_data:
            self._update_progress_description("Processing")
            df = self._process_data(df)

        self._close_progress()

        return df

    async def _async_request_data(
        self, start: datetime_like, end: datetime_like, session: RetryClient
    ) -> str:
        """
        Request minutely rainfall CSV data from Iowa mesonet

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime

        session: aiohttp_retry.RetryClient
            aiohttp_retry RetryClient to use for API calls

        Returns
        -------
        str
            Returns csv string from API request.
        """

        url = self._request_url(start, end, ["precip", "ptype"])
        # async json response
        # https://docs.aiohttp.org/en/stable/client_quickstart.html#json-response-content
        async with session.get(url) as response:
            text = await response.text()
            return "\n".join(text.split("\n")[1:])

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw asos data response into a cleaner dataframe.
        Routine drops NP data pulled from API, tried to convert UTC
        datetimes into local time based on info from HOMR, then reorders columns.

        Asos Data Flags
        -------------
        UP: Unknown Precipitation
        -R, R, +R: Liquid Rain (light, moderate, heavy)
        -FZR, FZR: Freezeing Rain (light, moderate)
        S-, S, S+: Frozen Rain (snow) (light moderate, heavy)
        NP: No precipitation

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame returned from _async_request_dataframe or _sync_request_dataframe

        Returns
        -------
        pd.DataFrame
            pandas DataFrame of cleaned data

        """

        df.loc[:, "dtUTC"] = pd.to_datetime(df.dtUTC)

        # drop zeros and NP values
        df = df.loc[(df.ptype != "NP") & (df.precip > 0)]

        if self.utc_offset is None:
            warn(
                "No timezone info set (self.utc_offset), cannot convert datetime to LST"
            )
            return df.reindex(["dtUTC", "precip", "ptype"], axis=1)

        else:
            df = df.reindex(["dtLocal", "dtUTC", "precip", "ptype"], axis=1)
            df.loc[:, "dtLocal"] = df.dtUTC + pd.Timedelta(f"{self.utc_offset}H")
            return df

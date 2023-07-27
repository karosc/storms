import ftplib
from io import BytesIO, StringIO
from typing import List, Sequence, Union

from aiohttp_retry import RetryClient,RetryOptionsBase,ExponentialRetry
import pandas as pd
import requests
from pytz import FixedOffset  # type: ignore

from storms._datasource import _DataSource, flatten
from storms._utils import timed_lru_cache, datetime_like
import os.path

_this_dir, _this_filename = os.path.split(__file__)

isd_history = os.path.join(_this_dir, "isd_history.txt")


heads = {
    "USAF": str,
    "WBAN": str,
    "STATION NAME": str,
    "CTRY": str,
    "ST": str,
    "CALL": str,
    "LAT": float,
    "LON": float,
    "ELEV(M)": float,
    "BEGIN": str,
    "END": str,
}

cols = (
    (0, 7),
    (7, 13),
    (13, 43),
    (43, 48),
    (48, 51),
    (51, 57),
    (57, 65),
    (65, 74),
    (74, 82),
    (82, 91),
    (91, 99),
)


def _getStationsFTP(only_complete_id: bool = False) -> pd.DataFrame:
    """Function to pull list of ISD stations and some limited meta data.

    TODO: Add in HOMR functionality for additional metadata

    Parameters
    ----------
    only_complete_id: bool, optional
        Switch to filter stations to only include those with
        with a USAF and WBAN ID number. Some older data only
        have WBAN and USAF set to 999999, defaults to False

    Returns
    -------
    pd.DataFrame
        DataFrame of ISD station and meta data

    """
    ftp = ftplib.FTP("ftp.ncdc.noaa.gov", "anonymous")
    r = BytesIO()
    ftp.retrbinary("RETR /pub/data/noaa/isd-history.txt", r.write)
    data = r.getvalue().decode().split("\n")[20:]
    del data[1]

    stations = pd.read_fwf(StringIO("\n".join(data)), converters=heads, colspecs=cols)
    stations.LAT = stations.LAT.astype(float).round(3)
    stations.LON = stations.LON.astype(float).round(3)

    if only_complete_id:
        stations = stations.loc[
            (stations.USAF != "999999") & (stations.WBAN != "99999")
        ]

    ftp.close()
    return stations


class GlobalHourly(_DataSource):
    def __init__(self, ID: str):
        """Methods for pulling data from NOAA's global hourly `ISD dataset`_.

        .. _ISD dataset: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database

        Parameters
        ----------
        ID: str
            ISD station ID to pull

        Returns
        -------

        """

        self.ID = ID
        self.URL = f"https://www.ncei.noaa.gov/access/services/data/v1?format=json&dataset=global-hourly&stations={self.ID}"
        self.qid = ID[6:]
        self.get_meta()

    # cache for 6 hours until next update (speed up repatative calls to table
    # https://gist.github.com/Morreski/c1d08a3afa4040815eafd3891e16b945
    @staticmethod
    @timed_lru_cache(seconds=60 * 60 * 6)
    def stations(only_complete_id: bool = False) -> pd.DataFrame:
        """Function to pull list of ISD stations and some limited meta data.

        TODO: Add in HOMR functionality for additional metadata

        Parameters
        ----------
        only_complete_id: bool, optional
            Switch to filter stations to only include those with
            with a USAF and WBAN ID number. Some older data only
            have WBAN and USAF set to 999999, defaults to False

        Returns
        -------
        pd.DataFrame
            DataFrame of ISD station and meta data

        Examples
        --------
        >>> from storms.precip import datasets as pds
        >>> pds.GlobalHourly.stations()
        """

        stations = pd.read_fwf(
            "https://www1.ncdc.noaa.gov/pub/data/noaa/isd-history.txt",
            # "https://web.archive.org/web/20220121062056/https://www1.ncdc.noaa.gov/pub/data/noaa/isd-history.txt",
            # isd_history,
            converters=heads,
            colspecs=cols,
            skiprows=20,
        )

        stations.LAT = stations.LAT.astype(float).round(3)
        stations.LON = stations.LON.astype(float).round(3)

        if only_complete_id:
            stations = stations.loc[
                (stations.USAF != "999999") & (stations.WBAN != "99999")
            ]
        return stations

    @staticmethod
    def map(only_complete_id: bool = True):
        """A folium map to help find global hourly stations. Using in
        jupyter notebook will display the map inline. Using in terminal
        will only produce a folium map object. The map can be saved to html
        with :python:`map.save('path_to_html_file.html')`

        Parameters
        ----------
        only_complete_id: bool, optional
            Switch to filter stations to only include those with
            with a USAF and WBAN ID number. Some older data only
            have WBAN and USAF set to 999999, defaults to False

        Returns
        -------
        folium.Map
            Interactive map of global hourly stations in NOAA's database.

        Raises
        ------
        ModuleNotFoundError
            Error if folium not installed. Since this function is not imperative to the
            the library, folium is not included in major requirements.txt file.

        Examples
        --------
        >>> from storms.precip import datasets as pds
        >>> pds.GlobalHourly.map()
        >>> # if you are not in juptyer, you'll need to save to html
        >>> map.save('ISD_Stations.html')
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

        df = GlobalHourly.stations(only_complete_id)
        valid_coords = ~df.loc[:, ["LAT", "LON"]].isnull().any(axis=1)
        # print(len(df))
        df = df.loc[valid_coords]
        # print(len(df))

        # print("valid")
        locations = df.loc[:, ["LAT", "LON"]].to_numpy(copy=True)
        popups = df.apply(
            lambda row: folium.Popup(  # type: ignore
                html="<br>".join(
                    [
                        "<b>ID:</b> " + row["USAF"] + row["WBAN"],
                        #'STATION NAME:'+row['STATION NAME'],
                        "<b>CALL:</b> " + str(row["CALL"]),
                        "<b>BEGIN:</b> " + str(row["BEGIN"]),
                        "<b>END:</b> " + str(row["END"]),
                    ]
                ),
                max_width="300",
            ),
            axis=1,
        ).to_list()

        m = folium.Map(  # type: ignore
            location=[40.742, -80.956],
            zoom_start=4,
            tiles="http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            attr="Carto",
        )
        folium.plugins.MarkerCluster(locations=locations, popups=popups).add_to(m)  # type: ignore

        folium.plugins.Fullscreen().add_to(m)  # type: ignore
        # folium.plugins.LocateControl().add_to(map_osm)
        return m

    def get_meta(self) -> None:
        """Function to pull meta data for station into object properties.

        Returns
        -------
        None
            Returns nothing, but sets object latitude, longitude, and timezone properties.

        Raises
        ------
        Exception
            If duplicate station ID found in stations table, raise exception.

        """
        usaf = self.ID[0:6]
        wban = self.ID[6:]
        stations = self.stations()

        homrURL = f"https://www.ncei.noaa.gov/access/homr/services/station/search?qid=:{self.ID[6:]}"
        homr = requests.get(homrURL).json()["stationCollection"]["stations"][0]

        df = stations.loc[
            (stations.USAF.astype(str) == usaf) & (stations.WBAN.astype(str) == wban)
        ]

        if len(df) > 1:
            raise Exception("ERROR, GageID not unique")

        else:
            # pull timezone based on station lat lon

            # self.latlon = (
            #     float(homr["header"]["latitude_dec"]),
            #     float(homr["header"]["longitude_dec"]),
            # )
            self.utc_offset = int(
                homr["location"]["geoInfo"]["utcOffsets"][0]["utcOffset"]
            )
            self.TZ = FixedOffset(self.utc_offset * 60)

            self.meta = df.iloc[0].to_dict()
            self.meta["BEGIN"] = pd.to_datetime(self.meta["BEGIN"]).strftime("%Y-%m-%d")
            self.meta["END"] = pd.to_datetime(self.meta["END"]).strftime("%Y-%m-%d")
            self.meta[
                "homrLNK"
            ] = f"https://www.ncei.noaa.gov/access/homr/#ncdcstnid={homr['ncdcStnId']}&tab=MSHR"

            # url = "http://api.geonames.org/timezoneJSON?formatted=true&lat={}&lng={}&username=karosc".format(
            #     self.meta.LAT, self.meta.LON
            # )
            # r = requests.get(url)
            # self.TZ = r.json()["timezoneId"]

    def request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        aSync: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Request precipitation DataFrame with annual requests to NOAA V1 API.
        For potential faster data pulls for longer time periods, use with aSync=True

        https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime.

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime.

        process_data: bool
            Switch to process NOAA rainfall data into hourly records or not.
            Set to False to get all the raw data pulled from NOAA.
            Set to True to get only FM-15 and SAO hourly records that are not
            suspect.

        aSync: bool
            Switch to pull data asynchronously (can be much faster for longer data pulls), defaults to False.

        progress: bool
            Switch to show tqdm bar for data download progress, defaults to True.

        **kwargs: dict
            Keyword arguments to pass onto _async_request_dataframe or _sync_request_dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame of precipitation records.

        Examples
        --------
        >>> from storms.precip import datasets as pds
        >>> bos = pds.GlobalHourly("72509014739")  #station ID for KBOS
        >>> # use aSync to download large datasets fast and process_data=True (default) to post process raw data into hourly records
        >>> df = bos.request_dataframe("1/1/2010","1/1/2021", aSync=True)
        """

        return super().request_dataframe(
            start, end, process_data, aSync, progress, **kwargs
        )

    def _request_url(
        self,
        start: datetime_like,
        end: datetime_like,
        datatype: Union[str, Sequence[str]] = "AA1",
    ) -> str:
        """Get a formatted global hourly request URL for a given datatype. Default data type is AA1 for hourly precip.
        See ISD format document for examples:
        http://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime
        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime
        datatype: str, optional
            ISD datatype to pull, defaults to 'AA1'

        Returns
        -------
        str
            NOAA v1 API URL for request data
            (https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation)

        Raises
        ------
        Exception
            If more than 10 years of data are requested, use pullRainfall.

        """
        st = pd.to_datetime(start).tz_localize(self.TZ)
        ed = pd.to_datetime(end).tz_localize(self.TZ)
        if (ed - st).total_seconds() / 60 / 60 / 24 / 365.25 > 10:
            raise Exception(
                "NOAA API can only serve up 10 years at a time. Use pullRainfall method "
                "to pull longer timeseries"
            )

        url = (
            self.URL + f"&startDate={st.isoformat()}&endDate={ed.isoformat()}"
        ).replace("+", "%2B")

        if datatype != None:
            url += f"&dataTypes={datatype}"
        return url

    def _sync_request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        pull_freq: str = "AS",
    ) -> pd.DataFrame:
        """Request precipitation DataFrame with synchronous annual requests to NOAA V1 API.
        For potential faster data pulls for longer time periods, use the
        asynchronous request method `_async_request_dataframe`.

        https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime.

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime.

        process_data: bool
            Switch to process NOAA rainfall data into hourly records or not.
            Set to False to get all the raw data pulled from NOAA.
            Set to True to get only FM-15 and SAO hourly records that are not
            suspect.

        pull_freq: str
            Pandas offset string for the length of time for each async data request, defaults to annual (AS).

        Returns
        -------
        pd.DataFrame
            DataFrame of precipitation records.

        """
        data = pd.DataFrame(
            flatten(self._sync_request_data_series(start, end, pull_freq))
        )

        if process_data:
            self._update_progress_description("Processing")
            data = self._process_data(data)

        self._close_progress()

        return data

    def _sync_request_data(
        self, start: datetime_like, end: datetime_like, session: requests.Session
    ) -> List[dict]:
        """Synchronously request hourly rainfall JSON data from NOAA's v1 API

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime.

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime.

        session: requests.Session
            requests Session to use for API calls.

        Returns
        -------
        list
            returns decoded JSON from API call. The global hourly API returns a JSON array of record

        """
        url = self._request_url(start, end, "AA1")
        with session.get(url) as response:
            return response.json()

    # async functions
    async def _async_request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        pull_freq: str = "AS",
        conn_limit: int = 30,
        retry_options: RetryOptionsBase = ExponentialRetry(attempts=5, start_timeout=0.1)

    ) -> pd.DataFrame:
        """
        Request precipitation DataFrame with asynchronous annual requests to NOAA V1 API.
        https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

        This is an async function and must be awaited as::

            df = await gage._async_request_dataframe(start = '1/1/2020', end = '1/1/2021')

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime

        process_data: bool
            Switch to process NOAA rainfall data into hourly records or not.
            Set to False to get all the raw data pulled from NOAA.
            Set to True to get only FM-15 and SAO hourly records that are not
            suspect.

        pull_freq: str
            Pandas offset string for the length of time for each async data request, defaults to annual (AS).

        conn_limit: int
            Connection limit for aiohttp session, defaults to 30

        retry_options: RetryOptionsBase
            Retry options to pass to aiohttp_retry client
            
        Returns
        -------
        pd.DataFrame
            DataFrame of precipitation records.

        """

        data = await self._async_request_data_series(start, end, pull_freq, conn_limit, retry_options,)
        data = pd.DataFrame(flatten(data))

        if process_data:
            self._update_progress_description("Processing")
            data = self._process_data(data)

        self._close_progress()

        return data

    async def _async_request_data(
        self, start: datetime_like, end: datetime_like, session: RetryClient
    ) -> List[dict]:
        """
        Request hourly rainfall JSON data from NOAA's v1 API

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
        List
           decoded JSON from API call. The global hourly API returns a JSON array of record
        """

        url = self._request_url(start, end, "AA1")
        # async json response
        # https://docs.aiohttp.org/en/stable/client_quickstart.html#json-response-content
        async with session.get(url) as response:
            return await response.json()

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw global hourly data response into hourly DataFrame.
        Applied basic QAQC and data filtering logic.

        Parameters
        ----------
        json: List[dict]
            list of global hourly records from decoded json

        Returns
        -------
        pd.DataFrame
            pandas DataFrame of filtered hourly data

        """
        data = data.dropna(subset=["AA1"]).reset_index(drop=True)
        data = pd.concat(
            [
                data.loc[:, ["DATE", "REPORT_TYPE", "STATION", "CALL_SIGN"]],
                data.AA1.str.split(",", expand=True),
            ],
            axis=1,
        )
        data.columns = [
            "UTC",
            "REPORT_TYPE",
            "STATION",
            "CALL_SIGN",
            "PERIOD",
            "DEPTH",
            "CONDITION",
            "QUALITY",
        ]
        # return data
        # convert data to local time zone
        data["UTC"] = pd.to_datetime(data["UTC"])

        data["LocalTime"] = (
            data["UTC"]
            .dt.tz_localize("UTC")
            .dt.tz_convert(self.TZ)
            .dt.tz_localize(None)
        )

        data["PERIOD"] = data["PERIOD"].astype(int)

        # 10ths of mm to inches
        data["Inches"] = round(data["DEPTH"].astype(float) / 10 / 25.4, 2)

        # basic QAQC filter
        # 1. only want FM-15 or SAO data
        # 2. only on non-zero precip
        # 3. ignore missing data with 9999
        # 4. only grab hourly periods
        # 5. remove suspect data that are not from NCEI source (Qual Flag of 2)
        # 6. remove erroneous data that are not from NCEI source (Qual Flag of 3)

        data = data.loc[
            (
                (data.REPORT_TYPE.str.strip() == "FM-15")
                | (data.REPORT_TYPE.str.strip() == "SAO")
            )
            & (data.Inches > 0)
            & (data.DEPTH.astype(int) < 9999)
            & (data.PERIOD == 1)
            & (~data.QUALITY.str.contains("2|3"))
            # & (~out.QUALITY.str.contains("1|2|I"))
        ]

        # round all data to hourly and keep last hourly measurement only (avoid dups)
        data["Hourly"] = data["LocalTime"].dt.floor("H")
        data = data.sort_values("Hourly", ascending=True).drop_duplicates(
            subset=["Hourly"], keep="last"
        )

        return data

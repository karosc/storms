from aiohttp_retry import RetryClient,RetryOptionsBase,ExponentialRetry
import pandas as pd
from storms._datasource import _DataSource, flatten
from urllib.parse import urlencode
from typing import List
import requests
from storms._utils import datetime_like


class NOAAtides(_DataSource):
    def __init__(self, stationID: str):
        """
        Methods for pulling from NOAAs CO-OPS tidal water levels data sets.

        :param stationID: CO-OPS station number from which to pull
        :tpye ID: str

        More info at:

        https://tidesandcurrents.noaa.gov/web_services_info.html

        """
        self.stationID = stationID
        self.URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
        self.frequency = {"water_level": "MS", "hourly_height": "MS"}

    @staticmethod
    def stations() -> pd.DataFrame:
        """
        Function to pull as list of CO-OPS stations with available tide data.

        Uses the tides and currents meta data api:

        https://api.tidesandcurrents.noaa.gov/mdapi/prod/


        :return: DataFrame of CO-OPS stations
        :rtype: pd.DataFrame

        :Example:

        >>> from storms.tide import datasets as tds
        >>> tds.NOAAtides.stations()

        """
        r = requests.get(
            "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=waterlevels"
        )
        df = pd.DataFrame(r.json()["stations"])
        return df

    @staticmethod
    def map():
        """A folium map to help find NOAA tide stations. Using in
        jupyter notebook will display the map inline. Using in terminal
        will only produce a folium map object. The map can be saved to html
        with `map.save('path_to_html_file.html')`

        :raises ModuleNotFoundError: Error if folium not installed. Since this function is not imperative to the
                                    the library, folium is not included in major requirements.txt file
        :return: interactive map of CO-OPS tide stations in NOAA's database
        :rtype: folium.Map

        :Example:

        >>> from storms.tide import datasets as tds
        >>> tds.NOAAtides.map()
        >>> # if you are not in juptyer, you'll need to save to html
        >>> map.save('NOAA_Stations.html')

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

        df = NOAAtides.stations()
        locations = df.loc[:, ["lat", "lng"]].to_numpy(copy=True)
        popups = df.apply(
            lambda row: folium.Popup(  # type: ignore
                html="<br>".join(
                    [
                        "<b>ID:</b> " + row["id"],
                        "<b>STATION NAME:</b>" + row["name"],
                        f'<b>lat,lon</b>: ({row["lat"]},{row["lng"]})',
                        f'<a target="_blank" href="https://tidesandcurrents.noaa.gov/stationhome.html?id={row["id"]}">NOAA home page</a>',
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

        return m

    def _request_url(  # type: ignore[override]
        self,
        start: datetime_like,
        end: datetime_like,
        datatype: str = "water_level",
        datum: str = "NAVD",
        units: str = "english",
        timeZone: str = "LST",
    ) -> str:
        st = pd.to_datetime(start).strftime("%Y%m%d")
        ed = pd.to_datetime(end).strftime("%Y%m%d")

        parameters = {
            "begin_date": st,
            "end_date": ed,
            "station": self.stationID,
            "product": datatype,
            "datum": datum.upper(),
            "units": units.lower(),
            "time_zone": timeZone.upper(),
            "format": "json",
        }
        return self.URL + urlencode(parameters, doseq=True)

    def request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        aSync=False,
        progress: bool = True,
        datatype: str = "water_level",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Request water level DataFrame with monthly requests to NOAA tides and currents API.
        For potential faster data pulls for longer time periods, use with aSync=True

        https://api.tidesandcurrents.noaa.gov/api/prod/

        :param start: Start date string that can be convert to Timestamp with pd.to_datetime
        :type start: datetime_like
        :param end: End date string that can be convert to Timestamp with pd.to_datetime
        :type end: datetime_like
        :param process_data: Switch to process NOAA tide data, implementing basic QAQC.
                            Set to False to get all the raw data pulled from NOAA.
        :type process_data: bool
        :param aSync: Switch to pull data asynchronously (can be much faster for longer data pulls)
        :type aSync: bool
        :return: DataFrame of tidal records.
        :rtype: pd.DataFrame

        :Example:

        >>> from storms.tide import datasets as tds
        >>> bos = pds.NOAATide("8443970")  #station ID for Boston Harbor
        >>> # use aSync to download large datasets
        >>> df = bos.request_dataframe("1/1/2020","1/1/2021", datatype="water_level", aSync=True)

        """

        pull_freq = self.frequency.get(datatype)
        if pull_freq is None:
            raise NotImplementedError(
                f"Sorry the NOOA product {datatype} is not supported by this tool"
            )
        return super().request_dataframe(
            start,
            end,
            process_data,
            aSync,
            progress,
            datatype=datatype,
            pull_freq=pull_freq,
            **kwargs,
        )

    def _sync_request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        pull_freq: str = "MS",
        datatype: str = "water_level",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Request tidal DataFrame with synchronous annual requests to NOAA CO-OPS API.
        For potential faster data pulls for longer time periods, use the
        asynchronous request method `_async_request_dataframe`.

        https://api.tidesandcurrents.noaa.gov/api/prod/

        :param start: Start date string that can be convert to Timestamp with pd.to_datetime
        :type start: datetime_like
        :param end: End date string that can be convert to Timestamp with pd.to_datetime
        :type end: datetime_like
        :param process_data: Switch to process NOAA tide data, implementing basic QAQC.
                            Set to False to get all the raw data pulled from NOAA.
        :type process_data: bool
        :return: DataFrame of tidal records.
        :rtype: pd.DataFrame
        """

        data = pd.DataFrame(
            flatten(
                self._sync_request_data_series(
                    start, end, pull_freq=pull_freq, datatype=datatype, **kwargs
                )
            )
        )

        if process_data:
            self._update_progress_description("Processing")
            data = self._process_data(data)

        self._close_progress()

        return data

    def _sync_request_data(
        self,
        start: datetime_like,
        end: datetime_like,
        session: requests.Session,
        datatype: str = "water_level",
        **kwargs,
    ) -> List[dict]:
        """
        Synchronously request tidal JSON data from NOAA's CO-OPS API

        :param start: Start date string that can be convert to Timestamp with pd.to_datetime
        :type start: datetime_like
        :param end: End date string that can be convert to Timestamp with pd.to_datetime
        :type end: datetime_like
        :param session: requests Session to use for API calls
        :type session: requests.Session
        :return: returns decoded JSON from API call. The CO-OPS API returns a JSON array
                 of records in the data property of the response.
        :rtype: list
        """

        url = self._request_url(start, end, datatype, **kwargs)
        with session.get(url) as response:
            return response.json()["data"]

    async def _async_request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        pull_freq: str = "AS",
        conn_limit: int = 30,
        retry_options: RetryOptionsBase = ExponentialRetry(attempts=5, start_timeout=0.1),
        datatype: str = "water_level",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Request tidal DataFrame with asynchronous annual requests to NOAA's CO-OPS API.

        https://api.tidesandcurrents.noaa.gov/api/prod/

        This is an async function and must be awaited as::

            df = await gage._async_request_dataframe(start = '1/1/2020', end = '1/1/2021')

        :param start: Start date string that can be convert to Timestamp with pd.to_datetime
        :type start: datetime_like
        :param end: End date string that can be convert to Timestamp with pd.to_datetime
        :type end: datetime_like
        :param process_data: Switch to process NOAA tide data, implementing basic QAQC.
                            Set to False to get all the raw data pulled from NOAA.
        :type process_data: bool
        :return: DataFrame of tidal records.
        :rtype: pd.DataFrame
        """

        data = await self._async_request_data_series(
            start, end, pull_freq, conn_limit, retry_options, datatype=datatype, **kwargs
        )
        data = pd.DataFrame(flatten(data))

        if process_data:
            self._update_progress_description("Processing")
            data = self._process_data(data)

        self._close_progress()

        return data

    async def _async_request_data(
        self,
        start: datetime_like,
        end: datetime_like,
        session: RetryClient,
        datatype: str = "water_level",
        **kwargs,
    ) -> List[dict]:
        """
        Asynchronously request tidal JSON data from NOAA's CO-OPS API

        :param start: Start date string that can be convert to Timestamp with pd.to_datetime
        :type start: datetime_like
        :param end: End date string that can be convert to Timestamp with pd.to_datetime
        :type end: datetime_like
        :param session: aiohttp_retry RetryClient to use for API calls
        :type session: aiohttp_retry.RetryClient
        :return: returns decoded JSON from API call. The CO-OPS API returns a JSON array
                 of records in the data property of the response.
        :rtype: list
        """

        url = self._request_url(start, end, datatype=datatype, **kwargs)

        # async json response
        # https://docs.aiohttp.org/en/stable/client_quickstart.html#json-response-content
        async with session.get(url) as response:
            data = await response.json()
            return data["data"]

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw tide data response into a nicer dataframe.
        Routine drops duplicate data pulled from API and renames columns to
        user friendly names.

        :param df: DataFrame returned from _async_request_dataframe or _sync_request_dataframe
        :type df: pd.DataFrame
        :return: pandas DataFrame of cleaned data
        :rtype: pd.DataFrame
        """
        df.drop_duplicates(inplace=True)
        df.loc[:, "t"] = pd.to_datetime(df.loc[:, "t"])
        df.columns = ["datetime", "waterlevel_ft", "sigma", "dataFlags", "quality"]
        return df

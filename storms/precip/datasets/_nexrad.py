from io import BytesIO
from typing import Tuple, Union
from math import ceil
import aiohttp
from aiohttp_retry import RetryClient,RetryOptionsBase,ExponentialRetry
import pandas as pd
import requests
from storms._datasource import _DataSource
from storms.precip.datasets._nexradmaps import n0rMAP, n0qMAP, transformRGB

from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer

from PIL import Image,UnidentifiedImageError
import numpy as np
from storms._utils import datetime_like


toProj = Transformer.from_crs(4326, 3857, always_xy=True).transform
toGeo = Transformer.from_crs(3857, 4326, always_xy=True).transform


def get_timezone_info(lat, lon):
    url = "http://api.geonames.org/timezoneJSON?formatted=true&lat={}&lng={}&username=karosc".format(
        lat, lon
    )
    r = requests.get(url)
    return r.json()


class NEXRAD(_DataSource):
    def __init__(self, lat: float, lon: float, averaging_distance: float = 500):
        """Methods for pulling reflectivity from NOAA's nexrad network.

        Parameters
        ----------
        ID: str
            name of nexrad source

        Returns
        -------

        """
        self.ID = "nexrad"
        self.lat = lat
        self.lon = lon
        self.bbox = NEXRAD._get_bbox(lat, lon, averaging_distance)
        self.resolution = ceil(averaging_distance / 1000) * 2
        self.bbox_str = ",".join([str(c) for c in self.bbox])
        tz_info = get_timezone_info(lat, lon)
        self.TZ = tz_info["timezoneId"]
        self.gmt_offset = tz_info["gmtOffset"]

    @staticmethod
    def _NORtoDBZ(rgb: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        rgb: np.ndarray:


        Returns
        -------

        """
        transformed = transformRGB(rgb)
        transformed[~np.isnan(transformed).any(axis=1)] = n0rMAP[
            transformed[~np.isnan(transformed).any(axis=1)].astype(int)
            ]
        return transformed

    @staticmethod
    def _NOQtoDBZ(rgb: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        rgb: np.ndarray:


        Returns
        -------

        """
        transformed = transformRGB(rgb)
        transformed[~np.isnan(transformed).any(axis=1)] = n0qMAP[
            transformed[~np.isnan(transformed).any(axis=1)].astype(int)
            ]
        return transformed

    @staticmethod
    def _get_bbox(lat: float, lon: float, buffer: float) -> Tuple[float, ...]:
        """

        Parameters
        ----------
        lat: float:

        lon: float:

        buffer: float:


        Returns
        -------

        """
        geo = Point(lon, lat)
        proj = transform(toProj, geo)
        bbox = proj.buffer(buffer, cap_style=1)
        bboxGeo = transform(toGeo, bbox)
        return bboxGeo.bounds

    # @lru_cache(maxsize=None)
    def request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        aSync: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Request nexrad level III reflectivity data from the Iowa mesonet nor or noq archive:
        https://mesonet.agron.iastate.edu/docs/nexrad_mosaic/

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime

        process_data: bool
            Switch to process nexrad data into average reflectivity or not.
            Set to False to get all image array data with rgba values of each radar mosaic pixcel.
            Set to True to map the rgba array to reflectivity mesurements and average the results to
            a single value per time step.

        aSync: bool
            Switch to pull data asynchronously (can be much faster for longer data pulls), defaults to False.

        progress: bool
            Switch to show tqdm bar for data download progress, defaults to True.

        **kwargs: dict
            Keyword arguments to pass onto _async_request_dataframe or _sync_request_dataframe.


        Returns
        -------
        pd.DataFrame
            DataFrame of precipitation records.

        """

        return super().request_dataframe(
            start, end, process_data, aSync, progress, **kwargs
        )

    def _request_url(self, start: datetime_like, *args) -> str:
        """Get a formatted nexrad request URL for the IOWA mesonet source.

        Parameters
        ----------
        start: datetime_like
            date string that can be converted to Timestamp with pd.to_datetime

        Returns
        -------
        str
            NEXRAD request url

        Raises
        ------
        Exception
            If start date is prior to 1995, no data before then.

        """
        dt = pd.to_datetime(start)  # .tz_localize(self.TZ)
        if dt < pd.Timestamp("1/1/1995"):
            raise Exception("No NEXRAD data prior to 1/1/1995")

        UTC = dt - pd.Timedelta(self.gmt_offset, unit="H")

        if UTC < pd.Timestamp("2011-03-01"):
            product = "n0r"
        else:
            product = "n0q"

        url = f"https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/{product}-t.cgi?&REQUEST=GetMap&TRANSPARENT=true&FORMAT=image/png&BGCOLOR=0x000000&VERSION=1.1.1&LAYERS=nexrad-{product}-wmst&STYLES=default&CRS=EPSG:4326&SRS=EPSG:4326&TIME={UTC.isoformat()}&BBOX={self.bbox_str}&WIDTH={self.resolution}&HEIGHT={self.resolution}"

        return url

    def _sync_request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        pull_freq: str = "5T",
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Request nexrad data DataFrame with synchronous requests to Iowa Mesonet.
        For potential faster data pulls for longer time periods, use the
        asynchronous request method `_async_request_dataframe`.

        Parameters
        ----------
        start: datetime_like
            Start date string that can be converted to Timestamp with pd.to_datetime.

        end: datetime_like
            End date string that can be converted to Timestamp with pd.to_datetime.

        process_data: bool
            Switch to process nexrad data into average reflectivity or not.
            Set to False to get all image array data with rgba values of each radar mosaic pixcel.
            Set to True to map the rgba array to reflectivity mesurements and average the results to
            a single value per time step.

        pull_freq: str
            Pandas offset string for the length of time for each async data request,
            defaults to every 5-min (the frequency of the mesonet datasource).

        Returns
        -------
        pd.DataFrame
            DataFrame of nexrad reflectivity records.

        """
        data = np.array(self._sync_request_data_series(start, end, pull_freq))
        index = pd.date_range(start, end, freq=pull_freq)

        if process_data:
            self._update_progress_description("Processing")
            data = pd.DataFrame(
                self._process_data(data, index), index=index, columns=["dbz"]
            )

        self._close_progress()

        return data

    def _sync_request_data(  # type: ignore[override]
        self, start: datetime_like, session: requests.Session, **kwargs
    ) -> np.ndarray:
        """Synchronously request nexrad data from Iowa Mesonet

        Parameters
        ----------
        start: datetime_like
            Datetime string that can be converted to Timestamp with pd.to_datetime.

        session: requests.Session
            requests Session to use for API calls.

        Returns
        -------
        list
            return: return array of rgba values for each pixel in the nexrad image

        """
        url = self._request_url(start)
        try:
            with session.get(url) as response:
                b = BytesIO(response.content)
                img = Image.open(b)
                arr = np.array(img.getdata())
        except UnidentifiedImageError:
            print(f'error pulling data for {start.isoformat()}. Filling with nan')
            arr = np.full((self.resolution**2,4),np.nan)
        return arr

    # async functions
    async def _async_request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        pull_freq: str = "5T",
        conn_limit: int = 30,
        retry_options: RetryOptionsBase = ExponentialRetry(attempts=5, start_timeout=0.1)
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Request nexrad data DataFrame with asynchronous requests to Iowa Mesonet.


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
            Switch to process nexrad data into average reflectivity or not.
            Set to False to get all image array data with rgba values of each radar mosaic pixcel.
            Set to True to map the rgba array to reflectivity mesurements and average the results to
            a single value per time step.

        pull_freq: str
            Pandas offset string for the length of time for each async data request,
            defaults to every 5-min (the frequency of the mesonet datasource).

        conn_limit: int
            Connection limit for aiohttp session, defaults to 30

        retry_options: RetryOptionsBase
            Retry options to pass to aiohttp_retry client

        Returns
        -------
        pd.DataFrame
            DataFrame of nexrad reflectivity records.

        """

        reqEnd = pd.to_datetime(end) + pd.Timedelta(pull_freq)
        data = np.array(
            await self._async_request_data_series(start, reqEnd, pull_freq, conn_limit, retry_options)
        )
        index = pd.date_range(start, end, freq=pull_freq)
        if process_data:
            self._update_progress_description("Processing")
            data = pd.DataFrame(
                self._process_data(data, index), index=index, columns=["dbz"]
            )

        self._close_progress()

        return data

    async def _async_request_data(  # type: ignore[override]
        self,
        start: datetime_like,
        session: RetryClient,
        **kwargs,
    ) -> np.ndarray:
        """
        Asynchronously request nexrad data from Iowa Mesonet


        Parameters
        ----------
        start: datetime_like
            Datetime string that can be converted to Timestamp with pd.to_datetime.

        session: aiohttp_retry.RetryClient
            aiohttp_retry RetryClient to use for API calls.

        Returns
        -------
        np.ndarray
            Array of rgba values for each pixel in the nexrad image
        """
        # start, dummy, session = args
        url = self._request_url(start)
        # async json response
        # https://docs.aiohttp.org/en/stable/client_quickstart.html#json-response-content
        error_counter = 0
        for i in range(5):
            try:
                async with session.get(url) as response:
                    b = BytesIO(await response.read())
                    img = Image.open(b)
                    arr = np.array(img.getdata())

            except aiohttp.ClientConnectionError:
                # print(
                #     "Oops, the connection was dropped before we finished, retrying..."
                # )
                error_counter += 1
                continue
            except UnidentifiedImageError:
                print(f'error pulling data for {start.isoformat()}. Filling with nan')
                arr = np.full((self.resolution**2,4),np.nan)
            except Exception as e:
                print(e)
                print(f"error pulling {url}, retrying...")
                error_counter += 1
                continue
            
            return arr


        raise Exception(f"error pulling {url} after 5 retries")

    def _process_data(self, data: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Process raw nexrad rgb values into reflectivity values using NOAA color maps
        and average values for the entire image.

        Parameters
        ----------
        data: np.ndarray
            Array of rgb values

        index: np.ndarray
            Array of datetimes for each rgb array


        Returns
        -------
        pd.DataFrame
            pandas DataFrame of filtered hourly data

        """
        nor = np.where(index < pd.Timestamp("2011-03-01"))[0]
        noq = np.where(index >= pd.Timestamp("2011-03-01"))[0]

        norD = NEXRAD._NORtoDBZ(data[nor])
        noqD = NEXRAD._NOQtoDBZ(data[noq])

        dbz = np.concatenate([norD, noqD], axis=0)

        # zero out negative dbz from n0q
        dbz[np.where(dbz < 0)] = 0
        dbz = np.mean(dbz, axis=1)

        return dbz

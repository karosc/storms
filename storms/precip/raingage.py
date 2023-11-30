from os import cpu_count
from typing import Optional, Sequence, Union, Tuple, Dict
from warnings import warn
from types import MethodType
from io import StringIO
import pathlib

import numpy as np
import pandas as pd
import requests

from numba import set_num_threads
from numpy.typing import ArrayLike
from pandas.tseries.frequencies import to_offset

from storms.precip._rainrank import (
    eventify,
    ns_intervals_parallel,
    Event,
)
from storms.precip.eva import EVA
from storms.precip.datasets import GlobalHourly, ASOS, NEXRAD
from storms.precip._disagg import continuous_deterministic, continuous_stochastic
from storms._utils import datetime_like
from storms.precip._dashboard import rg_2_dash
from tqdm.auto import tqdm

__HERE__ = pathlib.Path(__file__).parent
_noaa_ca_cert_path = str((__HERE__ / "hdsc-nws-noaa-gov-chain.pem").absolute())


def dbz_to_depth(
    dbz: np.ndarray, a: float = 200, b: float = 1.6, interval: float = 300, **kwargs
) -> np.ndarray:
    """
    Convert radar reflectivity to depth in inches using the Marshall-Palmer Equation

    https://glossary.ametsoc.org/wiki/Marshall-palmer_relation

    Parameters
    ----------
    dbz : np.ndarray
        array of dbz values
    a : float, optional
        Parameter a of the Z/R relationship
        Standard value according to Marshall-Palmer is a=200., by default 200.
    b : float, optional
        Parameter b of the Z/R relationship
        Standard value according to Marshall-Palmer is b=1.6, by default 1.6
    interval : float, optional
        time interval (s) the values of `dbz` represent, by default 300

    Returns
    -------
    np.ndarray
        array of precip depths in inches

    """
    z = 10.0 ** (dbz / 10.0)

    R = (z / a) ** (1.0 / b)

    return (R * interval / 3600.0) / 25.4


class Raingage(object):
    def __init__(
        self,
        data: pd.Series,
        freq: Optional[str] = None,
        latlon: Optional[Tuple[float, float]] = None,
        meta: Optional[dict] = {},
        ID: Optional[str] = None,
    ):
        """Raingage class for developing IDF stats and storm time histories

        Parameters
        ----------
        data: pd.Series
            Pandas series with timestamp/datetime index and precip values
        freq: str, optional
            Pandas frequency offset indicating the frequency of the timeseries data, defaults to None
        latlon: tuple, optional
            tuple of latitude and longitude. Used for pulling NOAA atlas 14 ARIs, defaults to None

        Returns
        -------

        Raises
        ------
        ValueError
            Error if lat lon is not a 2-elemnt tuple

        """

        if not np.issubdtype(data.index.dtype, np.datetime64):
            raise ValueError("Input pandas series must have a DatetimeIndex")
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Input pandas series must have a numeric dtype")

        self._events: Union[pd.DataFrame, Sequence[Event]]
        self._event_col: np.ndarray

        self.ID = ID
        self.meta = meta
        self.freq = freq if freq else pd.infer_freq(data.index)
        self.data = data.loc[data > 0]
        self.julian_datetime: np.ndarray = (
            self.data.loc[self.data > 0].index.to_julian_date().to_numpy(copy=True)
        )
        self.precip: np.ndarray = self.data.loc[self.data > 0].to_numpy(copy=True)
        self.datetime: np.ndarray = self.data.loc[self.data > 0].index.to_numpy(
            copy=True
        )
        self.latlon = latlon

        if self.latlon:
            if len(self.latlon) == 2:
                self.pfds = get_pfds(*self.latlon)
            else:
                raise ValueError(
                    "latlon should be a 2 element tuple or list with latitude followed by longitude"
                )
        else:
            self.pfds = None

        self._evds: Dict[str, EVA] = {}

        self.intervals: pd.DataFrame
        """DataFrame of sub-events found with the find_intervals method."""

        self._dbz_data: Dict[
            Tuple[pd.TimeStamp, pd.TimeStamp, float, Union[str, pd.DateOffset]],
            pd.Series,
        ] = {}

    @classmethod
    def from_asos(cls, ID: str, start: datetime_like, end: datetime_like, aSync=False):
        """Create raingage object with data from NOAA's ASOS dataset.

        Data are collected with asynchronous annual requests to the `Iowa Mesonet API`_.

        .. _Iowa Mesonet API: https://mesonet.agron.iastate.edu/request/asos/1min.phtml

        Parameters
        ----------
        ID: str
            FAA station code to pull.
        start: datetime_like
            Start date string that can be convert to Timestamp with pd.to_datetime.
        end: datetime_like
            End date string that can be convert to Timestamp with pd.to_datetime.
        aSync: bool
            Switch to pull data asynchronously (can be much faster for longer data pulls), defaults to False.

        Returns
        -------
        Raingage
            Raingage object

        """

        gage = ASOS(ID)
        df = gage.request_dataframe(start, end, aSync=aSync)
        if "dtLocal" in df.columns:
            df.set_index("dtLocal", inplace=True)
        else:
            df.set_index("dtUTC", inplace=True)
        return cls(df["precip"], freq="1T", latlon=(gage.latlon), ID=ID)

    @classmethod
    def from_GlobalHourly(
        cls, ID: str, start: datetime_like, end: datetime_like, aSync=False
    ):
        """Create raingage object with data from NOAA's ISD global hourly dataset.

        Data are collected with asynchronous annual requests to `NOAA V1 API`_.

        .. _NOAA V1 API: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

        Parameters
        ----------
        ID: str
            ISD station ID to pull
        start: datetime_like
            Start date string that can be convert to Timestamp with pd.to_datetime.
        end: datetime_like
            End date string that can be convert to Timestamp with pd.to_datetime.
        aSync: bool
            Switch to pull data asynchronously (can be much faster for longer data pulls), defaults to False.

        Returns
        -------
        raingage
            raingage object

        """
        gage = GlobalHourly(ID)
        df = gage.request_dataframe(start, end, aSync=aSync)
        data = df.set_index("Hourly")["Inches"]
        return cls(
            data.loc[data > 0],
            freq="1H",
            latlon=(gage.meta["LAT"], gage.meta["LON"]),
            meta=gage.meta,
            ID=ID,
        )

    @classmethod
    def from_swmm(cls, path: str, gage: Optional[str] = None, **kwargs):
        """Create raingage object from a swmm rainfall file. Expects freeform
        data file with gage, year, month, day, hour, minute, and precip columns.
        No headers allowed in input file, but line comments starting with ";" are allowed.

        Parameters
        ----------
        path: str
            Path to swmm dat file.

        gage: Optional[str]:
             Gage ID in swmm file to pull. If None, uses the first one found. Defaults to None.

        **kwargs: dict
            Keyword arguments to based to raingage constructor


        Returns
        -------
        raingage
            raingage object

        """
        data = pd.read_csv(
            path,
            delim_whitespace=True,
            names=["gage", "year", "month", "day", "hour", "minute", "precip"],
            comment=";",
            header=None,
            index_col=None,
        )
        if len(data.gage.unique()) > 1 and gage is None:
            raise Exception(
                "Multiple gages found in file and no gage specified",
                "To read this swmm file, please specify a gage",
            )
        elif len(data.gage.unique()) > 1 and gage is not None:
            data = data.loc[data.gage == gage]
            if len(data) == 0:
                raise Exception(f"No data found in file for requested gage {gage}")

        data.index = pd.to_datetime(
            data.loc[:, ["year", "month", "day", "hour", "minute"]]
        )
        return cls(data.loc[:, "precip"], **kwargs)

    @classmethod
    def from_csv(cls, path: str, **kwargs):
        """Create raingage object from csv file. First column is datetime used for index,
        second column is precip data. No headers allowed in input file.

        Parameters
        ----------
        path: str
            Path to csv file
        **kwargs: dict
            Keyword arguments to based to raingage constructor


        Returns
        -------
        raingage
            raingage object

        """
        data = pd.read_csv(
            path,
            sep=",",
            header=None,
            parse_dates=True,
            names=["datetime", "precip"],
            index_col=0,
            engine="c",
        )
        return cls(data.loc[:, "precip"], **kwargs)

    @classmethod
    def from_ff(cls, path: str, **kwargs):
        """Read free form timeseries data typical of SWMM timeseries dat files.
        Delimites input data using regex, matching either `/` or `:` or space as delimiter.
        Should parse month, day, year, hour, minute, precip, and gage into separate columns.
        Data must be in that order.

        Parameters
        ----------
        path: str
            path to input file
        **kwargs: dict
            Keyword arguments to based to raingage constructor


        Returns
        -------
        raingage
            raingage object

        """
        data = pd.read_csv(
            path,
            engine="python",
            comment=";",
            sep=r"\/|:|\s+",
            header=None,
            usecols=[0, 1, 2, 3, 4, 5, 6],
            names=["month", "day", "year", "hour", "minute", "precip", "gage"],
        )

        data.set_index(
            pd.to_datetime(data[["year", "month", "day", "hour", "minute"]]),
            inplace=True,
        )

        return cls(data.loc[:, "precip"], **kwargs)

    def __add__(self, other):
        if isinstance(other, (float, int)):
            data = self.data + other
            return Raingage(data, self.freq, self.latlon)

        elif isinstance(other, Raingage):
            data = sum(self._align_rain_gages(other))
            return Raingage(data, self.freq)

        else:
            raise ValueError(f"Method not compatible with type {type(other)}")

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            data = self.data - other
            return Raingage(data, self.freq, self.latlon)

        elif isinstance(other, Raingage):
            data = self._align_rain_gages(other)
            return Raingage(data[0] - data[1], self.freq)

        else:
            raise ValueError(f"Method not compatible with type {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            data = self.data * other
            return Raingage(data, self.freq, self.latlon)

        elif isinstance(other, Raingage):
            data = self._align_rain_gages(other)
            return Raingage(data[0] * data[1], self.freq)

        else:
            raise ValueError(f"Method not compatible with type {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            data = self.data / other
            return Raingage(data, self.freq, self.latlon)

        elif isinstance(other, Raingage):
            data = self._align_rain_gages(other)
            return Raingage(data[0] / data[1], self.freq)

        else:
            raise ValueError(f"Method not compatible with type {type(other)}")

    def __iadd__(self, other):
        if isinstance(other, (float, int)):
            self.data = self.data + other
            return self

        elif isinstance(other, Raingage):
            self.data = sum(self._align_rain_gages(other))
            return self

        else:
            raise ValueError(f"Method not compatible with type {type(other)}")

    def __isub__(self, other):
        if isinstance(other, (float, int)):
            self.data = self.data - other
            return self

        elif isinstance(other, Raingage):
            data = self._align_rain_gages(other)
            self.self = data[0] - data[1]
            return self

        else:
            raise ValueError(f"Method not compatible with type {type(other)}")

    def __imul__(self, other):
        if isinstance(other, (float, int)):
            self.data = self.data * other
            return self

        elif isinstance(other, Raingage):
            data = self._align_rain_gages(other)
            self.data = data[0] * data[1]
            return self

        else:
            raise ValueError(f"Method not compatible with type {type(other)}")

    def __itruediv__(self, other):
        if isinstance(other, (float, int)):
            self.data = self.data / other
            return self

        elif isinstance(other, Raingage):
            data = self._align_rain_gages(other)
            self.data = data[0] / data[1]
            return self

        else:
            raise ValueError(f"Method not compatible with type {type(other)}")

    def _align_rain_gages(self, other) -> tuple:
        """Helper function to align this object's timeries with those of another

        Parameters
        ----------
        other: storms.Raingage
            Raingage object with which to align data.

        Returns
        -------
        tuple
            Tuple of aligned data series in order of (self,other).

        Raises
        ------
        ValueError
            If either rain gage is missing a frequency
        ValueError
            If other rain gage has a different frequency

        """
        if self.freq is None or other.freq is None:
            raise ValueError(
                "The frequency of each Raingage object must be known to align them"
            )
        if self.freq != other.freq:
            raise ValueError("Raingages frequencies must be equal to align them")

        return self.data.align(other.data, join="outer", fill_value=0)

    @property
    def freq(self) -> pd.DateOffset:
        """Get pandas date offset of this object's data frequency.

        Returns
        -------
        pd.DateOffset
            Returns frequency of precip data as pandas offset.

        """
        return self._freq

    @freq.setter
    def freq(self, offset: str) -> pd.DateOffset:
        """Sets _freq pandas offset object from input offset string.
        Ultimately this function is used for inferring/storing the rainfall
        timeseries frequency.

        Parameters
        ----------
        offset: str
            Pandas offset string.

        Returns
        -------
        pd.DateOffset
            Pandas offset object.

        """
        self._freq = to_offset(offset)

    @property
    def ts_hours(self) -> float:
        """Return the timeseries timestep in hours.

        Returns
        -------
        float
            Time series time step (hours).

        """
        return self.freq.nanos / 1e9 / 60 / 60

    @property
    def num_years(self) -> int:
        """Calculate the nubmer of years in the rainfall timeseries.

        Returns
        -------
        int
            Number of years in dataset based on the first and last timestamps.

        """
        return round(
            (self.data.index.max() - self.data.index.min()) / np.timedelta64(365, "D")
        )

    def aggregate(self, freq: str, inplace: bool = True):
        """
        Aggregate a rainfall timeseries to a larger timestep
        using the pandas resample method

        Parameters
        ----------
        freq : str
            pandas offset alias or offset object describing the new
            time step frequency
        inplace: bool
            Switch to return new Raingage object or alter current, by default True.

        Raises
        ------
        Exception
            If target freq is smaller than existing freq.

        Returns
        -------
        Union[None, Raingage]
            None if aggregating inplace, otherwise returns a new Raingage
            at the requested frequency
        """
        out_freq = to_offset(freq)
        if out_freq < self.freq:
            raise Exception("Target freq is smaller than existing freq")

        series = self.data.resample(out_freq).sum()
        series = series.loc[series > 0]
        if inplace:
            self.__init__(series, freq, self.latlon, self.meta, self.ID)  # type: ignore
            return None
        else:
            return Raingage(series, freq, self.latlon, self.meta, self.ID)

    def disaggregate(
        self, freq: str, method: str, inplace: bool = True, **kwargs
    ) -> None:
        """Synthetically disaggregate rainfall data to another frequency.

        Supported methods:

        * uniform distribution
        * geometric similarity (`Ormsbee, 1989`_)

        .. _Ormsbee, 1989: https://doi.org/10.1061/(ASCE)0733-9429(1989)115:4(507)

        Parameters
        ----------
        freq: str
            Pandas offset string giving new output frequency.
        method: str
            One of ("uniform", "geometric_deterministic", "geometric_stochastic").
        inplace: bool, optional
            Switch to return new Raingage object or alter current, by default True.

        Raises
        ------
        ValueError
            If method argument not in accepted list.
        Exception
            If freq is not a factor of existing freq.
            (e.g. can't disagg 15-minute data to 10-minute data)

            If freq is larger than existing freq.

        Returns
        -------
        Raingage object with new freq or None if inplace=True.
        """

        methods = {
            "uniform": self._uniform_disaggregate,
            "geometric_deterministic": self._continuous_deter,
            "geometric_stochastic": self._continuous_stoch,
        }
        if method.lower() not in methods:
            raise ValueError(f"method must be one of {methods}")

        out_freq = to_offset(freq)
        num_bins = self.freq / out_freq

        if not num_bins.is_integer():
            raise Exception("Target freq is not a factor of existing freq")
        if out_freq > self.freq:
            raise Exception("Target freq is larger than existing freq")

        return methods[method](out_freq, num_bins, inplace, **kwargs)  # type: ignore

    def _uniform_disaggregate(
        self, out_freq: pd.DateOffset, num_bins: int, inplace: bool, **kwargs
    ):
        """Uniform disaggregation to evenly distribute rainfall between time steps at new interval given
        by out_freq.

        Parameters
        ----------
        out_freq: pd.DateOffset
            Pandas offset string giving new output frequency.
        num_bins: int
            Number of bins to split data into every time step.
        inplace: bool
            Switch to return new Raingage object or alter current, by default True.

        Returns
        -------
        Raingage object with new freq or None if inplace=True.
        """

        index = pd.DatetimeIndex(
            np.concatenate(
                [
                    pd.date_range(
                        st, st + self.freq, freq=out_freq, inclusive="left"
                    ).to_numpy()
                    for st in self.data.index
                ]
            )
        )
        data = self.data.to_numpy().repeat(num_bins) / num_bins
        series = pd.Series(data, index, name=self.data.name)
        if inplace:
            self.__init__(series, out_freq, self.latlon, self.meta, self.ID)  # type: ignore
            return None
        else:
            return Raingage(series, out_freq, self.latlon, self.meta, self.ID)

    def _continuous_deter(
        self, out_freq: pd.DateOffset, num_bins: int, inplace: bool, **kwargs
    ):
        """Continous deterministic disaggregation using geometric similarity to disaggregate
        data to new interval given by out_freq.

        Parameters
        ----------
        out_freq: pd.DateOffset
            Pandas offset string giving new output frequency.
        num_bins: int
            Number of bins to split data into every time step.
        inplace: bool
            Switch to return new Raingage object or alter current, by default True.

        Returns
        -------
        Raingage object with new freq or None if inplace=True.
        """

        in_ts = np.timedelta64(self.freq.nanos, "ns")
        out_ts = np.timedelta64(out_freq.nanos, "ns")
        index, data = continuous_deterministic(
            self.datetime, self.precip, int(num_bins), in_ts, out_ts
        )
        series = pd.Series(data, index, name=self.data.name)
        if inplace:
            self.__init__(series, out_freq, self.latlon, self.meta, self.ID)  # type: ignore
            return None
        else:
            return Raingage(series, out_freq, self.latlon, self.meta, self.ID)

    def _continuous_stoch(
        self,
        out_freq: pd.DateOffset,
        num_bins: int,
        inplace: bool,
        precision: int = 2,
        scale_factor: float = 0,
        **kwargs,
    ):
        """Continous stochastic disaggregation using geometric similarity to disaggregate
        data to new interval given by out_freq.

        Parameters
        ----------
        out_freq: pd.DateOffset
            Pandas offset string giving new output frequency.
        num_bins: int
            Number of bins to split data into every time step.
        inplace: bool
            Switch to return new Raingage object or alter current, by default True.
        precision: int
            The precision of output rainfall timeseries, by default 2 for hundreths of an inch.
        scale_factor: float
            Spiking factor used to randomly add higher peaks to output timeseries, by default 0.

        Returns
        -------
        Raingage object with new freq or None if inplace=True.
        """

        in_ts = np.timedelta64(self.freq.nanos, "ns")
        out_ts = np.timedelta64(out_freq.nanos, "ns")
        index, data = continuous_stochastic(
            self.datetime,
            self.precip,
            int(num_bins),
            in_ts,
            out_ts,
            precision,
            scale_factor,
        )
        series = pd.Series(data, index, name=self.data.name)
        if inplace:
            self.__init__(series[series > 0], out_freq, self.latlon, self.meta, self.ID)  # type: ignore
            return None
        else:
            return Raingage(
                series[series > 0], out_freq, self.latlon, self.meta, self.ID
            )

    def get_noaa_ari(self, dep: float, duration: float) -> float:
        """Pull NOAA ARI from NOAA PFDS assuming lat lon were provided during instantiation.

        Parameters
        ----------
        dep: float
            Rainfall depth in inches.
        duration: float
            Duration in hours.

        Returns
        -------
        float
            ARI in years.

        Raises
        ------
        ValueError
            If duration not in PFDS table.
        """

        if duration not in self.pfds.index:
            raise ValueError(f"{duration}-hours not in PFDS table")
        else:
            return _interp(
                dep, self.pfds.loc[duration].to_numpy(), self.pfds.columns.to_numpy()
            )

    def get_noaa_depth(self, ari: float, duration: float) -> float:
        """Pull NOAA Depth from NOAA PFDS assuming lat lon were provided during instantiation.

        Parameters
        ----------
        dep: float
            Rainfall depth in inches.
        duration: float
            Duration in hours.

        Returns
        -------
        float
            ARI in years.

        Raises
        ------
        ValueError
            If duration not in PFDS table.
        """

        if duration not in self.pfds.index:
            raise ValueError(f"{duration}-hours not in PFDS table")
        else:
            return _interp(
                ari, self.pfds.columns.to_numpy(), self.pfds.loc[duration].to_numpy()
            )

    @property
    def events(self) -> pd.DataFrame:
        """DataFrame of largest events. Depends on find_events already being run.

        Returns
        -------
        pd.DataFrame
            DataFrame of largest events.

        Raises
        ------
        Exception
            If find_events was not run, and thus data not available, raise exception.

        """
        if hasattr(self, "_events"):
            return self._events
        else:
            raise Exception("Events must first be found using find_events")

    @events.setter
    def events(self, data: list) -> None:
        self._events = pd.DataFrame(data)

    @property
    def intervals(self) -> pd.DataFrame:
        """
        DataFrame of sub-events at various time intervals. Depends on find_intervals already being run.

        Returns
        -------
        pd.DataFrame
            DataFrame of sub-events at various time intervals.

        Raises
        ------
        Exception
            If find_intervals was not run, and thus data not available, raise exception.
        """
        if hasattr(self, "_intervals"):
            return self._intervals
        else:
            raise Exception("Intervals must first be found using find_intervals")

    @intervals.setter
    def intervals(
        self,
        cols: Tuple[np.ndarray, ...],
        colNames: Tuple[str, ...] = (
            "event_num",
            "duration",
            "total",
            "event_start_index",
            "event_end_index",
            "start_date",
            "killing_event",
        ),
    ) -> None:
        if len(cols) != len(colNames):
            raise Exception(
                f"Inequal number of interval columns ({len(cols)}) and column names ({len(colNames)})"
            )

        df = pd.DataFrame.from_dict(dict(zip(colNames, cols)))
        self._intervals = df.loc[
            (df.killing_event == 0) & df.total > 0,
            [
                "event_num",
                "duration",
                "total",
                "event_start_index",
                "event_end_index",
                "start_date",
            ],
        ]

    def get_event(self, event_num: int) -> pd.Series:
        """Pull the hyetograph for a particular event based on the event number given.

        Parameters
        ----------
        event_num: int
            The event number to query.

        Returns
        -------
        pd.Series
            Hyetograph series for queried event.

        Raises
        ------
        Exception
            Data must first be binned into events with find_events
            before this function will work.

        """
        if self._event_col is None:
            raise Exception("Events must first be found using find_events")

        start, end = self.events.loc[
            self.events.event_num == event_num, ["event_start_index", "event_end_index"]
        ].iloc[0]

        return self.data.iloc[start:end]

    def get_event_by_rank(self, rank: int) -> pd.Series:
        """Pull the hyetograph for a particular event based on its rank in total rainfall.

        Parameters
        ----------
        rank: int
            The events rank in the period of record.

        Returns
        -------
        pd.Series
            Hyetograph series for queried event.

        Raises
        ------
        Exception
            Data must first be binned into events with find_events
            before this function will work.

        """
        if self.events is None:
            raise Exception("Events must first be found using find_events")

        start, end = self.events.loc[
            self.events.event_total.rank(ascending=False, method="first") == rank,
            ["event_start_index", "event_end_index"],
        ].iloc[0]

        return self.data.iloc[start:end]

    def find_events(
        self, inter_event_period: float = 6, threshold_depth: float = 0.25
    ) -> pd.DataFrame:
        """Run event finder. Rainfall data will be binned into descrete events based on
        the inter_event_period parameter in hours. If an event has less rainfall than threshold_depth. it will not
        be distinguished as a separate event, it will be bundled with the previous event.

        Parameters
        ----------
        inter_event_period: float, optional
            Interevent period in hours. The number of hours uses to
            separate distinct events, defaults to 6.

        threshold_depth: float, optional
            The minimum rainfall depth to qualify an "event", defaults to 0.25.

        Returns
        -------
        pd.DataFrame
            Creates and returns the events table stored in self.events.

        Raises
        ------
        Exception
            If data frequency (self.freq) is not known, data cannot be processed.

        """
        if self.freq is None:
            raise Exception(
                "data frequency could not be inferred or was not set,"
                "please set self.freq to a pandas offset alias "
                "(https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)"
            )

        self.events, self._event_col = eventify(
            self.precip,
            self.datetime,
            inter_event_period,
            threshold_depth,
            self.ts_hours,
        )

        return self.events

    def find_intervals(
        self,
        periods: ArrayLike = [1, 2, 3, 6, 12, 24, 48],
        threshold_depth: float = 0.25,
        eva_method: Optional[str] = "hybrid",
        constraints: Union[Sequence[float], bool] = True,
        cpus: int = cpu_count() - 1,  # type: ignore
        **kwargs,
    ) -> None:
        """Run interval sub-event finder. This function creates partial duration series'
        for various hour durations in `periods`. Sub-events with total rainfall less than threshold_depth
        are not counted.

        Parameters
        ----------
        periods: Sequence[float], optional
            Iterable of sub-event durations in hours to look for,
            defaults to [1, 2, 3, 6, 12, 24, 48].

        threshold_depth: float, optional
            Minimum amount of rainfall in inches to count a sub-event, defaults to 0.25.

        eva_method: Optional[str]
            EVA method to use, one of ("ams", "gev", "pds", "gpd", "hybrid", None), by default "hybrid".
            Setting to None skips EVA.

        constraints: Union[bool, Sequence[float]], optional
            Iterable of constraints to appy to each duration in period.
            These are really only for data greater than or equal to 1-hour
            frequency.

            If True, then use default constraints from Atlas 14 to
            calculate for given intervals.

            If False, don't constrain.

            Default constraits are derived from the `NOAA Atlas 14 Volume 10`_, where they
            are shown in tables 4.5.3 and 4.5.4

            defaults to True

            .. _NOAA Atlas 14 Volume 10: https://www.weather.gov/media/owp/oh/hdsc/docs/Atlas14_Volume10.pdf

        cpus: int, optional
            Number of threads to use when running JIT compiled function in parallel,
            defaults to cpu_count()-1.

        **kwargs: dict
            Keyword arguments to pass down to find_ari

        Returns
        -------
        None
            Returns nothing, but sets _intervals property to structured array with
            all sub-event data.

        Raises
        ------
        ValueError
            Raise error if periods cannot be cast to float64 array.
        ValueError
            Raise error if periods and constraints are not the same length.

        """
        if type(periods) != np.ndarray:
            try:
                periods = np.asarray(periods, dtype="<f8")
            except:
                raise ValueError("Could not convert periods to float64 array")
        # set num threads used in parallel processing
        set_num_threads(cpus)

        # calculate all interval totals
        self.intervals = ns_intervals_parallel(
            self.precip, self.datetime, self._event_col, periods, threshold_depth
        )

        self.find_ari(eva_method, periods, constraints, **kwargs)

    def find_ari(
        self,
        method: Optional[str] = "hybrid",
        durations: Union[Sequence[float], np.ndarray, None] = None,
        constraints: Union[Sequence[float], np.ndarray, bool] = True,
        nmom: int = 3,
        **kwargs,
    ) -> None:
        """Function that operates on self.intervals to calculate the ARI
        and lookup the Atlas 14 ARI of each event.

        Parameters
        ----------
        method: Optional[str]
            EVA method to use, one of ("ams", "gev", "pds", "gpd", "hybrid",None), by default "hybrid".
            Setting to None skips EVA.

        durations: Union[Sequence[float], np.ndarray, None], optional
            The periods in hours for which to calculate ARIs.

            If None, use all periods found in the duration column of self.intervals, defaults to None.

        constraints: Union[bool, Sequence[float]], optional
            Iterable of constraints to appy to each duration in period.
            These are really only for data greater than or equal to 1-hour
            frequency.

            If True, then use default constraints from Atlas 14 to
            calculate for given intervals.

            If False, dont constrain.

            Default constraits are derived from the `NOAA Atlas 14 Volume 10`_, where they
            are shown in tables 4.5.3 and 4.5.4

            .. _NOAA Atlas 14 Volume 10: https://www.weather.gov/media/owp/oh/hdsc/docs/Atlas14_Volume10.pdf

            Defaults to True

        nmom: int, optional
            Number of L-moments to use when fitting extreme value distribution, defaults to 3.

        Returns
        -------
        None
            Nothing, updates self.intervals.

        Raises
        ------
        Exception
            Must first run find_intervals.
        Exception
            Must provide list of periods if providing a list of constraints.
        ValueError
            If period does not pass Typecheck.
        ValueError
            If provided period and constraint lists are not of equal length.
        ValueError
            If constraint does not pass Typecheck.

        """

        ################## Typecheck periods #################
        if type(self.intervals) != pd.DataFrame:
            raise Exception("Intervals must first be found with find_intervals()")

        # if constraints without periods
        elif durations is None and not isinstance(constraints, (type(None), bool)):
            raise Exception(
                "Cannot provide constraints without periods."
                "The two must be in corresponding order."
            )
        # if both None, pull from existing intervals data
        elif durations is None:
            periods = self.intervals.duration.unique()

        elif isinstance(durations, (list, tuple, np.ndarray)):
            periods = np.asarray(durations)

        else:
            raise ValueError(f"Invalid periods argument {durations}")

        ############### Typecheck constraints ################
        if isinstance(constraints, (list, tuple, np.ndarray)):
            if len(periods) != len(constraints):
                raise ValueError(
                    f"There must be one constraint value for each period. "
                    f"{len(constraints)} constraints were input for {len(periods)} periods."
                )
            else:
                pass
        elif constraints is True:
            constraints = [self._get_constraint(period) for period in periods]

        elif constraints is False:
            constraints = np.ones(len(periods))
        else:
            raise ValueError(f"Invalid constraints argument {constraints}")

        ###########  check EVA suitability ###########
        valid_methods = ("ams", "gev", "pds", "gpd", "hybrid")
        if method is not None:
            self._evds = {}
            if method not in valid_methods:
                raise ValueError(f"method must be one of {valid_methods}")
            if nmom <= 0 or nmom > 5 or self.num_years < nmom:
                useEVA = False
                warn(
                    f"Not enough data for EVA. Must have at least {nmom} years of data, but more is recommended"
                )
            else:
                useEVA = True

            if useEVA:
                # loop over periods, fit EVA, and calculate ARIs
                for iperiod, period in enumerate(periods):
                    # # fill in rank column
                    # self.intervals.loc[
                    #     self.intervals.duration == period, "rank"
                    # ] = self.intervals.loc[self.intervals.duration == period, "total"].rank(
                    #     method="first", ascending=False
                    # )

                    # # fill in plotting position columns
                    # self.intervals.loc[
                    #     self.intervals.duration == period, "PLOTPOS"
                    # ] = self.intervals.loc[self.intervals.duration == period, "rank"].apply(
                    #     plotting_position, args=(self.num_years, 0.4)
                    # )

                    # fit EVA
                    inp = (
                        self.intervals.loc[self.intervals.duration == period]
                        .set_index("start_date")["total"]
                        .copy()
                    )
                    # fill in ARI column with EVA results
                    try:
                        g = EVA(
                            inp,
                            method,
                            constraints[iperiod],
                            nmom,
                            **kwargs,
                        )
                        self._evds[period] = g
                        # fill in ARI columns
                        self.intervals.loc[
                            self.intervals.duration == period, "ARI"
                        ] = self.intervals.loc[
                            self.intervals.duration == period, "total"
                        ].apply(
                            g.ARI
                        )

                    # if EVA error, fill in with N/A
                    except Exception as e:
                        print("Failed fitting EVA\n", str(e))
                        self.intervals.loc[
                            self.intervals.duration == period, "ARI"
                        ] = pd.NA

        # if lat/lon were provided and PFDS is available, calculate NOAA ARI for each interval event
        if self.pfds is not None:
            # add periods to PFDS index if they don't exist
            idx = set(periods.tolist() + self.pfds.index.to_list())
            self.pfds = self.pfds.reindex(idx).sort_index()

            # interpolate ARI depth between various durations linearly
            self.pfds.interpolate(how="index", in_place=True)

            # pull out ARI column headers to use for interpolation
            ARIs = self.pfds.columns.to_numpy()

            # loop over periods
            for period in periods:
                # apply numpy interp to depth and ARIs at each duration
                # interpolate depths between various ARIs logarithmically
                self.intervals.loc[
                    self.intervals.duration == period, "noaaARI"
                ] = self.intervals.loc[
                    self.intervals.duration == period, "total"
                ].apply(
                    _interp, args=(self.pfds.loc[period].to_numpy(), ARIs)
                )

    def _storm_dbz(
        self,
        start: datetime_like = None,
        end: datetime_like = None,
        event_num: Optional[int] = None,
        averaging_distance: float = 1000.0,
        freq: Union[str, pd.DateOffset] = None,
        progress: bool = False,
        nex: Optional[NEXRAD] = None,
    ) -> pd.Series:
        """Pull NEXRAD reflectivity for a given period or storm at this raingage's location.

        Can either specifiy start and end dates or give an event_num from the events table

        Parameters
        ----------
        start: datetime_like, optional
            start datetime, by default None

        end: datetime_like, optional
            end datetime, by default None

        event_num: Optional[int], optional
            event_num from events table, by default None

        averaging_distance: float, optional
            The distance from the Raingage coordinates in meters over which
            to average reflectivity, by default 1000.

        freq: Union[str,pd.DateOffset,None], optional
            The frequency of the reflectivity data to pull. Function will pull
            a reflectivity value at each timestep in the series with this frequency.
            Can be as low as 5 minutes

        progress: bool, optional
            Switch to show a tqdm bar for download progress, by default False

        nex: Optional[NEXRAD], optional
            Nexrad object to use for data pull. If None then create one using Raingage
            coordinates, by default None

        Returns
        -------
        pd.Series
            Series of reflectivity data
        """

        if event_num is not None:
            evt = self.events.set_index("event_num").loc[event_num]
            start = evt.start_date
            # end = self.datetime[evt.event_end_index - 1]
            end = (
                evt.start_date
                + pd.to_timedelta(evt.hours_duration - self.ts_hours, "H")
            ).round("T")
        else:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)

        pull_freq = freq if freq is not None else self.freq
        key = (start, end, averaging_distance, pull_freq)

        if self._dbz_data.get(key) is None:
            if nex is None:
                nex = NEXRAD(*self.latlon, averaging_distance)  # type: ignore

            data = nex.request_dataframe(
                start, end, aSync=True, pull_freq=pull_freq, progress=progress
            )
            self._dbz_data[key] = data

        return self._dbz_data.get(key)

    def _storm_dbz_inches(
        self,
        start: datetime_like = None,
        end: datetime_like = None,
        event_num: Optional[int] = None,
        averaging_distance: float = 1000.0,
        freq: Union[str, pd.DateOffset] = None,
        progress: bool = False,
        nex: Optional[NEXRAD] = None,
        **kwargs,
    ) -> pd.Series:
        """Pull NEXRAD rainfall for a given period or storm at this raingage's location.

        Rainfall is calculated from reflectivity using Marshall-Palmer equation defined in dbz_to_depth()

        Can either specifiy start and end dates or give an event_num from the events table

        Parameters
        ----------
        start: datetime_like, optional
            start datetime, by default None

        end: datetime_like, optional
            end datetime, by default None

        event_num: Optional[int], optional
            event_num from events table, by default None

        averaging_distance: float, optional
            The distance from the Raingage coordinates in meters over which
            to average reflectivity, by default 1000.

        freq: Union[str,pd.DateOffset,None], optional
            The frequency of the reflectivity data to pull. Function will pull
            a reflectivity value at each timestep in the series with this frequency.
            Can be as low as 5 minutes

        progress: bool, optional
            Switch to show a tqdm bar for download progress, by default False

        nex: Optional[NEXRAD], optional
            Nexrad object to use for data pull. If None then create one using Raingage
            coordinates, by default None

        **kwargs:dict
            Keyword arguments to pass to dbz_to_depth()

        Returns
        -------
        pd.Series
            Series of reflectivity data
        """
        freq = freq if freq is not None else self.freq
        freq_seconds = to_offset(freq).nanos / 1e9
        dbz = self._storm_dbz(
            start, end, event_num, averaging_distance, freq, progress, nex
        )
        kwargs["interval"] = freq_seconds
        inches = dbz_to_depth(dbz, **kwargs)
        return inches

    def _get_event_dbzs(self, averaging_distance: float = 1000.0, **kwargs) -> None:
        """Calculate rainfall depth in inches for each event in the events table
        using NEXRAD reflectivity data.

        Adds a new column to events table `dbz_inches` with NEXRAD estimated depths.


        Parameters
        ----------
        averaging_distance: float, optional
            The distance from the Raingage coordinates in meters over which
            to average reflectivity, by default 1000.

        **kwargs:dict
            Keyword arguments to pass to dbz_to_depth()

        Returns
        -------
        None
            Adds column to self.events DataFrame

        """

        # dummy progress update functions to override default NEXRAD class methods
        def _close_progress(self):
            pass

        def _increment_progress(self, task=None):
            pass

        nex = NEXRAD(*self.latlon, averaging_distance)  # type: ignore
        nex._close_progress = MethodType(_close_progress, nex)  # type: ignore
        nex._increment_progress = MethodType(_increment_progress, nex)  # type: ignore

        def applier(row):
            start = row.start_date
            end = (
                row.start_date
                + pd.to_timedelta(row.hours_duration - self.ts_hours, "H")
            ).round("T")

            depth = self._storm_dbz_inches(start, end, nex=nex, **kwargs)
            nex.bar.update(1)
            return depth.sum()

        nex.bar = tqdm(total=len(self.events))
        self.events["dbz_inches"] = self.events.apply(applier, axis=1)
        nex.bar.close()
        del nex

    def _get_constraint(self, interval: float) -> float:
        """Get a constraint value for a given interval from
        the default constraints applied in the Atlas 14

        Parameters
        ----------
        interval: float
            Interval duration in hours

        Returns
        -------
        float
            constraint to apply to EVA params

        """
        if not hasattr(self, "default_constraints"):
            self.default_constraints = np.array(
                [
                    [1, 1.08],
                    [2, 1.04],
                    [3, 1.02],
                    [6, 1.01],
                    [1 * 24, 1.11],
                    [2 * 24, 1.04],
                    [3 * 24, 1.03],
                    [4 * 24, 1.02],
                    [7 * 24, 1.01],
                ]
            )
            if self.ts_hours >= 24:
                pass
            elif self.ts_hours >= 1:
                self.default_constraints[4:, 1] = 1
            elif self.ts_hours < 1:
                self.default_constraints[:, 1] = 1
        return max(
            1,
            _interp(
                interval,
                self.default_constraints[:, 0],
                self.default_constraints[:, 1],
                log=False,
            ),
        )

    def IDF(
        self,
        ariYrs: Union[Sequence[float], np.ndarray, bool] = (
            1,
            2,
            5,
            10,
            25,
            50,
            100,
            500,
            1000,
        ),
    ) -> pd.DataFrame:
        """Generate and IDF table from the data computed using `find_events` and `find_intervals`.

        Parameters
        ----------
        ariYrs: Union[Sequence[float], np.ndarray, bool], optional
            description], defaults to ( 1, 2, 5, 10, 25, 50, 100, 500, 1000, )

        Returns
        -------
        pd.DataFrame
            IDF table with rainfall depths for ARIs given in ariYrs and durations
            pulled with find_intervals.

        Raises
        ------
        ValueError
            If find_intervals was not run yet.


        """

        if not hasattr(self, "intervals"):
            raise ValueError("Must run find_intervals to pull IDF curves")

        output = {}
        for duration, gev in self._evds.items():
            output[duration] = gev.idf(ariYrs)
        return pd.DataFrame.from_dict(output, orient="index", columns=ariYrs)

    def write_idf_report(
        self,
        file_path: str,
        idf_periods: Sequence[float] = (
            1 / 52,
            1 / 12,
            3 / 12,
            6 / 12,
            1,
            2,
            5,
            10,
            25,
            50,
            100,
            200,
            500,
            1000,
        ),
        dump_json: bool = True,
    ) -> None:
        rg_2_dash(
            file_path=file_path, rg=self, idf_periods=idf_periods, dump_json=dump_json
        )


def get_pfds(
    lat: float, lon: float, verify=_noaa_ca_cert_path, **kwargs
) -> pd.DataFrame:
    """Pull atlas 14 PFDS and return table as DataFrame

    Parameters
    ----------
    lat: float
        Latitude of station in decimal degrees
    lon: float
        Longitude of station in decimal degrees
    verify:
        verify argument to feed into requests.get
    **kwargs
        Additional kwargs to be fed into requests.get

    Returns
    -------
    pd.DataFrame
        DataFrame of rainfall depths at various durations and ARIs

    """
    # base url of noaa pfds
    try:
        url = f"https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/fe_text_mean.csv?lat={lat}&lon={lon}&data=depth&units=english&series=pds&"
        response = requests.get(url, verify=verify, **kwargs)

        # NOAA PFDS uses ASCII encoding rather than UTF
        df = pd.read_csv(
            StringIO(response.content.decode()),
            engine="python",
            encoding="cp1252",
            skiprows=13,
            skipfooter=2,
        )

        # convert string duration index to hours
        df.index = (
            pd.to_timedelta(
                df.iloc[:, 0].str.replace("-", "").str.replace(":", "")
            ).dt.total_seconds()
            / 60
            / 60
        )
        # drop duration column
        df = df.drop("by duration for ARI (years):", axis=1)

        # convert ARI column headers to floats so we can interpolate between them
        df.columns = df.columns.astype(float)

        return df
    except Exception as e:
        print("Error pulling PFDS")
        print(str(e))
        return None


def _masked_argmin(
    a: np.ndarray, lowerLimit: float
) -> int:  # Defining func for regular array based soln
    """
    Find index of minimum value in masked array.

    (from https://stackoverflow.com/a/37973409/3216752)

    Parameters
    ----------
    a: np.ndarray
        Numpy array to mask
    lowerLimit: float
        lower limit of the mask

    Returns
    -------
    int
        index of minimum value in array that is larger than mask
    """
    valid_idx = np.where(a > lowerLimit)[0]
    return valid_idx[a[valid_idx].argmin()]


def _interp(x: float, X: np.ndarray, Y: np.ndarray, log: bool = True) -> float:
    """Piecewise linear interpolation with switch to perform logarithmically

    Parameters
    ----------
    x: float
        x value to interpolate
    X: Sequence[float]
        Iterable of known Xs (must be numpy operable)
    Y: Sequence[float]
        Iterable of known Ys (must be numpy operable)
    log: bool, optional
        Bool switch to perform interpolation logarithmically, defaults to True

    Returns
    -------
    float
        Interpolated y-value given x

    """
    # if x larger than KnownXs, use last two KnownX to extrapolate
    if x >= np.max(X):
        i = len(X) - 1

    # if x larger than KnownXs, use frst two KnownX to extrapolate
    elif x <= np.min(X):
        i = 1
    # else use the nearest KnownXs around x to interpolate
    else:
        i = _masked_argmin(X, x)

    # interpolate logarithmically
    # change in x time slope of line between two surrounding KnownXs plus preceeding known Y
    if log:
        return np.exp(
            (np.log(x) - np.log(X[i - 1]))
            * (np.log(Y[i - 1]) - np.log(Y[i]))
            / (np.log(X[i - 1]) - np.log(X[i]))
            + np.log(Y[i - 1])
        )
    # interpolate linearly
    # change in x time slope of line between two surrounding KnownXs plus preceeding known Y
    else:
        return (x - X[i - 1]) * (Y[i] - Y[i - 1]) / (X[i] - X[i - 1]) + Y[i - 1]

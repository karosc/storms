from storms.precip.raingage import Raingage
from storms.precip._rainrank import eventify
import pandas as pd
from typing import NoReturn, Dict, List
from pandas.tseries.frequencies import to_offset


class Network:
    def __init__(self, *args: Raingage, **kwargs):
        self.gages: Dict[str, Raingage] = {}
        data: List[pd.Series] = []
        for i, gage in enumerate(args):
            if i == 0:
                self._freq = gage.freq
            self.validate_gage(gage)
            data.append(gage.data)
            if gage.ID is None:
                raise Exception("All gages in a network must have an ID set")
            self.gages[gage.ID] = gage

        self.gage_names = list(self.gages.keys())

        self.qaqc = kwargs.get("qaqc")
        if self.qaqc:
            self.validate_gage(self.qaqc)
            data.append(self.qaqc.data)
            self.gages[self.qaqc.ID] = self.qaqc
            self.gage_names.append(self.qaqc.ID)

        if self.freq is None:
            raise Exception(
                "Network must have a data frequency assigned, could not infer from gages."
            )

        self.data = pd.concat(data, axis=1).fillna(0)
        self.data.columns = self.gage_names
        self.precip = self.data.to_numpy()
        self.datetime = self.data.index.to_numpy()

    def validate_gage(self, gage: Raingage) -> None:
        if not isinstance(gage, Raingage):
            raise Exception("All network objects must be Raingages")
        if gage.ID is None:
            raise Exception("All Raingages must have an ID property")
        if gage.freq != self._freq:
            raise Exception("All gages in the network must have the same frequency.")

    @property
    def freq(self) -> pd.DateOffset:
        """
        get pandas date offset of this object's data frequency

        :return: Returns frequency of precip data as pandas offset
        :rtype: pd.DateOffset
        """
        return self._freq

    @freq.setter
    def freq(self, offset: str) -> pd.DateOffset:
        """
        sets _freq pandas offset object from input offset string.
        Ultimately this function is used for inferring/storing the rainfall
        timeseries frequency.

        :param offset: pandas offset string
        :type offset: str
        :return: pandas offset object
        :rtype: pd.DateOffset
        """
        self._freq = to_offset(offset)

    @property
    def ts_hours(self) -> float:
        """
        return the timeseries timestep in hours

        :return: time series time step (hours)
        :rtype: float
        """
        return self.freq.nanos / 1e9 / 60 / 60

    def find_events(
        self, inter_event_period: float = 6, threshold_depth: float = 0.25
    ) -> None:
        self._events, self._event_col = eventify(
            self.precip,
            self.datetime,
            inter_event_period,
            threshold_depth,
            self.ts_hours,
        )
        # create event dataframe
        self.events = pd.DataFrame(self._events)

        # expand event total and max to individual columns
        self.events.loc[
            :, [f"{gage}_total" for gage in self.gage_names]
        ] = self.events.event_total.apply(pd.Series).to_numpy()
        self.events.loc[
            :, [f"{gage}_max" for gage in self.gage_names]
        ] = self.events.event_max.apply(pd.Series).to_numpy()

        # drop unexpanded columns
        self.events.drop(["event_total", "event_max"], axis=1, inplace=True)

        # propagate event data back into gage objects
        for gage in self.gages:
            gage_obj = self.gages[gage]
            cols = [
                "event_num",
                f"{gage}_total",
                "start_date",
                "event_start_index",
                "event_end_index",
                "hours_duration",
                f"{gage}_max",
                "event_records",
            ]
            # rename gage_total and gage_max to event_total and event_max
            gage_obj._events = (
                self.events.loc[:, cols]
                .copy()
                .rename(
                    {f"{gage}_total": "event_total", f"{gage}_max": "event_max"}, axis=1
                )
            )
            # drop events that had zero rainfall at this gage
            # gage_obj._events = gage_obj._events.loc[
            #     gage_obj._events.event_total > 0
            # ].reset_index(drop=True)

            # calculate end date from durations
            endDate = (
                gage_obj._events.start_date
                + pd.to_timedelta(gage_obj._events.hours_duration - self.ts_hours, "h")
            ).round("T")

            # find endIndex in gage data
            gage_obj._events.event_end_index = gage_obj.data.index.get_indexer(
                endDate, method="ffill"
            )
            # find endIndex in gage data
            gage_obj._events.event_start_index = gage_obj.data.index.get_indexer(
                gage_obj._events.start_date, method="bfill"
            )
            # pull start_date from gage datetime series
            start_date = gage_obj.datetime[gage_obj._events.event_start_index]
            # pull end date from gage obj datetime series
            newEndDate = gage_obj.datetime[gage_obj._events.event_end_index - 1]
            # calculate new storm duration based on gage datetimes
            # gage_obj.events.hours_duration = (
            #     newEndDate - start_date
            # ) / hour_unit + gage_obj.ts_hours

            gageIDXinNetwork = self.data.index.get_indexer(gage_obj.datetime)
            gage_obj._event_col = self._event_col[gageIDXinNetwork]

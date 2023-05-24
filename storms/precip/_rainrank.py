from typing import Sequence, NamedTuple, Tuple
import numpy as np
from numba import njit, prange

sub_event_dtype = np.dtype(
    [
        ("event_num", "<i8"),
        ("duration", "<f8"),
        ("total", "<f8"),
        ("event_start_index", "<i8"),
        ("event_end_index", "<i8"),
        ("start_date", "datetime64[ns]"),
        # ("diff", "<f8"),
        ("killing_event", "<i8"),
        # ("killed", "<i8"),
    ]
)

hour_unit = np.timedelta64(1, "h")
minute_unit = np.timedelta64(1, "m")
ms_unit = np.timedelta64(1, "ms")
ns_unit = np.timedelta64(1, "ns")


class Event(NamedTuple):
    event_num: int
    event_total: np.float64
    start_date: np.datetime64
    event_start_index: int
    event_end_index: int
    hours_duration: np.float64
    event_max: np.float64
    event_records: int


################################################################################
# Helper functions to apply 1d functions along axis 0
# using numba https://github.com/numba/numba/issues/1269#issuecomment-702665837
###############################################################################
@njit(cache=True)
def nbround(arr, decimals):
    out = np.empty_like(arr)
    return np.round_(arr, decimals, out)


@njit(cache=True)
def apply_along_axis_0(func1d, arr):
    """Like calling func1d(arr, axis=0)"""
    if arr.size == 0:
        raise RuntimeError("Must have arr.size > 0")
    ndim = arr.ndim
    if ndim == 0:
        raise RuntimeError("Must have ndim > 0")
    elif 1 == ndim:
        return func1d(arr)
    else:
        result_shape = arr.shape[1:]
        out = np.empty(result_shape, arr.dtype)
        _apply_along_axis_0(func1d, arr, out)
        return out


@njit(cache=True)
def _apply_along_axis_0(func1d, arr, out):
    """Like calling func1d(arr, axis=0, out=out). Require arr to be 2d or bigger."""
    ndim = arr.ndim
    if ndim < 2:
        raise RuntimeError("_apply_along_axis_0 requires 2d array or bigger")
    elif ndim == 2:  # 2-dimensional case
        for i in range(len(out)):
            out[i] = func1d(arr[:, i])
    else:  # higher dimensional case
        for i, out_slice in enumerate(out):
            _apply_along_axis_0(func1d, arr[:, i], out_slice)


@njit(cache=True)
def nb_mean_axis_0(arr):
    return apply_along_axis_0(np.mean, arr)


@njit(cache=True)
def nb_std_axis_0(arr):
    return apply_along_axis_0(np.std, arr)


@njit(cache=True)
def nb_amax_axis_0(arr):
    return apply_along_axis_0(np.amax, arr)


################################################################
############## Event finding functions #########################
################################################################


@njit(cache=True)
def eventify1d(
    precip: np.ndarray,
    datetime: np.ndarray,
    inter_event_period: float = 6,
    threshold_depth: float = 0.25,
    ts: float = 1,
) -> Tuple[list, np.ndarray]:
    nRecords = len(precip)
    # initialize running event stats
    events = []
    event_num = 1
    event_total = precip[0]
    event_max = precip[0]
    event_start_index = 0
    event_records = 1

    # column of event number or each record in the precip data
    eventCol = np.empty(nRecords, dtype="<i8")
    eventCol[0] = event_num

    for i in range(1, nRecords):
        diff = (datetime[i] - datetime[i - 1]) / hour_unit

        # if new event
        new_event = diff > inter_event_period
        end_of_record = i == nRecords - 1
        if new_event or end_of_record:
            # add final record to event if at end of time series
            if end_of_record and not new_event:
                event_end_index = i
                event_total += precip[i]
                event_records += 1
                event_max = max(event_max, precip[i])
                eventCol[i] = event_num
            else:
                event_end_index = i - 1

            # if event exceeds threshold, save data and increment event num, otherwise, reset stats
            if round(event_total, 4) >= threshold_depth:
                hours_duration = max(
                    (
                        (datetime[event_end_index] - datetime[event_start_index])
                        / hour_unit
                    )
                    + ts,
                    ts,
                )
                events.append(
                    Event(
                        event_num,
                        round(event_total, 4),
                        datetime[event_start_index],
                        event_start_index,
                        i,
                        hours_duration,
                        event_max,
                        event_records,
                    )
                )

                event_num += 1

            # if at end of time series terminate loop
            if end_of_record:
                continue

            event_total = precip[i]
            event_max = precip[i]
            event_records = 1
            event_start_index = i
            eventCol[i] = event_num

        # else continue incrementing
        else:
            event_total += precip[i]
            event_records += 1
            event_max = max(event_max, precip[i])
            eventCol[i] = event_num

    return events, eventCol


@njit(cache=True)
def eventify2d(
    precip: np.ndarray,
    datetime: np.ndarray,
    inter_event_period: float = 6,
    threshold_depth: float = 0.25,
    ts: float = 1,
):
    nRecords = len(precip)
    # initialize running event stats
    events = []
    event_num = 1
    event_total = precip[0].copy()
    event_max = precip[0].copy()
    maxxer = np.zeros((2, precip[0].shape[0]))
    maxxer[0, :] = event_max.copy()
    event_start_index = 0
    event_records = 1

    eventCol = np.empty(nRecords, dtype="<i8")
    eventCol[0] = event_num

    for i in range(1, nRecords):
        diff = (datetime[i] - datetime[i - 1]) / hour_unit

        # if new event
        new_event = diff > inter_event_period
        end_of_record = i == nRecords - 1
        if new_event or end_of_record:
            # add final record to event if at end of time series
            if end_of_record and not new_event:
                event_total += precip[i]
                event_records += 1
                maxxer[1, :] = precip[i].copy()
                event_max = nb_amax_axis_0(maxxer)
                maxxer[0, :] = event_max.copy()
                # print(f"eor {event_start_index} {event_end_index}")
                event_end_index = i
                eventCol[i] = event_num
            else:
                event_end_index = i - 1

            # if event exceeds threshold, save data and increment event num, otherwise, reset stats
            if np.any(nbround(event_total, 4) >= threshold_depth):
                # print(f"eor {event_start_index} {event_end_index}")

                hours_duration = max(
                    (
                        (datetime[event_end_index] - datetime[event_start_index])
                        / hour_unit
                    )
                    + ts,
                    ts,
                )
                events.append(
                    Event(
                        event_num,
                        nbround(event_total, 4),
                        datetime[event_start_index],
                        event_start_index,
                        i,
                        hours_duration,
                        event_max,
                        event_records,
                    )
                )

                event_num += 1

            # if at end of time series terminate loop
            if end_of_record:
                continue

            event_total = precip[i].copy()
            event_max = precip[i].copy()
            event_records = 1
            event_start_index = i
            eventCol[i] = event_num
            maxxer[:, :] = 0
            maxxer[0, :] = event_max.copy()

        # else continue incrementing
        else:
            event_total += precip[i]
            event_records += 1
            maxxer[1, :] = precip[i].copy()
            event_max = nb_amax_axis_0(maxxer)
            maxxer[0, :] = event_max.copy()

            eventCol[i] = event_num

    return events, eventCol


def eventify(
    precip: np.ndarray,
    datetime: np.ndarray,
    inter_event_period: float = 6,
    threshold_depth: float = 0.25,
    ts: float = 1,
) -> Tuple[list, np.ndarray]:
    if len(precip.shape) > 1:
        return eventify2d(
            precip,
            datetime,
            inter_event_period,
            threshold_depth,
            ts,
        )
    else:
        return eventify1d(
            precip,
            datetime,
            inter_event_period,
            threshold_depth,
            ts,
        )


################################################################
############## Sub-event finding functions #####################
################################################################


@njit(nogil=True, cache=True)
def ns_intervals(
    precip: np.ndarray,
    datetime: np.ndarray,
    eventCol: np.ndarray,
    periods: np.ndarray = np.array([1, 2, 3, 6, 12, 24, 48]),
    threshold_depth: float = 0.25,
) -> np.ndarray:
    # allocate output array, we will calculate the largest event
    # for each periods starting from each record
    nPeriods = len(periods)
    nRecords = len(precip)
    maxEvents = nRecords * nPeriods
    # intervals = np.empty(maxEvents, dtype=sub_event_dtype)
    intervals = np.empty(maxEvents, dtype=sub_event_dtype)

    # outer loop over each record of precip
    for i in range(0, nRecords):
        # initialize stats
        subTotal = precip[i]
        eventStart = datetime[i]
        event_num = eventCol[i]
        diff = 0
        # eventidx = event_num - 1

        j = i

        # initialize period values for this record
        for iperiod, period in enumerate(periods):
            intervals["duration"][i * nPeriods + iperiod] = period
            intervals["event_num"][i * nPeriods + iperiod] = event_num
            intervals["event_start_index"][i * nPeriods + iperiod] = i
            intervals["start_date"][i * nPeriods + iperiod] = eventStart
            intervals["total"][i * nPeriods + iperiod] = 0
            intervals["killing_event"][i * nPeriods + iperiod] = 0

        # loop forward from record i until exceeding maximum period length
        copy_periods = periods.copy()
        while (diff < periods.max() - 0.01) and (j < nRecords):
            diff = (datetime[j] - datetime[i]) / hour_unit
            # print(diff)
            subTotal = round(precip[i : j + 1].sum(), 4)

            # loop over each period that is larger than current window, calculate
            # stats for each and assign to period values
            for period in periods[periods - 0.01 > round(diff, 4)]:
                iperiod = np.where(periods == period)[0][0]
                intervals["event_end_index"][i * nPeriods + iperiod] = j + 1
                if (
                    intervals["total"][i * nPeriods + iperiod] < subTotal
                    and round(subTotal, 3) >= threshold_depth
                ):
                    intervals["total"][i * nPeriods + iperiod] = subTotal
            j += 1

    # sort interval data (smallest to largest)
    sortedIDX = np.argsort(intervals["total"], kind="mergesort")
    intervals = intervals[sortedIDX]

    for period in periods:
        # pull out indices of the period we are examining
        # invert array to get sorted from largest to smallest
        periodIndicies = np.where(intervals["duration"] == period)[0][::-1]

        # loop over subevents, zeroing out precip values for smaller events that either
        # 1. overlap in time with larger event
        # 2. are have start date within period-hours of a larger event
        # This effort is to avoid double counting rainfall
        for i, largerIDX in enumerate(periodIndicies):
            # skip event if it was already killed
            if intervals["killing_event"][largerIDX] > 0:
                continue
            # larger event is the current storm (earlier in sort list)
            # all storms after this event are smaller
            smallerIDX = periodIndicies[i + 1 :]

            # calculate difference in start time between larger event and smaller events in hours
            diff = (
                np.abs(
                    (
                        intervals["start_date"][smallerIDX]
                        - intervals["start_date"][largerIDX]
                    )
                )
                / hour_unit
            )

            # create empty array for diffs to round (for some reason numba can't round in place using array.round())
            roundDiff = np.empty_like(diff)
            # round hours to 4 decimal places (~0.5 second)
            np.round_(diff, 4, roundDiff)

            # find smaller events that are within 1 period of larger event
            tooClose = roundDiff < period

            # find smaller events that have same event number as larger event
            sameEvent = (
                intervals["event_num"][smallerIDX] == intervals["event_num"][largerIDX]
            )

            # indices to remove are smaller storms that are too close in time
            # or are the same meteorological event
            remove = tooClose | sameEvent  # & largerThanOne

            intervals["killing_event"][smallerIDX[remove]] = intervals["event_num"][
                largerIDX
            ]

    return intervals


@njit(parallel=True, nogil=True, cache=True)
def ns_intervals_parallel(
    precip: np.ndarray,
    datetime: np.ndarray,
    eventCol: np.ndarray,
    periods: np.ndarray = np.array([1, 2, 3, 6, 12, 24, 48]),
    threshold_depth: float = 0.25,
) -> Sequence[np.ndarray]:
    # allocate output array, we will calculate the largest event
    # for each periods starting from each record
    nPeriods = len(periods)
    nRecords = len(precip)
    maxEvents = nRecords * nPeriods
    # intervals = np.zeros(maxEvents, dtype=sub_event_dtype)
    event_num = np.empty(maxEvents, dtype="<i8")
    duration = np.empty(maxEvents, dtype="<f8")
    total = np.empty(maxEvents, dtype="<f8")
    event_start_index = np.empty(maxEvents, dtype="<i8")
    event_end_index = np.empty(maxEvents, dtype="<i8")
    start_date = np.empty(maxEvents, dtype="datetime64[ns]")
    killing_event = np.empty(maxEvents, dtype="<i8")

    # outter loop over each record of precip
    for i in prange(0, nRecords):
        # initialize stats
        subTotal = precip[i]
        eventStart = datetime[i]
        evNum = eventCol[i]
        diff = 0
        # eventidx = event_num - 1

        j = i

        # initialize period values for this record
        for iperiod in prange(len(periods)):
            period = periods[iperiod]

            duration[i * nPeriods + iperiod] = period
            event_num[i * nPeriods + iperiod] = evNum
            event_start_index[i * nPeriods + iperiod] = i
            start_date[i * nPeriods + iperiod] = eventStart
            total[i * nPeriods + iperiod] = 0
            killing_event[i * nPeriods + iperiod] = 0

        # loop forward from record i until exceeding maximum period length
        while (diff < periods.max() - 0.01) and (j < nRecords):
            diff = (datetime[j] - datetime[i]) / hour_unit
            # print(diff)
            subTotal = round(precip[i : j + 1].sum(), 4)

            # loop over each period that is larger than current window, calculate
            # stats for each and assign to period values

            for period in periods[periods - 0.01 > round(diff, 4)]:
                iperiod = np.where(periods == period)[0][0]
                event_end_index[i * nPeriods + iperiod] = j + 1

                if (
                    total[i * nPeriods + iperiod] < subTotal
                    and round(subTotal, 3) >= threshold_depth
                ):
                    total[i * nPeriods + iperiod] = subTotal
            j += 1

    # sort interval data (smallest to largest)
    sortedIDX = np.argsort(total, kind="mergesort")
    event_num = event_num[sortedIDX]
    duration = duration[sortedIDX]
    total = total[sortedIDX]
    event_start_index = event_start_index[sortedIDX]
    event_end_index = event_end_index[sortedIDX]
    start_date = start_date[sortedIDX]
    killing_event = killing_event[sortedIDX]

    for iperiod in prange(len(periods)):
        period = periods[iperiod]

        # pull out indices of the period we are examining
        # invert array to get sorted from largest to smallest
        periodIndicies = np.where(duration == period)[0][::-1]

        # loop over subevents, zeroing out precip values for smaller events that either
        # 1. overlap in time with larger event
        # 2. are have start date within period-hours of a larger event
        # This effort is to avoid double counting rainfall
        for i, largerIDX in enumerate(periodIndicies):
            # loops += 1
            # skip event if it was already killed
            if killing_event[largerIDX] > 0:
                continue
            # larger event is the current storm (earlier in sort list)
            # all storms after this event are smaller
            smallerIDX = periodIndicies[i + 1 :]

            # calculate difference in start time between larger event and smaller events in hours
            diff = np.abs((start_date[smallerIDX] - start_date[largerIDX])) / hour_unit

            # create empty array for diffs to round (for some reason numba can't round in place using array.round())
            roundDiff = np.empty_like(diff)
            # round hours to 4 decimal places (~0.5 second)
            np.round_(diff, 4, roundDiff)

            # find smaller events that are within 1 period of larger event
            tooClose = roundDiff < period

            # find smaller events that have same event number as larger event
            sameEvent = event_num[smallerIDX] == event_num[largerIDX]

            # indices to remove are smaller storms that are too close in time
            # or are the same meteorological event
            remove = tooClose | sameEvent  # & largerThanOne

            # self.intervals["total"][smallerIDX[remove]] = 0
            killing_event[smallerIDX[remove]] = event_num[largerIDX]

    return (
        event_num,
        duration,
        total,
        event_start_index,
        event_end_index,
        start_date,
        killing_event,
    )

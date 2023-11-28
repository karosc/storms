import numpy as np
from numba import njit
from numba.extending import overload
from numba.types import Array as nbArray, float64 as nbfloat64, int64 as nbint64

###################################
######### numba utils #############
###################################

hour_unit = np.timedelta64(1, "h")
minute_unit = np.timedelta64(1, "m")


@overload(np.nan_to_num)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """Numba implementation of nan_to_num
    https://github.com/numba/numba/issues/6857
    """
    if isinstance(x, nbArray):

        def nan_to_num_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
            if copy:
                out = np.copy(x).reshape(-1)
            else:
                out = x.reshape(-1)
            for i in range(len(out)):
                if np.isnan(out[i]):
                    out[i] = nan
                if posinf is not None and np.isinf(out[i]) and out[i] > 0:
                    out[i] = posinf
                if neginf is not None and np.isinf(out[i]) and out[i] < 0:
                    out[i] = neginf
            return out.reshape(x.shape)

    else:

        def nan_to_num_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
            if np.isnan(x):
                return nan
            if posinf is not None and np.isinf(x) and x > 0:
                return posinf
            if neginf is not None and np.isinf(x) and x < 0:
                return neginf
            return x

    return nan_to_num_impl


@njit(cache=True)
def _is_not_in(master: np.ndarray, search: np.ndarray):
    """Numba function for determining if certain values are in an array"""
    return ~(
        np.searchsorted(master, search, side="right")
        != np.searchsorted(master, search, side="left")
    )


@njit(cache=True)
def nbround(x, decimals):
    "https://github.com/numba/numba/issues/2648"
    out = np.empty_like(x)
    return np.round_(x, decimals, out)


####################################################################
######### Ormsbee Geometric Similarity Disagg Methods  #############
##### https://doi.org/10.1061/(ASCE)0733-9429(1989)115:4(507) ######
####################################################################


@njit(cache=True)
def _vstar(Vt, tstar):
    """Ormsbee eq 18c"""
    if len(Vt) != 3:
        raise Exception("Vt must be 3 elements")
    return 30 * (Vt[1] + Vt[2]) - (tstar / 2) * (Vt[2] - Vt[0])


@njit(cache=True)
def _tstar(Vt):
    """Time parameter in ormsbee algorithm

    Ormsbee eqs 20 - 23
    """
    # if flat rainfall pattern
    if (np.abs(Vt[0] - Vt[1]) < 1e-5) and (np.abs(Vt[1] - Vt[2]) < 1e-5):
        return 60

    # Type 1 or 2
    elif ((Vt[0] >= Vt[1]) and (Vt[1] >= Vt[2])) or (
        (Vt[1] >= Vt[0]) and (Vt[2] >= Vt[1])
    ):
        return 60 * ((Vt[0] - Vt[1]) / (Vt[0] - Vt[2]))

    # Type 3 or 4
    else:
        return 60 * ((Vt[0] - Vt[1]) / (Vt[0] + Vt[2] - 2 * Vt[1]))


@njit(cache=True)
def _pdf_small(t, Vt, tstar, Vstar):
    """Orsmbee eq 19a"""
    return np.nan_to_num(
        ((Vt[0] * t) / Vstar) - (((Vt[0] - Vt[1]) * t**2) / (2 * Vstar * tstar))
    )


@njit(cache=True)
def _pdf_big(t, Vt, tstar, Vstar):
    """Orsmbee eq 19b"""
    return np.nan_to_num(
        (((Vt[1] + Vt[0]) * tstar) / (2 * Vstar))
        + ((Vt[1] * (t - tstar)) / (Vstar))
        - (((Vt[1] - Vt[2]) * (t - tstar) ** 2) / (2 * Vstar * (60 - tstar)))
    )


@njit(cache=True)
def _pdf(t, Vt):
    """Ormsbee eq 19"""
    tstar = _tstar(Vt)
    Vstar = _vstar(Vt, tstar)
    weights = np.empty_like(t)

    mask1 = t <= tstar
    weights[mask1] = _pdf_small(t[mask1], Vt, tstar, Vstar)

    weights[~mask1] = _pdf_big(t[~mask1], Vt, tstar, Vstar)
    return weights


@njit(cache=True)
def continuous_deterministic(
    datetime: np.ndarray,
    precip: np.ndarray,
    num_bins: int,
    in_ts: np.timedelta64,
    out_ts: np.timedelta64,
):
    num_recs = len(precip) * num_bins
    diss_precip = np.empty(num_recs, precip.dtype)
    diss_dt = np.empty(num_recs, datetime.dtype)

    working_hours = np.empty(3, datetime.dtype)
    t = np.arange(start=1, stop=num_bins + 1) * out_ts / minute_unit

    for i, hour in enumerate(datetime):
        working_hours[0] = hour - in_ts
        working_hours[1] = hour
        working_hours[2] = hour + in_ts

        # determine which datetimes are missing
        # https://stackoverflow.com/questions/31789187/find-indices-of-large-array-if-it-contains-values-in-smaller-array
        # fill data gaps with zeros (not raining)
        idx = np.searchsorted(datetime, working_hours)
        idx[(idx >= len(precip))] = len(precip) - 1
        Vt = precip[idx]
        Vt[_is_not_in(datetime, working_hours)] = 0

        # Ormsbee eq 24
        pdf = _pdf(t, Vt)
        pdf_m1 = np.roll(pdf, 1)
        pdf_m1[0] = 0

        # dissagg data based on pdfs
        diss_precip[i * num_bins : (i + 1) * num_bins] = precip[i] * (pdf - pdf_m1)
        diss_dt[i * num_bins : (i + 1) * num_bins] = hour + (t * minute_unit - out_ts)

    return diss_dt, diss_precip


# @njit(cache=True)
def continuous_stochastic(
    datetime: np.ndarray,
    precip: np.ndarray,
    num_bins: int,
    in_ts: np.timedelta64,
    out_ts: np.timedelta64,
    precision: int = 2,
    scale_factor: float = 0,
):
    # seed numpy for reproducible randomness
    np.random.seed(42)

    ## setup data structures ##
    num_recs = len(precip) * num_bins
    diss_precip = np.empty(num_recs, precip.dtype)
    diss_dt = np.empty(num_recs, datetime.dtype)

    working_hours = np.empty(3, datetime.dtype)

    t = np.arange(start=0, stop=num_bins) * out_ts / minute_unit
    for i, hour in enumerate(datetime):
        working_hours[0] = hour - in_ts
        working_hours[1] = hour
        working_hours[2] = hour + in_ts

        # determine which datetimes are missing
        # https://stackoverflow.com/questions/31789187/find-indices-of-large-array-if-it-contains-values-in-smaller-array
        # fill data gaps with zeros (not raining)
        idx = np.searchsorted(datetime, working_hours)
        idx[(idx >= len(precip))] = len(precip) - 1
        Vt = precip[idx]
        Vt[_is_not_in(datetime, working_hours)] = 0

        ## setup data structures ##
        # number of discrete rainfall data points resulting
        num_pulses = int(round(Vt[1] / 10**-precision, 0))
        # pdf of current input window
        pdf = _pdf(t, Vt)
        rand = np.random.rand(num_pulses)
        bins = np.searchsorted(pdf, rand, side="right") - 1
        # empty array to hold dissagged window
        diss = np.zeros_like(pdf)
        for j in bins:
            diss[j] += 0.01

        total_without_max = diss.sum() - diss.max()
        if total_without_max > 0.0001:
            rand_val = np.random.rand()
            # scale up peak rainfall depth by percentage of the total rainfall without the max
            # perturb scale factor by random value
            spiked_max = np.round(
                diss.max() + scale_factor * rand_val * total_without_max, precision
            )
            # reduce all rainfall values by the increase in the spiked max
            reduction_factor = (diss.sum() - spiked_max) / total_without_max
            diss = nbround(diss * reduction_factor, precision)
            # sub in spiked max
            diss[np.argmax(diss)] = spiked_max
            # correct for any floating point arithmetic error
            # add/subtract any error in the total rainfall to/from the max
            diss[np.argmax(diss)] = spiked_max + Vt[1] - diss.sum()

        # assign dissagg data to output array
        diss_precip[i * num_bins : (i + 1) * num_bins] = diss
        diss_dt[i * num_bins : (i + 1) * num_bins] = hour + (
            (t + out_ts / minute_unit) * minute_unit - out_ts
        )

    return diss_dt, diss_precip

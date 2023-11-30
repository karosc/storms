from typing import Union, Callable

import numpy as np
from numpy.typing import ArrayLike
from pandas import Series

from scipy.stats import genextreme, genpareto, rv_continuous
from lmoments3 import distr, lmom_ratios


class EVA:
    def __init__(
        self,
        data: Series,
        model: str,
        constraint: float = 1,
        nmom: int = 3,
        useLMU: bool = False,
        regionalize: bool = True,
        **kwargs,
    ):
        """

        A class to help create a fitted extreme value distribution with which to calculate the
        ARI of storm events.

        Parameters
        ----------
        data : pd.Series
            Timeseries of events with datetime index and numeric values

        model : str
            The type of extreme value model to use. One of ("ams", "gev", "pds", "gpd", "hybrid", None)

        constraint : float, optional
            Constraint to apply to extreme value model params.
            This should be used when the data are hourly greater in frequency, by default 1

        nmom : int, optional
            Number of l-moments to use when fitting data, by default 3

        useLMU : bool, optional
            Switch to Calculates the "unbiased" sample L-moment ratios of a data
            set, by a more direct method, by default False

        regionalize : bool, optional
            Switch to regionalize sample L-moment ratios of a data, by default True
        """
        self.ser = data
        self.model = model
        self.data = np.sort(data)
        # self.num_years = num_years
        self.constraint = constraint
        self.nmom = nmom
        self.useLMU = useLMU
        self.regionalize = regionalize

        if model in ("ams", "gev"):
            self.model = genextreme
            self.fitter = distr.gev._lmom_fit
            self.evs = np.sort(data.groupby(data.index.year).max().to_numpy(copy=True))
        elif model in ("pds", "gpd"):
            self.model = genpareto
            self.fitter = distr.gpa._lmom_fit
            self.evs = np.sort(data.to_numpy(copy=True))[-self.num_years :]
        elif model in ("hybrid"):
            self.model = genextreme
            self.fitter = distr.gev._lmom_fit
            self.evs = np.sort(data.to_numpy(copy=True))[-self.num_years :]

        self._model = EVModel(
            self.evs,
            self.num_years,
            self.model,
            self.constraint,
            self.fitter,
            self.nmom,
            self.useLMU,
            self.regionalize,
        )

    @property
    def num_years(self) -> int:
        """Calculate the nubmer of years in the event timeseries.

        Parameters
        ----------

        Returns
        -------
        int
            Number of years in dataset based on the first and last timestamps.

        """
        return round(
            (self.ser.index.max() - self.ser.index.min()) / np.timedelta64(365, "D")
        )

    def ARI(self, d: float) -> Union[float, np.ndarray]:
        """
        Calculate the ARI of an event based on the fitted GEV.
        This uses the CDF of the GEV distribution to calculated cumulative probability,
        then finds the probability of exceedance (1 - CDF result), then inverts that result.

        ex. ( 1 / ( 1 - CDF(d) ) )

        Parameters
        ----------
        d: float
            Depth of storm event.

        Returns
        -------
        Union[float,np.ndarray]
            ARI in years.
        """
        # return 1.0 / (
        #     1.0
        #     - math.exp(
        #         -1.0
        #         * (
        #             1.0
        #             - self.paras["c"] / self.paras["scale"] * (d - self.paras["loc"])
        #         )
        #         ** (1.0 / self.paras["c"])
        #     )
        # )
        ARI = np.atleast_1d(1.0 / (1 - self._model.cdf(d)))
        pps = np.atleast_1d(
            plotting_position(
                len(self.data) - np.searchsorted(self.data, d), self.num_years
            )
        )

        # where EV model is limited (low ARIs) use the plotting position
        ARI[ARI == 1] = pps[ARI == 1]

        # return float if only 1 ARI requested
        return ARI if len(ARI) > 1 else ARI[0]

    def idf(
        self,
        ariYrs: ArrayLike = np.array([1.05, 2, 5, 10, 25, 50, 100, 200, 500, 1000]),
    ) -> Union[float, np.ndarray]:
        """Compute depth at various return intervals in years

        Parameters
        ----------
        ariYrs : ArrayLike, optional
            return intervals for which to compute depths, by default np.array([1.05, 2, 5, 10, 25, 50, 100, 200, 500, 1000])

        Returns
        -------
        Union[float, np.ndarray]
            Depth(s)
        """

        ariYrs = np.asarray(ariYrs)
        gevYrs = ariYrs[ariYrs > 1]
        ppYrs = ariYrs[ariYrs <= 1]

        ppDeps = (
            Series(self.data)
            .reindex(len(self.data) - plotting_positionInv(ppYrs, self.num_years))
            .to_numpy()
        )
        # ppDeps = self.data[len(self.data) - plotting_positionInv(ppYrs, self.num_years)]
        gevDeps = self._model.ppf(1 - 1 / gevYrs)

        return np.concatenate([ppDeps, gevDeps])

    def CI95(
        self,
        ariYrs: np.ndarray = np.array([1.05, 2, 5, 10, 25, 50, 100, 200, 500, 1000]),
        generations: int = 1000,
    ) -> np.ndarray:
        """Calculate 5th and 95th percentile confidence limits for various ARIs in ariYrs input.

        Parameters
        ----------
        ariYrs : np.ndarray, optional
            Array of ARIs for which to calculate confidence limits, by default np.array([1.05, 2, 5, 10, 25, 50, 100, 200, 500, 1000])
        generations : int, optional
            Number of generations to run when bootstrap resampling, by default 1000

        Returns
        -------
        np.ndarray
            2d array with [[5th percentile,95th percentile],] limits for each ARI
        """

        pps = 1 - 1 / ariYrs
        al = []
        for i in range(generations):
            data = self._model.rvs(size=self.num_years)
            # newFit = EVA(self.ser, self.self.num_years)

            newFit = EVModel(
                data,
                self.num_years,
                self.model,
                1,
                self.fitter,
                self.nmom,
                self.useLMU,
                self.regionalize,
            )

            al.append(newFit.ppf(pps))

        arr = np.vstack(al)
        return np.percentile(arr, [5, 95], axis=0).transpose()


class EVModel:
    """

    A class to help unify working with multiple types of EVA methods and distributions.

    Parameters
    ----------
    evs : np.ndarray
        Timeseries of events with datetime index and numeric values

    model : str
        The type of extreme value model to use. One of ("ams", "gev", "pds", "gpd", "hybrid", None)

    constraint : float, optional
        Constraint to apply to extreme value model params.
        This should be used when the data are hourly greater in frequency

    nmom : int, optional
        Number of l-moments to use when fitting data

    useLMU : bool, optional
        Switch to Calculates the "unbiased" sample L-moment ratios of a data
        set, by a more direct method

    regionalize : bool, optional
        Switch to regionalize sample L-moment ratios of a data
    """

    def __init__(
        self,
        evs: np.ndarray,
        num_years: int,
        model: rv_continuous,
        constraint: float,
        fitter: Callable,
        nmom: int,
        useLMU: bool = False,
        regionalize: bool = True,
    ):
        self.evs = np.sort(evs)
        self.num_years = num_years
        self.model = model
        self.constraint = constraint
        self.fitter = fitter
        self.nmom = nmom
        self.useLMU = useLMU
        self.regionalize = regionalize

        self.LMU() if self.useLMU else self.LMR()

    def LMR(self):
        self._ratios = _samlmr(self.evs * self.constraint, nmom=3)
        if self.regionalize:
            self._regionalRatios = _reglmr([self._ratios], [self.num_years])
            self.paras = self.fitter(self._regionalRatios)
            self.paras["loc"] *= self._ratios[0]
            self.paras["scale"] *= self._ratios[0]
        else:
            self.paras = self.fitter(self._ratios)

    def LMU(self):
        self._ratios = lmom_ratios(self.evs * self.constraint, nmom=3)
        if self.regionalize:
            self._regionalRatios = _reglmr([self._ratios], [self.num_years])
            self.paras = self.fitter(self._regionalRatios)
            self.paras["loc"] *= self._ratios[0]
            self.paras["scale"] *= self._ratios[0]
        else:
            self.paras = self.fitter(self._ratios)

    def rvs(self, size: ArrayLike = 1) -> np.ndarray:
        """Random variates of given type

        Parameters
        ----------
        size: ArrayLike
            Defining number of random variate, defaults to 1.

        Returns
        -------
        np.ndarray
            Random variates of given `size`.
        """
        return self.model.rvs(size=size, **self.paras)

    def ppf(self, q: ArrayLike) -> Union[float, np.ndarray]:
        """Percent point function (inverse of `cdf`) at q of the given RV

        Parameters
        ----------
        q: ArrayLike
            lower tail probability

        Returns
        -------
        Union[float, np.ndarray]
            Percent point function evaluated at q
        """
        return self.model.ppf(q, **self.paras)

    def cdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Cumulative distribution function of the given

        Parameters
        ----------
        x : ArrayLike
            quantiles

        Returns
        -------
        Union[float, np.ndarray]
            Cumulative distribution function evaluated at `x`
        """
        return self.model.cdf(x, **self.paras)

    def pdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """
        Probability density function at x of the given RV.

        Parameters
        ----------
        x : ArrayLike
            quantiles

        Returns
        -------
        Union[float, np.ndarray]
            Probability density function evaluated at `x`
        """
        return self.model.cdf(x, **self.paras)


def plotting_position(
    M: Union[int, np.int64],
    nyrs: Union[int, np.int64],
    A: Union[float, np.float64] = 0.4,
) -> float | np.float64:
    """
    Empirical  return  period  (plotting  position)  is  calculated  by  the  general  equation
    first proposed by Gringorten (1963) and analyzed by Cunnane (1978). From `SWMM4 Manual`_ pg 299.

    .. _SWMM4 Manual: http://www.dynsystem.com/netstorm/docs/swmm4manuals.pdf

    Parameters
    ----------
    M: float
        Rank of event in descending order

    nyrs: float
        Number of years in dataset

    A: float, optional
        Emperical coefficient. A value of A = 0 gives the familiar Weibull
        plotting position (Gumbel, 1958), defaults to 0.4

    Returns
    -------
    float
        Return period in years
    """
    return (nyrs + 1 - 2 * A) / (M - A)


def plotting_positionInv(T: int, nyrs: int, A: float = 0.4) -> float:
    """
    Empirical  return  period  (plotting  position)  is  calculated  by  the  general  equation
    first proposed by Gringorten (1963) and analyzed by Cunnane (1978). From `SWMM4 Manual`_ pg 299.

    .. _SWMM4 Manual: http://www.dynsystem.com/netstorm/docs/swmm4manuals.pdf

    Parameters
    ----------
    T : float
        ARI in yrs

    nyrs : float
        Number of years in dataset

    A : float, optional
        Emperical coefficient. A value of A = 0 gives the familiar Weibull
        plotting position (Gumbel, 1958), defaults to 0.4

    Returns
    -------
    float
        Rank of storm in PDS
    """
    return np.round((nyrs + 1 - 2 * A + T * A) / T, 0).astype(int)


def _samlmr(x: np.ndarray, nmom: int = 5, A: float = -0.35, B: float = 0.0) -> list:
    N = len(x)
    SMALL = 1e-5

    if nmom > 20 or nmom > N:
        raise Exception(
            f"Parameter nmom ({nmom}) is invalid, it must be <20 and greater than N ({N})"
        )
    if A <= -1 or A >= B:
        raise Exception(
            "Plotting-position parameters invalid, A must by > -1 and B must be < A"
        )
    Asum = np.zeros(20)
    if np.abs(A) > SMALL or np.abs(B) > SMALL:
        Asum[0] = x.sum()
        ppos = (np.arange(1, N + 1, 1) + A) / (N + B)
        for i in range(1, nmom):
            Asum[i] = (x * ppos**i).sum()
        Asum[:nmom] = Asum[:nmom] / N

    else:
        Asum[0] = x.sum()
        Z = np.arange(1, N + 1, 1)
        inp = np.ones(len(Z))
        for i in range(1, nmom):
            inp = inp * (Z - i)
            Asum[i] = (x * inp).sum()

        Y = N
        Z = N
        Asum[0] = Asum[0] / Z
        for j in range(1, nmom):
            Y = Y - 1
            Z = Z * Y
            Asum[j] = Asum[j] / Z

    K = nmom
    P0 = -1.0 if nmom % 2 == 1 else 1.0
    for KK in range(2, nmom + 1):
        AK = K
        P0 = -P0
        P = P0
        Temp = P * Asum[0]

        for i in range(K - 1):
            I = i + 1
            AI = I
            P = P * (AK + AI - 1) * (AI - AK) / (AI**2)
            Temp = Temp + P * Asum[i + 1]
        Asum[K - 1] = Temp
        K -= 1
    Xmom = [Asum[0]]
    if nmom == 1:
        return Xmom
    Xmom.append(Asum[1])

    if np.abs(Asum[1]) < SMALL:
        raise Exception("All data values equal")
    if nmom == 2:
        return Xmom

    for mom in Asum[2:]:
        Xmom.append(mom / Asum[1])

    return Xmom[:3]


def _reglmr(lmom_ratios: Union[list, np.ndarray], weights: list) -> np.ndarray:
    try:
        xMom = np.asarray(lmom_ratios)
    except Exception as e:
        print("Failed casting lmom_ratios to array", str(e))
    if len(xMom.shape) != 2:
        raise Exception(
            f"lmom_ratios array must have 2 dimentions not {len(xMom.shape)}"
        )

    nSite = 1
    SMALL = 1e-5
    nMom = len(xMom[0])
    Rmom = np.zeros(nMom)
    WSum = 0
    for iSite in range(nSite):
        sMean = xMom[iSite][0]
        if abs(sMean) < SMALL:
            raise ValueError(f"ZERO MEAN AT SITE {iSite}")
        W = weights[iSite]
        WSum = WSum + W
        Rmom[1] = Rmom[1] + W * xMom[iSite][1] / sMean
        if nMom == 2:
            break
        Rmom[2:nMom] = Rmom[2:nMom] + W * xMom[iSite][2:nMom]
    if WSum <= 0:
        raise ValueError("Sum of weights is negative or zero")
    Rmom[0] = 1
    Rmom[1] = Rmom[1] / WSum
    if nMom == 2:
        return Rmom
    Rmom[2:nMom] = Rmom[2:nMom] / WSum
    return Rmom

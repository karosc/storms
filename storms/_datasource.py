import asyncio
import logging
import sys
from types import SimpleNamespace
from typing import Any, List, Sequence, Union

import pandas as pd
from aiohttp import ClientSession, TCPConnector, TraceRequestStartParams, TraceConfig
from aiohttp_retry import ExponentialRetry, RetryClient
from numpy import ndarray
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from storms._utils import datetime_like, async_runner
from tqdm.autonotebook import tqdm

handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(handlers=[handler])
logger = logging.getLogger(__name__)

sync_retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[502, 503, 504])


async def on_request_start(
    session: ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: TraceRequestStartParams,
    # retry_options: ExponentialRetry
) -> None:
    current_attempt = trace_config_ctx.trace_request_ctx["current_attempt"]
    print(f"attempt {current_attempt -1}")
    # if retry_options.attempts <= current_attempt:
    #     logger.warning("Wow! We are in last attempt")


def flatten(t):
    return [item for sublist in t for item in sublist]


class _DataSource(object):
    @staticmethod
    def map():
        raise NotImplementedError

    def __init__(self, ID: str):
        self.ID = ID
        self.URL = ""
        self.progress: bool = True

    def _increment_progress(self, task=None):
        if self.progress:
            self.bar.update(1)

    def _init_progress(self, totalDuration):
        if self.progress and not hasattr(self, "bar"):
            self.bar = tqdm(total=totalDuration)

    def _update_progress_description(self, description):
        if self.progress:
            self.bar.set_description(description)

    def _close_progress(self):
        if self.progress:
            self.bar.close()
            del self.bar

    def request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool = True,
        aSync: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        self.progress = progress

        if aSync:
            data = async_runner(
                self._async_request_dataframe, start, end, process_data, **kwargs
            )

        # if not running async
        else:
            data = self._sync_request_dataframe(start, end, process_data, **kwargs)

        return data

    def _request_url(
        self,
        start: datetime_like,
        end: datetime_like,
        datatype: Union[str, Sequence[str]],
    ) -> Any:
        return NotImplementedError

    def _sync_request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool,
        pull_freq: str,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _sync_request_data_series(
        self, start: datetime_like, end: datetime_like, pull_freq: str = "AS", **kwargs
    ) -> pd.DataFrame:
        # convert string inputs to datetime-like
        dStart = pd.to_datetime(start)
        dEnd = pd.to_datetime(end)

        # yearly pulls from start of year
        # was getting some weird API behaviour when given
        # requesting date ranges that span the new year
        # freq = "AS"

        # set up annual date range, appending start date which may not be Jan 1
        dRange = pd.date_range(dStart, dEnd, freq=pull_freq).insert(0, dStart)
        # append end date, which may not be Jan 1 and drop dups in case it was
        dRange = dRange.insert(len(dRange), dEnd).drop_duplicates()
        self._init_progress(totalDuration=len(dRange) - 1)
        self._update_progress_description("Downloading")
        with Session() as session:
            session.mount("http://", HTTPAdapter(max_retries=sync_retries))
            session.mount("https://", HTTPAdapter(max_retries=sync_retries))

            data = []
            for i in range(len(dRange) - 1):
                data.append(
                    self._sync_request_data(
                        start=dRange[i],
                        end=dRange[i + 1] - pd.Timedelta("1 minute"),
                        session=session,
                        **kwargs,
                    )
                )
                self._increment_progress()
        return data

    def _sync_request_data(
        self, start: datetime_like, end: datetime_like, session: Session
    ) -> Union[ndarray, str, List[dict]]:
        raise NotImplementedError

    async def _async_request_dataframe(
        self,
        start: datetime_like,
        end: datetime_like,
        process_data: bool,
        pull_freq: str,
        conn_limit: int,
    ) -> pd.DataFrame:
        raise NotImplementedError

    async def _async_request_data_series(
        self,
        start: datetime_like,
        end: datetime_like,
        pull_freq: str = "AS",
        conn_limit: int = 30,
        retry_options: ExponentialRetry = ExponentialRetry(
            attempts=5, start_timeout=0.1
        ),
        **kwargs,
    ) -> pd.DataFrame:
        # convert string inputs to datetime-like
        dStart = pd.to_datetime(start)
        dEnd = pd.to_datetime(end)

        # yearly pulls from start of year
        # was getting some weird API behaviour when given
        # requesting date ranges that span the new year
        # set up annual date range, appending start date which may not be Jan 1
        dRange = pd.date_range(dStart, dEnd, freq=pull_freq).insert(0, dStart)
        # append end date, which may not be Jan 1 and drop dups in case it was
        dRange = dRange.insert(len(dRange), dEnd).drop_duplicates()

        self._init_progress(len(dRange) - 1)
        # set up request session
        async with TCPConnector(limit=conn_limit) as connector:
            # async with aiohttp.ClientSession(connector=connector) as session:
            # trace_config = TraceConfig()
            # trace_config.on_request_start.append(on_request_start)
            async with RetryClient(
                connector=connector,
                retry_options=retry_options,
                # trace_configs=[trace_config],
            ) as session:
                # create async tasks
                # https://docs.python.org/3/library/asyncio-task.html#creating-tasks
                tasks = [
                    asyncio.ensure_future(
                        self._async_request_data(
                            start=dRange[i],
                            end=dRange[i + 1] - pd.Timedelta("1 minute"),
                            session=session,
                            **kwargs,
                        )
                    )
                    for i in range(len(dRange) - 1)
                ]
                for task in tasks:
                    task.add_done_callback(self._increment_progress)
                # run tasks concurrently and return list of responses
                # https://docs.python.org/3/library/asyncio-task.html#running-tasks-concurrently
                self._update_progress_description("Downloading")
                return await asyncio.gather(*tasks)

    async def _async_request_data(
        self, start: datetime_like, end: datetime_like, session: RetryClient
    ) -> Union[ndarray, str, List[dict]]:
        raise NotImplementedError

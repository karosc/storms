from functools import lru_cache, wraps
from time import monotonic_ns
from datetime import datetime
from pandas._libs.tslibs.timestamps import Timestamp
from typing import Callable, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor


datetime_like = Union[str, datetime, Timestamp]


def timed_lru_cache(
    _func=None, *, seconds: int = 600, maxsize: int = 128, typed: bool = False
):
    """
    From https://realpython.com/lru-cache-python/

    Extension of functools lru_cache with a timeout

    Parameters
    ----------
    seconds (int): Timeout in seconds to clear the WHOLE cache, default = 10 minutes
    maxsize (int): Maximum Size of the Cache
    typed (bool): Same value of different type will be a different entry

    """

    def wrapper_cache(f):
        f = lru_cache(maxsize=maxsize, typed=typed)(f)
        f.delta = seconds * 10**9
        f.expiration = monotonic_ns() + f.delta

        @wraps(f)
        def wrapped_f(*args, **kwargs):
            if monotonic_ns() >= f.expiration:
                f.cache_clear()
                f.expiration = monotonic_ns() + f.delta
            return f(*args, **kwargs)

        wrapped_f.cache_info = f.cache_info
        wrapped_f.cache_clear = f.cache_clear
        return wrapped_f

    # To allow decorator to be used without arguments
    if _func is None:
        return wrapper_cache
    else:
        return wrapper_cache(_func)


def async_runner(func: Callable, *args, **kwargs):
    # try to get existing running loop
    # this gets around issues with jupyter
    # which keeps its own loop running
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # If loop is running (i.e. in a jupyter session)
    # Run async routine in a separate event loop on
    # a separate thread
    # https://gist.github.com/minrk/feaf2022bf43d1a94e03ceaf9a4ef355
    if loop and loop.is_running():
        with ThreadPoolExecutor(1) as pool:
            result = pool.submit(lambda: asyncio.run(func(*args, **kwargs))).result()

    # if not in jupyter session, run event loop in current thread
    else:
        result = asyncio.run(func(*args, **kwargs))

    return result

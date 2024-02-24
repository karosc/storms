#!/usr/bin/env python

"""Tests for `storms` package."""

import os
from time import time

import pytest
from pandas import Timestamp

from storms import Raingage

dir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="module")
def BostonRaingage():
    """
    Load boston historical data

    """
    print("loading data")
    start = time()
    gage = Raingage.from_ff(
        os.path.join(dir, "data/Logan.1h"), freq="h", latlon=(42.3606, -71.0097)
    )
    gage.find_events()
    gage.find_intervals(periods=[24], constraints=[1])
    print(
        f"{time()-start} seconds to load data, get events, and pull 24-hour intervals"
    )
    return gage


def testLargestEvent(BostonRaingage):
    """Check largest event is found properly"""
    largest24hrEvent = [
        11.94,
        Timestamp("1955-08-18 06:00:00"),
        41482,
        41518,
        40.0,
        1.41,
        36,
    ]
    assert (
        BostonRaingage.events.set_index("event_num")
        .sort_values("event_total")
        .iloc[-1]
        .to_list()
        == largest24hrEvent
    )

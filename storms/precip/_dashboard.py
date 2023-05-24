from typing import Sequence
import pandas as pd
import json
from jinja2 import Environment, FileSystemLoader
from storms.precip import datasets as pds
import pathlib

_HERE_ = pathlib.Path(__file__).parent


# handy function for converting hour differentials
# to humanized text
def yrs2text(val):
    if int(val) > 0:
        return f"{val:,.0f} year"
    elif int(val * 12) > 0:
        return f"{val * 12:,.0f} month"
    elif int(val * 52) > 0:
        return f"{val * 52:,.0f} week"
    else:
        return f"{val * 365:,.0f} day"


def rg_2_dash(
    file_path: str,
    rg,
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
    meta = rg.meta.copy() if rg.meta else {}

    meta["start_date"] = pd.to_datetime(rg.datetime.min()).strftime("%Y-%m-%d")
    meta["endDate"] = pd.to_datetime(rg.datetime.max()).strftime("%Y-%m-%d")
    meta["nRecords"] = f"{len(rg.data):,}"
    meta["interEvent"] = "6 hours"
    maxDroughtIDX = (rg.datetime[1:] - rg.datetime[:-1]).argmax()
    meta[
        "largestDrought"
    ] = f"{str(pd.to_timedelta(rg.datetime[maxDroughtIDX+1]-rg.datetime[maxDroughtIDX]))} before {pd.to_datetime(rg.datetime[maxDroughtIDX+1]).strftime('%Y-%m-%d')}"
    meta["steps_per_hour"] = int(pd.Timedelta("1h") / rg.freq)
    meta["LAT"], meta["LON"] = rg.latlon

    idf = rg.IDF(idf_periods)

    idf.sort_index(axis=1, ascending=False, inplace=True)
    # hour columns to text
    idf.columns = [yrs2text(i) for i in idf.columns]
    # dump idf table to dict
    idfDict = idf.to_dict("list")
    # add durations to dict
    idfDict["durations"] = idf.index.to_list()

    atlas = rg.pfds.sort_index(axis=1, ascending=False)
    # hour columns to text
    atlas.columns = [yrs2text(i) for i in atlas.columns]
    # dump atlas table to dict
    atlasDict = atlas.to_dict("list")
    # add durations to dict
    atlasDict["durations"] = atlas.index.to_list()

    # create json dict and add data,dataetime, and idf
    js = {}
    js["data"] = rg.data.to_list()
    js["datetime"] = rg.data.index.strftime("%Y-%m-%d %H:%M:%s").to_list()
    js["idf"] = idfDict
    js["atlas"] = atlasDict
    # pivot intervals table on duration
    t = rg.intervals.pivot(
        index="event_num",
        columns="duration",
        values=["total", "event_start_index", "event_end_index", "noaaARI", "ARI"],
    )
    # swap levels to put param as top level header
    t.columns = t.columns.swaplevel(0, 1)

    # copy and slice largest events
    events = rg.events.copy()
    # convert datetime to string
    events.start_date = events.start_date.dt.strftime("%Y-%m-%d %H:%M:%s")

    # index by event number and slice out important columns
    events.set_index("event_num", inplace=True)
    events = events.loc[
        :,
        [
            "event_total",
            "start_date",
            "event_start_index",
            "event_end_index",
            "hours_duration",
        ],
    ]

    # add multi columns to events table
    events.columns = pd.MultiIndex.from_tuples(
        [
            ("event", col)
            for col in [
                "total",
                "start_date",
                "event_start_index",
                "event_end_index",
                "hours_duration",
            ]
        ]
    )
    # join in intervals to events table
    out = events.join(t)

    # find max ARI for each event and add to table
    out.loc[:, ("event", "ARI")] = out.xs("ARI", level=1, axis=1).max(axis=1)
    out.loc[:, ("event", "noaaARI")] = out.xs("noaaARI", level=1, axis=1).max(axis=1)

    # sort by event total
    out.sort_values(("event", "total"), ascending=False, inplace=True)
    # dump to dict
    outDict = {
        level: out.xs(level, level=1, axis=1).to_dict("list")
        for level in out.columns.levels[1]
    }
    # add in event_num index
    outDict["event_num"] = out.index.to_list()
    # add large events table to json dict
    js["events"] = outDict

    # calculate the AMS for the dataset
    # group events
    gpd = out.groupby(pd.to_datetime(out.loc[:, ("event", "start_date")]).dt.year)
    # ams=gpd.max().xs('total',level=1,axis=1).reset_index().rename({('event', 'start_date'):'start_date'},axis=1).to_dict('list')

    amsData = out.loc[gpd.idxmax(numeric_only=True).loc[:, ("event", "total")]]
    eventstart_dates = pd.to_datetime(
        amsData.loc[:, ("event", "start_date")]
    ).dt.strftime("%m/%d/%Y")
    ams = amsData.xs("total", level=1, axis=1).join(
        eventstart_dates.rename("start_date")
    )
    # .to_dict('list')

    js["ams"] = ams.to_dict(orient="list")

    js["meta"] = meta

    jsstr = json.dumps(js, separators=(",", ":"))

    jsstr = jsstr.replace("NaN", "null").replace("Infinity", "null")

    env = Environment(loader=FileSystemLoader(str(_HERE_.absolute())))
    template = env.get_template("index.html")

    # render output html
    output_from_parsed_template = template.render(data=jsstr)
    # save html to file
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write(output_from_parsed_template)
    if dump_json:
        with open(file_path.replace("html", "json"), "w", encoding="utf-8") as fh:
            fh.write(jsstr)

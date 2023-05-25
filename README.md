<img src="https://raw.githubusercontent.com/karosc/storms/main/docs/storms_logo_with_text.png" width=100%></img>


# storms: a simple and effective storm event analysis toolkit
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOCS:Pages](https://github.com/karosc/storms/actions/workflows/documentation.yaml/badge.svg)](https://www.karosc.com/storms/)
[![Latest PyPI version](https://img.shields.io/pypi/v/storms.svg)](https://pypi.python.org/pypi/storms/)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
<!-- [![PyPI Monthly Downloads](https://img.shields.io/badge/dynamic/json.svg?label=Downloads&url=https%3A%2F%2Fpypistats.org%2Fapi%2Fpackages%2Fstorms%2Frecent&query=%24.data.last_month&colorB=green&suffix=%20last%20month)](https://pypi.python.org/pypi/storms/)
 -->

## Features

- Download hourly rainfall timeseries from [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) 
- Bin rainfall data timeseries into descrete events
- Develop a partial duration series of rainfall from metorologically independent events for any duration
- Calculate the ARI of historical events at various timeseries using GEV or plotting position
- Interpolate NOAA Atlas 14 ARI for events based on station location and event depth
- Provide pandas DataFrame interface to all these data 

## Installation


```sh
#pip with git
pip install git+http://github.com/karosc/storms.git
```

```sh
#pip without git
pip install http:/github.com/karosc/storms/archive/main.zip
```

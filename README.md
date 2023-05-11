<img src="docs/storms_logo_with_text.png" width=100%></img>

---

<!--
[![image](https://img.shields.io/pypi/v/storms.svg)](https://pypi.python.org/pypi/storms)
[![image](https://img.shields.io/travis/karosc/storms.svg)](https://travis-ci.com/karosc/storms)
[![Documentation Status](https://readthedocs.org/projects/storms/badge/?version=latest)](https://storms.readthedocs.io/en/latest/?version=latest)
-->
# storms: a simple and effective storm event analysis toolkit
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOCS:Pages](https://github.com/karosc/storms/actions/workflows/documentation.yaml/badge.svg)](https://www.karosc.com/storms/)


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

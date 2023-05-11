"""
A module for analysis of rain gage data. Particularly for event by event analysis
and calculation of storm ARIs
"""
from storms.precip.raingage import Raingage, get_pfds
from storms.precip.network import Network
import storms.precip.datasets as datasets

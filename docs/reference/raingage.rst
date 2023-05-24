=========
Raingage
=========
.. currentmodule:: storms

The Raingage class provides an interface for analyzing water equivalent precipitation
observed at a single raingage. Data can be read from NOAA data sources or local files.   

Constructors
~~~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: api/

   Raingage
   Raingage.from_GlobalHourly
   Raingage.from_asos
   Raingage.from_ff
   Raingage.from_csv
   Raingage.from_swmm

Properties
~~~~~~~~~~~
.. autosummary::
   :nosignatures:
   :toctree: api/

   Raingage.freq
   Raingage.ts_hours
   Raingage.num_years
   Raingage.events
   Raingage.intervals

Methods
~~~~~~~~
   
.. autosummary::
   :nosignatures:
   :toctree: api/
   
   Raingage.disaggregate
   Raingage.find_events
   Raingage.find_intervals
   Raingage.find_ari
   Raingage.IDF
   Raingage.get_event
   Raingage.get_event_by_rank
   Raingage.get_noaa_ari
   



   
   


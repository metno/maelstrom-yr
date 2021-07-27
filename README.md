## maelstrom-a1

A dataset plugin for climetlab for the dataset maelstrom-a1.

## Datasets description

Contains gridded weather forecasts and analyses (truth) for the Nordics.

## Using climetlab to access the data

The data can be loaded by the climetlab package (https://github.com/ecmwf/climetlab). The dataset has the
following arguments:
- size: Which dataset to load (currently 5GB is supported, but in the future a 5TB dataset will be added)
- parameter: Which predictand to load (currently "air_temperature" is supported)
- dates: If left blank, the whole dataset is loaded. Otherwise, provide a list of dates in "YYYY-MM-DD"
format to load a subset

Here is an example of how to load the data:
```
!pip install climetlab climetlab_maelstrom_a1
import climetlab as cml
ds = cml.load_dataset("maelstrom-a1", size="5GB", parameter="air_temperature", dates=['2020-06-29'])
ds.to_xarray()
```

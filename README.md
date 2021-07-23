## maelstrom-a1

A dataset plugin for climetlab for the dataset maelstrom-a1.


Features
--------

In this README is a description of how to get the maelstrom-a1.

## Datasets description

There are two datasets: 

### 1 : `a1`


### 2
TODO


## Using climetlab to access the data (supports grib, netcdf and zarr)

See the demo notebooks here (https://github.com/ecmwf-lab/climetlab_maelstrom_a1/notebooks

https://github.com/ecmwf-lab/climetlab_maelstrom_a1/notebooks/demo_a1.ipynb
[nbviewer] (https://nbviewer.jupyter.org/github/climetlab_maelstrom_a1/blob/main/notebooks/demo_a1.ipynb) 
[colab] (https://colab.research.google.com/github/climetlab_maelstrom_a1/blob/main/notebooks/demo_a1.ipynb) 

The climetlab python package allows easy access to the data with a few lines of code such as:
```

!pip install climetlab climetlab_maelstrom_a1
import climetlab as cml
ds = cml.load_dataset("maelstrom-a1", size="5GB", "air_temperature", dates=['2020-06-29'])
ds.to_xarray()
```

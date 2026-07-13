# API Examples

`ml_starter_geocube_api.py` demonstrates how to use the GeoCube-ML Python API to:

- discover collection roots under a workspace
- print collection, cube, grid, and layer metadata
- load selected layers such as `soil_ph` and `soil_soc`
- stack selected layers into a `feature, y, x` xarray object for ML workflows
- create quick spatial plots when `matplotlib` is installed

List available collections:

```bash
python tests/examples/ml_starter_geocube_api.py \
  --search-root /chrysaor/remotesensing/jbk/climate/ngee/src \
  --no-plot
```

Load and plot selected layers:

```bash
python tests/examples/ml_starter_geocube_api.py \
  --search-root /chrysaor/remotesensing/jbk/climate/ngee/src \
  --collection-root /chrysaor/remotesensing/jbk/climate/ngee/src/arctic_geocubes \
  --cube-name arctic_30sec \
  --layers soil_ph soil_soc \
  --plot-output /tmp/arctic_30sec_soil_layers.png
```

The load example expects the selected layers to exist in the cube.

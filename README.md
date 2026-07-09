# GeoCube-ML

> A lightweight framework for building **analysis-ready ancillary data cubes** for ecological, environmental, and geospatial machine learning.

GeoCube-ML converts collections of GeoTIFF and NetCDF layers into standardized, reusable **Zarr data cubes** that can be queried with **xarray** and processed lazily with **Dask**.

It is designed for workflows where many gridded ancillary datasets are repeatedly reprojected, resampled, clipped, stacked, and extracted at ecological point observation locations.

Instead of preprocessing the same rasters for every modeling project, GeoCube lets you process them once, store them in an analysis-ready format, and selectively load only the layers needed for Random Forests, neural networks, XGBoost, spatial prediction, or exploratory analysis.

---

## Why GeoCube-ML?

Ecological upscaling and spatial prediction workflows often require many predictor layers:

* elevation
* slope
* aspect
* soil properties
* land cover
* tree cover
* NDVI
* precipitation
* temperature
* geology
* hydrology
* remotely sensed indices

A typical workflow repeatedly performs the same preprocessing:

1. Read source rasters
2. Reproject to a common CRS
3. Resample to a common resolution
4. Clip to a study region
5. Align all layers to the same grid
6. Mask missing values
7. Extract predictor values at observation points
8. Build machine learning tables

GeoCube separates the expensive preprocessing step from model development.

Raw rasters are ingested once into a reusable cube. Modeling workflows then query, load, and extract layers directly from the cube.

---

## Core design

A GeoCube-ML project is organized as a **CubeCollection**.

A collection can contain many region- and resolution-specific cubes.

```text
my_geocube_collection/

  collection.json

  grids/
    global_1km.json
    amazon_1km.json
    amazon_250m.json

  cubes/
    global_1km.zarr
    amazon_1km.zarr
    amazon_250m.zarr

  catalog/
    catalog.json
```

Each individual cube has:

* one CRS
* one extent
* one resolution
* one grid
* many predictor layers

This keeps every cube internally consistent.

---

## CubeCollection abstraction

The `CubeCollection` is the top-level registry for managing many cubes.

It records:

* cube name
* cube path
* grid path
* region label
* resolution label
* description

Example cubes:

```text
global_5km
global_1km
north_america_1km
amazon_1km
amazon_250m
europe_100m
```

This allows the same project to support multiple analysis scales without mixing incompatible grids in one Zarr store.

---

## Data model

Each cube is stored as an `xarray.Dataset` in Zarr format.

Example:

```text
amazon_1km.zarr

Dimensions:
  y
  x

Variables:
  elevation
  slope
  soil_ph
  soil_clay
  annual_precip
  mean_temperature
  landcover
  ndvi
```

All variables share the same:

* `x` coordinates
* `y` coordinates
* CRS
* resolution
* extent
* array shape

This makes the cube immediately suitable for stacking predictors.

---

## Coordinate reference system

GeoCube defaults to WGS84 latitude/longitude:

```text
EPSG:4326
```

Note that EPSG:4326 is the standard WGS84 lat/lon CRS.

---

## Region-specific alignment

Every ingested layer is forced onto the target cube grid.

This means:

* data outside the target region is excluded
* missing data inside the target region is filled with a configured missing value
* output shape always matches the cube grid
* output coordinates always match the cube grid
* resolution always matches the cube grid
* CRS always matches the cube grid

This ensures that a region-specific cube truly represents only the defined region and resolution.

---

## Provenance tracking

GeoCube-ML tracks provenance for every layer.

Each ingested layer stores provenance metadata in the Zarr variable attributes and in the STAC catalog.

Provenance includes:

* source path
* source file SHA256 checksum
* source NetCDF variable, if applicable
* layer name
* cube name
* grid name
* region
* CRS
* extent
* resolution
* resampling method
* source nodata value
* output missing value
* ingest timestamp
* software version
* Python version

This makes each layer reproducible and auditable.

Example provenance:

```json
{
  "source_path": "/data/raw/soil_ph.tif",
  "source_sha256": "abc123...",
  "source_variable": null,
  "layer_name": "soil_ph",
  "cube_name": "amazon_1km",
  "grid_name": "amazon_1km",
  "region": "amazon",
  "crs": "EPSG:4326",
  "resolution_degrees": 0.0083333333,
  "extent": [-80, -25, -45, 10],
  "resampling": "bilinear",
  "source_nodata": -9999,
  "missing_value": -9999,
  "ingested_at_utc": "2026-07-06T12:00:00+00:00",
  "software": "geocube-ml",
  "software_version": "0.1.0",
  "python_version": "3.12.4"
}
```

---

## STAC catalog

GeoCube-ML maintains a lightweight STAC catalog alongside the cubes.

Each layer is represented as a STAC Item with assets for:

* the Zarr cube
* the original source file

The STAC properties include:

* layer name
* cube name
* region
* grid name
* resolution
* CRS
* source path
* provenance metadata

This makes the cube discoverable without opening every Zarr store.

---

## Installation

For development:

```bash
git clone https://github.com/your-org/geocube-ml
cd geocube-ml
pip install -e .
```

Example dependencies:

```toml
[project]
name = "geocube-ml"
version = "0.1.0"
dependencies = [
  "xarray",
  "rioxarray",
  "rasterio",
  "dask",
  "zarr",
  "netcdf4",
  "pystac",
  "shapely",
  "geopandas",
  "pandas",
  "numpy",
  "typer"
]

[project.scripts]
geocube-ml = "geocube_ml.cli:app"
```

---

## Quick start

Initialize a collection:

```bash
geocube-ml collection-init my_collection
```

Add a cube:

```bash
geocube-ml collection-add-cube my_collection \
  --name amazon_1km \
  --region amazon \
  --resolution-label 1km \
  --resolution 0.0083333333 \
  --xmin -80 \
  --ymin -25 \
  --xmax -45 \
  --ymax 10 \
  --crs EPSG:4326 \
  --description "Amazon basin 1 km ancillary predictor cube"
```

Ingest a GeoTIFF:

```bash
geocube-ml collection-ingest my_collection soil_ph.tif \
  --cube-name amazon_1km \
  --layer soil_ph \
  --resampling bilinear \
  --missing-value -9999
```

Ingest a NetCDF variable:

```bash
geocube-ml collection-ingest my_collection climate.nc \
  --cube-name amazon_1km \
  --layer annual_precip \
  --variable precip \
  --resampling bilinear \
  --missing-value -9999
```

List layers:

```bash
geocube-ml collection-layers my_collection --cube-name amazon_1km
```

Inspect provenance:

```bash
geocube-ml provenance my_collection/cubes/amazon_1km.zarr soil_ph
```
---

# High-performance blockwise ingest

GeoCube-ML is designed to ingest datasets that are much larger than available system memory.

Instead of loading an entire raster into memory, GeoCube-ML performs ingest **one target cube block at a time** using Rasterio's `WarpedVRT`.

```
Source Raster
(GeoTIFF / NetCDF)

        │

        ▼

 Reproject to Target Grid
        │
        ▼

 Read Target Block
 (e.g. 512 × 512)

        │
        ▼

 Fill Missing Values

        │
        ▼

 Write Block to Zarr

        │
        ▼

 Repeat for Next Block
```

Peak memory usage is approximately one target block plus Rasterio's internal buffers rather than the entire raster.

This allows ingest of global datasets on modest workstations and login nodes.

---

# Choosing chunk sizes

Every cube defines a chunk size used for

- blockwise ingest
- Zarr storage
- Dask computation

Typical recommendations are

| Environment | Recommended Chunk Size |
|-------------|----------------------:|
| Laptop / login node | 256 × 512 |
| Desktop workstation | 512 × 512 |
| HPC compute node | 1024 × 1024 |

The default is

```json
"chunks": [512, 512]
```

Smaller chunks reduce peak memory usage.

Larger chunks generally improve sequential I/O performance but require more memory.

---

# Progress reporting

Long-running ingests display a progress bar.

Example

```text
Ingesting annual_precip

███████████████████████████████ 100%

756 / 756 blocks
```

This makes it easy to monitor ingest progress for very large rasters.

---

# Automatic layer statistics

During ingest GeoCube-ML computes summary statistics without performing a second pass over the data.

Statistics include

- valid pixel count
- missing pixel count
- total pixel count
- minimum
- maximum
- mean
- standard deviation

These statistics are stored in

- layer metadata
- cube manifest
- STAC catalog
- provenance record

Example

```json
{
  "statistics": {
    "valid_count": 134522003,
    "missing_count": 832913,
    "total_count": 135354916,
    "min": 3.72,
    "max": 9.84,
    "mean": 6.43,
    "std": 0.91
  }
}
```

This allows quick inspection of predictor layers without loading the cube.

---

# Automatic validation

Every layer is validated immediately after ingest.

Validation checks

- output dimensions
- target CRS
- grid alignment
- output data type
- unexpected NaN values
- chunk structure

If validation fails, ingest aborts with an informative error message.

---

# Cube manifest

Every Zarr cube contains a lightweight manifest describing its contents.

```
amazon_1km.zarr/

    .geocube_ml_manifest.json
```

The manifest records

- cube metadata
- creation time
- last update
- software version
- number of layers
- summary information for every layer

Example

```json
{
  "cube_version": "0.1.0",
  "layer_count": 37,
  "last_updated_utc": "...",

  "layers": {
    "soil_ph": {

      "region": "amazon",

      "resolution_degrees": 0.008333333,

      "statistics": {

        "valid_count": 12345678,

        "mean": 6.41
      }
    }
  }
}
```

The manifest provides a fast summary of a cube without scanning every Zarr variable.

View the manifest

```bash
geocube-ml manifest my_collection/cubes/amazon_1km.zarr
```

Validate the manifest

```bash
geocube-ml validate-manifest my_collection/cubes/amazon_1km.zarr
```

---

# Batch ingest

GeoCube-ML can ingest an entire directory of predictor rasters.

Example

```bash
geocube-ml collection-ingest-dir \
    my_collection \
    raw_predictors \
    --cube-name amazon_1km \
    --pattern "*.tif"
```

or

```bash
geocube-ml collection-ingest-dir \
    my_collection \
    climate \
    --cube-name amazon_1km \
    --pattern "*.nc" \
    --variable precip
```

Features include

- automatic layer naming
- progress reporting
- continue after failures
- summary report
- manifest updates
- STAC updates
- provenance tracking

Example output

```text
Batch ingest complete

✓ elevation

✓ slope

✓ soil_ph

✓ annual_precip

✗ ndvi
    Missing CRS information

Summary

4 successful

1 failed
```

---

# Incremental updates

GeoCube-ML is designed to efficiently maintain long-lived ancillary data collections.

Rather than rebuilding an entire cube, GeoCube-ML compares every source dataset against the provenance information already stored for each layer.

Each layer records

- source file path
- SHA256 checksum
- ingest timestamp
- software version
- processing parameters

Before ingesting a layer, GeoCube-ML determines whether anything has changed.

```
Source File

        │

        ▼

Compute SHA256

        │

        ▼

Compare with Manifest

        │

   ┌────┴─────┐

   │          │

Same      Different

   │          │

Skip     Re-ingest
```

Only modified datasets are reprocessed.

---

## Layer registry and version history

In addition to the cube manifest, GeoCube-ML maintains a persistent **layer registry** that records the complete history of every predictor layer.

Each cube contains:

```text
my_collection/

    cubes/

        amazon_1km.zarr/

            .geocube_ml_manifest.json

            .geocube_ml_layer_registry.json
```

The layer registry is the authoritative record of every ingest operation.

Unlike the cube manifest, which summarizes the current contents of a cube, the registry maintains a complete version history for each layer.

Each registry entry records:

* layer name
* version number
* ingest timestamp
* source dataset
* SHA256 checksum
* source variable
* cube and grid definition
* processing parameters
* provenance metadata
* summary statistics
* validation results

Example:

```json
{
  "layers": {
    "soil_ph": {
      "current_version": 3,
      "history": [
        {
          "version": 1,
          "ingested_at_utc": "...",
          "build_spec": {
            "source_sha256": "4af7..."
          }
        },
        {
          "version": 2,
          "ingested_at_utc": "...",
          "build_spec": {
            "source_sha256": "9cb2..."
          }
        },
        {
          "version": 3,
          "ingested_at_utc": "...",
          "build_spec": {
            "source_sha256": "d8fa..."
          }
        }
      ]
    }
  }
}
```

The registry provides a complete audit trail for every predictor layer and makes it possible to determine exactly which source data and processing parameters were used for any version of the cube.

---

## Incremental ingest

By default, GeoCube-ML performs **checksum-aware incremental ingest**.

Before processing a source dataset, GeoCube-ML compares the current build specification with the most recent registry entry.

The comparison includes:

* source dataset checksum
* source variable
* cube definition
* target grid
* CRS
* spatial extent
* resolution
* resampling method
* missing-value flag
* GeoCube-ML version

If none of these have changed, the layer is skipped automatically.

```text
Source Dataset

        │

        ▼

Calculate SHA256

        │

        ▼

Read Layer Registry

        │

   ┌────┴────┐

   │         │

Unchanged  Changed

   │         │

 Skip     Rebuild
```

This dramatically reduces update times for large collections where only a small number of datasets have changed.

---

## Update modes

GeoCube-ML supports four update modes.

| Mode                   | Description                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `checksum` *(default)* | Rebuild only layers whose source checksum or processing specification has changed. |
| `missing`              | Only ingest layers that do not already exist.                                      |
| `skip`                 | Skip all existing layers regardless of source changes.                             |
| `overwrite`            | Always rebuild every layer.                                                        |

Example:

```bash
geocube-ml collection-ingest-dir raw_predictors \
    --cube-name global_1km \
    --update-mode checksum
```

---

## Dry-run mode

Use `--dry-run` to preview what GeoCube-ML would do without modifying the cube.

```bash
geocube-ml collection-ingest-dir raw_predictors \
    --cube-name global_1km \
    --dry-run
```

Example output:

```text
Checking registry...

112 layers unchanged

5 layers require rebuilding

3 new layers detected

Dry run complete.

No changes were written.
```

Dry-run mode is particularly useful before launching long ingest jobs on HPC systems.

---

## Registry inspection

Display the complete registry:

```bash
geocube-ml registry my_collection/cubes/global_1km.zarr
```

Display the complete history for a single layer:

```bash
geocube-ml layer-history \
    my_collection/cubes/global_1km.zarr \
    soil_ph
```

This returns every recorded version of the layer together with its provenance, statistics, validation report, and processing configuration.

---

## Example maintenance workflow

Assume a cube currently contains 150 predictor layers.

After downloading updated climate data:

* 143 layers are unchanged
* 5 climate layers have been updated
* 2 new predictor layers have been added

Running:

```bash
geocube-ml collection-ingest-dir \
    raw_predictors \
    --cube-name global_1km
```

produces:

```text
Checking existing registry...

143 unchanged

5 updated

2 new

Processing...

✓ annual_precip

✓ annual_temperature

✓ vapour_pressure

✓ solar_radiation

✓ wind_speed

✓ tree_cover

✓ wetland_fraction

Done.

Processed: 7

Skipped: 143
```

Instead of rebuilding every predictor layer, GeoCube-ML only processes the datasets that have changed.

This makes maintaining large, long-lived ancillary data collections both efficient and fully reproducible.

---

# Update modes

GeoCube-ML supports several update modes.

| Mode | Description |
|------|-------------|
| `skip` | Skip existing layers regardless of source changes. |
| `checksum` *(default)* | Re-ingest only if the source checksum has changed. |
| `overwrite` | Always rebuild the layer. |
| `missing` | Only ingest layers that do not already exist. |

For example

```bash
geocube-ml collection-ingest-dir \
    predictors \
    raw_data \
    --cube-name global_1km \
    --update-mode checksum
```

Only datasets whose checksum differs from the previous ingest will be processed.

---

# Change detection

A layer is automatically rebuilt when any of the following changes:

- source dataset checksum
- NetCDF variable
- target grid
- cube resolution
- CRS
- resampling method
- missing-value flag
- GeoCube-ML version (optional)

This guarantees that every layer remains consistent with the current cube definition.

---

# Example update workflow

Suppose a cube contains 120 predictor layers.

```
120 Layers

│

├── 112 unchanged

├── 5 updated

└── 3 new
```

Running

```bash
geocube-ml collection-ingest-dir predictors raw_data \
    --cube-name global_1km
```

would result in

```
Checking existing layers...

112 unchanged

5 updated

3 new

Processing...

✓ annual_precip

✓ vapour_pressure

✓ tree_cover

✓ ndvi

✓ land_surface_temp

✓ soil_carbon

✓ population_density

Done.

Processed: 8

Skipped: 112
```

Instead of rebuilding all 120 layers, only the eight new or modified datasets are ingested.

---

# Force rebuilding

To rebuild every layer regardless of provenance

```bash
geocube-ml collection-ingest-dir \
    predictors \
    raw_data \
    --cube-name global_1km \
    --update-mode overwrite
```

---

# Dry-run mode

A dry-run allows inspection of pending updates without modifying the cube.

```bash
geocube-ml collection-ingest-dir \
    predictors \
    raw_data \
    --cube-name global_1km \
    --dry-run
```

Example output

```
Would update

annual_precip

soil_carbon

Would add

population_density

Would skip

112 unchanged layers
```

This is useful for verifying updates before launching a long ingest job.

---

# Provenance-aware reproducibility

Because every layer stores its provenance and checksum, GeoCube-ML can answer questions such as:

- Which source dataset produced this layer?
- Has this source dataset changed since ingest?
- Which layers need rebuilding?
- Which software version produced this predictor?
- Which resampling method was used?
- When was this layer last updated?

This makes GeoCube-ML suitable for maintaining large, evolving ancillary data repositories while preserving full reproducibility.

---

# Provenance and reproducibility

Every ingested layer contains complete provenance information.

Recorded metadata includes

- source dataset
- SHA256 checksum
- source variable
- ingest time
- target cube
- target grid
- region
- resampling method
- missing value
- software version
- Python version
- summary statistics
- validation status

This makes every predictor layer fully reproducible and auditable.

---

# End-to-end workflow

```
Raw GeoTIFFs / NetCDF

          │

          ▼

 Blockwise Ingest

          │

          ▼

 Reproject

          │

          ▼

 Clip to Region

          │

          ▼

 Align to Cube Grid

          │

          ▼

 Fill Missing Values

          │

          ▼

 Compute Statistics

          │

          ▼

 Validate Layer

          │

          ▼

 Update Manifest

          │

          ▼

 Update STAC

          │

          ▼

 Ready for Analysis
```

---

## Python API

Create a collection:

```python
from geocube_ml.collection import CubeCollection
from geocube_ml.grid import CubeGrid

collection = CubeCollection("my_collection")
```

Add a cube:

```python
grid = CubeGrid(
    name="amazon_1km",
    resolution=0.0083333333,
    xmin=-80,
    ymin=-25,
    xmax=-45,
    ymax=10,
    crs="EPSG:4326",
)

collection.add_cube(
    name="amazon_1km",
    grid=grid,
    region="amazon",
    resolution_label="1km",
    description="Amazon basin 1 km ancillary predictor cube",
)
```

Ingest layers:

```python
collection.ingest(
    cube_name="amazon_1km",
    source_path="soil_ph.tif",
    layer_name="soil_ph",
    resampling="bilinear",
    missing_value=-9999,
)

collection.ingest(
    cube_name="amazon_1km",
    source_path="landcover.nc",
    layer_name="landcover",
    variable="lc_class",
    resampling="nearest",
    missing_value=-9999,
)
```

List available layers:

```python
layers = collection.layers("amazon_1km")

for layer in layers:
    print(layer.name, layer.region, layer.resolution_degrees)
```

Load selected layers:

```python
ds = collection.load(
    cube_name="amazon_1km",
    layers=[
        "soil_ph",
        "landcover",
        "annual_precip",
    ],
)

print(ds)
```

Access one layer:

```python
soil = ds["soil_ph"]
```

Trigger computation with Dask:

```python
mean_soil_ph = soil.where(soil != -9999).mean().compute()
```

---

## Query provenance in Python

```python
from geocube_ml.cube import get_layer_provenance

prov = get_layer_provenance(
    "my_collection/cubes/amazon_1km.zarr",
    "soil_ph",
)

print(prov["source_path"])
print(prov["source_sha256"])
print(prov["resampling"])
```

---

## Extract point observations

```python
import geopandas as gpd
from geocube_ml.extract import extract_points

points = gpd.read_file("observations.gpkg")

training = extract_points(
    cube_path="my_collection/cubes/amazon_1km.zarr",
    points=points,
    layers=[
        "soil_ph",
        "landcover",
        "annual_precip",
    ],
)

training.to_parquet("training_data.parquet")
```

The resulting table can be used directly with:

* scikit-learn
* XGBoost
* LightGBM
* PyTorch
* TensorFlow
* statsmodels

---

## Recommended resampling choices

Use `nearest` for categorical data:

* land cover
* geology class
* soil class
* biome
* ecoregion

Use `bilinear` for continuous data:

* elevation
* temperature
* precipitation
* soil pH
* NDVI
* slope

Use `average` when aggregating from finer to coarser grids:

* tree cover percentage
* population density
* fractional vegetation cover

---

## Missing values

GeoCube-ML uses a configurable missing value, defaulting to:

```text
-9999
```

During ingest:

* NaN values are replaced with the missing value
* source nodata is respected when available
* gaps inside the target region are filled with the missing value
* data outside the target region is discarded

For analysis:

```python
da = ds["soil_ph"]
valid = da.where(da != -9999)
```

---

## CLI reference

Initialize a collection:

```bash
geocube-ml collection-init COLLECTION_ROOT
```

Add a cube:

```bash
geocube-ml collection-add-cube COLLECTION_ROOT \
  --name CUBE_NAME \
  --region REGION \
  --resolution-label LABEL \
  --resolution RESOLUTION \
  --xmin XMIN \
  --ymin YMIN \
  --xmax XMAX \
  --ymax YMAX
```

Ingest a source raster:

```bash
geocube-ml collection-ingest COLLECTION_ROOT SOURCE \
  --cube-name CUBE_NAME \
  --layer LAYER_NAME
```

List layers:

```bash
geocube-ml collection-layers COLLECTION_ROOT
```

List layers in one cube:

```bash
geocube-ml collection-layers COLLECTION_ROOT --cube-name CUBE_NAME
```

Show provenance:

```bash
geocube-ml provenance CUBE_PATH LAYER_NAME
```

---

## Example workflow

```bash
geocube-ml collection-init ecological_predictors

geocube-ml collection-add-cube ecological_predictors \
  --name amazon_1km \
  --region amazon \
  --resolution-label 1km \
  --resolution 0.0083333333 \
  --xmin -80 \
  --ymin -25 \
  --xmax -45 \
  --ymax 10

geocube-ml collection-ingest ecological_predictors elevation.tif \
  --cube-name amazon_1km \
  --layer elevation \
  --resampling bilinear

geocube-ml collection-ingest ecological_predictors landcover.tif \
  --cube-name amazon_1km \
  --layer landcover \
  --resampling nearest

geocube-ml collection-layers ecological_predictors --cube-name amazon_1km
```

Then in Python:

```python
from geocube_ml.collection import CubeCollection

collection = CubeCollection("ecological_predictors")

ds = collection.load(
    "amazon_1km",
    layers=["elevation", "landcover"],
)

print(ds)
```

---

## Design philosophy

GeoCube-ML is not intended to replace GIS software.

It is a lightweight bridge between raw geospatial rasters and machine learning workflows.

The goal is simple:

> Process once. Reuse everywhere. Load only what you need.

---

## Roadmap

Planned features:

* parallel batch ingest
* STAC search API
* layer versioning
* time dimension support
* temporal predictor cubes
* cloud object storage support
* COG export
* raster statistics summaries
* ML feature set definitions
* data validation reports
* cube comparison tools
* provenance diffing
* Dask cluster examples
* map preview utilities

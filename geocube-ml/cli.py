from pathlib import Path
import json
import typer
import geopandas as gpd

from .grid import CubeGrid
from .collection import CubeCollection
from .cube import list_layers, get_layer_provenance, load_layers
from .extract import extract_points

app = typer.Typer(help="Build and query analysis-ready ancillary Zarr cubes.")


@app.command("collection-init")
def collection_init(
    root: str = typer.Argument(..., help="Collection root directory."),
):
    collection = CubeCollection(root)
    collection.save()
    typer.echo(f"Initialized GeoCube collection at {root}")


@app.command("collection-add-cube")
def collection_add_cube(
    root: str = typer.Argument(...),
    name: str = typer.Option(...),
    region: str = typer.Option(...),
    resolution_label: str = typer.Option(...),
    resolution: float = typer.Option(...),
    xmin: float = typer.Option(...),
    ymin: float = typer.Option(...),
    xmax: float = typer.Option(...),
    ymax: float = typer.Option(...),
    crs: str = typer.Option("EPSG:4326"),
    description: str | None = typer.Option(None),
):
    collection = CubeCollection(root)

    grid = CubeGrid(
        name=name,
        resolution=resolution,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        crs=crs,
    )

    collection.add_cube(
        name=name,
        grid=grid,
        region=region,
        resolution_label=resolution_label,
        description=description,
    )

    typer.echo(f"Added cube {name} to collection {root}")


@app.command("collection-ingest")
def collection_ingest(
    root: str = typer.Argument(...),
    cube_name: str = typer.Option(...),
    source: str = typer.Argument(...),
    layer: str = typer.Option(...),
    variable: str | None = typer.Option(None),
    resampling: str = typer.Option("bilinear"),
    nodata: float | None = typer.Option(None),
    missing_value: float = typer.Option(-9999.0),
    overwrite: bool = typer.Option(True),
):
    collection = CubeCollection(root)

    collection.ingest(
        cube_name=cube_name,
        source_path=source,
        layer_name=layer,
        variable=variable,
        resampling=resampling,
        nodata=nodata,
        missing_value=missing_value,
        overwrite=overwrite,
    )

    typer.echo(f"Ingested {layer} into cube {cube_name}")


@app.command("collection-layers")
def collection_layers(
    root: str = typer.Argument(...),
    cube_name: str | None = typer.Option(None),
):
    collection = CubeCollection(root)
    layers = collection.layers(cube_name)

    if cube_name:
        for layer in layers:
            typer.echo(
                f"{layer.name} | cube={layer.cube_name} | region={layer.region} | "
                f"grid={layer.grid_name} | res={layer.resolution_degrees}"
            )
        return

    for cube, cube_layers in layers.items():
        typer.echo(f"\n{cube}")
        for layer in cube_layers:
            typer.echo(f"  - {layer.name} | region={layer.region} | res={layer.resolution_degrees}")


@app.command("provenance")
def provenance_cmd(
    cube: str = typer.Argument(...),
    layer: str = typer.Argument(...),
):
    prov = get_layer_provenance(cube, layer)
    typer.echo(json.dumps(prov, indent=2))

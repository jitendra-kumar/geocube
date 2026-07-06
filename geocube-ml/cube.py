from dataclasses import dataclass
import json
import xarray as xr


@dataclass
class LayerInfo:
    name: str
    dims: tuple
    dtype: str
    region: str
    cube_name: str
    grid_name: str
    resolution_degrees: str | float
    crs: str
    provenance: dict | None


def open_cube(cube_path: str, chunks: dict | str | None = "auto") -> xr.Dataset:
    return xr.open_zarr(cube_path, chunks=chunks)


def get_layer_provenance(cube_path: str, layer_name: str) -> dict:
    ds = open_cube(cube_path)
    if layer_name not in ds:
        raise KeyError(f"Layer not found: {layer_name}")

    raw = ds[layer_name].attrs.get("provenance")
    if not raw:
        return {}

    return json.loads(raw)


def list_layers(cube_path: str) -> list[LayerInfo]:
    ds = open_cube(cube_path)

    layers = []
    for name, da in ds.data_vars.items():
        raw_prov = da.attrs.get("provenance")
        provenance = json.loads(raw_prov) if raw_prov else None

        layers.append(
            LayerInfo(
                name=name,
                dims=tuple(da.dims),
                dtype=str(da.dtype),
                region=da.attrs.get("region", "unspecified"),
                cube_name=da.attrs.get("cube_name", "unknown"),
                grid_name=da.attrs.get("grid_name", "unknown"),
                resolution_degrees=da.attrs.get("resolution_degrees", "unknown"),
                crs=da.attrs.get("crs", "unknown"),
                provenance=provenance,
            )
        )

    return layers


def load_layers(
    cube_path: str,
    layers: list[str] | None = None,
    region: str | None = None,
    chunks: dict | str | None = "auto",
) -> xr.Dataset:
    ds = open_cube(cube_path, chunks=chunks)

    if region:
        matching = [
            name for name, da in ds.data_vars.items()
            if da.attrs.get("region") == region
        ]
        ds = ds[matching]

    if layers:
        missing = [layer for layer in layers if layer not in ds.data_vars]
        if missing:
            raise KeyError(f"Missing layers in cube: {missing}")
        ds = ds[layers]

    return ds

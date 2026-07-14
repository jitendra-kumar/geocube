from dataclasses import dataclass, asdict
from pathlib import Path
import json
import xarray as xr

from .grid import CubeGrid
from .cube import (
    delete_layer_data,
    get_layer_provenance,
    list_layers,
    load_layers,
    rename_layer_data,
)
from .ingest import ingest_layer, list_netcdf_variables
from .manifest import remove_layer_manifest, rename_layer_manifest
from .catalog import delete_stac_item, upsert_stac_item
from .registry import mark_layer_deleted, rename_layer_registry
from .storage import layer_group_name


@dataclass
class CubeRecord:
    name: str
    path: str
    grid_path: str
    region: str
    resolution_label: str
    description: str | None = None


class CubeCollection:
    """
    Manage multiple region/resolution-specific Zarr cubes.

    A collection is a lightweight registry that knows where each cube lives,
    what grid it uses, and how to discover/load layers across cubes.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.registry_path = self.root / "collection.json"
        self.cubes_dir = self.root / "cubes"
        self.grids_dir = self.root / "grids"
        self.catalog_dir = self.root / "catalog"

        self.cubes_dir.mkdir(parents=True, exist_ok=True)
        self.grids_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_dir.mkdir(parents=True, exist_ok=True)

        self.records = self._load_registry()

    def _load_registry(self) -> dict[str, CubeRecord]:
        if not self.registry_path.exists():
            return {}

        data = json.loads(self.registry_path.read_text())
        return {name: CubeRecord(**record) for name, record in data["cubes"].items()}

    def save(self) -> None:
        data = {
            "version": "0.1.0",
            "cubes": {
                name: asdict(record)
                for name, record in self.records.items()
            },
        }
        self.registry_path.write_text(json.dumps(data, indent=2))

    def add_cube(
        self,
        name: str,
        grid: CubeGrid,
        region: str,
        resolution_label: str,
        description: str | None = None,
    ) -> CubeRecord:
        grid_path = self.grids_dir / f"{name}.json"
        cube_path = self.cubes_dir / f"{name}.zarr"

        grid_path.write_text(json.dumps(asdict(grid), indent=2))

        record = CubeRecord(
            name=name,
            path=str(cube_path),
            grid_path=str(grid_path),
            region=region,
            resolution_label=resolution_label,
            description=description,
        )

        self.records[name] = record
        self.save()
        return record

    def get_cube(self, name: str) -> CubeRecord:
        if name not in self.records:
            raise KeyError(f"Cube not found in collection: {name}")
        return self.records[name]

    def load_grid(self, cube_name: str) -> CubeGrid:
        record = self.get_cube(cube_name)
        data = json.loads(Path(record.grid_path).read_text())
        return CubeGrid(**data)

    def ingest(
        self,
        cube_name: str,
        source_path: str,
        layer_name: str,
        description: str | None = None,
        variable: str | None = None,
        resampling: str = "bilinear",
        nodata: float | None = None,
        missing_value: float = -9999.0,
        overwrite: bool = True,
        update_mode: str = "checksum",
        dry_run: bool = False,
    ):
        record = self.get_cube(cube_name)
        grid = self.load_grid(cube_name)

        return ingest_layer(
            source_path=source_path,
            cube_path=record.path,
            cube_name=cube_name,
            grid=grid,
            layer_name=layer_name,
            description=description,
            variable=variable,
            region=record.region,
            resampling=resampling,
            nodata=nodata,
            missing_value=missing_value,
            overwrite=overwrite,
            update_mode=update_mode,
            dry_run=dry_run,
            stac_dir=str(self.catalog_dir),
        )

    def ingest_dir(
        self,
        cube_name: str,
        source_dir: str,
        pattern: str = "*",
        description: str | None = None,
        variable: str | None = None,
        resampling: str = "bilinear",
        nodata: float | None = None,
        missing_value: float = -9999.0,
        overwrite: bool = True,
        continue_on_error: bool = True,
        update_mode: str = "checksum",
        dry_run: bool = False,
    ) -> list[dict]:
        source_dir = Path(source_dir)
    
        candidates = sorted(
            p for p in source_dir.glob(pattern)
            if p.suffix.lower() in [".tif", ".tiff", ".nc", ".nc4", ".netcdf"]
        )
    
        results = []
    
        for path in candidates:
            layer_name = path.stem
    
            try:
                result = self.ingest(
                    cube_name=cube_name,
                    source_path=str(path),
                    layer_name=layer_name,
                    description=description,
                    variable=variable,
                    resampling=resampling,
                    nodata=nodata,
                    missing_value=missing_value,
                    overwrite=overwrite,
                    update_mode=update_mode,
                    dry_run=dry_run,
                )
    
                results.append(
                    {
                        "source": str(path),
                        "layer": layer_name,
                        "status": result["status"],
                        "reason": result.get("reason"),
                        "changed_keys": result.get("changed_keys", []),
                        "error": None,
                    }
                )
    
            except Exception as exc:
                result = {
                    "source": str(path),
                    "layer": layer_name,
                    "status": "failed",
                    "error": str(exc),
                }
                results.append(result)
    
                if not continue_on_error:
                    raise
    
        return results

    def netcdf_variables(self, source_path: str) -> list[str]:
        """List data variables available in a NetCDF source file."""
        return list_netcdf_variables(source_path)

    def ingest_netcdf(
        self,
        cube_name: str,
        source_path: str,
        variables: list[str] | None = None,
        layer_names: list[str] | None = None,
        layer_prefix: str | None = None,
        description: str | None = None,
        resampling: str = "bilinear",
        nodata: float | None = None,
        missing_value: float = -9999.0,
        overwrite: bool = True,
        continue_on_error: bool = True,
        update_mode: str = "checksum",
        dry_run: bool = False,
    ) -> list[dict]:
        """
        Ingest one or more variables from a NetCDF file as separate layers.

        If variables is omitted, all NetCDF data variables are ingested. By
        default each layer name matches its source variable name. Provide
        layer_names for a one-to-one variable-to-layer mapping, or layer_prefix
        to prefix generated layer names.
        """
        available = self.netcdf_variables(source_path)
        selected = list(variables) if variables is not None else available

        missing = [name for name in selected if name not in available]
        if missing:
            raise ValueError(
                f"Missing NetCDF variables: {missing}. Available variables: {available}"
            )

        if layer_names is not None and layer_prefix is not None:
            raise ValueError("Use layer_names or layer_prefix, not both.")

        if layer_names is not None:
            if len(layer_names) != len(selected):
                raise ValueError("layer_names must match the number of variables.")
            target_layers = list(layer_names)
        elif layer_prefix:
            target_layers = [f"{layer_prefix}_{name}" for name in selected]
        else:
            target_layers = selected

        results = []
        for variable, layer_name in zip(selected, target_layers, strict=True):
            try:
                result = self.ingest(
                    cube_name=cube_name,
                    source_path=source_path,
                    layer_name=layer_name,
                    description=description,
                    variable=variable,
                    resampling=resampling,
                    nodata=nodata,
                    missing_value=missing_value,
                    overwrite=overwrite,
                    update_mode=update_mode,
                    dry_run=dry_run,
                )

                results.append(
                    {
                        "source": source_path,
                        "source_variable": variable,
                        "layer": layer_name,
                        "status": result["status"],
                        "reason": result.get("reason"),
                        "changed_keys": result.get("changed_keys", []),
                        "error": None,
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "source": source_path,
                        "source_variable": variable,
                        "layer": layer_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

                if not continue_on_error:
                    raise

        return results

    def update_layer(
        self,
        cube_name: str,
        source_path: str,
        layer_name: str,
        description: str | None = None,
        variable: str | None = None,
        resampling: str = "bilinear",
        nodata: float | None = None,
        missing_value: float = -9999.0,
        dry_run: bool = False,
    ):
        return self.ingest(
            cube_name=cube_name,
            source_path=source_path,
            layer_name=layer_name,
            description=description,
            variable=variable,
            resampling=resampling,
            nodata=nodata,
            missing_value=missing_value,
            overwrite=True,
            update_mode="checksum",
            dry_run=dry_run,
        )

    def overwrite_layer(
        self,
        cube_name: str,
        source_path: str,
        layer_name: str,
        description: str | None = None,
        variable: str | None = None,
        resampling: str = "bilinear",
        nodata: float | None = None,
        missing_value: float = -9999.0,
        dry_run: bool = False,
    ):
        return self.ingest(
            cube_name=cube_name,
            source_path=source_path,
            layer_name=layer_name,
            description=description,
            variable=variable,
            resampling=resampling,
            nodata=nodata,
            missing_value=missing_value,
            overwrite=True,
            update_mode="overwrite",
            dry_run=dry_run,
        )

    def delete_layer(self, cube_name: str, layer_name: str) -> dict:
        record = self.get_cube(cube_name)

        delete_layer_data(record.path, layer_name)
        remove_layer_manifest(record.path, layer_name)
        mark_layer_deleted(record.path, layer_name)
        delete_stac_item(str(self.catalog_dir), cube_name, layer_name)

        return {
            "cube": cube_name,
            "layer": layer_name,
            "status": "deleted",
        }

    def rename_layer(
        self,
        cube_name: str,
        old_name: str,
        new_name: str,
    ) -> dict:
        record = self.get_cube(cube_name)
        grid = self.load_grid(cube_name)

        rename_layer_data(record.path, old_name, new_name)
        rename_layer_manifest(record.path, old_name, new_name)
        rename_layer_registry(record.path, old_name, new_name)

        provenance = get_layer_provenance(record.path, new_name)
        source_path = provenance.get("source_path", record.path)

        delete_stac_item(str(self.catalog_dir), cube_name, old_name)
        upsert_stac_item(
            stac_dir=str(self.catalog_dir),
            cube_path=record.path,
            cube_name=cube_name,
            layer_name=new_name,
            grid=grid,
            source_path=source_path,
            region=record.region,
            description=provenance.get("description"),
            provenance=provenance,
            zarr_group=layer_group_name(new_name),
        )

        return {
            "cube": cube_name,
            "old_layer": old_name,
            "new_layer": new_name,
            "status": "renamed",
        }

    def layers(self, cube_name: str | None = None):
        if cube_name:
            return list_layers(self.get_cube(cube_name).path)

        out = {}
        for name, record in self.records.items():
            if Path(record.path).exists():
                out[name] = list_layers(record.path)
            else:
                out[name] = []
        return out

    def load(
        self,
        cube_name: str,
        layers: list[str] | None = None,
        chunks: dict | str | None = "auto",
    ) -> xr.Dataset:
        record = self.get_cube(cube_name)
        return load_layers(record.path, layers=layers, chunks=chunks)

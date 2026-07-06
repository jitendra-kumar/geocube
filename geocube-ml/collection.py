from dataclasses import dataclass, asdict
from pathlib import Path
import json
import xarray as xr

from .grid import CubeGrid
from .cube import list_layers, load_layers
from .ingest import ingest_layer


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
        variable: str | None = None,
        resampling: str = "bilinear",
        nodata: float | None = None,
        missing_value: float = -9999.0,
        overwrite: bool = True,
    ):
        record = self.get_cube(cube_name)
        grid = self.load_grid(cube_name)

        return ingest_layer(
            source_path=source_path,
            cube_path=record.path,
            cube_name=cube_name,
            grid=grid,
            layer_name=layer_name,
            variable=variable,
            region=record.region,
            resampling=resampling,
            nodata=nodata,
            missing_value=missing_value,
            overwrite=overwrite,
            stac_dir=str(self.catalog_dir),
        )

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

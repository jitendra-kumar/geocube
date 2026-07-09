from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import xarray as xr


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_path(cube_path: str | Path) -> Path:
    return Path(cube_path) / ".geocube_ml_manifest.json"


def load_manifest(cube_path: str | Path) -> dict:
    path = manifest_path(cube_path)

    if not path.exists():
        return {
            "software": "geocube-ml",
            "cube_version": "0.1.0",
            "created_at_utc": now_utc(),
            "last_updated_utc": now_utc(),
            "layer_count": 0,
            "layers": {},
        }

    return json.loads(path.read_text())


def update_manifest(
    cube_path: str | Path,
    layer_name: str,
    layer_attrs: dict,
    stats: dict | None = None,
) -> dict:
    cube_path = Path(cube_path)
    manifest = load_manifest(cube_path)

    manifest["last_updated_utc"] = now_utc()

    manifest["layers"][layer_name] = {
        "name": layer_name,
        "region": layer_attrs.get("region"),
        "cube_name": layer_attrs.get("cube_name"),
        "grid_name": layer_attrs.get("grid_name"),
        "crs": layer_attrs.get("crs"),
        "resolution_degrees": layer_attrs.get("resolution_degrees"),
        "extent": layer_attrs.get("extent"),
        "missing_value": layer_attrs.get("missing_value"),
        "resampling": layer_attrs.get("resampling"),
        "source_path": layer_attrs.get("source_path"),
        "statistics": stats or {},
        "updated_at_utc": now_utc(),
    }

    manifest["layer_count"] = len(manifest["layers"])

    path = manifest_path(cube_path)
    path.write_text(json.dumps(manifest, indent=2))

    return manifest


def validate_cube_manifest(cube_path: str | Path) -> dict:
    ds = xr.open_zarr(cube_path, chunks={})
    manifest = load_manifest(cube_path)

    manifest_layers = set(manifest.get("layers", {}).keys())
    actual_layers = set(ds.data_vars)

    return {
        "cube_path": str(cube_path),
        "manifest_layer_count": len(manifest_layers),
        "actual_layer_count": len(actual_layers),
        "missing_from_manifest": sorted(actual_layers - manifest_layers),
        "missing_from_cube": sorted(manifest_layers - actual_layers),
        "ok": manifest_layers == actual_layers,
    }

#!/usr/bin/env python
"""
Starter example for using the GeoCube-ML Python API in ML workflows.

This script demonstrates how to:

1. Discover GeoCube-ML collections under a workspace.
2. Print collection, cube, grid, and layer metadata.
3. Load selected layers such as soil_ph and soil_soc.
4. Stack selected layers into a feature cube suitable for ML workflows.
5. Make quick spatial plots of the selected layers.

Example:

    python tests/examples/ml_starter_geocube_api.py \
        --search-root /chrysaor/remotesensing/jbk/climate/ngee/src \
        --collection-root /chrysaor/remotesensing/jbk/climate/ngee/src/arctic_geocubes \
        --cube-name arctic_30sec \
        --layers soil_ph soil_soc \
        --plot-output /tmp/arctic_30sec_soil_layers.png

If the selected layers have not been ingested yet, the script will still print
collection summaries and then exit with a clear message at the load step.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "geocube_ml").exists():
    sys.path.insert(0, str(REPO_ROOT))

from geocube_ml.collection import CubeCollection  # noqa: E402


DEFAULT_LAYERS = ["soil_ph", "soil_soc"]


def is_geocube_collection_root(path: Path) -> bool:
    """Return True when path looks like a GeoCube-ML collection root."""
    registry_path = path / "collection.json"
    try:
        data = json.loads(registry_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    return isinstance(data.get("cubes"), dict)


def discover_collections(search_root: Path) -> list[Path]:
    """Return directories containing a GeoCube-ML collection.json file."""
    return sorted(
        path.parent
        for path in search_root.rglob("collection.json")
        if path.is_file() and is_geocube_collection_root(path.parent)
    )


def summarize_collection(collection_root: Path) -> dict:
    """Collect basic collection, cube, grid, and layer metadata."""
    collection = CubeCollection(str(collection_root))
    cubes = []

    for cube_name, record in collection.records.items():
        grid = collection.load_grid(cube_name)
        layers = collection.layers(cube_name) if Path(record.path).exists() else []

        cubes.append(
            {
                "name": record.name,
                "region": record.region,
                "resolution_label": record.resolution_label,
                "description": record.description,
                "path": record.path,
                "grid": {
                    **asdict(grid),
                    "width": grid.width,
                    "height": grid.height,
                },
                "layer_count": len(layers),
                "layers": [
                    {
                        "name": layer.name,
                        "description": layer.description,
                        "dtype": layer.dtype,
                        "dims": layer.dims,
                        "region": layer.region,
                        "grid_name": layer.grid_name,
                        "crs": layer.crs,
                        "resolution_degrees": layer.resolution_degrees,
                    }
                    for layer in layers
                ],
            }
        )

    return {
        "collection_root": str(collection_root),
        "cube_count": len(cubes),
        "cubes": cubes,
    }


def print_collection_summary(summary: dict) -> None:
    """Print a compact human-readable collection summary."""
    print(f"\nCollection: {summary['collection_root']}")
    print(f"Cube count: {summary['cube_count']}")

    for cube in summary["cubes"]:
        grid = cube["grid"]
        print(
            f"  Cube: {cube['name']} | region={cube['region']} | "
            f"resolution={cube['resolution_label']} | layers={cube['layer_count']}"
        )
        if cube["description"]:
            print(f"    description: {cube['description']}")
        print(
            f"    grid: crs={grid['crs']} extent="
            f"({grid['xmin']}, {grid['ymin']}, {grid['xmax']}, {grid['ymax']}) "
            f"resolution={grid['resolution']} shape={grid['height']}x{grid['width']} "
            f"chunks={tuple(grid['chunks'])}"
        )

        if not cube["layers"]:
            print("    layers: none")
            continue

        print("    layers:")
        for layer in cube["layers"]:
            line = (
                f"      - {layer['name']} | dtype={layer['dtype']} | "
                f"dims={tuple(layer['dims'])} | crs={layer['crs']}"
            )
            if layer["description"]:
                line += f" | description={layer['description']}"
            print(line)


def load_feature_cube(
    collection_root: Path,
    cube_name: str,
    layer_names: list[str],
):
    """Load layers and stack them as feature, y, x for ML-style processing."""
    collection = CubeCollection(str(collection_root))
    cube_record = collection.get_cube(cube_name)

    if not Path(cube_record.path).exists():
        raise FileNotFoundError(
            f"Cube store does not exist yet: {cube_record.path}. "
            "Ingest at least one layer before loading data."
        )

    available = {layer.name for layer in collection.layers(cube_name)}
    missing = [layer for layer in layer_names if layer not in available]

    if missing:
        raise ValueError(
            f"Missing requested layers in cube {cube_name}: {missing}. "
            f"Available layers: {sorted(available)}"
        )

    dataset = collection.load(cube_name=cube_name, layers=layer_names)
    feature_cube = dataset[layer_names].to_array(dim="feature")

    if {"y", "x"}.issubset(feature_cube.dims):
        feature_cube = feature_cube.transpose("feature", "y", "x")

    return dataset, feature_cube


def plot_layers(dataset, layer_names: list[str], output: Path | None = None) -> None:
    """Make quick spatial plots for selected layers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or omit plotting."
        ) from exc

    fig, axes = plt.subplots(
        1,
        len(layer_names),
        figsize=(6 * len(layer_names), 5),
        constrained_layout=True,
    )

    if len(layer_names) == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, layer_names, strict=True):
        dataset[layer_name].plot(ax=ax, robust=True)
        ax.set_title(layer_name)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"\nWrote plot: {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GeoCube-ML Python API starter for collection discovery, "
        "layer loading, feature stacking, and plotting."
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path.cwd(),
        help="Workspace directory to search recursively for collection.json files.",
    )
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=None,
        help="Specific collection root to load selected layers from.",
    )
    parser.add_argument(
        "--cube-name",
        default="arctic_30sec",
        help="Cube name to load from when --collection-root is provided.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Layer names to load and stack.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional output image path for spatial layer plots.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Load and stack layers, but do not make plots.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print discovered collection summaries as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collections = discover_collections(args.search_root)

    if not collections:
        print(f"No GeoCube-ML collections found under: {args.search_root}")
    elif args.json:
        print(json.dumps([summarize_collection(path) for path in collections], indent=2))
    else:
        print(f"Discovered {len(collections)} GeoCube-ML collection(s).")
        for collection_root in collections:
            print_collection_summary(summarize_collection(collection_root))

    if args.collection_root is None:
        print(
            "\nProvide --collection-root to load selected layers and build an "
            "ML-ready feature cube."
        )
        return

    print(
        f"\nLoading layers {args.layers} from cube {args.cube_name} "
        f"in {args.collection_root}"
    )
    try:
        dataset, feature_cube = load_feature_cube(
            collection_root=args.collection_root,
            cube_name=args.cube_name,
            layer_names=args.layers,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stdout.flush()
        print(f"\nCould not load requested layers: {exc}", flush=True)
        raise SystemExit(1) from exc

    print("\nLoaded xarray.Dataset:")
    print(dataset)
    print("\nStacked feature cube:")
    print(feature_cube)
    print(
        "\nML starter: convert feature_cube to tabular samples, extract point "
        "values, or pass chunks into model training workflows."
    )

    if not args.no_plot:
        plot_layers(dataset, args.layers, output=args.plot_output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results_new"
DEFAULT_CLIMATOLOGY_STORE = (
    REPO_ROOT
    / "data/climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative_MWE.zarr"
)


def parse_csv_strings(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class ForecastSpec:
    name: str
    path: Path


def parse_checkpoint_specs(specs: Sequence[str]) -> List[CheckpointSpec]:
    parsed: List[CheckpointSpec] = []
    for raw in specs:
        if "=" in raw:
            name_raw, path_raw = raw.split("=", 1)
            name = sanitize_name(name_raw)
            path = Path(path_raw).expanduser().resolve()
        else:
            path = Path(raw).expanduser().resolve()
            name = sanitize_name(path.parent.name or path.stem)

        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        parsed.append(CheckpointSpec(name=name, path=path))
    return parsed


def parse_forecast_specs(specs: Sequence[str]) -> List[ForecastSpec]:
    parsed: List[ForecastSpec] = []
    for raw in specs:
        if "=" in raw:
            name_raw, path_raw = raw.split("=", 1)
            name = sanitize_name(name_raw)
            path = Path(path_raw).expanduser().resolve()
        else:
            path = Path(raw).expanduser().resolve()
            if path.parent.name == "forecast" and path.parent.parent.name.startswith("checkpoint_"):
                name = sanitize_name(path.parent.parent.name[len("checkpoint_") :])
            else:
                name = sanitize_name(path.parent.name or path.stem)

        if not path.is_file():
            raise FileNotFoundError(f"Forecast NetCDF file not found: {path}")
        parsed.append(ForecastSpec(name=name, path=path))
    return parsed


def discover_forecasts(results_root: Path) -> List[ForecastSpec]:
    found: List[ForecastSpec] = []
    pattern = results_root.glob("checkpoint_*/forecast/forecast.nc")
    for nc_path in sorted(pattern):
        checkpoint_dir = nc_path.parents[1]
        name = checkpoint_dir.name[len("checkpoint_") :] if checkpoint_dir.name.startswith("checkpoint_") else checkpoint_dir.name
        found.append(ForecastSpec(name=sanitize_name(name), path=nc_path.resolve()))
    return found


def build_checkpoint_dirs(results_root: Path, checkpoint_name: str) -> Tuple[Path, Path, Path]:
    root = Path(results_root).expanduser().resolve()
    checkpoint_dir = root / f"checkpoint_{sanitize_name(checkpoint_name)}"
    forecast_dir = checkpoint_dir / "forecast"
    metrics_dir = checkpoint_dir / "metrics"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, forecast_dir, metrics_dir

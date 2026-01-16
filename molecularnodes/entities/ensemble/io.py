from pathlib import Path
from typing import TYPE_CHECKING
from .cellpack import CellPack
from .cryosparc import CsMeta
from .star import StarFile

if TYPE_CHECKING:
    from .base import Ensemble


def load_starfile(file_path, node_setup=True, world_scale=0.01) -> StarFile:
    ensemble = StarFile.from_starfile(file_path)
    ensemble.create_object(
        name=Path(file_path).name, node_setup=node_setup, world_scale=world_scale
    )

    return ensemble


def load_cryosparc_metadata(file_path, node_setup=True, world_scale=0.01) -> CsMeta:
    ensemble = CsMeta.from_csfile(file_path)
    ensemble.create_object(
        name=Path(file_path).name, node_setup=node_setup, world_scale=world_scale
    )

    return ensemble


_metadata_load_functions = {
    ".star": load_starfile,
    ".cs": load_cryosparc_metadata,
}


def load_metadata(
    file_path: Path | str,
    node_setup: bool = True,
    world_scale: float = 0.01,
    file_type: str | None = None,
) -> "Ensemble":
    file_path = Path(file_path)
    if file_type is None:
        file_type = file_path.suffix
    load_fn = _metadata_load_functions.get(file_type.lower())
    if load_fn is None:
        raise ValueError(f"Unsupported metadata file type: {file_type.lower()}")
    return load_fn(
        file_path=file_path,
        node_setup=node_setup,
        world_scale=world_scale,
    )


def load_cellpack(
    file_path,
    name="NewCellPackModel",
    node_setup=True,
    world_scale=0.01,
    fraction: float = 1,
):
    ensemble = CellPack(file_path)
    ensemble.create_object(
        name=name, node_setup=node_setup, world_scale=world_scale, fraction=fraction
    )

    return ensemble

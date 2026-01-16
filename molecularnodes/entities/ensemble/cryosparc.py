from pathlib import Path
from typing import TYPE_CHECKING
import databpy
import numpy as np
from pandas import DataFrame
from ... import blender as bl
from ...nodes import nodes
from .base import Ensemble, EnsembleDataFrame, EntityType

if TYPE_CHECKING:
    from typing import Sequence
    from bpy.types import Object


class CsDataFrame(EnsembleDataFrame):
    def __init__(
        self,
        data: DataFrame,
        coord_columns: "Sequence[str] | None",
        rot_columns: "Sequence[str] | None",
        shift_columns: "Sequence[str] | None",
    ):
        super().__init__(
            data,
            coord_columns=coord_columns,
            rot_columns=rot_columns,
            shift_columns=shift_columns,
            rotation_convention="rotvec",
            image_id_columns="uid",
        )
        self.type = "cryosparc"


class CsMeta(Ensemble):
    def __init__(self, file_path: str | Path) -> None:
        super().__init__(file_path=file_path)
        self.type = "cryosparc"
        self.current_image = -1
        self._entity_type = EntityType.ENSEMBLE_CRYOSPARC

    @classmethod
    def from_csfile(cls, file_path: str | Path) -> "CsMeta":
        self = cls(file_path)
        self._read()

        return self

    def _read(self):
        cs = np.load(self.file_path)
        # cryosparc UIDs are too large for C ints
        df_columns = {"uid": cs["uid"].astype(str)}
        coord_columns = []
        shift_columns = []
        rot_columns = []
        cs_fields = cs.dtype.names
        prefixes = set(f.split("/")[0] for f in cs_fields)
        if "alignments3D" in prefixes:
            rot_columns = ["rotvec_x", "rotvec_y", "rotvec_z"]
            for i, title in enumerate(rot_columns):
                df_columns[title] = cs["alignments3D/pose"][:, i]
            shift_columns = ["shift_x", "shift_y", "shift_z"]
            for i, title in enumerate(shift_columns[:2]):
                df_columns[title] = cs["alignments3D/shift"][:, i]
            df_columns["shift_z"] = np.zeros_like(df_columns["shift_x"])
        if "location/micrograph_uid" in prefixes:
            self.image_id_columns = ["micrograph_uid"]
            df_columns["micrograph_uid"] = cs["location/micrograph_uid"].astype(str)
        if "location/center_x_frac" in prefixes:
            coord_columns = ["center_x_frac", "center_y_frac", "center_z_frac"]
            for title in coord_columns[:2]:
                df_columns[title] = cs[f"location/{title}"]
            df_columns["center_z_frac"] = np.zeros_like(
                df_columns["location/center_x_frac"]
            )
        df = DataFrame(df_columns)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category")
        self.data_reader = df
        self.data_frame = CsDataFrame(
            df,
            coord_columns=coord_columns if coord_columns else None,
            rot_columns=rot_columns if rot_columns else None,
            shift_columns=shift_columns if shift_columns else None,
        )

    def create_object(
        self,
        name: str = "CryoSPARCObject",
        node_setup: bool = True,
        world_scale: float = 0.01,
        fraction: float = 1,
        simplify=False,
    ) -> "Object":
        if self.data_frame is None:
            raise ValueError("DataFrame not assigned. Call from_csfile() first.")

        self.object = databpy.create_object(
            self.data_frame.coordinates_scaled * world_scale,
            collection=bl.coll.mn(),
            name=name,
        )
        self.object.mn.entity_type = self._entity_type.value
        self.data_frame.store_data_on_object(self.object)

        if node_setup:
            nodes.create_starting_nodes_cryosparc(self.object)

        self.object["csfile_path"] = str(self.file_path)
        return self.object

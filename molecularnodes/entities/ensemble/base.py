from abc import ABCMeta
from pathlib import Path
from typing import TYPE_CHECKING, Union
import bpy
import numpy as np
from databpy import AttributeTypes, BlenderObject
from pandas import CategoricalDtype
from scipy.spatial.transform import Rotation
from ... import blender as bl
from ..base import EntityType, MolecularEntity

if TYPE_CHECKING:
    from typing import Literal, Sequence
    from pandas import DataFrame


class Ensemble(MolecularEntity, metaclass=ABCMeta):
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize an Ensemble object.

        Parameters
        ----------
        file_path : Union[str, Path]
            The path to the file.
        """
        super().__init__()
        self._entity_type = EntityType.ENSEMBLE
        self.type: str = "ensemble"
        self.file_path: Path = bl.path_resolve(file_path)

    @property
    def instance_collection(self) -> bpy.types.Collection:
        """
        The instances of the ensemble.

        Returns
        -------
        bpy.types.Collection
            The collection containing the ensemble instances.
        """
        return bpy.data.collections[self._instance_collection_name]

    @instance_collection.setter
    def instance_collection(self, value: bpy.types.Collection) -> None:
        """
        Set the instance collection.

        Parameters
        ----------
        value : bpy.types.Collection
            The collection to set as the instance collection.

        Raises
        ------
        ValueError
            If the value is not a bpy.types.Collection.
        """
        if not isinstance(value, bpy.types.Collection):
            raise ValueError("The instances must be a bpy.types.Collection.")
        self._instance_collection_name = value.name

    def create_object(
        self,
        name: str = "NewEnsemble",
        node_setup: bool = True,
        world_scale: float = 0.01,
        fraction: float = 1.0,
        simplify=False,
    ) -> bpy.types.Object:
        """
        Create a 3D object for the ensemble.

        Parameters
        ----------
        name : str, optional
            The name of the model, by default "NewEnsemble"
        node_setup : bool, optional
            Whether to setup nodes for the data and instancing objects, by default True
        world_scale : float, optional
            Scaling transform for the coordinates before loading in to Blender, by default 0.01
        fraction : float, optional
            The fraction of the instances to display on loading. Reducing can help with performance, by default 1.0
        simplify : bool, optional
            Whether to instance the given models or simplify them for debugging and performance, by default False

        Notes
        -----
        Creates a data object which stores all of the required instancing information. If
        there are molecules to be instanced, they are also created in their own data collection.
        """
        pass


class EnsembleDataFrame:
    def __init__(
        self,
        data: "DataFrame",
        coord_columns: "Sequence[str] | None" = None,
        shift_columns: "Sequence[str] | None" = None,
        rot_columns: "Sequence[str] | None" = None,
        rotation_convention: "Literal['rotvec', 'ZYZ'] | None" = None,
        image_id_columns: "str | Sequence[str] | None" = None,
    ) -> None:
        self.data = data
        self._coord_columns = coord_columns
        self._rot_columns = rot_columns
        self._rotation_convention = rotation_convention
        if rot_columns is not None and rotation_convention is None:
            raise ValueError(
                "If metadata includes rotations the convention must be specified"
            )
        self._shift_columns = shift_columns
        if isinstance(image_id_columns, str):
            self._image_id_columns = [image_id_columns]
        elif image_id_columns is None:
            self._image_id_columns = []
        else:
            self._image_id_columns = image_id_columns

    @property
    def coordinates(self) -> np.ndarray:
        if self._coord_columns is None:
            return np.zeros((len(self.data), 3))

        coord = self.data[self._coord_columns].to_numpy()

        if self._shift_columns is not None:
            try:
                shift = self.data[self._shift_columns].to_numpy()
                coord -= shift
            except KeyError:
                pass

        return coord

    @property
    def scale(self) -> np.ndarray:
        arr = np.zeros((len(self.data), 1), dtype=np.float32)
        arr[:] = 1.0
        return arr

    @property
    def coordinates_scaled(self) -> np.ndarray:
        return self.coordinates * self.scale

    def rotation_as_quaternion(self) -> np.ndarray:
        if self._rot_columns is None:
            return np.zeros((len(self.data), 4))
        rot_columns = self.data[self._rot_columns].to_numpy()
        if self._rotation_convention == "ZYZ":
            quaternions = (
                Rotation.from_euler("ZYZ", rot_columns, degrees=True)
                .inv()
                .as_quat(scalar_first=True)
            )
        elif self._rotation_convention == "rotvec":
            quaternions = (
                Rotation.from_rotvec(rot_columns).inv().as_quat(scalar_first=True)
            )
        else:
            raise ValueError(
                f"Unsupported rotation convention {self._rotation_convention}"
            )
        return quaternions

    def image_id_values(self) -> np.ndarray:
        for col_name in self._image_id_columns:
            try:
                return self.data[col_name].cat.codes.to_numpy()
            except KeyError:
                continue
            except AttributeError:
                return self.data[col_name].to_numpy()

        return np.zeros(len(self.data), dtype=int)

    def store_data_on_object(self, obj: bpy.types.Object) -> None:
        bob = BlenderObject(obj)
        bob.store_named_attribute(
            self.rotation_as_quaternion(),
            name="rotation",
            atype=AttributeTypes.QUATERNION,
        )

        bob.store_named_attribute(
            self.image_id_values(),
            name="image_id",
            atype=AttributeTypes.INT,
        )

        for col in self.data.columns:
            if isinstance(self.data[col].dtype, CategoricalDtype):
                bob.object[f"{col}_categories"] = list(self.data[col].cat.categories)
                data = self.data[col].cat.codes.to_numpy()
                bob.store_named_attribute(data, name=col, atype=AttributeTypes.INT)
            else:
                bob.store_named_attribute(self.data[col].to_numpy(), name=col)

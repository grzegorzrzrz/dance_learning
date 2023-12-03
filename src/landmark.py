import numpy as np
from constants import NODES_NAME
from math import isclose


class Landmark:
    def __init__(self, id: int, x: float, y: float, z: float) -> None:
        """Class containing data for single point in 3D skeleton.

        Args:
            id (int): id of the point. This id is unique within single skeleton.
            x (float): Value of coordinate x of point.
            y (float): Value of coordinate y of point.
            z (float): Value of coordinate z of point.
        """
        self._id = id
        self._x = x
        self._y = y
        self._z = z
        self._name = NODES_NAME[id]

    @property
    def id(self):
        return self._id

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def name(self):
        """
        Name of the landmark based on its id.
        List of names is in constants.py file
        Returns:
            _type_: Name of landmark.
        """
        return self._name

    def __bool__(self):
        return bool(self.x is not None and self.y is not None and self.z is not None)

    def __eq__(self, __value: object) -> bool:
        if not self or not __value:
            return self.id == __value.id
        return self.id == __value.id and isclose(self.x, __value.x) and isclose(self.y, __value.y) and isclose(self.z, __value.z)

class RawLandmark(Landmark):
    def __init__(self, id: int, x: float, y: float, z:float) -> None:
        """Landmark which is created when human pose is captured.
        Contains exactly the same data, as when this point was detected in a image.
        functionally, this class is exactly the same as Landmark class.

        Args:
            id (int): id of the point. This id is unique within single skeleton.
            x (float): Value of coordinate x of point.
            y (float): Value of coordinate y of point.
            z (float): Value of coordinate z of point.
        """
        super().__init__(id, x, y, z)


class EmptyLandmark(Landmark):

    def __init__(self, id: int) -> None:
        """Class, which is used when human pose could not be estimated correctly.
        This class is needed to create full skeletons, despite lacking sufficient data,
        and is placeholder for normal Landmarks.

        Args:
            id (int): Id of this point, as if it would be a proper Landmark.
        """
        super().__init__(id, None, None, None)


class SkeletonLandmark(Landmark):
    def __init__(self, raw_landmark: RawLandmark, parent_raw_landmark: RawLandmark,
                 parent_normalized_landmark, normalized_distance: float) -> None:
        """Type of Landmark which is used for Skeletons, when capturing data from image.
        This class converts RawLandmarks into points, which have constant, normalized distances between anoher points.

        Args:
            raw_landmark (RawLandmark): RawLandmark whose position will be normalized to Skeleton
            parent_raw_landmark (RawLandmark): Parent of raw_landmark. Distance between them will be normalized.
            parent_normalized_landmark (SkeletonLandmark): Patent of this Landmark. Distance between them is normalized.
            normalized_distance (float): Distance between this Landmark and its parent.
        """

        x1 = parent_raw_landmark.x
        y1 = parent_raw_landmark.y
        z1 = parent_raw_landmark.z

        x2 = raw_landmark.x
        y2 = raw_landmark.y
        z2 = raw_landmark.z

        raw_distance = np.sqrt(np.power((x2-x1), 2) + np.power((y2-y1), 2) + np.power((z2-z1), 2))
        normalized_x = (x2-x1) * normalized_distance / raw_distance
        normalized_y = (y2-y1) * normalized_distance / raw_distance
        normalized_z = (z2-z1) * normalized_distance / raw_distance

        self._x = parent_normalized_landmark.x + normalized_x
        self._y = parent_normalized_landmark.y + normalized_y
        self._z = parent_normalized_landmark.z + normalized_z

        self._id = raw_landmark.id
        self._parent_landmark = parent_normalized_landmark
        self._distance = normalized_distance


    def parent_landmark(self):
        return self._parent_landmark


    def distance(self):
        return self._distance


class AnchorSkeletonLandmark(SkeletonLandmark):

    def __init__(self) -> None:
        """Type of SkeletonLandmark, which is used when creating Skeletons from image.
        AnchorSkeletonLandmark is first Landmark of skeleton, based upon which a whole skeleton will be created.
        Its coordiates are (0, 0, 0) and its id is -1.
        """
        self._x = 0
        self._y = 0
        self._z = 0
        self._id = -1
        self._parent_landmark = None
        self._distance = 0

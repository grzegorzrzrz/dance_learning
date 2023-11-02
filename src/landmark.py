import numpy as np
from constants import NODES_NAME


class Landmark:
    def __init__(self, id, x, y, z) -> None:
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
        return self._name


class RawLandmark(Landmark):
    def __init__(self, id, x, y, z) -> None:
        super().__init__(id, x, y, z)


class SkeletonLandmark(Landmark):
    def __init__(self, raw_landmark: RawLandmark, parent_raw_landmark: RawLandmark,
                 parent_normalized_landmark, normalized_distance) -> None:

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
        self._x = 0
        self._y = 0
        self._z = 0
        self._id = -1
        self._parent_landmark = None
        self._distance = 0

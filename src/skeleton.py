import csv
from landmark import *

class Skeleton3D:
    def __init__(self, skeleton_data_file, raw_landmarks_data, timestamp) -> None:
        self._raw_landmarks = []
        for raw_landmark_data in raw_landmarks_data:
            id, x, y, z = raw_landmark_data
            new_landmark = RawLandmark(id, x, y, z)
            self._raw_landmarks.append(new_landmark)
        self._timestamp = timestamp

        anchor = AnchorSkeletonLandmark()
        self._landmarks = [anchor]
        with open(skeleton_data_file) as handle:
            csv_reader = csv.DictReader(handle, delimiter=",")
            for row in csv_reader:
                raw_child    = self.get_raw_landmark_by_id(int(row["child"]))
                raw_parent   = self.get_raw_landmark_by_id(int(row["parent"]))
                norm_parent  = self.get_landmark_by_id(int(row["parent"]))
                distance     = float(row["distance"])
                new_landmark = SkeletonLandmark(raw_child, raw_parent, norm_parent, distance)
                self._landmarks.append(new_landmark)

        self._landmarks.sort(key=lambda x: x.id)


    def landmarks(self):
        return self._landmarks

    def raw_landmarks(self):
        return self._raw_landmarks

    def get_landmark_by_id(self, id):
        return self._get_landmark_by_id(self.landmarks(), id)

    def get_raw_landmark_by_id(self, id):
        return self._get_landmark_by_id(self.raw_landmarks(), id)

    def _get_landmark_by_id(self, landmark_list, id):
        for landmark in landmark_list:
            if landmark.id == id:
                return landmark

    @property
    def timestamp(self):
        return self._timestamp

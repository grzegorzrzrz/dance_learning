import csv
from landmark import *
from math import isclose


class Skeleton:
    def __init__(self, landmarks_data, timestamp: float) -> None:
        """A class for containg data about pose estimation from single image.
        This class is a container for Landmarks got from pose.

        Args:
            landmarks_data (List[int, float, float, float]): A list of containers, which have 4 elements
            describing single Landmark: its id and x, y and z coordnates. From this list. This data must be already normalized,
            and it should not be taken directly from image pose estimation.
            a list of Landmarks will be created.
            timestamp (float): A number, describing in which second there was a pose, which is described by this class.
        """
        anchor = AnchorSkeletonLandmark()
        self._landmarks = [anchor]
        self._timestamp = timestamp
        for landmark_data in landmarks_data:
            id, x, y, z = landmark_data
            if x and y and z:
                self._landmarks.append(Landmark(id, float(x), float(y), float(z)))
            else:
                self._landmarks.append(EmptyLandmark(id))

    def landmarks(self):
        """Returns a list of Landmarks of this Skeleton.
        """
        return self._landmarks

    def get_landmark_by_id(self, id) -> Landmark:
        """Returns one of Landmarks of this Skeleton, which has the given id.
        """
        return self._get_landmark_by_id(self.landmarks(), id)

    def _get_landmark_by_id(self, landmark_list, id):
        for landmark in landmark_list:
            if landmark.id == id:
                return landmark

    @property
    def timestamp(self):
        return self._timestamp

    def __lt__(self, __value: object) -> bool:
        return self.timestamp < __value.timestamp

    def __eq__(self, __value: object) -> bool:
        for landmark in self.landmarks():
            check_id = landmark.id
            other_lm = __value.get_landmark_by_id(check_id)
            if not other_lm:
                return False
            if landmark != other_lm:
                return False
        return isclose(self.timestamp, __value.timestamp)



class RawSkeleton(Skeleton):

    def __init__(self, skeleton_data_file: str, raw_landmarks_data, timestamp: float) -> None:
        """A type of Skeleton, which is suitable for creating pose directly form image pose estimator.
        This class gets a data rquried to created list of Landmarks, as normal Skeleon class, but it also normalizes
        distances of these Landmarks based on skeleton_data_file.

        Args:
            skeleton_data_file (str): Path to csv file, which contain data about how to build a skeleton.
            raw_landmarks_data (List[int, float, float, float]): A list of containers, which have 4 elements
            describing single Landmark: its id and x, y and z coordnates. From this list.
            timestamp (float): A number, describing in which second there was a pose, which is described by this class.
        """
        self._raw_landmarks = []
        for raw_landmark_data in raw_landmarks_data:
            id, x, y, z = raw_landmark_data
            new_landmark = RawLandmark(id, float(x), float(y), float(z))
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


    def raw_landmarks(self):
        """Returns a list of RawLandmarks which have data taken directly from image.
        """
        return self._raw_landmarks

    def get_raw_landmark_by_id(self, id) -> RawLandmark:
        """Returns one of RawLandmarks of this Skeleton, which has the given id.
        """
        return self._get_landmark_by_id(self.raw_landmarks(), id)


class EmptySkeleton(Skeleton):

    def __init__(self, skeleton_data_file: str, timestamp: float) -> None:
        """Class, which is used when human pose could not be estimated correctly from image.
        This class is a placeholder for Dance class, so that there will be information that there wasn't
        detected pose on given time.

        Args:
            skeleton_data_file (_type_):  Path to csv file, which contain data about how to build a skeleton.
            timestamp (float): A number, describing in which second pose has not been detected.
        """
        self._timestamp = timestamp

        anchor = EmptyLandmark(-1)
        self._landmarks = [anchor]
        with open(skeleton_data_file) as handle:
            csv_reader = csv.DictReader(handle, delimiter=",")
            for row in csv_reader:
                self._landmarks.append(EmptyLandmark(int(row["child"])))

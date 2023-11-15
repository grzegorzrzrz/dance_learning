from src.skeleton import Skeleton
from typing import List
from src.pose_estimation import estaminate_from_frame, create_skeleton_from_raw_pose_landmarks, reverse_dictionary
from src.data_writer import write_data_to_csv_file
from src.constants import NODES_NAME, SKELETON_FILE, DEFAULT_PROJECTION
import cv2
import csv
import math
import time

class Dance:
    def __init__(self, skeleton_table: List[Skeleton], name="") -> None:
        """A class which represents a dance, as a list of Skeletons created in time.
        This class can be created by complete list of Skeletons, but it also can be created without initial data, which
        can be added after creating an intance of this class.
        Args:
            skeleton_table (List[Skeleton]): Initial list of Skeletons, representing a dance.
            name (str, optional): @TODO do name
        """
        self._skeleton_table = skeleton_table
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def skeleton_table(self) -> List[Skeleton]:
        return self._skeleton_table

    def get_skeleton_by_timestamp(self, timestamp) -> Skeleton:
        """Returns a Skeleton, which has a timestamp closest to a timestamp given as an argument.
        """
        return min(self._skeleton_table, key= lambda x: abs(x.timestamp-timestamp))

    def add_skeleton(self, skeleton: Skeleton):
        """Add a new Skeleton to list of Skeletons.
        Skeleton can be added only if it wasn't in the list before.
        """
        if skeleton not in self.skeleton_table:
            self._skeleton_table.append(skeleton)
            self._skeleton_table.sort()

    def get_last_skeleton(self) -> Skeleton:
        """Returns a Skeleton, which has the biggest timestamp from all Skeletons in this Dance.
        Returns None if there isn't any Skeleon in this list.
        """
        return self._skeleton_table[-1] if self._skeleton_table else None


class DanceManager:
    def __init__(self, dance_data_path: Dance, camera: cv2.VideoCapture) -> None:
        """A class which main purpose is to comepare dance from dance_data_path
        to a data gathered by the camera.

        Args:
            pattern_dance (Dance): A Dance class object, which contain data about dance from file dance_wideo_path.
            camera (cv2.VideoCapture): Object representing camera, from which we can get live video with dance.
        """

        self._dance_data_path = dance_data_path
        self._pattern_dance = create_dance_from_data_file(dance_data_path)
        self._actual_dance = Dance([])
        self._camera = camera
        self._displayer_timestamp = 0
        self._is_video_being_played = False

    @property
    def pattern_dance(self) -> Dance:
        """Returns a Dance object with data about dance from video
        """
        return self._pattern_dance

    @property
    def actual_dance(self) -> Dance:
        """Returns a Dance object with data about dance from camera
        """
        return self._actual_dance

    @property
    def dance_data_path(self) -> str:
        return self._dance_data_path

    @property
    def camera(self) -> cv2.VideoCapture:
        return self._camera

    @property
    def displayer_timestamp(self) -> float:
        """Returns a number which describes the current time of video
        """
        return self._displayer_timestamp

    @property
    def is_video_being_played(self):
        """returs True if viedo about dance is being played
        """
        return self._is_video_being_played

    def set_flag_is_video_being_played(self, value: bool):
        self._is_video_being_played = value

    def set_displayer_timestamp(self, value: float):
        self._displayer_timestamp = value

    def compare_dances(self, dimension = DEFAULT_PROJECTION):
        """A method, which starts two treads, one from displaying viedo, second for gathering data from camera
        and continuously compares dances while viedo is being played.
        """
        self._is_video_being_played = True
        start_time = time.time()
        video_length = self.pattern_dance.get_last_skeleton().timestamp
        self._actual_dance = Dance([])
        self.set_displayer_timestamp(0)

        while self._is_video_being_played and self.displayer_timestamp < video_length:
            ret, frame = self.camera.read()
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = estaminate_from_frame(imgRGB)
            self.set_displayer_timestamp(time.time() - start_time)
            skeleton = create_skeleton_from_raw_pose_landmarks(result.pose_world_landmarks, self.displayer_timestamp, dimension)
            self.actual_dance.add_skeleton(skeleton)
            self._compare_recent_dance()

            if not ret:
                #Something is wrong with camera
                break

    def save_actual_dance(self, file_name):
        """Sace dance form camera as a csv file named file_name.
        """
        write_data_to_csv_file(self.actual_dance, file_name, SKELETON_FILE)


    # def _get_dance_data_from_camera(self, dimension = DEFAULT_PROJECTION):
    #     """A method responsible gathering data from camera and updating
    #     actual_dance Dance class object with new Skeletons bases on its data.
    #     Method works until viedo is being played.

    #     Args:
    #         dimension (str, optional): Describes in how many dimensions should data be generated. Possible values are "2D" and "3D".
    #         Defaults to DEFAULT_PROJECTION ("2D").
    #     """

    def _compare_recent_dance(self):
        """
        Get the comparison of recent dance, based on gathered data from camera and data about dance from viedo.
        """
        last_frame = self.actual_dance.get_last_skeleton()
        if not last_frame:
            return
        pattern_frame = self.pattern_dance.get_skeleton_by_timestamp(last_frame.timestamp)

        for lm in last_frame.landmarks():
            landmark_id = lm.id
            patt_lm = pattern_frame.get_landmark_by_id(landmark_id)
            error = 0
            if not lm:
                #Camera could not find a person
                pass
            elif not patt_lm:
                #Person could not be found in reference video
                pass
            else:
                landmark_error = math.sqrt((lm.x - patt_lm.x)**2 + (lm.y - patt_lm.y)**2 + (lm.z - patt_lm.z)**2)
                error += landmark_error

        print(f"{last_frame.timestamp}: {error}")

def create_dance_from_data_file(data_file):

    nodes_name_dict = reverse_dictionary(NODES_NAME)
    headlines = []
    skeleton_list = []
    with open(data_file, "r") as handle:
        raw_dance_data = csv.DictReader(handle, delimiter=",")

        raw_headlines = raw_dance_data.fieldnames


        for raw_headline in raw_headlines[1:]:
            headline_name = raw_headline[:-2]
            if headline_name not in headlines:
                headlines.append(headline_name)

        next(raw_dance_data)
        for line in raw_dance_data:
            current_raw_skeleton = []
            timestamp = float(line["timestamp"])
            n_nodes = (len(line)-1)//3

            for node in range(n_nodes):
                id = nodes_name_dict[headlines[node]]
                x = line[raw_headlines[3*node + 1]]
                y = line[raw_headlines[3*node + 2]]
                z = line[raw_headlines[3*node + 3]]
                current_raw_skeleton.append([id, x, y, z])
            skeleton_list.append(Skeleton(current_raw_skeleton, timestamp))
    return Dance(skeleton_list)


def get_dance_data_from_video(video_path, dimension = DEFAULT_PROJECTION):

    data = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    while True:
        success, img = cap.read()
        if not success:
            return Dance(data)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = estaminate_from_frame(imgRGB)
        timestamp = current_frame / fps
        skeleton = create_skeleton_from_raw_pose_landmarks(results.pose_world_landmarks, timestamp, dimension)
        data.append(skeleton)
        current_frame += 1

class MockDanceManager(DanceManager):
    def __init__(self, pattern_dance: str, actual_dance: Dance) -> None:
        """Mock testing class for DanceManager.

        Args:
            pattern_dance (Dance): A Dance class object, which mocks data from video.
            actual_dance (Dance): A Dance class object, which mocks data from camera.
        """

        self._dance_video_path = None
        self._pattern_dance = pattern_dance
        self._actual_dance = actual_dance
        self._dance_displayer_thread = None
        self._dance_data_getter_thread = None
        self._displayer_timestamp = 0
        self._is_video_being_played = False

    def compare_dances(self):
        self._is_video_being_played = True
        time_start = time.time()
        dance_time = self.pattern_dance.get_last_skeleton().timestamp

        while self._is_video_being_played:
            self._compare_recent_dance()
            self._displayer_timestamp = time.time() - time_start
            if self._displayer_timestamp > dance_time:
                self._is_video_being_played = False

    def _compare_recent_dance(self):
        """
        Get the comparison of recent dance, based on gathered data from camera and data about dance from viedo.
        """
        last_frame = self.actual_dance.get_skeleton_by_timestamp(self.displayer_timestamp)
        if not last_frame:
            return
        pattern_frame = self.pattern_dance.get_skeleton_by_timestamp(last_frame.timestamp)

        for lm in last_frame.landmarks():
            landmark_id = lm.id
            patt_lm = pattern_frame.get_landmark_by_id(landmark_id)
            error = 0
            if not lm:
                #Camera could not find a person
                pass
            elif not patt_lm:
                #Person could not be found in reference video
                pass
            else:
                landmark_error = math.sqrt((lm.x - patt_lm.x)**2 + (lm.y - patt_lm.y)**2 + (lm.z - patt_lm.z)**2)
                error += landmark_error

        print(f"{last_frame.timestamp}: {error}")

# def dance(data_path, dance_path):
#     dance = create_dance_from_data_file(data_path)
#     dance_manager = DanceManager(dance_path, dance)
#     dance_manager.compare_dances()
#     dance_manager.save_actual_dance("src/atemp.csv")

# if __name__ == "__main__":
#     # test = get_dance_data_from_video("src/test (1).mp4")
#     # write_data_to_csv_file(test, "src/temp.csv")
#     # dance("src/temp.csv", "src/test (1).mp4")
#     pd = create_dance_from_data_file("static/pattern.csv")
#     ad = create_dance_from_data_file("static/actual.csv")
#     dm = MockDanceManager(pd, ad)
#     dm.compare_dances()
#     # d = get_dance_data_from_video("static/d1v3.mp4")
#     # write_data_to_csv_file(d, "static/actual.csv", SKELETON_FILE)
#     # pass
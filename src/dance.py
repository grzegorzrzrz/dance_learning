from skeleton import Skeleton
from typing import List
from pose_estimation import estaminate_from_frame, create_skeleton_from_raw_pose_landmarks, reverse_dictionary
from data_writer import write_data_to_csv_file
from constants import NODES_NAME, SKELETON_FILE, DEFAULT_PROJECTION
import cv2
import csv
import threading
import math

import time #@TODO Remove after testing

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
    def __init__(self, dance_video_path: str, pattern_dance: Dance, camera: cv2.VideoCapture) -> None:
        """A class which main purpose is to show a video of the dance, and comepare this dance
        to a data gathered by the camera.

        Args:
            dance_video_path (str): A path to a video which will be shown and which from which we want
            a comparison with dance from camera.
            pattern_dance (Dance): A Dance class object, which contain data about dance from file dance_wideo_path.
        """
        self._dance_video_path = dance_video_path
        self._pattern_dance = pattern_dance
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
    def dance_video_path(self) -> str:
        return self._dance_video_path

    @property
    def dance_displayer_thread(self):
        return self._dance_displayer_thread

    @property
    def dance_data_getter_thread(self):
        return self._dance_data_getter_thread

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

    def compare_dances(self):
        """A method, which starts two treads, one from displaying viedo, second for gathering data from camera
        and continuously compares dances while viedo is being played.
        """
        self._is_video_being_played = True
        self._dance_data_getter_thread.start()

        while self._is_video_being_played:
            self._compare_recent_dance()

    def save_actual_dance(self, file_name):
        """Sace dance form camera as a csv file named file_name.
        """
        write_data_to_csv_file(self.actual_dance, file_name, SKELETON_FILE)

    def _TEMP_show_dance_pattern(self):
        """A method responsible for displaying video. Method works until video is finished.
        """
        cap = cv2.VideoCapture(self._dance_video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = int(1000/fps)
        current_frame = 0
        while True:
            success, img = cap.read()

            self._displayer_timestamp = current_frame / fps
            if success:
                cv2.putText(img, str(round(self.displayer_timestamp, 4)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)

            else:
                #Video is finished
                self._is_video_being_played = False
                break

            cv2.imshow("Image", img)
            cv2.waitKey(frame_time)
            current_frame += 1


    def _get_dance_data_from_camera(self, dimension = DEFAULT_PROJECTION):
        """A method responsible gathering data from camera and updating
        actual_dance Dance class object with new Skeletons bases on its data.
        Method works until viedo is being played.

        Args:
            dimension (str, optional): Describes in how many dimensions should data be generated. Possible values are "2D" and "3D".
            Defaults to DEFAULT_PROJECTION ("2D").
        """
        cap = cv2.VideoCapture(0)
        self._dance_displayer_thread.start()
        while self._is_video_being_played:
            ret, frame = cap.read()
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = estaminate_from_frame(imgRGB)
            skeleton = create_skeleton_from_raw_pose_landmarks(result.pose_world_landmarks, self.displayer_timestamp, dimension)
            self.actual_dance.add_skeleton(skeleton)

            if not ret:
                #Something is wrong with camera
                break

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

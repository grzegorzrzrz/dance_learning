from skeleton import Skeleton
from typing import List
from pose_estimation import estaminate_from_frame, create_skeleton_from_raw_pose_landmarks, reverse_dictionary
from data_writer import write_data_to_csv_file
from constants import NODES_NAME, SKELETON_FILE
import cv2
import csv
import threading
import math

class Dance:
    def __init__(self, skeleton_table: List[Skeleton]) -> None:
        self._skeleton_table = skeleton_table

    @property
    def skeleton_table(self) -> List[Skeleton]:
        return self._skeleton_table

    def get_skeleton_by_timestamp(self, timestamp) -> Skeleton:
        return min(self._skeleton_table, key= lambda x: abs(x.timestamp-timestamp))

    def add_skeleton(self, skeleton: Skeleton):
        if skeleton not in self.skeleton_table:
            self._skeleton_table.append(skeleton)
            self._skeleton_table.sort()

    def get_last_skeleton(self):
        return self._skeleton_table[-1] if self._skeleton_table else None


class DanceManager:
    def __init__(self, dance_video_path, pattern_dance: Dance) -> None:
        self._dance_video_path = dance_video_path
        self._pattern_dance = pattern_dance
        self._actual_dance = Dance([])
        self._dance_displayer_thread = threading.Thread(target=self._TEMP_show_dance_pattern)
        self._dance_data_getter_thread = threading.Thread(target=self._get_dance_data_from_camera)
        self._displayer_timestamp = 0
        self._is_video_being_played = False

    @property
    def pattern_dance(self):
        return self._pattern_dance

    @property
    def actual_dance(self):
        return self._actual_dance

    @property
    def dance_video_path(self):
        return self._dance_video_path

    @property
    def dance_displayer_thread(self):
        return self._dance_displayer_thread

    @property
    def dance_data_getter_thread(self):
        return self._dance_data_getter_thread

    @property
    def displayer_timestamp(self):
        return self._displayer_timestamp

    @property
    def is_video_being_played(self):
        return self._is_video_being_played

    def compare_dances(self):
        self._is_video_being_played = True
        self._dance_data_getter_thread.start()

        while self._is_video_being_played:
            self._compare_recent_dance()

    def save_actual_dance(self, file_name):
        write_data_to_csv_file(self.actual_dance, file_name, SKELETON_FILE)

    def _TEMP_show_dance_pattern(self):
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


    def _get_dance_data_from_camera(self):
        cap = cv2.VideoCapture(0)
        self._dance_displayer_thread.start()
        while self._is_video_being_played:
            ret, frame = cap.read()
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = estaminate_from_frame(imgRGB)
            skeleton = create_skeleton_from_raw_pose_landmarks(result.pose_landmarks, self.displayer_timestamp, "2D")
            self.actual_dance.add_skeleton(skeleton)

            if not ret:
                #Something is wrong with camera
                break

    def _compare_recent_dance(self):
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


def get_dance_data_from_video(video_path, dimension = "3D"):

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
        skeleton = create_skeleton_from_raw_pose_landmarks(results.pose_landmarks, timestamp, dimension)
        data.append(skeleton)
        current_frame += 1

def dance(data_path, dance_path):
    dance = create_dance_from_data_file(data_path)
    dance_manager = DanceManager(dance_path, dance)
    dance_manager.compare_dances()
    dance_manager.save_actual_dance("src/atemp.csv")

if __name__ == "__main__":
    test = get_dance_data_from_video("src/test (1).mp4", "2D")
    write_data_to_csv_file(test, "src/temp.csv")
    dance("src/temp.csv", "src/test (1).mp4")

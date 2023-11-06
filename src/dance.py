from skeleton import Skeleton
from typing import List
from pose_estimation import estaminate_from_frame, create_skeleton_from_raw_pose_landmarks, reverse_dictionary
from constants import NODES_NAME
import cv2
import csv
import time
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
        self._skeleton_table.append(skeleton)
        self._skeleton_table.sort()

    def get_last_skeleton(self):
        return self._skeleton_table[-1] if self._skeleton_table else None


class DanceManager:
    def __init__(self, dance_video_path, pattern_dance: Dance) -> None:
        self._dance_video_path = dance_video_path
        self._pattern_dance = pattern_dance
        self._actual_dance = Dance([])

    @property
    def pattern_dance(self):
        return self._pattern_dance

    @property
    def actual_dance(self):
        return self._actual_dance

    @property
    def dance_video_path(self):
        return self._dance_video_path

    def compare_dances(self):

        input_dance =  threading.Thread(target=self._TEMP_show_dance_pattern)
        output_dance = threading.Thread(target=self._get_dance_data_from_camera)

        input_dance.start()
        output_dance.start()

        while True:
            self._compare_recent_dance()


    def _TEMP_show_dance_pattern(self):
        cap = cv2.VideoCapture(self._dance_video_path)
        sTime = time.time()
        while True:
            success, img = cap.read()

            cTime = time.time()
            vTime = cTime - sTime

            if success:
                cv2.putText(img, str(round(vTime, 4)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)

            else:
                break

            cv2.imshow("Image", img)
            cv2.waitKey(1)



    def _get_dance_data_from_camera(self):
        cap = cv2.VideoCapture(0)
        sTime = time.time()

        while True:
            ret, frame = cap.read()
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = estaminate_from_frame(imgRGB)
            cTime = time.time()
            vTime = cTime - sTime
            skeleton = create_skeleton_from_raw_pose_landmarks(result.pose_landmarks, vTime, "2D")
            self.actual_dance.add_skeleton(skeleton)

            if not ret:
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
            if lm and patt_lm:
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

def dance(data_path, dance_path):
    dance = create_dance_from_data_file(data_path)
    dance_manager = DanceManager(dance_path, dance)
    dance_manager.compare_dances()

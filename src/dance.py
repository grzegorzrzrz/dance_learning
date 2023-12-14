from skeleton import Skeleton
from typing import List
from pose_estimation import estaminate_from_frame, create_skeleton_from_raw_pose_landmarks, reverse_dictionary
from data_writer import write_data_to_csv_file
from constants import NODES_NAME, SKELETON_FILE, DEFAULT_PROJECTION, ACTUAL_DANCE_DATA_PATH, DEFAULT_SCORING_TIMESTEP
from datetime import datetime
import cv2
import csv
import math
import time
import numpy
import matplotlib.pyplot as plt
import os

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
    def __init__(self, camera: cv2.VideoCapture) -> None:
        """A class which main purpose is to comepare dance from dance_data_path
        to a data gathered by the camera.

        Args:
            pattern_dance (Dance): A Dance class object, which contain data about dance from file dance_wideo_path.
            camera (cv2.VideoCapture): Object representing camera, from which we can get live video with dance.
        """

        self._actual_dance = Dance([])
        self._camera = camera
        self._displayer_timestamp = 0
        self._is_video_being_played = False
        self._is_camera_checked = False

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

    @property
    def is_camera_checked(self):
        return self._is_camera_checked

    def set_flag_is_video_being_played(self, value: bool):
        self._is_video_being_played = value

    def set_flag_is_camera_checked(self, value: bool):
        self._is_camera_checked = value

    def set_displayer_timestamp(self, value: float):
        self._displayer_timestamp = value

    def check_camera(self, checking_time):
        time_start = time.time()
        current_time = time.time()
        while current_time < time_start + checking_time:
            ret, frame = self.camera.read()
            if not ret:
                #Singal that camera could not be found
                time_start = time.time()
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = estaminate_from_frame(imgRGB)
            skeleton = create_skeleton_from_raw_pose_landmarks(result.pose_world_landmarks, self.displayer_timestamp, DEFAULT_PROJECTION)
            if not skeleton:
                #Singal that person could not be found on camera
                time_start = time.time()
            current_time = time.time()

        self.set_flag_is_camera_checked(True)


    def compare_dances(self, dance_data_path: str, timestep= DEFAULT_SCORING_TIMESTEP,
                       save_actual_dance = True, dimension = DEFAULT_PROJECTION):
        """A method, which continuously compares dances while viedo is being played.
        """

        self._dance_data_path = dance_data_path
        self._pattern_dance = create_dance_from_data_file(dance_data_path)
        self._is_video_being_played = True
        start_time = time.time()
        video_length = self.pattern_dance.get_last_skeleton().timestamp
        self._actual_dance = Dance([], name=self.pattern_dance.name)
        self.set_displayer_timestamp(0)

        values = []
        inv_values =[]
        base_values = []
        t = []
        t_0 = start_time
        value_record = []
        depth = 0.5
        n = 10
        radius = numpy.linspace(0,-depth,n)

        while self._is_video_being_played and self.displayer_timestamp < video_length:
            ret, frame = self.camera.read()
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = estaminate_from_frame(imgRGB)
            self.set_displayer_timestamp(time.time() - start_time)
            skeleton = create_skeleton_from_raw_pose_landmarks(result.pose_world_landmarks, self.displayer_timestamp, dimension)
            self.actual_dance.add_skeleton(skeleton)

            results = []

            for delay in radius:
                res = self._compare_recent_dance(delay)
                if res:
                    score,output = res
                    #print(res)
                    #results.append(score)
                    if score != None:
                        results.append((score,output))

            # if not results:
            #     print("check2")

            values.append(min(results))
            inv_values.append(max(results))
            base_values.append(results[0])

            t.append(self._displayer_timestamp)
            #print(f"time: {self._displayer_timestamp}, score:{min(results)}, inv_score{max(results)}")
            #print(f"{self.displayer_timestamp} t_0: {t_0}")
            if (time.time() - t_0) >= timestep:#when we output the result
                t_0 = time.time()
                values = getNthTupleElementFromList(values, 0)
                value_record.extend(values)
                avg_value = sum(values)/len(values)
                # report is what we want the user to see, here we print it
                # report = getGrade(avg_value)
                # print(report)

                values.clear()

            if not ret:
                #Something is wrong with camera
                break

        if save_actual_dance:
            self.save_actual_dance()

    def save_actual_dance(self, file_name=None):
        """Sace dance form camera as a csv file named file_name.
        """
        if not file_name:
            file_name = f"{ACTUAL_DANCE_DATA_PATH}/{add_current_timestamp_to_filename(self.pattern_dance.name)}.csv"
        write_data_to_csv_file(self.actual_dance, file_name, SKELETON_FILE)


    # def _get_dance_data_from_camera(self, dimension = DEFAULT_PROJECTION):
    #     """A method responsible gathering data from camera and updating
    #     actual_dance Dance class object with new Skeletons bases on its data.
    #     Method works until viedo is being played.

    #     Args:
    #         dimension (str, optional): Describes in how many dimensions should data be generated. Possible values are "2D" and "3D".
    #         Defaults to DEFAULT_PROJECTION ("2D").
    #     """

    def _compare_recent_dance(self, delay):
        """
        Get the comparison of recent dance, based on gathered data from camera and data about dance from viedo.
        """
        last_frame = self.actual_dance.get_skeleton_by_timestamp(self.displayer_timestamp)  #Returns a Skeleton, which has the biggest timestamp from all Skeletons in this Dance. Returns None if there isn't any Skeleon in this list.

        pattern_frame = self.pattern_dance.get_skeleton_by_timestamp(last_frame.timestamp - delay)#what if timestam is negative?

        if not last_frame or not pattern_frame:# triggers when there is no dance data
            #print("no last frame or pattern frame")
            return#returns a none which causes issues in alt_dance_manager

        # if isinstance(EmptySkeleton, )
        #last_frame is the skeleton of the user
        #pattern_frame is the skeleton generated form the video for that time instance

        limbs = {#orientation from their perspective
            #key -> [points, weight(for cumulative error)]
            "right_arm":  [[14,12,24], 0.7],
            "right_hand": [[16,12,24], 1],
            "left_arm":   [[13,11,23], 0.7],
            "left_hand":  [[15,11,23], 1],

            "right_leg":  [[23,24,26], 0.2],
            "right_foot": [[23,24,28], 0.2],
            "left_leg":   [[24,23,25], 0.2],
            "left_foot":  [[24,23,27], 0.2]
            }

        #practice error calculation for one limb
        a_cos, a_sin = last_frame.get_cossin(limbs["right_arm"][0])
        p_cos, p_sin = pattern_frame.get_cossin(limbs["right_arm"][0])
        error_rignt_arm = min(abs(a_cos - p_cos), abs(a_sin - p_sin))

        #actual error calculation for all limbs
        error = 0
        sum = 0
        output = ""
        for limb in limbs:
            a_cos, a_sin = last_frame.get_cossin(limbs[limb][0])
            p_cos, p_sin = pattern_frame.get_cossin(limbs[limb][0])
            #error += min(abs(a_cos - p_cos), abs(a_sin - p_sin)) * limbs[limb][1]
            angle = abs(math.degrees(math.acos(a_cos) - math.acos(p_cos)))
            even = " "
            if angle < 10:
                even = "  "
            elif angle >= 100:
                even = ""
            output += f"{limb}: {int(angle)}{even} | "
            error += angle*limbs[limb][1]
            sum += limbs[limb][1]# update total weight

        output = f"{int(self.displayer_timestamp*1000)}ms, Res (deg): | {output}"
        error = error/sum#adjust error for weight
        #print(a_cos, p_cos)
        #print(error_rignt_arm)
        #print(error)
        return error, output#lets plot it and see if it makes sense


def get_dance_name_from_path(path: str):
    file_name = os.path.basename(path)
    file_name = file_name.split(".")[0]
    return file_name

def add_current_timestamp_to_filename(filename):
    now = datetime.now()
    datetime_format = "%Y-%m-%d_%H-%M-%S"
    datetime_text = now.strftime(datetime_format)
    new_filename = f"{filename}_{datetime_text}"

    return new_filename


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
    return Dance(skeleton_list, name=get_dance_name_from_path(data_file))


def get_dance_data_from_video(video_path, dimension = DEFAULT_PROJECTION):

    data = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    while True:
        success, img = cap.read()
        if not success:
            return Dance(data, name=get_dance_name_from_path(video_path))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = estaminate_from_frame(imgRGB)
        timestamp = current_frame / fps
        skeleton = create_skeleton_from_raw_pose_landmarks(results.pose_world_landmarks, timestamp, dimension)
        data.append(skeleton)
        current_frame += 1

def getNthTupleElementFromList(L: list, n:int):
    #when u have a list of tuples this function will return only the n-th element of each tuple as a list
    R = []
    for element in L:
        R.append(element[n])
    return R

def getGrade(score: int):
    #returns a string name for a score you are given
    #fell free to adjust
    scoring_system = {
        #!keys have to be increasing and in [0,180]
        10: "very good",
        30: "good",
        50: "almost",
        100: "are u sure you are dancing?",
        180: "u are not good at this"
    }
    for limit, result in scoring_system.items():
        if score <= limit:
            return result + ", " + str(int((score/180)*100)) + "% pose difference"
    return "ERROR score so terrible it was out of bounds"


# class MockDanceManager(DanceManager):
#     def __init__(self, pattern_dance: str, actual_dance: Dance) -> None:
#         """Mock testing class for DanceManager.

#         Args:
#             pattern_dance (Dance): A Dance class object, which mocks data from video.
#             actual_dance (Dance): A Dance class object, which mocks data from camera.
#         """

#         self._dance_video_path = None
#         self._pattern_dance = pattern_dance
#         self._actual_dance = actual_dance
#         self._dance_displayer_thread = None
#         self._dance_data_getter_thread = None
#         self._displayer_timestamp = 0
#         self._is_video_being_played = False



#     # def compare_dances(self):
#         #last_frame = self.actual_dance.get_last_skeleton()
#         # if not last_frame:
#         #     return
#         # pattern_frame = self.pattern_dance.get_skeleton_by_timestamp(last_frame.timestamp)

#         # for lm in last_frame.landmarks():
#         #     landmark_id = lm.id
#         #     patt_lm = pattern_frame.get_landmark_by_id(landmark_id)
#         #     error = 0
#         #     if not lm:
#         #         #Camera could not find a person
#         #         pass
#         #     elif not patt_lm:
#         #         #Person could not be found in reference video
#         #         pass
#         #     else:
#         #         landmark_error = math.sqrt((lm.x - patt_lm.x)**2 + (lm.y - patt_lm.y)**2 + (lm.z - patt_lm.z)**2)
#         #         error += landmark_error

#         # print(f"{last_frame.timestamp}: {error}")
#     #     self._is_video_being_played = True
#     #     time_start = time.time()
#     #     dance_time = self.pattern_dance.get_last_skeleton().timestamp

#     #     values = []
#     #     inv_values =[]
#     #     base_values = []
#     #     t = []

#     #     while self._is_video_being_played:
#     #         depth = 0.5
#     #         n = 10
#     #         radius = numpy.linspace(0,-depth,n)
#     #         results = []

#     #         for delay in radius:
#     #             res = self.alt_compare_recent_dance(delay)
#     #             if res:
#     #                 score,output = res
#     #                 #print(res)
#     #                 #results.append(score)
#     #                 if score != None:
#     #                     results.append((score,output))

#     #         if not results:
#     #             print("check2")

#     #         values.append(min(results))
#     #         inv_values.append(max(results))
#     #         base_values.append(results[0])

#     #         t.append(self._displayer_timestamp)
#     #         #print(f"time: {self._displayer_timestamp}, score:{min(results)}, inv_score{max(results)}")
#     #         self._displayer_timestamp = time.time() - time_start
#     #         if self._displayer_timestamp > dance_time:
#     #             self._is_video_being_played = False

#     #     #values is a list of tuples first one being score second one being output
#     #     #so now since we dont need it we need to strip the second element form the tupels in the list
#     #     values = getNthTupleElementFromList(values, 0)
#     #     base_values = getNthTupleElementFromList(base_values, 0)
#     #     inv_values = getNthTupleElementFromList(inv_values, 0)

#     #     avg_value = sum(values)/len(values)
#     #     inv_avg_value = sum(inv_values)/len(inv_values)
#     #     avg_base = sum(base_values)/len(base_values)

#     #     print(f"average score is: {avg_value}, average worst score is: {inv_avg_value}, base score is {avg_base}")
#     #     print(f"your score is: {getGrade(avg_value)}")

#     #     if len(values) == len(inv_values) == len(base_values) == len(t):
#     #         plt.plot(t,inv_values, linewidth = 1, label = "worst")
#     #         plt.plot(t,base_values, linewidth = 1, label = "base")
#     #         plt.plot(t,values, linewidth = 1,label = "best")
#     #         plt.legend()
#     #         plt.show()
#     #     else:
#     #         print(f"Pyplot lists not matching {len(values)}, {len(inv_values)}, {len(base_values)}, {len(t)}")
#     #     #PLOT THE VALUES

#     def compare_dances_live(self, timestep: float):
#         # self._is_video_being_played = True
#         # time_start = time.time()
#         # dance_time = self.pattern_dance.get_last_skeleton().timestamp

#         # values = []
#         # inv_values =[]
#         # base_values = []
#         # t = []
#         # t_0 = time_start
#         # value_record = []

#         # while self._is_video_being_played:
#         #     results = []

#         #     for delay in radius:
#         #         res = self.alt_compare_recent_dance(delay)
#         #         if res:
#         #             score,output = res
#         #             #print(res)
#         #             #results.append(score)
#         #             if score != None:
#         #                 results.append((score,output))

#         #     if not results:
#         #         print("check2")

#         #     values.append(min(results))
#         #     inv_values.append(max(results))
#         #     base_values.append(results[0])

#         #     t.append(self._displayer_timestamp)
#         #     #print(f"time: {self._displayer_timestamp}, score:{min(results)}, inv_score{max(results)}")
#         #     #print(f"{self.displayer_timestamp} t_0: {t_0}")
#         #     if (time.time() - t_0) >= timestep:#when we output the result
#         #         t_0 = time.time()
#         #         values = getNthTupleElementFromList(values, 0)
#         #         value_record.extend(values)
#         #         avg_value = sum(values)/len(values)
#         #         #report is what we want the user to see, here we print it
#         #         report = getGrade(avg_value)
#         #         print(report)

#         #         values.clear()



#         #     self._displayer_timestamp = time.time() - time_start
#         #     if self._displayer_timestamp > dance_time:
#         #         self._is_video_being_played = False

#         #values is a list of tuples first one being score second one being output
#         #so now since we dont need it we need to strip the second element form the tupels in the list
#         # values = getNthTupleElementFromList(values, 0)
#         # base_values = getNthTupleElementFromList(base_values, 0)
#         # inv_values = getNthTupleElementFromList(inv_values, 0)

#         # avg_value = sum(values)/len(values)
#         # inv_avg_value = sum(inv_values)/len(inv_values)
#         # avg_base = sum(base_values)/len(base_values)
#         return



#     # def _compare_recent_dance(self):
#     #     """
#     #     Get the comparison of recent dance, based on gathered data from camera and data about dance from viedo.
#     #     """
#     #     last_frame = self.actual_dance.get_skeleton_by_timestamp(self.displayer_timestamp)
#     #     if not last_frame:
#     #         return
#     #     pattern_frame = self.pattern_dance.get_skeleton_by_timestamp(last_frame.timestamp)

#     #     for lm in last_frame.landmarks():
#     #         landmark_id = lm.id
#     #         patt_lm = pattern_frame.get_landmark_by_id(landmark_id)
#     #         error = 0
#     #         if not lm:
#     #             #Camera could not find a person
#     #             pass
#     #         elif not patt_lm:
#     #             #Person could not be found in reference video
#     #             pass
#     #         else:
#     #             landmark_error = math.sqrt((lm.x - patt_lm.x)**2 + (lm.y - patt_lm.y)**2 + (lm.z - patt_lm.z)**2)
#     #             error += landmark_error

#     #     print(f"{last_frame.timestamp}: {error}")

#     def alt_compare_recent_dance(self, delay) -> float:#delay can be set to account dancers dealy
#         last_frame = self.actual_dance.get_skeleton_by_timestamp(self.displayer_timestamp)  #Returns a Skeleton, which has the biggest timestamp from all Skeletons in this Dance. Returns None if there isn't any Skeleon in this list.

#         pattern_frame = self.pattern_dance.get_skeleton_by_timestamp(last_frame.timestamp - delay)#what if timestam is negative?

#         if not last_frame or not pattern_frame:# triggers when there is no dance data
#             #print("no last frame or pattern frame")
#             return#returns a none which causes issues in alt_dance_manager

#         # if isinstance(EmptySkeleton, )
#         #last_frame is the skeleton of the user
#         #pattern_frame is the skeleton generated form the video for that time instance

#         limbs = {#orientation from their perspective
#             #key -> [points, weight(for cumulative error)]
#             "right_arm":  [[14,12,24], 0.7],
#             "right_hand": [[16,12,24], 1],
#             "left_arm":   [[13,11,23], 0.7],
#             "left_hand":  [[15,11,23], 1],

#             "right_leg":  [[23,24,26], 0.2],
#             "right_foot": [[23,24,28], 0.2],
#             "left_leg":   [[24,23,25], 0.2],
#             "left_foot":  [[24,23,27], 0.2]
#             }

#         #practice error calculation for one limb
#         a_cos, a_sin = last_frame.get_cossin(limbs["right_arm"][0])
#         p_cos, p_sin = pattern_frame.get_cossin(limbs["right_arm"][0])
#         error_rignt_arm = min(abs(a_cos - p_cos), abs(a_sin - p_sin))

#         #actual error calculation for all limbs
#         error = 0
#         sum = 0
#         output = ""
#         for limb in limbs:
#             a_cos, a_sin = last_frame.get_cossin(limbs[limb][0])
#             p_cos, p_sin = pattern_frame.get_cossin(limbs[limb][0])
#             #error += min(abs(a_cos - p_cos), abs(a_sin - p_sin)) * limbs[limb][1]
#             angle = abs(math.degrees(math.acos(a_cos) - math.acos(p_cos)))
#             even = " "
#             if angle < 10:
#                 even = "  "
#             elif angle >= 100:
#                 even = ""
#             output += f"{limb}: {int(angle)}{even} | "
#             error += angle*limbs[limb][1]
#             sum += limbs[limb][1]# update total weight

#         output = f"{int(self.displayer_timestamp*1000)}ms, Res (deg): | {output}"
#         error = error/sum#adjust error for weight
#         #print(a_cos, p_cos)
#         #print(error_rignt_arm)
#         #print(error)
#         return error, output#lets plot it and see if it makes sense

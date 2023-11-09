import cv2
import mediapipe as mp
import time
from constants import LEFT_ANCHOR_CREATOR_NODE, RIGHT_ANCHOR_CREATOR_NODE, SKELETON_FILE, DEFAULT_PROJECTION, MODEL_PATH
from skeleton import *


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE)

landmarker = PoseLandmarker.create_from_options(options)

mpDraw = mp.solutions.drawing_utils

def estaminate_from_frame(frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    return landmarker.detect(mp_image)

def create_skeleton_from_raw_pose_landmarks(pose_landmarks, timestamp, dimension=DEFAULT_PROJECTION) -> RawSkeleton:
    if pose_landmarks:
        current_frame_data = []
        for id, lm in enumerate(pose_landmarks[0]):
            current_frame_data.append([id, lm.x, lm.y, lm.z])
        left_anchor = current_frame_data[LEFT_ANCHOR_CREATOR_NODE]
        right_anchor = current_frame_data[RIGHT_ANCHOR_CREATOR_NODE]

        anchor_landmark_x = (left_anchor[1] + right_anchor[1]) / 2
        anchor_landmark_y = (left_anchor[2] + right_anchor[2]) / 2

        if dimension == "3D":
            anchor_landmark_z = (left_anchor[3] + right_anchor[3]) / 2
        elif dimension == "2D":
            anchor_landmark_z = 0

        current_frame_data.append([-1, anchor_landmark_x, anchor_landmark_y, anchor_landmark_z])
        frame_skeleton = RawSkeleton(SKELETON_FILE, current_frame_data, timestamp)

    else:
        frame_skeleton = EmptySkeleton(SKELETON_FILE, timestamp)

    return frame_skeleton


def show_video_with_estimation(path):
    cap = cv2.VideoCapture(path)
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = landmarker.detect(imgRGB)
        print(results.pose_landmarks)
        if results.pose_landmarks:
            # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w,c = img.shape
                print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


def get_pose_data_from_single_frame():

    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = estaminate_from_frame(imgRGB)
        print(results.pose_world_landmarks)
        if results.pose_world_landmarks:
            # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_world_landmarks[0]):
                h, w,c = img.shape
                print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        cv2.imshow("Test", imgRGB)
    ret, frame = cap.read()

    cap.release()

def reverse_dictionary(dictionary):
    reversed_dict = {}
    for key, value in dictionary.items():
        if value in reversed_dict:
            raise ValueError("Values in the dictionary are not unique.")
        reversed_dict[value] = key
    return reversed_dict

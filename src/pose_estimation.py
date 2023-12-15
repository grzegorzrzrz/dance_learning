import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import time
import numpy as np
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
            if dimension == "3D":
                current_frame_data.append([id, lm.x, lm.y, lm.z])
            elif dimension == "2D":
                current_frame_data.append([id, lm.x, lm.y, 0])
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

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
      pose_landmarks = pose_landmarks_list[idx]

      # Draw the pose landmarks.
      pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
      ])
      mp.solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        mp.solutions.pose.POSE_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def show_video_with_estimation(path):
    cap = cv2.VideoCapture(path)
    cam = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time_ms = int(1000/fps)
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = estaminate_from_frame(imgRGB)
        # annotated_image = imgRGB
        annotated_image = draw_landmarks_on_image(imgRGB, result)
        cv2.imshow("Actual", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        # time.sleep(1/fps)
        success, img = cam.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = estaminate_from_frame(imgRGB)
        # annotated_image = imgRGB
        annotated_image = draw_landmarks_on_image(imgRGB, result)
        cv2.imshow("Pattern", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        # time.sleep(1/fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

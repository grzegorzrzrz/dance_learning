import cv2
import mediapipe as mp
import time
from constants import LEFT_ANCHOR_CREATOR_NODE, RIGHT_ANCHOR_CREATOR_NODE
from skeleton import Skeleton3D, Skeleton2D

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def show_video_with_estimation():
    cap = cv2.VideoCapture('src/a.mp4')
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
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


def get_data_for_plot_display():

    data = []

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    cap = cv2.VideoCapture('src/c.mp4')
    while True:
        success, img = cap.read()
        if not success:
            return data
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            current_frame_data = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                current_frame_data.append([lm.x, lm.y, lm.z])
            data.append(current_frame_data)


def get_3d_pose_data_from_video(video_path):

    data = []

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    cap = cv2.VideoCapture(video_path)
    while True:
        success, img = cap.read()
        if not success:
            return data
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            current_frame_data = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                current_frame_data.append([id, lm.x, lm.y, lm.z])
            left_anchor = current_frame_data[LEFT_ANCHOR_CREATOR_NODE]
            right_anchor = current_frame_data[RIGHT_ANCHOR_CREATOR_NODE]
            anchor_landmark_x = (left_anchor[1] + right_anchor[1]) / 2
            anchor_landmark_y = (left_anchor[2] + right_anchor[2]) / 2
            anchor_landmark_z = (left_anchor[3] + right_anchor[3]) / 2
            current_frame_data.append([-1, anchor_landmark_x, anchor_landmark_y, anchor_landmark_z])
            frame_skeleton = Skeleton3D("src/skeleton.csv", current_frame_data)
            data.append(frame_skeleton)


def get_2d_pose_data_from_video(video_path):

    data = []

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    cap = cv2.VideoCapture(video_path)
    while True:
        success, img = cap.read()
        if not success:
            return data
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            current_frame_data = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                current_frame_data.append([id, lm.x, lm.y])
            left_anchor = current_frame_data[LEFT_ANCHOR_CREATOR_NODE]
            right_anchor = current_frame_data[RIGHT_ANCHOR_CREATOR_NODE]
            anchor_landmark_x = (left_anchor[1] + right_anchor[1]) / 2
            anchor_landmark_y = (left_anchor[2] + right_anchor[2]) / 2
            current_frame_data.append([-1, anchor_landmark_x, anchor_landmark_y])
            frame_skeleton = Skeleton2D("src/skeleton.csv", current_frame_data)
            data.append(frame_skeleton)
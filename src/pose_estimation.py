import cv2
import mediapipe as mp
import time
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
    cap = cv2.VideoCapture('src/a.mp4')
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




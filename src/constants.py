LEFT_ANCHOR_CREATOR_NODE = 23
RIGHT_ANCHOR_CREATOR_NODE = 24
MODEL_PATH = "src/pose_landmarker_heavy.task"
SKELETON_FILE = "src/skeleton.csv"
DEFAULT_PROJECTION = "2D"
DEFAULT_CHECKING_CAMERA_TIME = 3

DANCE_VIDEOS_PATH = "data/dance_videos"
PATTERN_DANCE_DATA_PATH = "data/pattern_dance_data"
ACTUAL_DANCE_DATA_PATH = "data/actual_dance_data"

NODES_NAME = {

    -1: "anchor",
    0: "nose",
    1: "left_eye_inner",
    2: "left eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left shoulder",
    12: "right_shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right_wrist",
    17: "left pinky",
    18: "right pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left hip",
    24: "right_hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel",
    31: "left foot index",
    32: "right foot index"

}

N_RAW_NODES = len(NODES_NAME)

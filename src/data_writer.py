from skeleton import *
import csv
from typing import List
from constants import NODES_NAME, SKELETON_FILE
from dance import Dance

from pose_estimation import *

def write_data_to_csv_file(dance_data: Dance, path: str, skeleton_file=SKELETON_FILE):

    with open(skeleton_file, "r") as handle:
        csv_reader = csv.DictReader(handle, delimiter=",")
        used_nodes = [int(row["child"]) for row in csv_reader]
        used_nodes.sort()
        used_nodes = [NODES_NAME[id] for id in used_nodes]

    csv_file_names = ["timestamp"]
    for node in used_nodes:
        csv_file_names.append(f"{node}_x")
        csv_file_names.append(f"{node}_y")
        csv_file_names.append(f"{node}_z")

    data_to_write = [csv_file_names]
    for skeleton in dance_data.skeleton_table:
        timestamp = skeleton.timestamp
        current_skeleton_data = [timestamp]
        for landmark in skeleton.landmarks()[1:]:
            current_skeleton_data += [landmark.x, landmark.y, landmark.z]
        data_to_write.append(current_skeleton_data)

    with open(path, "w") as handle:
        writer = csv.writer(handle, delimiter=',')

        for row in data_to_write:
            writer.writerow(row)

# test = get_pose_data_from_video("src/test (1).mp4", "2D")
# write_data_to_csv_file(test, "src/temp.csv")

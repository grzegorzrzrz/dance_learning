from skeleton import *
import csv
from typing import List
from constants import NODES_NAME

from pose_estimation import *

def write_data_to_csv_file(skeleton_list: List[Skeleton3D], path: str, skeleton_file="src/skeleton.csv"):

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
    iter = 0
    for skeleton in skeleton_list:
        current_skeleton_data = [iter]
        for landmark in skeleton.landmarks()[1:]:
            current_skeleton_data += [landmark.x, landmark.y, landmark.z]
        iter += 1
        data_to_write.append(current_skeleton_data)

    with open(path, "w") as handle:
        writer = csv.writer(handle, delimiter=',')

        for row in data_to_write:
            writer.writerow(row)

test = get_3d_pose_data_from_video("src/test (1).mp4")
write_data_to_csv_file(test, "src/temp.csv")

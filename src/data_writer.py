from skeleton import *
import csv
from typing import List

from pose_estimation import *

def write_data_to_csv_file(skeleton_list: List[Skeleton3D], path):
    data_to_write = []
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
write_data_to_csv_file(test, "src/temp.txt")

import sys
from data_writer import write_data_to_csv_file
from dance import get_dance_data_from_video
from constants import PATTERN_DANCE_DATA_PATH
if __name__ == "__main__":
    dance = get_dance_data_from_video(sys.argv[1])
    if len(sys.argv) <= 2:
        data_path = f"{PATTERN_DANCE_DATA_PATH}/{dance.name}.csv"
        write_data_to_csv_file(dance, data_path)
    else:
        write_data_to_csv_file(dance, sys.argv[2])
import sys
from src.data_writer import write_data_to_csv_file
from src.dance import get_dance_data_from_video
if __name__ == "__main__":
    dance = get_dance_data_from_video(sys.argv[1])
    write_data_to_csv_file(dance, sys.argv[2])
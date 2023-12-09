from pose_display import plot_ineractive_double_dance_2d
import sys
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        plot_ineractive_double_dance_2d(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        plot_ineractive_double_dance_2d(sys.argv[1])

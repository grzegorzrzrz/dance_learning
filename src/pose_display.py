import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pose_estimation import *
from dance import get_dance_data_from_video, create_dance_from_data_file, Dance




def update_3d_plot(frame_numer, ax, data):
    plt.cla()
    ax.set(xlim3d=(-2, 2), xlabel='X')
    ax.set(ylim3d=(-2, 2), ylabel='Y')
    ax.set(zlim3d=(-2, 2), zlabel='Z')
    x_data = [landmark.x for landmark in data[frame_numer].landmarks()]
    y_data = [landmark.y for landmark in data[frame_numer].landmarks()]
    z_data = [landmark.z for landmark in data[frame_numer].landmarks()]
    ax.scatter(x_data, y_data, z_data)

def update_2d_plot(frame_number, ax, data):
    plt.cla()
    ax.set(xlim=(-4, 4), xlabel='X')
    ax.set(ylim=(-4, 4), ylabel='Y')
    plot_points_for_2d_plot(ax, data[frame_number])

def update_double_2d_plot(frame_number, ax, p_data: Dance, a_data: Dance):
    plt.cla()
    ax.set(xlim=(-4, 4), xlabel='X')
    ax.set(ylim=(-4, 4), ylabel='Y')

    plot_points_for_2d_plot(ax, p_data.skeleton_table[frame_number], "b")
    a_skeleton = a_data.get_skeleton_by_timestamp(p_data.skeleton_table[frame_number].timestamp)
    for landmark in a_skeleton.landmarks():
        if not landmark:
            return
    plot_points_for_2d_plot(ax, a_skeleton, "r")

def plot_points_for_2d_plot(ax, data: Skeleton, color=""):
    x_data = [landmark.x for landmark in data.landmarks()]
    y_data = [-landmark.y for landmark in data.landmarks()]
    ax.plot(x_data, y_data, color + ".")

def plot_data_from_3d_skeleton(dance_file):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    skeleton = create_dance_from_data_file(dance_file)
    data = skeleton.skeleton_table
    ani = animation.FuncAnimation(fig, update_3d_plot, len(data), fargs=([ax, data]), interval=17)

    plt.show()

def plot_data_from_2d_skeleton(dance_file):
    fig, ax = plt.subplots()

    skeleton = create_dance_from_data_file(dance_file)
    data = skeleton.skeleton_table
    ani = animation.FuncAnimation(fig, update_2d_plot, len(data), fargs=([ax, data]), interval=17)

    plt.show()

def compare_dances_from_file(pattern_dance_path, actual_dance_path):
    fig, ax = plt.subplots()
    pattern_dance = create_dance_from_data_file(pattern_dance_path)
    actual_dance = create_dance_from_data_file(actual_dance_path)
    ani = animation.FuncAnimation(fig, update_double_2d_plot,
                                  len(pattern_dance.skeleton_table), fargs=([ax, pattern_dance, actual_dance]), interval=17)

    plt.show()

compare_dances_from_file("src/temp.csv", "src/atemp.csv")

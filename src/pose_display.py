import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from pose_estimation import *
import numpy as np
from dance import create_dance_from_data_file, Dance




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

def plot_data_from_2d_skeleton(dance_file):
    fig, ax = plt.subplots()

    skeleton = create_dance_from_data_file(dance_file)
    data = skeleton.skeleton_table
    ani = animation.FuncAnimation(fig, update_2d_plot, len(data), fargs=([ax, data]), interval=17)

    plt.show()

def plot_ineractive_double_dance_2d(pattern_dance_path, actual_dance_path):

    def get_data_to_plot(dance: Dance, timestamp):
        x_data = []
        y_data = []
        id_data = []
        skeleton = dance.get_skeleton_by_timestamp(timestamp)
        for landmark in skeleton.landmarks():
            if landmark:
                x_data.append(landmark.x)
                y_data.append(-landmark.y)
                id_data.append(landmark.id)
            else:
                x_data.append(0)
                y_data.append(0)
                id_data.append("X")

        return x_data, y_data, id_data

    fig, ax = plt.subplots(figsize=(5,10))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-4.5, 4.5)
    axtime = fig.add_axes([0.25, 0.93, 0.65, 0.03])
    pattern_dance = create_dance_from_data_file(pattern_dance_path)
    actual_dance = create_dance_from_data_file(actual_dance_path)
    max_time = pattern_dance.get_last_skeleton().timestamp

    time_slider = Slider(axtime, label="timestamp",
                         valmin=0, valmax=max_time, valinit=0)

    patt_x, patt_y, patt_id = get_data_to_plot(pattern_dance, 0)
    act_x, act_y, act_id = get_data_to_plot(actual_dance, 0)
    patt_texts = []
    act_texts = []
    for i in range(len(patt_x)):
        patt_texts.append(ax.text(patt_x[i], patt_y[i], patt_id[i], c="b"))
        act_texts.append(ax.text(act_x[i], act_y[i], act_id[i], c="r"))

    ax.set_title(f"Time: 0")

    def update(val):
        timestamp = time_slider.val
        patt_x, patt_y, patt_id = get_data_to_plot(pattern_dance, timestamp)
        act_x, act_y, act_id = get_data_to_plot(actual_dance, timestamp)
        for i in range(len(patt_x)):
            patt_texts[i].set_x(patt_x[i])
            patt_texts[i].set_y(patt_y[i])
            patt_texts[i].set_text(patt_id[i])
            act_texts[i].set_x(act_x[i])
            act_texts[i].set_y(act_y[i])
            act_texts[i].set_text(act_id[i])

        ax.set_title(f"Time: {timestamp}")

        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    plt.show()

def compare_dances_from_file(pattern_dance_path, actual_dance_path):
    fig, ax = plt.subplots()
    pattern_dance = create_dance_from_data_file(pattern_dance_path)
    actual_dance = create_dance_from_data_file(actual_dance_path)
    ani = animation.FuncAnimation(fig, update_double_2d_plot,
                                  len(pattern_dance.skeleton_table), fargs=([ax, pattern_dance, actual_dance]), interval=17)

    plt.show()

plot_ineractive_double_dance_2d("static/pattern.csv", "static/actual.csv")
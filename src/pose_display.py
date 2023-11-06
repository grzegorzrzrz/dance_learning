import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pose_estimation import *




def update_3d_plot(frame_numer, ax, data):
    plt.cla()
    ax.set(xlim3d=(-2, 2), xlabel='X')
    ax.set(ylim3d=(-2, 2), ylabel='Y')
    ax.set(zlim3d=(-2, 2), zlabel='Z')
    x_data = [landmark.x for landmark in data[frame_numer].landmarks()]
    y_data = [landmark.y for landmark in data[frame_numer].landmarks()]
    z_data = [landmark.z for landmark in data[frame_numer].landmarks()]
    ax.scatter(x_data, y_data, z_data)
    for landmark in data[frame_numer].landmarks():
        if landmark.parent_landmark():
            x = [landmark.x, landmark.parent_landmark().x]
            y = [landmark.y, landmark.parent_landmark().y]
            z = [landmark.z, landmark.parent_landmark().z]
            ax.plot(x, y, z)

def update_2d_plot(frame_numer, ax, data):
    plt.cla()
    ax.set(xlim=(-4, 4), xlabel='X')
    ax.set(ylim=(-4, 4), ylabel='Y')
    x_data = [landmark.x for landmark in data[frame_numer].landmarks()]
    y_data = [-landmark.y for landmark in data[frame_numer].landmarks()]
    ax.plot(x_data, y_data, ".")
    for landmark in data[frame_numer].landmarks():
        if landmark.parent_landmark():
            x = [landmark.x, landmark.parent_landmark().x]
            y = [-landmark.y, -landmark.parent_landmark().y]
            ax.plot(x, y)

def plot_raw_data():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    data = get_data_for_plot_display()
    ani = animation.FuncAnimation(fig, update_3d_plot, len(data), fargs=([ax, data]), interval=17)

    plt.show()

def plot_data_from_3d_skeleton():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    data = get_dance_data_from_video("src/test (1).mp4")
    ani = animation.FuncAnimation(fig, update_3d_plot, len(data), fargs=([ax, data]), interval=17)

    plt.show()

def plot_data_from_2d_skeleton():
    fig, ax = plt.subplots()

    data = get_dance_data_from_video("src/test (1).mp4", "2D")
    ani = animation.FuncAnimation(fig, update_2d_plot, len(data), fargs=([ax, data]), interval=17)

    plt.show()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pose_estimation import get_data_for_plot_display




def update_plot(frame_numer, data):
    plt.cla()
    ax.set(xlim3d=(0, 1), xlabel='X')
    ax.set(ylim3d=(0, 1), ylabel='Y')
    ax.set(zlim3d=(0, 1), zlabel='Z')
    x_data = [point[0] for point in data[frame_numer]]
    y_data = [point[1] for point in data[frame_numer]]
    z_data = [point[2] for point in data[frame_numer]]
    ax.scatter(x_data, y_data, z_data)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

data = get_data_for_plot_display()
ani = animation.FuncAnimation(fig, update_plot, len(data), fargs=([data]), interval=17)

plt.show()

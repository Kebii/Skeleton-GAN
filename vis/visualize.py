import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

num_node = 25
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
graph = [(i - 1, j - 1) for (i, j) in inward_ori_index]

def visualise(data, graph=None, is_3d=True, k=0):
    '''
    Visualise skeleton data using matplotlib

    Args:
        data: tensor of shape (B x C x T x V)
        graph: graph representation for skeleton
        is_3d: set true for 3d skeletons
    '''
    N, C, T, V = data.shape

    plt.ion()
    fig = plt.figure()
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    if graph is None:
        pose = ax.plot(np.zeros(V), np.zeros(V), 'b.')[0]
        ax.axis([-300, 300, -300, 300])
        for t in range(T):
            pose.set_xdata(data[k, 0, t, :])
            pose.set_ydata(data[k, 1, t, :])
            fig.canvas.draw()
            plt.pause(0.01)
    else:
        edge = graph
        pose = []
        for i in range(len(edge)):
            if is_3d:
                pose.append(ax.plot(np.zeros(3), np.zeros(3), 'b-')[0])
            else:
                pose.append(ax.plot(np.zeros(2), np.zeros(2), 'b-')[0])
        ax.axis([-300, 300, -300, 300])
        if is_3d:
            ax.set_zlim3d(-300, 300)
        for t in range(T):
            for i, (v1, v2) in enumerate(edge):
                x1 = data[k, :2, t, v1]
                x2 = data[k, :2, t, v2]
                if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                    pose[i].set_xdata(data[k, 0, t, [v1, v2]])
                    pose[i].set_ydata(data[k, 1, t, [v1, v2]])
                    if is_3d:
                        pose[i].set_3d_properties(data[k, 2, t, [v1, v2]])
            fig.canvas.draw()
            plt.pause(0.01)

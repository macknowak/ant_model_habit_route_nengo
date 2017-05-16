import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_bg(data_path):
    """Plot basal ganglia."""
    plt.figure()

    bg = np.loadtxt(os.path.join(data_path, "bg.txt"))

    plt.plot(bg[:,0], bg[:,1:-1])
    plt.plot(bg[:,0], bg[:,-1], label="default")
    plt.xlabel("Time (s)")
    plt.ylim(-0.2, 1.2)
    plt.ylabel("Utility")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("bg")


def plot_bot_alv(data_path):
    """Plot average landmark vector."""
    plt.figure()

    bot_alv = np.loadtxt(os.path.join(data_path, "botalv.txt"))

    plt.plot(bot_alv[:,0], bot_alv[:,1], label="x")
    plt.plot(bot_alv[:,0], bot_alv[:,2], label="y")
    plt.xlabel("Time (s)")
    plt.ylim(-1.2, 1.2)
    plt.ylabel("Component (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("bot_alv")


def plot_bot_orient(data_path):
    """Plot robot orientation."""
    plt.figure()

    bot_orient = np.loadtxt(os.path.join(data_path, "botorient.txt"))

    plt.plot(bot_orient[:,0], bot_orient[:,1])
    plt.xlabel("Time (s)")
    plt.ylim(-180.0, 180.0)
    plt.yticks(np.arange(-180, 181, 45))
    plt.ylabel("Orientation (deg)")
    plt.title("bot_orient")


def plot_bot_pos(data_path):
    """Plot robot position."""
    plt.figure()

    bot_pos = np.loadtxt(os.path.join(data_path, "botpos.txt"))

    plt.plot(bot_pos[:,0], bot_pos[:,1], label="x")
    plt.plot(bot_pos[:,0], bot_pos[:,2], label="y")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("bot_pos")


def plot_bot_pos_aerial(data_path):
    """Plot robot position (aerial view)."""
    plt.figure()

    bot_pos = np.loadtxt(os.path.join(data_path, "botpos.txt"))
    waypts_pos = np.loadtxt(os.path.join(data_path, "wayptpos.txt"))
    lands_pos = np.loadtxt(os.path.join(data_path, "landpos.txt"))

    plt.plot(bot_pos[:,1], bot_pos[:,2], '-', color='midnightblue')
    plt.plot(waypts_pos[:-1,0], waypts_pos[:-1,1], 'o',
             markerfacecolor='chocolate')
    plt.plot(waypts_pos[-1,0], waypts_pos[-1,1], 'o', markerfacecolor='maroon')
    plt.plot(lands_pos[:,0], lands_pos[:,1], 'og', markerfacecolor='darkgreen')
    plt.xlabel("Distance (m)")
    plt.ylabel("Distance (m)")
    plt.axis('scaled')
    plt.axis((-2.5, 2.5, -2.5, 2.5))
    plt.title("bot_pos")


def plot_diff_alv(data_path):
    """Plot difference between the average landmark vector and the average
       landmark vector at the next waypoint."""
    plt.figure()

    diff_alv = np.loadtxt(os.path.join(data_path, "diffalv.txt"))

    plt.plot(diff_alv[:,0], diff_alv[:,1], label="x")
    plt.plot(diff_alv[:,0], diff_alv[:,2], label="y")
    plt.xlabel("Time (s)")
    plt.ylim(-1.2, 1.2)
    plt.ylabel("Component (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("diff_alv")


def plot_gain_diff_alv(data_path):
    """Plot gain of the difference between the average landmark vector and the
       average landmark vector at the next waypoint."""
    plt.figure()

    gain_diff_alv = np.loadtxt(os.path.join(data_path, "gaindiffalv.txt"))

    plt.plot(gain_diff_alv[:,0], gain_diff_alv[:,1])
    plt.xlabel("Time (s)")
    plt.ylim(-0.2, 1.2)
    plt.ylabel("Gain")
    plt.title("gain_diff_alv")


def plot_next_waypt(data_path):
    """Plot next waypoint."""
    plt.figure()

    next_waypt = np.loadtxt(os.path.join(data_path, "nextwaypt.txt"))
    waypts_names = np.loadtxt(os.path.join(data_path, "wayptname.txt"),
                              dtype=str)

    plt.plot(next_waypt[:,0], next_waypt[:,1:])
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, 1.5)
    plt.ylabel("Similarity")
    plt.legend(waypts_names, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("next_waypt")


def plot_prev_waypt(data_path):
    """Plot previous waypoint."""
    plt.figure()

    prev_waypt = np.loadtxt(os.path.join(data_path, "prevwaypt.txt"))
    waypts_names = np.loadtxt(os.path.join(data_path, "wayptname.txt"),
                              dtype=str)

    plt.plot(prev_waypt[:,0], prev_waypt[:,1:])
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, 1.5)
    plt.ylabel("Similarity")
    plt.legend(waypts_names, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("prev_waypt")


def plot_thal(data_path):
    """Plot thalamus."""
    plt.figure()

    thal = np.loadtxt(os.path.join(data_path, "thal.txt"))

    plt.plot(thal[:,0], thal[:,1:-1])
    plt.plot(thal[:,0], thal[:,-1], label="default")
    plt.xlabel("Time (s)")
    plt.ylim(-0.2, 1.2)
    plt.ylabel("Activation")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("thal")


# Process command line arguments
n_args = len(sys.argv) - 1
if n_args < 1:
    sys.stderr.write("{}: error: too few arguments\n".format(sys.argv[0]))
    sys.exit(2)
if n_args > 1:
    sys.stderr.write("{}: error: too many arguments\n".format(sys.argv[0]))
    sys.exit(2)
data_path = sys.argv[1]

# Make plots
plot_bot_pos_aerial(data_path)
plot_bot_pos(data_path)
plot_bot_orient(data_path)
plot_bot_alv(data_path)
plot_prev_waypt(data_path)
plot_next_waypt(data_path)
plot_diff_alv(data_path)
plot_gain_diff_alv(data_path)
plot_bg(data_path)
plot_thal(data_path)

# Show figures
plt.show()

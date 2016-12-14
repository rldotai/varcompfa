import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

##############################################################################
# Plotting
##############################################################################
def plot_phase(trajectory, smoothing=5):
    """Plotting the phase space of a trajectory."""
    trajectory = np.apply_along_axis(window_avg, 0, trajectory, n=smoothing)
    _x, _y = trajectory.T
    plt.plot(_x, _y)
    plt.show()


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

##############################################################################
# Plotting
##############################################################################
def plot_phase(trajectory, smoothing=5):
    """Plotting the phase space of a trajectory."""
    # TODO: Have this return the figure and axes
    trajectory = np.apply_along_axis(window_avg, 0, trajectory, n=smoothing)
    _x, _y = trajectory.T
    plt.plot(_x, _y)
    plt.show()

def __plotting_state_values():
    # TODO: Make this into a proper function
    # Plotting
    g_run = run_df.groupby(level='run')

    fig, ax = plt.subplots()
    colors = mpl.cm.viridis(np.linspace(0, 1, len(states)))
    colors = colors[:, :3] # Remove alpha values

    for name in g_run.groups:
        group = g_run.get_group(name)
        xdata = group.index.get_level_values('t')
        ydata = np.vstack(group['direct_variances']).T
        for ix, s in enumerate(states):
            ax.plot(xdata, ydata[ix], color=(*colors[ix], 0.4))

    ax.set_ylim([0, 1.5])
    ax.set_title("Direct $\lambda$={lmbda} $\lambda$_bar={lmbda_bar}".format(**params))

    # True values
    x_lim = ax.get_xlim()
    for s, ix in enumerate(states):
        ax.hlines(summary_df['var_kp_return'][s], *x_lim, colors=(*colors[ix], 1), linestyles='--')

    plt.show()
    ##########

import os
 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns

import numpy as np

#cmp: 'summer', 'rainbow', 'viridis', cm.coolwarm

def plot_contour(X, Y, Z, levels, scatter_func=None):
  if levels is None:
    vmin, vmax, vlevel = 0.1, 10, 0.5
    levels = np.arange(vmin, vmax, vlevel)
  # --------------------------------------------------------------------
  # Plot 2D contours
  # --------------------------------------------------------------------
  fig = plt.figure()
  CS = plt.contour(X, Y, Z, cmap=cm.coolwarm, levels=levels)
  plt.clabel(CS, inline=1, fontsize=8)
  if scatter_func is not None:
    optim_idx = scatter_func(Z)
    plt_scatter = plt.scatter(X.reshape(-1)[optim_idx], Y.reshape(-1)[optim_idx], s=100 , c='red', marker='*', alpha=0.8)
    x_idx, y_idx = X.reshape(-1)[optim_idx], Y.reshape(-1)[optim_idx]
    plt.annotate("(%.2f,%.2f)"%(x_idx, y_idx), xy = (x_idx, y_idx), xytext = (0, 10), textcoords='offset points', color='black', weight='heavy', bbox=dict(boxstyle='round,pad=0.5', fc='red', ec='k', lw=1, alpha=0.5))
#    plt_scatter.set_zorder(1)
    plt.setp(plt_scatter, 'zorder', 3)
#    plt.setp(CS, 'zorder', 0)
  return fig

def plot_2D(X, Y, Z, landscape_dir, landscape_name, levels=None, show=False, azim=-60, elev=30, scatter_func=None):
  if show:
    matplotlib.use('Agg')

#  if levels is None:
#    vmin, vmax, vlevel = 0.1, 10, 0.5
#    levels = np.arange(vmin, vmax, vlevel)
#  # --------------------------------------------------------------------
#  # Plot 2D contours
#  # --------------------------------------------------------------------
#  fig = plt.figure()
#  CS = plt.contour(X, Y, Z, cmap=cm.coolwarm, levels=levels)
#  plt.clabel(CS, inline=1, fontsize=8)
#  if scatter_func is not None:
#    optim_idx = scatter_func(Z)
#    plt_scatter = plt.scatter(X.reshape(-1)[optim_idx], Y.reshape(-1)[optim_idx], s=100 , c='red', marker='*', alpha=0.8)
#    x_idx, y_idx = X.reshape(-1)[optim_idx], Y.reshape(-1)[optim_idx]
#    plt.annotate("(%.2f,%.2f)"%(x_idx, y_idx), xy = (x_idx, y_idx), xytext = (0, 10), textcoords='offset points', color='black', weight='heavy', bbox=dict(boxstyle='round,pad=0.5', fc='red', ec='k', lw=1, alpha=0.5))
##    plt_scatter.set_zorder(1)
#    plt.setp(plt_scatter, 'zorder', 5)
##    plt.setp(CS, 'zorder', 0)
  fig = plot_contour(X, Y, Z, levels, scatter_func)


  save_name = os.path.join(landscape_dir, '%s_2dcontour.pdf'%landscape_name)
  fig.savefig(save_name, dpi=300,
              bbox_inches='tight', format='pdf')
  
  fig = plt.figure()
  save_name = os.path.join(landscape_dir, '%s_2dcontourf.pdf'%landscape_name)
  print(save_name)
  CS = plt.contourf(X, Y, Z, cmap='summer', levels=levels)
  fig.savefig(save_name, dpi=300,
              bbox_inches='tight', format='pdf')
  plt.close(fig)
  
  # --------------------------------------------------------------------
  # Plot 2D heatmaps
  # --------------------------------------------------------------------
  fig = plt.figure() 
  sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=levels.min(), vmax=levels.max(),
                         xticklabels=False, yticklabels=False)
  sns_plot.invert_yaxis()
  save_name = os.path.join(landscape_dir, '%s_2dheat.pdf'%landscape_name)
  sns_plot.get_figure().savefig(save_name,
                                dpi=300, bbox_inches='tight', format='pdf')
  plt.close(fig)

  # --------------------------------------------------------------------
  # Plot 3D surface
  # --------------------------------------------------------------------
  fig = plt.figure()
  ax = Axes3D(fig, azim=azim, elev=elev)
  projections = []
  def on_click(event):
    azim, elev = ax.azim, ax.elev
    projections.append((azim, elev))
    print(azim, elev)

  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  ax.contourf(X, Y, Z, zdir='Z', offset=Z.min()-0.1, cmap=cm.coolwarm)
#  ax.set_zlim(0, 1.5)
  fig.colorbar(surf, shrink=0.5, aspect=5)

  save_name = os.path.join(landscape_dir, '%s_3dsurface.pdf'%landscape_name)
  print(save_name)

  fig.savefig(save_name, dpi=300,
              bbox_inches='tight', format='pdf')
  if show:
    cid = fig.canvas.mpl_connect('button_release_event', on_click)
    plt.show()
  plt.close(fig)


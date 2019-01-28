import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def twod_plot(d, quantity, region=[0, 0, 0, 0], zoom=None, pos='ll',
              bbox=(2.1, 0.5), alpha=1.0, ec='k'):
    """
    Plot the given quantity in a x-y plot. Add a zoom-in to a certain region.

    Arguments:
    ----------

    quantity : array
        which quantity to plot


    Keywords:
    ---------
    d : pylacompass dataset

    region : list
        list of [x0, x1, y0, y1] into which to zoom

    zoom : float
        zoom level, none adds no zoom-in

    pos : str
        where to put the zoom-in:
        'll': lower left
        'ul': upper left
        'lr': lower right
        'ur': upper right
        'r': right next to the main axes

    bbox : 2 element tuple
        bounding box position where to place the box in case of pos=='r'

    ec : color
        color of the lines connecting the inset

    alpha : float
        alpha value of the lines connecting the inset

    Output:
    -------

    f : figure

    """

    x0, x1, y0, y1 = region

    f, ax = plt.subplots(figsize=(6, 5))

    cc = ax.pcolormesh(d.xy1, d.xy2, np.log10(quantity.T + 1e-45), rasterized=True)
    plt.colorbar(cc)
    ax.set_aspect(1)

    if zoom is not None:

        if pos == 'll':
            loc1 = 2
            loc2 = 4
            loc  = 3
        elif pos == 'ur':
            loc1 = 2
            loc2 = 4
            loc = 1
        elif pos == 'ul':
            loc1 = 1
            loc2 = 3
            loc = 2
        elif pos == 'lr':
            loc1 = 1
            loc2 = 3
            loc = 4
        elif pos == 'r':
            loc1 = 2
            loc2 = 3
            loc = 5

        if pos == 'r':
            axins = zoomed_inset_axes(ax, zoom, loc=5, bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
        else:
            axins = zoomed_inset_axes(ax, zoom, loc=loc)

        axins.pcolormesh(d.xy1, d.xy2, np.log10(quantity.T + 1e-45), rasterized=True)
        axins.set_aspect(1)
        axins.set_xlim(x0, x1)
        axins.set_ylim(y0, y1)

        axins.axes.get_xaxis().set_visible(False)
        axins.axes.get_yaxis().set_visible(False)

        mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc='none', ec=ec, alpha=alpha)

    return f

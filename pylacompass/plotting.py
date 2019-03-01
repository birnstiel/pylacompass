import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def twod_plot(d, quantity, region=[0, 0, 0, 0], zoom=None, pos='ll',
              bbox=(2.1, 0.5), alpha=1.0, ec='k', fct='pcolormesh',
              r_unit=1, cb_orientation='vertical',
              **kwargs):
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
        values assumed to be already scaled with r_unit

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

    fct : bound method
        with bound method of the axis to use for plotting, default pcolormesh

    r_unit : float
        dividing radius by this quantity, default 1, to be used for cgs->AU

    cb_orientation : str
        orientation parameter passed to the colorbar: vertical or horizontal

    **kwargs : keywords
        other keywords to be passed to the plotting function fct

    Output:
    -------

    f : figure

    """

    x0, x1, y0, y1 = region

    if pos == 'r':
        f, ax = plt.subplots(figsize=(10, 5))
    else:
        f, ax = plt.subplots(figsize=(6, 5))

    ax.set_aspect(1)

    cc = getattr(ax, fct)(
        d.xy1 / r_unit, d.xy2 / r_unit,
        np.log10(quantity.T + 1e-45), rasterized=True, **kwargs)

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

        getattr(axins, fct)(
            d.xy1 / r_unit, d.xy2 / r_unit,
            np.log10(quantity.T + 1e-45), rasterized=True, **kwargs)
        axins.set_aspect(1)
        axins.set_xlim(x0, x1)
        axins.set_ylim(y0, y1)

        axins.axes.get_xaxis().set_visible(False)
        axins.axes.get_yaxis().set_visible(False)

        mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc='none', ec=ec, alpha=alpha)

        plt.colorbar(cc, orientation=cb_orientation)

    return f

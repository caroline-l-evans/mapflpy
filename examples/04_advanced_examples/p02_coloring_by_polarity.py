"""
Calculating Field Line Polarity
===============================

Classify field line output and plot the results.

This example demonstrates how to use the :func:`~mapflpy.utils.get_fieldline_polarity` function
to classify tracing output into the following categories:

- Open field lines connected to positive polarity regions at the domain inner boundary
- Open field lines connected to negative polarity regions at the domain inner boundary
- Closed field lines
- Disconnected field lines
- Field lines that failed to fully resolve within the tracing buffer size
"""
# sphinx_gallery_start_ignore
import os
if 'SPHINX_GALLERY_BUILD' not in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')
# sphinx_gallery_end_ignore

import matplotlib
import matplotlib.pyplot as plt
from psi_io import np_interpolate_slice_from_hdf

from mapflpy.tracer import Tracer
from mapflpy.utils import get_fieldline_polarity, plot_traces, plot_sphere
from psi_data import fetch_mas_data

# %%
# Load in the magnetic field files and instantiate the Tracer

magnetic_field_files = fetch_mas_data(domains="cor", variables="br,bt,bp")
tracer = Tracer(*magnetic_field_files)

# %%
# Here we use the default launch points – a fibonacci lattice at 1.01 Rsun – and perform
# forward tracing of 256 field lines

traces = tracer.trace_fwd(n=256)

# %%
# The :func:`~mapflpy.utils.get_fieldline_polarity` function is used to classify the traces;
# the result of this operation yields an array of :class:`~mapflpy.globals.Polarity` enum values
# (where the possible values are listed in the table below):
#
# +-------------------+---------------+------------------------------------------------------------+
# | Enum Value        | Integer Value | Description                                                |
# +===================+===============+============================================================+
# | :data:`R0_R1_NEG` | -2            | Open field line with negative polarity at inner boundary   |
# +-------------------+---------------+------------------------------------------------------------+
# | :data:`R0_R0`     | -1            | Closed field line                                          |
# +-------------------+---------------+------------------------------------------------------------+
# | :data:`ERROR`     | 0             | Field line that failed to resolve within the buffer size   |
# +-------------------+---------------+------------------------------------------------------------+
# | :data:`R1_R1`     | 1             | Disconnected field line                                    |
# +-------------------+---------------+------------------------------------------------------------+
# | :data:`R0_R1_POS` | 2             | Open field line with positive polarity at inner boundary   |
# +-------------------+---------------+------------------------------------------------------------+
#
# To properly classify the field lines, we need to provide the inner and outer radii of the
# domain and the filepath to the magnetic field file *viz.* used to determine the polarity at
# the inner boundary (here the radial component of the magnetic field).
#
# .. note::
#    The ``atol`` ("absolute tolerance") parameter is used to determine whether a field line
#    footpoint is sufficiently close to the inner or outer boundary to be considered "anchored" there.
#    This keyword argument is passed to :func:`~numpy.isclose` when comparing the footpoint radius
#    to the boundary radii.

polarity = get_fieldline_polarity(1,
                                  30,
                                  magnetic_field_files.cor_br,
                                  traces,
                                  atol=1e-2)
print(f"Polarity array of size: {polarity.size}")
# %%
# Finally, we can visualize the classified field lines by assigning colors based on their
# polarity. Here, we create a custom colormap where:
#
# - Blue corresponds to negative polarity open field lines
# - Grey corresponds to closed field lines
# - Black corresponds to error/undefined field lines
# - Green corresponds to disconnected field lines
# - Red corresponds to positive polarity open field lines

polarity_cmap = matplotlib.colors.ListedColormap(
    ['blue', 'grey', 'black', 'green', 'red'],
    name='Polarity',
    N=5
)
colors = polarity_cmap((polarity.astype(int) + 2) / 4)  # Normalize to [0, 1] for colormap

# %%
# `psi-io <https://predsci.com/doc/psi-io/>`_ :py:func:`~psi_io.psi_io.np_interpolate_slice_from_hdf` is
# used to linearly interpolate a 2D slice of data at the solar surface (r=1.0 Rs), using the
# radial component of the magnetic field.
ax = plt.figure().add_subplot(projection='3d')
plot_traces(traces, ax=ax, colors=colors)
values, theta_scale, phi_scale = np_interpolate_slice_from_hdf(magnetic_field_files.cor_br,
                                                           1.0, None, None,)
plot_sphere(values.T, 1.0, theta_scale, phi_scale, clim=(-10, 10), ax=ax)

FOV = 1.5  # Rsun
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()
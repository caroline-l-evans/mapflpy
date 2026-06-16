"""
Trace Forward
=============

Perform forward tracing of magnetic field lines.

This example demonstrates how to use the :func:`~mapflpy.scripts.run_forward_tracing`
function to trace magnetic field lines forward from a set of default starting points.
It also shows how to load magnetic field data files and visualize the traced field lines in 3D.
"""
# sphinx_gallery_start_ignore
import os
CONTEXT = 'spawn'
if 'SPHINX_GALLERY_BUILD' not in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')
    CONTEXT = 'fork'
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt

from mapflpy.scripts import run_forward_tracing
from mapflpy.utils import plot_traces
from psi_data import fetch_mas_data

# %%
# Load in the magnetic field files
#
# The :func:`~psi_data.fetch_mas_data` function returns a named tuple of file paths
# corresponding to the radial, theta, and phi components of the magnetic field data.
magnetic_field_files = fetch_mas_data(domains="cor", variables="br,bt,bp")

# %%
# Run forward tracing using the default launch points
#
# .. note::
#    By default, if no launch points are provided, the function will use a set of 128
#    predefined launch points distributed in a Fibonacci lattice at a radius of 1.01 Rsun.
traces = run_forward_tracing(*magnetic_field_files, context=CONTEXT)
print("Geometry shape:", traces.geometry.shape)

# %%
# The shape of the resulting traces geometry is an **M x 3 x N** array, where **M** is the
# field line length (*i.e.* the ``buffer_size``), **N** is the number of launch points
# (here 128), and the second dimension corresponds to the radial-theta-phi coordinates.
#
# The utility functions provided in :mod:`~mapflpy.utils` are designed to work with this
# contiguous memory layout for efficient processing and visualization.

# %%
# Plot traces using the :func:`~mapflpy.utils.plot_traces` utility function and adjust
# the field of view to be 4 Solar Radii in each direction
ax = plt.figure().add_subplot(projection='3d')
plot_traces(traces, ax=ax)

FOV = 4.0  # Rsun
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()
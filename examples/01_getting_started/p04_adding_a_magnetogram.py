"""
Adding a Magnetogram
====================

Perform forward tracing of magnetic field lines and adding a magnetogram.

This example demonstrates how to use :func:`~mapflpy.scripts.run_forward_tracing`
and :func:`~mapflpy.utils.plot_traces` function to plot magnetic field lines, along
with :func:`~mapflpy.utils.plot_sphere` to add a magnetogram at the solar surface.
"""
# sphinx_gallery_start_ignore
import os
CONTEXT = 'spawn'
if 'SPHINX_GALLERY_BUILD' not in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')
    CONTEXT = 'fork'
# sphinx_gallery_end_ignore

from psi_io import np_interpolate_slice_from_hdf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from mapflpy.scripts import run_forward_tracing
from mapflpy.utils import plot_traces, plot_sphere
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
traces = run_forward_tracing(*magnetic_field_files, context='fork')
print("Geometry shape:", traces.geometry.shape)

# %%
# The shape of the resulting traces geometry is an **M x 3 x N** array, where **M** is the
# field line length (*i.e.* the ``buffer_size``), **N** is the number of launch points
# (here 128), and the second dimension corresponds to the radial-theta-phi coordinates.
#
# The utility functions provided in :mod:`~mapflpy.utils` are designed to work with this
# contiguous memory layout for efficient processing and visualization.

# %%
# `psi-io <https://predsci.com/doc/psi-io/>`_ :py:func:`~psi_io.psi_io.np_interpolate_slice_from_hdf` is
# used to linearly interpolate a 2D slice of data at the solar surface (r=1.0 Rs), using the
# radial component of the magnetic field.
values, theta_scale, phi_scale = np_interpolate_slice_from_hdf(magnetic_field_files.cor_br,
                                                               1.0, None, None,)

# %%
# Plot traces using the :func:`~mapflpy.utils.plot_traces` utility function and adjust
# the field of view to be 2.5 Solar Radii in each direction
rsample = np.random.random_sample(size=traces.geometry.shape[-1])
colors = matplotlib.colormaps['hsv'](rsample)

ax = plt.figure().add_subplot(projection='3d')
plot_sphere(values.T, 1.0, theta_scale, phi_scale, clim=(-10, 10), ax=ax)
plot_traces(traces, ax=ax, colors=colors)
FOV = 2.5  # Rsun
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()
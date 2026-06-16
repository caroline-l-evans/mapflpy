"""
Setting Tracing Direction
=========================

Perform forward and backward tracing of magnetic field lines.

This example demonstrates how to use the :class:`~mapflpy.tracer.Tracer` class to perform
forward and backward tracing of magnetic field lines using the
:meth:`~mapflpy.tracer._Tracer.set_tracing_direction`
method to explicitly set the tracing direction.

.. attention::
    The :class:`~mapflpy.tracer.Tracer` class enforces a singleton pattern to manage issues that
    arise from the underlying :mod:`mapflpy_fortran` object not being thread-safe. As a result, it is
    recommended to use the :class:`~mapflpy.tracer.Tracer` class in single-threaded contexts only
    *viz.* instantiating one instance of the class at a time.
"""
# sphinx_gallery_start_ignore
import os
if 'SPHINX_GALLERY_BUILD' not in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt
import numpy as np

from mapflpy.tracer import Tracer
from mapflpy.utils import plot_traces
from psi_data import fetch_mas_data

# %%
# Load in the magnetic field files
#
# The :func:`~psi_data.fetch_mas_data` function returns a named tuple of file paths
# corresponding to the radial, theta, and phi components of the magnetic field data
# (with fields like ``cor_br``, ``cor_bt``, ``cor_bp``).
magnetic_field_files = fetch_mas_data(domains="cor", variables="br,bt,bp")

# %%
# The :class:`~mapflpy.tracer.Tracer` class can be instantiated directly with the magnetic
# field file paths (along with any additional "mapfl params" as keyword arguments).
#
# For now, the default ``mapfl`` configuration is used *i.e.*
# :data:`~mapflpy.globals.DEFAULT_PARAMS`.

tracer = Tracer(*magnetic_field_files)

# %%
# Define launch points in spherical coordinates (r, theta, phi); here we define a ring of
# points at r=15 Rs around the equatorial plane. The resultant array should have the shape (N, 3),
# where **N** is the number of launch points.

rvalues = 15
thetas = np.pi/2
phis = np.linspace(0, 2 * np.pi, 180)

rr, tt, pp = np.meshgrid(rvalues, thetas, phis, indexing='ij')
launch_points = np.column_stack((rr, tt, pp))[0,...]


# %%
# To explicitly set the tracing direction, use the
# :meth:`~mapflpy.tracer._Tracer.set_tracing_direction` method – **'f'** for forward,
# **'b'** for backward – before calling the :meth:`~mapflpy.tracer._Tracer.trace` method.
#
# .. note::
#    The :meth:`~mapflpy.tracer._Tracer.trace_fwd` and :meth:`~mapflpy.tracer._Tracer.trace_bwd`
#    methods are simply convenience wrappers around this functionality.

tracer.set_tracing_direction('f')
fwd_traces = tracer.trace(launch_points=launch_points)
tracer.set_tracing_direction('b')
bwd_traces = tracer.trace(launch_points=launch_points)

# %%
# Plot traces using the :func:`~mapflpy.utils.plot_traces` utility function and adjust
# the field of view to be 30 Solar Radii in each direction.
#
# Here we plot the forward traces in the default color and the backward traces in red.

ax = plt.figure().add_subplot(projection='3d')
plot_traces(fwd_traces, ax=ax)
plot_traces(bwd_traces, ax=ax, color='red')

FOV = 30.0  # Rsun
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()
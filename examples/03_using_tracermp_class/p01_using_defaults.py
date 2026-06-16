"""
Using TracerMP Defaults
=======================

Perform simple tracing using the TracerMP class.

This example demonstrates how to use the :class:`~mapflpy.tracer.TracerMP` class to perform
forward tracing of magnetic field lines from a set of default starting points.
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

from mapflpy.tracer import TracerMP
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
# The :class:`~mapflpy.tracer.TracerMP` class can be instantiated directly with the magnetic
# field file paths (along with any additional "mapfl params" as keyword arguments).
#
# For now, the default ``mapfl`` configuration is used *i.e.*
# :data:`~mapflpy.globals.DEFAULT_PARAMS`.

with TracerMP(*magnetic_field_files, context=CONTEXT) as tracer:
    traces = tracer.trace_fwd()

# %%
# .. note::
#    The :class:`~mapflpy.tracer.TracerMP` class utilizes multiprocessing to instantiate a
#    new :mod:`mapflpy_fortran` object in a worker process, and communicates with it via
#    inter-process communication. The :py:func:`~multiprocessing.Pipe` protocol requires that
#    the :py:class:`multiprocessing.connection.Connection` is properly opened and closed to
#    avoid resource leaks. Therefore, it is recommended to use the :class:`~mapflpy.tracer.TracerMP`
#    class within a context manager (the ``with`` statement) to ensure that resources are
#    properly managed.
#
#    Alternatively, the :meth:`~mapflpy.tracer.TracerMP.connect` and
#    :meth:`~mapflpy.tracer.TracerMP.disconnect` methods can be used to manually manage the
#    connection lifecycle.

# %%
# The shape of the resulting traces geometry is an **M x 3 x N** array, where **M** is the
# field line length (*i.e.* the ``buffer_size``), **N** is the number of launch points
# (here 128), and the second dimension corresponds to the radial-theta-phi coordinates.
#
# The utility functions provided in :mod:`~mapflpy.utils` are designed to work with this
# contiguous memory layout for efficient processing and visualization.

# %%
# Traces can be plotted using the :func:`~mapflpy.utils.plot_traces` utility function

ax = plt.figure().add_subplot(projection='3d')
plot_traces(traces, ax=ax)

FOV = 4.0  # Rsun
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()
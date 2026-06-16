"""
Multiple TracerMP Instances
===========================

Perform simple tracing using the TracerMP class.

This example demonstrates how to use the :class:`~mapflpy.tracer.TracerMP`
class to perform forward tracing of magnetic field lines from a set of default starting points
across both Coronal and Heliospheric magnetic field models.
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
import numpy as np

from mapflpy.tracer import TracerMP
from mapflpy.utils import plot_traces
from psi_data import fetch_mas_data

# %%
# Load in the magnetic field files
#
# The :func:`~psi_data.fetch_mas_data` function returns a named tuple of file paths
# corresponding to the radial, theta, and phi components of the magnetic field data
# (with fields like ``cor_br``, ``cor_bt``, ``cor_bp``).

cor_magnetic_field = fetch_mas_data(domains="cor", variables="br,bt,bp")
hel_magnetic_field = fetch_mas_data(domains="hel", variables="br,bt,bp")

# %%
# The :class:`~mapflpy.tracer.TracerMP` class can be instantiated directly with the magnetic
# field file paths (along with any additional "mapfl params" as keyword arguments).
#
# For now, the default ``mapfl`` configuration is used *i.e.*
# :data:`~mapflpy.globals.DEFAULT_PARAMS`.

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

tracer_cor = TracerMP(*cor_magnetic_field, context=CONTEXT)
tracer_hel = TracerMP(*hel_magnetic_field, context=CONTEXT)

tracer_cor.connect()
tracer_hel.connect()

traces_cor = tracer_cor.trace_fwd()

# %%
# Forward trace field lines from the coronal model, then use the end points at r=30 Rs
# as launch points for the heliospheric model. When tracing with the heliospheric model,
# we need to set the minimum domain radius to 30 Rs using the ``domain_r_min_`` parameter.

outerboundary = np.isclose(traces_cor.end_pos[0,...], 30)
hel_launch_points = traces_cor.end_pos[:, outerboundary]
tracer_hel['domain_r_min_'] = 30.0
traces_hel = tracer_hel.trace_fwd(launch_points=hel_launch_points)

tracer_cor.disconnect()
tracer_hel.disconnect()

# %%
# Traces can be plotted using the :func:`~mapflpy.utils.plot_traces` utility function

ax = plt.figure().add_subplot(projection='3d')
plot_traces(traces_cor, ax=ax)
plot_traces(traces_hel, ax=ax, color='orange')

FOV = 100  # Rsun
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()
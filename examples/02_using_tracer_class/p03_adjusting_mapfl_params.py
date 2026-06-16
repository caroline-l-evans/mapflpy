"""
Adjusting mapfl Parameters
==========================

Adjust the default mapfl parameters

This example demonstrates how to use the :class:`~mapflpy.tracer.Tracer` class to perform
fieldline tracing using non-default mapfl parameters. Furthermore, it explores the *"Tracer
as dictionary"* design pattern to update mapfl parameters after instantiation.

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
# The :class:`~mapflpy.tracer.Tracer` class is, for demonstration purposes, instantiated
# without arguments to illustrate how to set the magnetic field files post-initialization.

tracer = Tracer()
tracer.br = magnetic_field_files.cor_br
tracer.bt = magnetic_field_files.cor_bt
tracer.bp = magnetic_field_files.cor_bp

# %%
# Define launch points in spherical coordinates (r, theta, phi); here we define a grid of
# points at r=1 Rs covering a range of latitudes and longitudes.

rvalues = 15
thetas = np.linspace(0, np.pi, 15)
phis = np.linspace(0, 2 * np.pi, 15)

rr, tt, pp = np.meshgrid(rvalues, thetas, phis, indexing='ij')
launch_points = np.vstack((rr.ravel(), tt.ravel(), pp.ravel()))

traces = tracer.trace_fbwd(launch_points=launch_points)

# %%
# Plot traces using the :func:`~mapflpy.utils.plot_traces` utility function and adjust
# the field of view to be 30 Solar Radii in each direction.

ax = plt.figure().add_subplot(projection='3d')
plot_traces(traces, ax=ax, color='c')

FOV = 30  # Rsun
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()

# %%
# Now, adjust some of the mapfl parameters using the Tracer-as-dictionary pattern.
#
# .. attention::
#    The base :class:`~mapflpy.tracer._Tracer` class extends the
#    :class:`~collections.abc.MutableMapping` interface, allowing users to interact with a tracer's
#    :data:`~mapflpy.tracer._Tracer.params` property as if it were a dictionary.

tracer['domain_r_min_'] = 10    # Directly set the minimum radius parameter
tracer['domain_r_max_'] = 20    # Directly set the maximum radius parameter
traces = tracer.trace_fbwd(launch_points=launch_points)

ax = plt.figure().add_subplot(projection='3d')
plot_traces(traces, ax=ax, color='m')

FOV = 20  # Rsun
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()



# %%
# Similarly, one can use any of the native :any:`dict` methods to manipulate the mapfl parameters,
# such as :meth:`~dict.update`.

tracer.update(domain_r_min_=1, domain_r_max_=30, cubic_=False, dsmult_=2)

# %%
# Similarly, one can fetch views of the current parameters
# using the :meth:`~dict.items`, :meth:`~dict.keys`, and :meth:`~dict.values` methods.
print("Tracer parameter keys:\n")
for key in tracer.keys():
    print(" - " + key)

# %%
# Once again, perform forward-backward tracing using the adjusted parameters.

traces = tracer.trace_fbwd(launch_points=launch_points)

ax = plt.figure().add_subplot(projection='3d')
plot_traces(traces, ax=ax, color='y')

FOV = 30  # Rsun
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()



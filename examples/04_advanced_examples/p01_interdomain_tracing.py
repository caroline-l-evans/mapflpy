"""
Interdomain Tracing
===================

Perform tracing across coronal and heliospheric domains.

This example demonstrates how to use the :func:`~mapflpy.scripts.inter_domain_tracing`
function to trace magnetic field lines across domain boundaries.
"""
import os

# sphinx_gallery_start_ignore
CONTEXT = 'spawn'
if 'SPHINX_GALLERY_BUILD' not in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')
    CONTEXT = 'fork'
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt

from mapflpy.scripts import inter_domain_tracing
from mapflpy.utils import plot_traces, fetch_default_launch_points
from psi_data import fetch_mas_data

# %%
# Load in the coronal and heliospheric magnetic field files

cor_files = fetch_mas_data(domains="cor", variables="br,bt,bp")
hel_files = fetch_mas_data(domains="hel", variables="br,bt,bp")

# %%
# Define 100 launch points using the fibonacci lattice method (at 1 Rsun)
launch_points = fetch_default_launch_points(n=100, r=1)

# %%
# Call the interdomain tracing function with the provided coronal and heliospheric
# magnetic field files and the defined launch points; otherwise, the default keyword
# arguments are used.
#
# .. note::
#    Refer to the :func:`~mapflpy.scripts.inter_domain_tracing` documentation for
#    additional keyword arguments that can be passed to this function, *viz.* to finetune the
#    way in which traces cross (or recross) the domain boundary. Of particular interest are
#
#    - ``maxiter``
#    - ``r_interface``
#    - ``helio_shift``
#    - ``rtol``

final_traces, traced_to_boundary, boundary_recross = inter_domain_tracing(
    *cor_files,
    *hel_files,
    launch_points=launch_points,
    context=CONTEXT)

# %%
# This call returns a tuple where the first element is the trace geometry. The remaining
# two elements are boolean arrays indicating whether a trace successfully travel from the
# coronal inner boundary to the heliospheric outer boundary, and whether a trace recrossed
# the domain boundary.

# %%
# .. warning::
#    The ``final_traces`` that are returned from :func:`~mapflpy.scripts.inter_domain_tracing`
#    are *not* a single contiguous (``nan`` buffered) array. **Unlike** other tracing protocols found
#    throughout the **mapflpy** package (which return :class:`~mapflpy.globals.Traces` objects),
#    these traces are a list of heterogeneously sized arrays, one per launch point *i.e.*
#    an **N x 3 x M** :sub:`i…n` list.

for i, trace in enumerate(final_traces):
    print(f"Trace {i}, shape: {trace.shape}")

# %%
# Plot and adjust the field of view to be 200 Solar Radii in each direction so that both domains
# are visible.
FOV = 200.0  # Rsun

ax = plt.figure().add_subplot(projection='3d')
plot_traces(*final_traces, ax=ax)
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()

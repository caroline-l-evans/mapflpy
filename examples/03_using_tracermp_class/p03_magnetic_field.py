"""
Setting the Magnetic Field
==========================

Explicitly set the magnetic field files using the TracerMP class.

This example demonstrates how to use explicitly set the :class:`~mapflpy.tracer.TracerMP`
magnetic field data *i.e.* :attr:`~mapflpy.tracer._Tracer.br`, :attr:`~mapflpy.tracer._Tracer.bt`,
and :attr:`~mapflpy.tracer._Tracer.bp` attributes.
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
# corresponding to the radial, theta, and phi components of the magnetic field data for the
# requested domain (with fields like ``cor_br``, ``cor_bt``, ``cor_bp`` for the coronal domain
# and ``hel_br``, ``hel_bt``, ``hel_bp`` for the heliospheric domain).

magnetic_field_files = fetch_mas_data(domains="cor", variables="br,bt,bp")
hel_magnetic_field_files = fetch_mas_data(domains="hel", variables="br,bt,bp")

# %%
# The :class:`~mapflpy.tracer.TracerMP` class is, for demonstration purposes, instantiated
# without arguments to illustrate how to set the magnetic field files post-initialization.
#
# .. note::
#    As with :class:`~mapflpy.tracer.Tracer`, the magnetic field data can be set using the
#    :attr:`~mapflpy.tracer._Tracer.br`, :attr:`~mapflpy.tracer._Tracer.bt`,
#    and :attr:`~mapflpy.tracer._Tracer.bp` attributes, or by passing the file paths to
#    :meth:`~mapflpy.tracer._Tracer.load_fields`
#
# .. warning::
#    When setting the magnetic field data with :class:`~mapflpy.tracer.TracerMP`, only a
#    filepath can be supplied *i.e.* not a NumPy array, as with :class:`~mapflpy.tracer.Tracer`.
#    This is due to the inter-process communication mechanism used by
#    :class:`~mapflpy.tracer.TracerMP` and the prohibitive cost of passing magnetic field data
#    over the pipe.
#
# Here we pass along the ``n`` and ``r`` parameters to the :meth:`~mapflpy.tracer._Tracer.trace_fwd` and
# :meth:`~mapflpy.tracer._Tracer.trace_bwd` methods to specify the number of field lines to trace
# and the starting radius respectively.

with TracerMP(context=CONTEXT) as tracer:
    with TracerMP(context=CONTEXT) as tracer_hel:
        tracer.br = magnetic_field_files.cor_br
        tracer.bt = magnetic_field_files.cor_bt
        tracer.bp = magnetic_field_files.cor_bp

        tracer_hel.load_fields(*hel_magnetic_field_files)
        tracer_hel['domain_r_min_'] = 30

        traces = tracer.trace_fwd(n=128, r=1)
        traces_hel = tracer_hel.trace_bwd(n=32, r=200)

# %%
# Plot traces using the :func:`~mapflpy.utils.plot_traces` utility function and adjust
# the field of view to be 200 Solar Radii in each direction.

ax = plt.figure().add_subplot(projection='3d')
plot_traces(traces, ax=ax)
plot_traces(traces_hel, ax=ax, color='m')

FOV = 200.0  # Rsun
for dim in 'xyz':
    getattr(ax, f'set_{dim}lim3d')((-FOV, FOV))
ax.set_box_aspect([1, 1, 1])

plt.show()
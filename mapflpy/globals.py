"""
Module defining custom types, enumerations, and constants for the mapflpy package.
"""


from __future__ import annotations

from collections import namedtuple
from enum import IntEnum
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeVar, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    'Traces',
    'Polarity',
    'MAGNETIC_FIELD_PATHS',
    'DEFAULT_FIELDS',
    'DEFAULT_PARAMS',
]


# ------------------------------------------------------------------------------
# Named tuple for storing trace information.
# This is used to return the results of tracing operations.
# It contains the geometry of the traces, their start and end positions,
# and whether they were traced to a boundary.
#------------------------------------------------------------------------------
Traces = namedtuple('Traces', ['geometry', 'start_pos', 'end_pos', 'traced_to_boundary', 'integral'])
Traces.__doc__ = (
    """Named tuple for storing magnetic fieldline trace information.
    
    Attributes
    ----------
    geometry : ndarray
        Array of shape (N, M, 3) containing the 3D coordinates of the traced fieldlines,
        where N is the number of fieldlines and M is the number of points per fieldline.
    start_pos : ndarray
        Array of shape (N, 3) containing the starting positions of each fieldline.
    end_pos : ndarray
        Array of shape (N, 3) containing the ending positions of each fieldline.
    traced_to_boundary : ndarray
        Boolean array of shape (N,) indicating whether each fieldline was traced to a boundary.
    integral : ndarray
        Array of shape (N,) containing the field aligned integral. By default the integral
        is just the field line length, but certain mapflpy options can change this behavior, 
        currently `integrate_along_fl_`, `scalar_input_file_`, `weight_integral_by_area_`, and 
        max_along_fl_`.
    """
)

# ------------------------------------------------------------------------------
# Named tuple for storing mapping information produced by some of the mapping
# scripts in `mapflpy.scripts`
# ------------------------------------------------------------------------------
Mapping = namedtuple('Mapping', ['r', 't', 'p', 'traced_to_boundary', 'integral'])
Mapping.__doc__ = (
    """Named tuple for storing magnetic fieldline mapping information.

    Attributes
    ----------
    r : ndarray
        Array containing the radial positions of the mapping endpoints [R_sun]).
    t : ndarray
        Array containing the theta (co-latitude) positions of the mapping endpoints [radians].
    p : ndarray
        Array containing the phi (longitude) positions of the mapping endpoints [radians].
    traced_to_boundary : ndarray
        Boolean array indicating whether each fieldline was traced to a boundary.
    integral : ndarray
        Array containing the values of the field line integral. By default the integral
        is just the field line length, but certain mapflpy options can change this behavior, 
        currently `integrate_along_fl_`, `scalar_input_file_`, `weight_integral_by_area_`, and 
        `max_along_fl_`.
    """
)

# ------------------------------------------------------------------------------
# Named tuple for storing squashing factor produced by the
# scripts in `mapflpy.scripts`
# ------------------------------------------------------------------------------
SquashingFactor = namedtuple('Squashing_Factor', ['q', 'p', 't'])
Mapping.__doc__ = (
    """Named tuple for storing q and the field on which it was calculated.

    Attributes
    ----------
    q : ndarray
        Array containing the squashing factor
    p : ndarray
        Array containing the phi (longitude) positions for q in radians.
    t : ndarray
        Array containing the theta (co-latitude) positions for q in radians.

    """
)

# ------------------------------------------------------------------------------
# Type aliases for improved code readability.
# ------------------------------------------------------------------------------
NumberType = TypeVar(
    'NumberType',
    bound=np.floating | np.integer | np.bool_ | float | int | bool,
)
"""Type variable for numeric types, including NumPy and built-in types."""

DirectionType: TypeAlias = Literal["f", "b"]
"""Type alias for tracing direction, *viz.* 'f' (forward) or 'b' (backward)."""

MagneticFieldLabelType: TypeAlias = Literal["br", "bt", "bp"]
"""Type alias for magnetic field component labels."""

ContextType: TypeAlias = Literal["fork", "spawn", "forkserver"]
"""Type alias for multiprocessing context types."""

PathType: TypeAlias = str | Path | PathLike[str]
"""Type alias for file path representations."""

ArrayType: TypeAlias = npt.NDArray[NumberType]
"""Type alias for NumPy arrays containing numeric types."""

MagneticFieldArrayType: TypeAlias = tuple[ArrayType, ArrayType, ArrayType, ArrayType] | PathType
"""Type alias for magnetic field data, either as a tuple of NumPy arrays
    for (Br, Bt, Bp, r) or as a file path to HDF5 data."""

DEFAULT_BUFFER_SIZE = 2000
"""Default buffer size for mapfl traces."""

MAGNETIC_FIELD_LABEL = ('br', 'bt', 'bp')
"""Labels for magnetic field components, *viz.* radial (br), theta (bt), and phi (bp)."""

DIRECTION = ('f', 'b')
"""Tracing directions, *viz.* 'f' (forward) or 'b' (backward)."""


class Polarity(IntEnum):
    """
    Enumeration of magnetic fieldline polarity classifications.

    This enumeration is used to label the connectivity of magnetic
    fieldlines based on their endpoints. The values encode both the
    type of connectivity (open, closed, or invalid) and, for open
    lines, the sign of the radial magnetic field at the footpoint.

    Attributes
    ----------
    R0_R1_NEG : int (-2)
        Open fieldline connecting from the inner boundary (R0)
        to the outer boundary (R1) with a **negative** radial
        magnetic field (Br < 0) at the inner boundary footpoint.

    R0_R0 : int (-1)
        Closed fieldline with both endpoints anchored at the
        inner boundary (R0).

    ERROR : int (0)
        Indicates an undefined or unclassified trace, typically used
        when the endpoints do not terminate at a boundary.

    R1_R1 : int (1)
        Closed fieldline with both endpoints anchored at the
        outer boundary (R1) *e.g.* "Disconnected" trace.

    R0_R1_POS : int (2)
        Open fieldline connecting from the inner boundary (R0)
        to the outer boundary (R1) with a **positive** radial
        magnetic field (Br > 0) at the inner boundary footpoint.

    Notes
    -----
    The sign convention assumes that the radial magnetic field (Br)
    is evaluated at the **inner boundary** footpoint of the fieldline.
    """
    R0_R1_NEG = -2
    R0_R0 = -1
    ERROR = 0
    R1_R1 = 1
    R0_R1_POS = 2


# ------------------------------------------------------------------------------
# Base dictionary for maintaining magnetic field filepaths.
# Used to pass field data within the TracerMP class.
# ------------------------------------------------------------------------------
MAGNETIC_FIELD_PATHS = MappingProxyType({
    'br': '',
    'bt': '',
    'bp': '',
})
"""Base dictionary for magnetic field file paths *viz.* used in
:class:`~mapflpy.tracer.TracerMP`."""


# ------------------------------------------------------------------------------
# Base dictionary for maintaining magnetic field array/scale data.
# Used to pass field data within the Tracer class.
# ------------------------------------------------------------------------------
DEFAULT_FIELDS = MappingProxyType({
    'br': None,
    'br_r': None,
    'br_nr': None,
    'br_t': None,
    'br_nt': None,
    'br_p': None,
    'br_np': None,
    'bt': None,
    'bt_r': None,
    'bt_nr': None,
    'bt_t': None,
    'bt_nt': None,
    'bt_p': None,
    'bt_np': None,
    'bp': None,
    'bp_r': None,
    'bp_nr': None,
    'bp_t': None,
    'bp_nt': None,
    'bp_p': None,
    'bp_np': None,
})
"""Base dictionary for magnetic field arrays and scales."""

# ------------------------------------------------------------------------------
# The following MAPFL_PARAMS are modeled after the mapfl.in file. However, the
# following key-value pairs are not exhaustive; rather, they are a *working*
# configuration that will suit most general uses of the _Tracer class.
# ------------------------------------------------------------------------------
DEFAULT_PARAMS = MappingProxyType({
    'integrate_along_fl_': False,
    'scalar_input_file_': '',
    'weight_integral_by_area_': False,
    'max_along_fl_': False,
    'verbose_': False,
    'cubic_': False,
    'trace_fwd_': False,
    'trace_bwd_': False,
    'trace_3d_': False,
    'trace_slice_': False,
    'compute_ch_map_': False,
    'debug_level_': 0,
    'use_analytic_function_': False,
    'function_params_file_': ' ',
    'domain_r_min_': 1,
    'domain_r_max_': 300,
    'bfile_r_': 'br002.h5',
    'bfile_t_': 'bt002.h5',
    'bfile_p_': 'bp002.h5',
    'ds_variable_': True,
    'ds_over_rc_': 0.0025,
    'ds_min_': 0.00005,
    'ds_max_': 10.0,
    'ds_limit_by_local_mesh_': True,
    'ds_local_mesh_factor_': 1.0,
    'ds_lmax_': 5000.0,
    'set_ds_automatically_': True,
    'dsmult_': 1.0,
    'rffile_': ' ',
    'tffile_': ' ',
    'pffile_': ' ',
    'effile_': ' ',
    'kffile_': ' ',
    'qffile_': ' ',
    'lffile_': ' ',
    'rbfile_': ' ',
    'tbfile_': ' ',
    'pbfile_': ' ',
    'ebfile_': ' ',
    'kbfile_': ' ',
    'qbfile_': ' ',
    'lbfile_': ' ',
    'new_r_mesh_': True,
    'mesh_file_r_': ' ',
    'nrss_': 1,
    'r0_': 0.0,
    'r1_': 0.0,
    'new_t_mesh_': True,
    'mesh_file_t_': ' ',
    'ntss_': 2,
    't0_': 0.5,
    't1_': 0.6,
    'new_p_mesh_': True,
    'mesh_file_p_': ' ',
    'npss_': 2,
    'p0_': 0.1,
    'p1_': 0.2,
    'volume3d_output_file_r_': ' ',
    'volume3d_output_file_t_': ' ',
    'volume3d_output_file_p_': ' ',
    'slice_coords_are_xyz_': False,
    'trace_slice_direction_is_along_b_': True,
    'compute_q_on_slice_': False,
    'q_increment_h_': 0.0001,
    'slice_input_file_r_': 'lp_r.hdf',
    'slice_input_file_t_': 'lp_t.hdf',
    'slice_input_file_p_': 'lp_p.hdf',
    'trace_from_slice_forward_': True,
    'slice_output_file_forward_r_': 'rf.hdf',
    'slice_output_file_forward_t_': 'tf.hdf',
    'slice_output_file_forward_p_': 'pf.hdf',
    'trace_from_slice_backward_': True,
    'slice_output_file_backward_r_': 'rb.hdf',
    'slice_output_file_backward_t_': 'tb.hdf',
    'slice_output_file_backward_p_': 'pb.hdf',
    'slice_q_output_file_': 'q_output.hdf',
    'slice_length_output_file_': ' ',
    'ch_map_r_': 1.0,
    'ch_map_output_file_': 'ch_output.hdf',
    'compute_ch_map_3d_': False,
    'ch_map_3d_output_file_': ' ',
    'write_traces_to_hdf_': False,
    'write_traces_root_': 'fl',
    'write_traces_as_xyz_': True,
})
"""Base dictionary for MAPFL parameters used in tracing operations.

For general use of the :class:`~mapflpy.tracer._Tracer` class (and its subclasses), 
these parameters should be sufficient.

.. warning::
    This set of parameters is modeled after the ``mapfl.in`` file. With that said, 
    ``mapflpy`` does not yet implement all possible functionality from the original
    ``MAPFL`` Fortran. Therefore, some parameters may be ignored or not yet implemented.
    Future releases will aim to expand this functionality, and the documentation will
    be extended to articulate which of these parameters can be manipulated.  
"""

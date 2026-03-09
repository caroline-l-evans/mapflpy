"""
Utility functions for processing fieldline traces.
"""
from __future__ import annotations

import math
import random
from typing import Tuple, List, Any, Optional, TypeVar, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from psi_io import interpolate_positions_from_hdf

from mapflpy.globals import Traces, Polarity, ArrayType, PathType

__all__ = [
    'shift_phi_lps',
    'shift_phi_traces',
    'fibonacci_lattice',
    'fetch_default_launch_points',
    'combine_fwd_bwd_traces',
    'get_fieldline_polarity',
    'get_fieldline_endpoints',
    'get_fieldline_npoints',
    'trim_fieldline_nan_buffer'
]


def shift_phi_lps(lp: np.ndarray,
                  phi_shift: float = 0.0
                  ) -> np.ndarray:
    """
    Shift a Fortran ordered (3,N) launch point array in longitude by phi_shift radians.

    Parameters
    ----------
    lp : ArrayLike
        Launch points for fieldline tracing. Here we assume `lp[2,:]` is the phi coordinate.
    phi_shift : float
        The longitudinal shift in radians. Defualt is 0.0.

    Returns
    -------
    lp : ArrayLike
        A copy of lp shifted in longitude.

    """
    if np.isclose(phi_shift, 0.0):
        return lp
    else:
        lp[2, :] = np.mod(lp[2, :] + phi_shift, 2 * np.pi)
        return lp


def shift_phi_traces(traces: Sequence[np.ndarray],
                     phi_shift: float = 0.0
                     ) -> Sequence[np.ndarray]:
    """
    Shift mapflpy traces in longitude by phi_shift radians.

    Parameters
    ----------
    traces : list of ndarray
        List of mapflpy traces. It assumes that for each `trace` in `traces` that `trace[2,:]`
        is the phi coordinate.
    phi_shift : float
        The longitudinal shift in radians. Defualt is 0.0.

    Returns
    -------
    traces : list of ndarray
        A copy of the traces list shifted in longitude.

    """
    if np.isclose(phi_shift, 0.0):
        return traces
    else:
        for trace in traces:
            trace[2, :] = np.mod(trace[2, :] + phi_shift, 2 * np.pi)
        return traces


def fibonacci_lattice(
    n: int = 100,
    radius: float = 1.0,
    randomize: bool = False,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an approximately uniform set of spherical points using a Fibonacci lattice.

    This is a vectorized “Fibonacci sphere” sampler that places ``n`` points on a
    sphere using the golden-angle increment. The result is returned in spherical
    coordinates using the **colatitude** convention:

    - ``r`` is constant (set to ``radius``).
    - ``t`` (theta) is the colatitude from +z in ``[0, π]``.
    - ``p`` (phi) is the azimuth from +x toward +y in ``[0, 2π)``.

    Parameters
    ----------
    n : int, optional
        Number of sample points to generate.
    radius : float, optional
        Radius assigned to each generated point.
    randomize : bool, optional
        If True, apply a random phase offset to the golden-angle sequence to
        avoid always starting from the same meridian. This preserves the
        quasi-uniform spacing but changes the rotation of the point set.
    seed : int or None, optional
        RNG seed used only when ``randomize=True`` for reproducibility.

    Returns
    -------
    r, t, p : ndarray
        Arrays of shape ``(n,)`` containing spherical coordinates of each point.

    Notes
    -----
    - Internally, points are generated on a unit sphere and then assigned ``r=radius``.
      The returned ``t`` and ``p`` reflect the angular distribution of the lattice.
    - ``p`` is wrapped into ``[0, 2π)`` using modulo arithmetic.
    - This sampler is “approximately uniform” (low-discrepancy) rather than truly
      random; it is often preferred for deterministic coverage of the sphere.

    Examples
    --------
    >>> r, t, p = fibonacci_lattice(n=1000, radius=2.0, randomize=True, seed=0)
    >>> r.shape, t.shape, p.shape
    ((1000,), (1000,), (1000,))
    """
    # index array
    i = np.arange(n, dtype=np.float64)

    # random phase
    rnd = 1.0
    if randomize:
        rng = np.random.default_rng(seed)
        rnd = rng.random() * n

    offset = 2.0 / n
    increment = np.pi * (3.0 - np.sqrt(5.0))

    # y in [-1, 1] (approximately), and radial factor in xz-plane
    y = (i * offset - 1.0) + (offset / 2.0)
    r_xz = np.sqrt(np.maximum(0.0, 1.0 - y * y))  # guard tiny negatives

    # golden-angle increment around
    phi = np.mod(i + rnd, n) * increment

    x = np.cos(phi) * r_xz
    z = np.sin(phi) * r_xz

    # radius (should be 1 for unit sphere)
    r = np.full_like(x, radius)

    # theta (co-latitude)
    t = np.arccos(np.clip(z, -1.0, 1.0))

    # p computed as atan2(y, x), wrapped to [0, 2pi)
    p = np.mod(np.arctan2(y, x), 2.0 * np.pi)

    return r, t, p


def fetch_default_launch_points(n: int = 128,
                                r: float = 1.01
                                ) -> NDArray[np.float64]:
    """Generate a default set of N launch points for mapfl.

    The N launch points will roughly uniformly sample the sphere
    at a given radius using the `fibonacci_lattice` algorithm (default 1.01).

    Parameters
    ----------
    n : int
        Number of launch points to generate.
    r : float, optional
        Radius at which to place the launch points. Defaults to 1.01.

    Returns
    -------
    launch_points : ndarray
        A 3xN array of launch points.

    """
    rtp = fibonacci_lattice(n, r)
    return np.stack(rtp)


def combine_fwd_bwd_traces(fwd_traces: Traces,
                           bwd_traces: Traces
                           ) -> Traces:
    """
    Convenience function for combining forward and backward traces.

    Parameters
    ----------
    fwd_traces : Traces
        Output from :meth:`~mapflpy.tracer._Tracer.trace()` when
        tracing direction is set to forward.
    bwd_traces : Traces
        Output from :meth:`~mapflpy.tracer._Tracer.trace()` when
        tracing direction is set to backward.

    Returns
    -------
    combined_traces : Traces
        Combined forward and backward traces (with redundant coordinates removed).

    Notes
    -----
    This function assumes that the launch points for the two ``trace()`` calls are
    the same, and that for each :class:`Traces` object, the :data:`Traces.geometry`
    is a 3D array of size (M, 3, N), where

    - M is the number of points along the field line,
    - N is the number of field lines, and
    - the second dimension corresponds to the coordinates (r, t, p).
    """
    combined_traces = Traces(np.concatenate([np.flip(bwd_traces.geometry, axis=0)[:-1, :, :],
                                             fwd_traces.geometry]),
                             bwd_traces.end_pos,
                             fwd_traces.end_pos,
                             fwd_traces.traced_to_boundary & bwd_traces.traced_to_boundary)
    return combined_traces


def get_fieldline_polarity(inner_boundary: float,
                           outer_boundary: float,
                           br_filepath: PathType,
                           *traces: Tuple[Traces] | Tuple[ArrayType, ArrayType],
                           **kwargs
                           ) -> NDArray[Polarity]:
    """
    Determine the polarity of traced magnetic field lines based on their endpoints.

    This function classifies field lines into different polarity categories based on the
    radial position of their endpoints relative to two spherical boundaries. Optionally,
    if field lines are open (connecting inner to outer boundary), the sign of the Br field
    at the inner boundary is used to determine the polarity direction.

    Parameters
    ----------
    inner_boundary : float
        The radial distance of the inner spherical boundary (e.g., 1 R_sun).
    outer_boundary : float
        The radial distance of the outer spherical boundary (e.g., 30 R_sun).
    br_filepath : PathType
        Path to an HDF file containing the Br (radial magnetic field) component data.
        This is used to determine the sign of open field lines.
    *traces : Traces or tuple of ndarray
        Either a single `Traces` object containing field line tracing results,
        or a tuple of two arrays `(start_points, end_points)` with shape (3, N),
        where N is the number of field lines.
    **kwargs : dict
        Additional keyword arguments passed to `np.isclose()` for numerical comparison
        of endpoint radii with the inner/outer boundaries (e.g., `atol=1e-2`).

    Returns
    -------
    polarity : ndarray of Polarity
        An array of integer enum values representing the polarity of each field line:

            - `Polarity.R0_R1_POS`: Open field line from inner to outer boundary with positive Br.
            - `Polarity.R0_R1_NEG`: Open field line from inner to outer boundary with negative Br.
            - `Polarity.R0_R0`: Closed field line that begins and ends on the inner boundary.
            - `Polarity.R1_R1`: Disconnected field line (begins and ends on the outer boundary).
            - `Polarity.ERROR`: Field line does not reach one or both of the inner/outer boundaries.

    Raises
    ------
    ValueError
        If the number or type of arguments in `traces` is invalid.
    ImportError
        If the optional :py:mod:`scipy` library is not installed, which is required for
        :py:func:`psi_io.psi_io.interpolate_positions_from_hdf`.

    Notes
    -----
    - This function assumes that the :attr:`br_filepath` contains the radial magnetic field component.
    - If one has done both a forward and backward trace, the traces should be merged first
      using :func:`combine_fwd_bwd_traces` before checking polarity.
    """
    # Create `endpoints` which is a (2, 3, N) array where N is the number of field lines.
    # The first dimension is for start and end points, the second for coordinates (r, t, p)
    match len(traces):
        case 1:
            if isinstance(traces[0], Traces):
                endpoints = np.stack((traces[0].start_pos, traces[0].end_pos))
            else:
                # Assume traces[0] is a (3, N) array of fieldline geometry.
                # i.e. calling `Traces.geometry`.
                endpoints = get_fieldline_endpoints(traces[0])
        case 2:
            endpoints = np.stack(traces)
        case _:
            raise ValueError(f'Expected either the fieldline geometry, or an array of starting positions '
                             f'and an array of ending positions, but got {len(traces)} arguments.')
    polarity = np.full(endpoints.shape[-1], Polarity.ERROR, dtype=Polarity)

    # Check if the endspoints start and end at the same boundary, i.e. are "closed" field lines.
    closed_fls = np.isclose(endpoints[0, 0, ...], endpoints[1, 0, ...], **kwargs)

    # If the closed fiedlines start and end at the inner boundary, they are true "closed" field lines.
    polarity[closed_fls & np.isclose(
        endpoints[0, 0, ...], inner_boundary, **kwargs)] = Polarity.R0_R0

    # If the closed fieldlines start and end at the outer boundary, they are disconnected field lines.
    polarity[closed_fls & np.isclose(
        endpoints[0, 0, ...], outer_boundary, **kwargs)] = Polarity.R1_R1

    # Check for open field lines, i.e. field lines that start at one boundary and end at the other.
    open_fls = np.isclose(endpoints[0, 0, ...], inner_boundary, **kwargs) & np.isclose(
        endpoints[1, 0, ...], outer_boundary, **kwargs)
    open_inv = np.isclose(endpoints[0, 0, ...], outer_boundary, **kwargs) & np.isclose(
        endpoints[1, 0, ...], inner_boundary, **kwargs)
    if np.any(open_fls | open_inv):
        # np.concatenate(...).T call creates a (M, 3) array, where M is the number of open field lines.
        # and (for open field lines that started at the outer boundary and ended at the inner boundary),
        # the appropriate endpoint is used, i.e. the endpoint at the inner boundary.
        br_values = interpolate_positions_from_hdf(str(br_filepath),
                                                   *np.concatenate((endpoints[0, :, open_fls],
                                                                    endpoints[1, :,
                                                                    open_inv])).T,)
        pvalues = np.sign(br_values)
        polarity[
            open_fls | open_inv] = np.where(pvalues < 0, Polarity.R0_R1_NEG, Polarity.R0_R1_POS)
    return polarity


def get_fieldline_endpoints(traces: Traces | np.ndarray) -> np.ndarray:
    """
    Extract the start and end positions of each valid fieldline.

    This function identifies the first and last non-NaN entries along the
    buffer axis (axis 0) of a fieldline array and returns those positions
    as the fieldline endpoints.

    Parameters
    ----------
    traces : Traces | ndarray
        A `Traces` object or a 3D NumPy array of shape (M, 3, N), where:

        - M is the buffer length (number of points along a fieldline),
        - 3 represents the coordinates (e.g., r, t, p),
        - N is the number of fieldlines.

        NaN values represent unused buffer space.

    Returns
    -------
    endpoints : ndarray
        An array of shape (2, 3, N) containing the start and end points
        of each fieldline. The first index is 0 for start and 1 for end.
    """
    # Extract geometry array if input is a Traces object
    fls = traces.geometry if isinstance(traces, Traces) else traces

    # Get the index of the first non-NaN entry for each fieldline (along axis 0)
    spos_idx = np.argmin(np.isnan(fls), axis=0)

    # Get the index of the last non-NaN entry by reversing axis 0
    fpos_idx = fls.shape[0] - 1 - np.argmin(np.isnan(fls[::-1, ...]), axis=0)

    # Stack indices for start and end: shape → (2, 3, N)
    idx = np.stack((spos_idx, fpos_idx), axis=0)  # → shape (2, N, P)

    # Extract the positions from `fls` at the corresponding indices (`idx`)
    return np.take_along_axis(fls, idx, axis=0)


def get_fieldline_npoints(traces: Traces | np.ndarray) -> np.ndarray:
    """
    Count the number of valid (non-NaN) points in each fieldline.

    Parameters
    ----------
    traces : Traces | ndarray
        A ``Traces`` object or a 3D array of shape (M, 3, N),
        where NaNs indicate unused portions of the buffer.

    Returns
    -------
    npoints : ndarray of int
        An array of shape (N,) indicating the number of valid points
        in each of the N fieldlines.
    """
    # Extract geometry array if input is a Traces object
    fls = traces.geometry if isinstance(traces, Traces) else traces
    # Count valid (non-NaN) positions by checking for NaNs along axis 1 (r, t, p)
    return np.sum(~np.isnan(fls).any(axis=1), axis=0)


def trim_fieldline_nan_buffer(traces: Traces | np.ndarray) -> List[np.ndarray]:
    """
    Remove NaN buffer regions from fieldlines.

    This function trims unused buffer slots from each fieldline and
    returns a list of individual 2D fieldline arrays.

    Parameters
    ----------
    traces : Traces | ndarray
        A `Traces` object or a 3D array of shape (M, 3, N).

    Returns
    -------
    trimmed : list of ndarray
        A list of N arrays, each of shape (3, n_i), where n_i is the
        number of valid points in the i-th fieldline. All NaN values
        along axis 0 (point index) are removed.

    Notes
    -----
    This function is mainly provided as a convenience for users who want to
    process individual field lines after tracing. However, the NaN-buffered
    field lines – a homogeneous 3D array of shape (M, 3, N) – can be used directly
    with vectorized numpy operations, and (in spite of the increased memory overhead)
    is more performant than iterating over a heterogeneous list of arrays.
    """
    fls = traces.geometry if isinstance(traces, Traces) else traces
    return [v[:, ~np.isnan(v).any(axis=0)] for v in fls.T]


def combine_and_pad_fieldlines(arrs: list | tuple,
                               to_trace: bool = False,
                               traced_to_boundary: Optional[NDArray[np.bool_]] = None
                               ) -> np.ndarray | Traces:
    """
    NaN-pad a list of variable-length 3D point arrays into a single dense array.

    This is a convenience utility for combining a list of arrays shaped
    ``(3, Ni)`` (e.g., polyline/trajectory points) into a single array with a
    uniform length along the point dimension by padding shorter entries with
    ``np.nan``.

    Each input array is interpreted as 3 coordinate components (x/y/z or
    r/t/p, etc.) by ``Ni`` samples. The output is arranged as
    ``(max_len, 3, n)`` where ``n = len(arrs)`` and ``max_len = max(Ni)``.

    Parameters
    ----------
    arrs : Sequence[ArrayLike]
        List (or other sequence) of arrays, each with shape ``(3, Ni)``. The
        second dimension (``Ni``) may vary across arrays.
    to_trace : bool, optional
        If True, return a `Traces` object instead of a raw array.
    traced_to_boundary : ndarray | None, optional
        If ``to_trace=True``, this array it used to set the `traced_to_boundary` attribute
        of the returned `Traces` object.
        *This utility function is principally designed to be used in conjunction with*
        :func:`~mapflpy.scripts.inter_domain_tracing`; *as such, this ndarray of boolean values
        is returned from the `inter_domain_tracing` function and indicates which field lines
        were traced to the boundary.*



    Returns
    -------
    out : ndarray
        Array of shape ``(max_len, 3, n)`` with dtype ``float``. For the i-th
        input array with length ``Ni``, the first ``Ni`` rows of ``out[:, :, i]``
        contain the transposed data (so that the point index is the first axis),
        and rows ``Ni:`` are filled with ``np.nan``.

        Concretely, for each i:

        - ``out[:Ni, :, i] == arrs[i].T``
        - ``out[Ni:, :, i] == np.nan``

    Raises
    ------
    ValueError
        If any element of ``arrs`` does not have shape ``(3, Ni)`` (i.e. the
        first dimension is not 3).

    Notes
    -----
    - The returned array is always floating-point because NaN padding requires
      a float dtype.
    - The input arrays are copied into the output; the result is not a view.
    """
    # arrs: list of arrays, each (3, Ni)
    number_of_traces = len(arrs)
    max_trace_length = max(a.shape[1] for a in arrs)
    shape_err = any(a.shape[0] != 3 for a in arrs)
    if shape_err:
        raise ValueError("All input arrays must have shape (3, Ni)")

    geometry = np.full((max_trace_length, 3, number_of_traces), np.nan, dtype=float)

    if not to_trace:
        for i, a in enumerate(arrs):
            geometry[:a.shape[1], :, i] = a.T   # (3, ni) -> (ni, 3)
        return geometry
    else:
        start_pos = np.full((3, number_of_traces), np.nan, dtype=float)
        end_pos = np.full((3, number_of_traces), np.nan, dtype=float)
        for i, a in enumerate(arrs):
            geometry[:a.shape[1], :, i] = a.T   # (3, ni) -> (ni, 3)
            start_pos[:, i] = a[:,0]
            end_pos[:, i] = a[:, -1]
        return Traces(geometry, start_pos, end_pos, traced_to_boundary)





def s2c(r, t, p):
    """
    convert numpy arrays of r,t,p (spherical) to x,y,z (cartesian)
    - r, t, p are numpy arrays of any shape (must match). t and p must be in radians.
    - x, y, z are returned in the same units as r.
    """
    ct = np.cos(t)
    st = np.sin(t)
    cp = np.cos(p)
    sp = np.sin(p)
    x = r * cp * st
    y = r * sp * st
    z = r * ct
    return x, y, z


def plot_traces(*iargs,
                ax=None,
                **kwargs):
    """
    Quick and dirty 3D plot of fieldlines using matplotlib.

    Parameters
    ----------
    *iargs : Traces or ndarray
        One or more `Traces` objects or 3D arrays of shape (M, 3, N).
        Each argument will be plotted in a different color.

    Notes
    -----
    This function is intended for quick visualization of fieldline traces.
    For more advanced plotting capabilities, consider using libraries like
    PyVista or Mayavi.
    """
    import matplotlib
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')

    for traces in iargs:
        fls = traces.geometry if isinstance(traces, Traces) else traces
        if fls.ndim == 3:
            x, y, z = s2c(fls[:, 0, :], fls[:, 1, :], fls[:, 2, :])
            x, y, z = x.T, y.T, z.T

            segments = [np.column_stack([x[i], y[i], z[i]]) for i in range(x.shape[0])]

            # Choose colormap
            colors = matplotlib.colormaps['hsv'](np.random.random_sample(len(segments)))

            lc = Line3DCollection(segments, linewidths=1.0, **kwargs)
            ax.add_collection3d(lc)
        else:
            x, y, z = s2c(fls[0, :], fls[1, :], fls[2, :])
            ax.plot3D(x, y, z)

    return ax


def plot_sphere(values, r, t, p, clim=None, ax=None):
    """
    Quick and dirty 3D plot of a spherical surface using matplotlib.

    Parameters
    ----------
    values : ndarray
        A 2D array of shape (len(t), len(p)) representing the values on the spherical surface.
    r : float
        The radius at which to plot the spherical surface.
    t : ndarray
        A 1D array of theta (co-latitude) positions [radians, 0-pi].
    p : ndarray
        A 1D array of phi (longitude) positions [radians, 0-2pi].

    Notes
    -----
    This function is intended for quick visualization of spherical surfaces.
    For more advanced plotting capabilities, consider using libraries like
    PyVista or Mayavi.
    """
    import matplotlib
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')

    # Create a meshgrid for theta and phi
    T, P = np.meshgrid(t, p, indexing='ij')

    # Convert spherical to Cartesian coordinates
    X, Y, Z = s2c(r, T, P)

    cmin = clim[0] if clim is not None else values.min()
    cmax = clim[1] if clim is not None else values.max()

    # Plot the surface with the provided values as color mapping
    # Plot sphere surface with colormap
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=matplotlib.colormaps['seismic']((values - cmin) / (cmax - cmin)),
        rstride=1, cstride=1,
        linewidth=0, antialiased=False, shade=False
    )

    return ax

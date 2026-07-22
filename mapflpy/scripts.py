"""
Standalone functions for running mapflpy tracing routines.

These functions provide a simplified interface for performing forward, backward, and
forward-backward tracing using the :any:`TracerMP` class. They handle the initialization
and execution of the tracing processes, allowing users to easily obtain tracing results
without needing to manage the underlying tracer objects directly.

This module also includes a specialized function for inter-domain tracing, which coordinates
tracing between two different magnetic domains (*viz.*, coronal and heliospheric) using
multiprocessing. This function manages the complexities of boundary conditions and trace
concatenation.

.. note::
   This module is designed to encapsulate standard "one-off" tracing workflows, *i.e.*:

   - a :class:`~mapflpy.tracer.TracerMP` object(s) is instantiated with the appropriate magnetic field files and parameters,
   - one or more tracing calls are performed on a set of launch points (with or without additional post-processing logic),
   - and, lastly, the tracing object(s) is destroyed and the resultant traces are returned.

"""
from __future__ import annotations
import concurrent.futures
import copy
from contextlib import ExitStack
from functools import partial
from typing import Optional, Iterable, Tuple, Callable

import numpy as np
from numpy.typing import NDArray, ArrayLike

from mapflpy.globals import DEFAULT_BUFFER_SIZE, Traces, Mapping, PathType, DirectionType, ContextType, SquashingFactor
from mapflpy.tracer import TracerMP
from mapflpy.utils import shift_phi_traces, shift_phi_lps, fetch_default_launch_points, modulo_twopi, get_half_mesh, calc_jacobian, calc_q
from psi_io import interpolate_positions_from_hdf

__all__ = [
    "run_forward_tracing",
    "run_backward_tracing",
    "run_fwdbwd_tracing",
    "inter_domain_tracing",
    "_inter_domain_tracing"
]


def run_forward_tracing(br: PathType,
                        bt: PathType,
                        bp: PathType,
                        launch_points: Optional[ArrayLike] = None,
                        buffer_size: int = DEFAULT_BUFFER_SIZE,
                        **kwargs
                        ) -> Traces:
    """
    Run forward tracing using TracerMP.

    This function initializes a :class:`~mapflpy.tracer.TracerMP` instance and calls the
    :meth:`~mapflpy.tracer._Tracer.trace_fwd` method to perform forward tracing from the specified
    launch points. The buffer size can be adjusted to control the number of points in the trace geometry.

    .. note::
       If ``launch_points`` is None, default launch points will be used – a Fibonacci lattice
       generated through the :func:`~mapflpy.utils.fetch_default_launch_points` function.

       If ``launch_points`` is specified, it should be an array-like object that can be reshaped
       into a (3, N) :class:`~numpy.ndarray`, where N is the number of launch points. Each column of this array
       should represent a launch point in spherical coordinates (radius, theta, phi).

    Parameters
    ----------
    br, bt, bp : PathType
        Paths to hdf4 or hdf5 magnetic field files (:math:`B_r`, :math:`B_{\\theta}`, :math:`B_{\\phi}`).
    launch_points : ArrayLike | None, optional
        Launch points used by the tracer (see above note).
    buffer_size : int, optional
        Buffer size for trace geometry. Default is 2000.
    **kwargs
        Additional keyword arguments to be passed to the :any:`TracerMP` initialization.

    Returns
    -------
    Traces
        A `Traces` object containing the results of the forward tracing.

    See Also
    --------
    :meth:`~mapflpy.tracer._Tracer.trace_fwd`
    """
    with TracerMP(br, bt, bp, **kwargs) as tracer:
        return tracer.trace_fwd(launch_points, buffer_size)


def run_backward_tracing(br: PathType,
                         bt: PathType,
                         bp: PathType,
                         launch_points: Optional[ArrayLike] = None,
                         buffer_size: int = DEFAULT_BUFFER_SIZE,
                         **kwargs
                         ) -> Traces:
    """
    Run backward tracing using TracerMP.

    This function initializes a :class:`~mapflpy.tracer.TracerMP` instance and calls the
    :meth:`~mapflpy.tracer._Tracer.trace_bwd` method to perform backward tracing from the specified
    launch points. The buffer size can be adjusted to control the number of points in the trace geometry.

    .. note::
       If ``launch_points`` is None, default launch points will be used – a Fibonacci lattice
       generated through the :func:`~mapflpy.utils.fetch_default_launch_points` function.

       If ``launch_points`` is specified, it should be an array-like object that can be reshaped
       into a (3, N) :class:`~numpy.ndarray`, where N is the number of launch points. Each column of this array
       should represent a launch point in spherical coordinates (radius, theta, phi).

    Parameters
    ----------
    br, bt, bp : PathType
        Paths to hdf4 or hdf5 magnetic field files (:math:`B_r`, :math:`B_{\\theta}`, :math:`B_{\\phi}`).
    launch_points : ArrayLike | None, optional
        Launch points used by the tracer (see above note).
    buffer_size : int, optional
        Buffer size for trace geometry. Default is 2000.
    **kwargs
        Additional keyword arguments to be passed to the :any:`TracerMP` initialization.

    Returns
    -------
    Traces
        A `Traces` object containing the results of the forward tracing.

    See Also
    --------
    :meth:`~mapflpy.tracer._Tracer.trace_bwd`
    """
    with TracerMP(br, bt, bp, **kwargs) as tracer:
        return tracer.trace_bwd(launch_points, buffer_size)


def run_fwdbwd_tracing(br: PathType,
                       bt: PathType,
                       bp: PathType,
                       launch_points: Optional[ArrayLike] = None,
                       buffer_size: int = DEFAULT_BUFFER_SIZE,
                       **kwargs
                       ) -> Traces:
    """
    Run forward and backward tracing using TracerMP.

    This function initializes a :class:`~mapflpy.tracer.TracerMP` instance and calls the
    :meth:`~mapflpy.tracer._Tracer.trace_fbwd` method to perform forward and backward tracing
    from the specified launch points. The buffer size can be adjusted to control the number
    of points in the trace geometry.

    .. note::
       If ``launch_points`` is None, default launch points will be used – a Fibonacci lattice
       generated through the :func:`~mapflpy.utils.fetch_default_launch_points` function.

       If ``launch_points`` is specified, it should be an array-like object that can be reshaped
       into a (3, N) :class:`~numpy.ndarray`, where N is the number of launch points. Each column of this array
       should represent a launch point in spherical coordinates (radius, theta, phi).

    Parameters
    ----------
    br, bt, bp : PathType
        Paths to hdf4 or hdf5 magnetic field files (:math:`B_r`, :math:`B_{\\theta}`, :math:`B_{\\phi}`).
    launch_points : ArrayLike | None, optional
        Launch points used by the tracer (see above note).
    buffer_size : int, optional
        Buffer size for trace geometry. Default is 2000.
    **kwargs
        Additional keyword arguments to be passed to the :any:`TracerMP` initialization.

    Returns
    -------
    Traces
        A `Traces` object containing the results of the forward tracing.

    See Also
    --------
    :meth:`~mapflpy.tracer._Tracer.trace_fbwd`
    """
    with TracerMP(br, bt, bp, **kwargs) as tracer:
        return tracer.trace_fbwd(launch_points, buffer_size)


def map_field_lines_in_parallel(trace_function: Callable,
                                r_in: ArrayLike,
                                t_in: ArrayLike,
                                p_in: ArrayLike,
                                nproc: int = 4):
    """Map field lines from r, t, p positions in parallel using a given tracing script.

    The `trace_function` is a callable function that wraps the desired mapflpy script that uses TracerMP in
    the background. It must only accept one argument (`launch_points`) and otherwise wraps a tracing
    script that is defined with any specific keywords that determine how the trace is done.
    Currently only :func:`run_forward_tracing`, :func:`run_backward_tracing` make sense to wrap here.

    The r,t,p input arrays can be any shape (1D, 2D, 3D, etc.) as long as it is consistent.

    .. warning::
       The wrapped trace function should specify the `buffer_size` keyword to a small number (e.g. 3)
       since the trace geometry is not used anyway. This is essential for keeping the memory footprint
       small and for the mapping to compute as quickly as possible.

    Parameters
    ----------
    trace_function : Callable
        A function that wraps the mapflpy forward or backwards tracing script to call in parallel
        (:func:`run_forward_tracing`, :func:`run_backward_tracing`) and sets the desired parameters to
        run the wrapped script with.
        Only the endpoints are used, so only one of forward or backwards mapping makes sense.
    r_in : ArrayLike
        An N-Dimensional array of radial positions to map from.
    t_in : ArrayLike
        An N-Dimensional array of theta positions to map from (shape must match `r_in`).
    p_in : ArrayLike
        An N-Dimensional array of phi positions to map from (shape must match `r_in`).
    nproc : int, optional
        The number of processes to spawn. This should be equal to or less than the number of threads
        that can be used on the machine.

    Returns
    -------
    mapping : :class:`~mapflpy.globals.Mapping`
        A namedtuple containing the mapping results (:class:`mapflpy.globals.Mapping`).

    See Also
    --------
    :func:`map_pt_forward`
    :func:`map_pt_backward`
    :func:`run_forward_tracing`
    :func:`run_backward_tracing`
    """

    # convert the r,t,p positions to a list of launchpoints split into nproc chunks
    r1d = r_in.ravel()
    t1d = t_in.ravel()
    p1d = p_in.ravel()
    lp_all = np.stack([r1d, t1d, p1d], axis=0)
    lp_chunks = np.array_split(lp_all, nproc, axis=1)

    # run the mapping function on nprocs
    with concurrent.futures.ThreadPoolExecutor(max_workers=nproc) as executor:
        results = list(executor.map(trace_function, lp_chunks))

    # collect the results back into one array
    end_pos_1d = np.concatenate([result.end_pos for result in results], axis=1)
    traced_to_boundary_1d = np.concatenate([result.traced_to_boundary for result in results])
    integral_1d = np.concatenate([result.integral for result in results])

    # reshape and return the Mapping object
    r_out = np.reshape(end_pos_1d[0, :], r_in.shape)
    t_out = np.reshape(end_pos_1d[1, :], t_in.shape)
    p_out = np.reshape(end_pos_1d[2, :], p_in.shape)
    traced_to_boundary = np.reshape(traced_to_boundary_1d, r_in.shape)
    integral = np.reshape(integral_1d, r_in.shape)

    return Mapping(r_out, t_out, p_out, traced_to_boundary, integral)


def map_pt_forward(br: PathType,
                   bt: PathType,
                   bp: PathType,
                   p1d: ArrayLike,
                   t1d: ArrayLike,
                   radius: float = 1.0,
                   timeout: float = 3600.,
                   context: Optional[ContextType] = 'fork',
                   nproc: int = 4,
                   **mapfl_params):
    """Map field lines forwards on a phi, theta grid at constant radius.

    The inputs are similar to instantiating a Tracer class except one also
    specifies 1D arrays that define a grid in phi, theta and a radius to map from.

    Currently the mapping radius defaults to 1.0 or must be set. In the future
    this will default to the inner radius of the main mesh of the B files.

    Notes
    -----
    This is a very specific helper function for a common task. Generic mappings
    can be built by following this as an example for wrapping :func:`map_field_lines_in_parallel`.

    Parameters
    ----------
    br : PathType
        Path to the Br magnetic field file.
    bt : PathType
        Path to the Bt magnetic field file.
    bp : PathType
        Path to the Bp magnetic field file.
    p1d : ArrayLike
        A 1D numpy array of phi (longitude) positions in radians that will be used to
        build the mapping grid.
    t1d : ArrayLike
        A 1D numpy array of theta (co-latitude) positions in radians that will be used to
        build the mapping grid.
    radius : float, optional
        Radius to map forward from. Default 1.0.
    timeout : float, optional
        Timeout in seconds for interprocess communication. Default is 3600 seconds.
    context : ContextType, optional
        The multiprocessing context to use when spawning the subprocess. Since many
        processes are launched, generally 'fork' is recommended. Behavior may depend
        on the system architecture. Default is 'fork'.
    nproc : int, optional
        The number of processes to spawn. This should be equal to or less than the number of threads
        that can be used on the machine.
    **mapfl_params : dict
        Additional tracing parameters passed to the subprocess.

    Returns
    -------
    mapping : :class:`~mapflpy.globals.Mapping`
        A namedtuple containing the mapping results (:class:`mapflpy.globals.Mapping`).

    See Also
    --------
    :func:`map_field_lines_in_parallel`
    """

    # build a 2D grid of p and t locations from the input 1D arrays
    p2d, t2d = np.meshgrid(p1d, t1d)
    # the r locations are on the specified radius
    r2d = np.ones_like(p2d)*radius

    # define the wrapped trace function
    def trace_function(launch_points):
        traces = run_forward_tracing(br, bt, bp, launch_points=launch_points,
                                     buffer_size=3, context=context, timeout=timeout,
                                     **mapfl_params)
        return traces

    # compute the mapping
    mapping = map_field_lines_in_parallel(trace_function, r2d, t2d, p2d, nproc=nproc)

    return mapping


def map_pt_backward(br: PathType,
                    bt: PathType,
                    bp: PathType,
                    p1d: ArrayLike,
                    t1d: ArrayLike,
                    radius: float = 30.0,
                    timeout: float = 3600.,
                    context: Optional[ContextType] = 'fork',
                    nproc: int = 4,
                    **mapfl_params):
    """Map field lines backwards on a phi, theta grid at constant radius.

    The inputs are similar to instantiating a Tracer class except one also
    specifies 1D arrays that define a grid in phi, theta and a radius to map from.

    Currently the mapping radius defaults to 30 or must be set. In the future
    this will default to the outer radius of the main mesh of the B files.

    Notes
    -----
    This is a very specific helper function for a common task. Generic mappings
    can be built by following this as an example for wrapping :func:`map_field_lines_in_parallel`.

    Parameters
    ----------
    br : PathType
        Path to the Br magnetic field file.
    bt : PathType
        Path to the Bt magnetic field file.
    bp : PathType
        Path to the Bp magnetic field file.
    p1d : ArrayLike
        A 1D numpy array of phi (longitude) positions in radians that will be used to
        build the mapping grid.
    t1d : ArrayLike
        A 1D numpy array of theta (co-latitude) positions in radians that will be used to
        build the mapping grid.
    radius : float, optional
        Radius to map backwards from. Default 30.0.
    timeout : float, optional
        Timeout in seconds for interprocess communication. Default is 3600 seconds.
    context : ContextType, optional
        The multiprocessing context to use when spawning the subprocess. Since many
        processes are launched, generally 'fork' is recommended. Behavior may depend
        on the system architecture. Default is 'fork'.
    nproc : int, optional
        The number of processes to spawn. This should be equal to or less than the number of threads
        that can be used on the machine.
    **mapfl_params : dict
        Additional tracing parameters passed to the subprocess.

    Returns
    -------
    mapping : :class:`~mapflpy.globals.Mapping`
        A namedtuple containing the mapping results (:class:`mapflpy.globals.Mapping`).

    See Also
    --------
    :func:`map_field_lines_in_parallel`
    """


    # build a 2D grid of p and t locations from the input 1D arrays
    p2d, t2d = np.meshgrid(p1d, t1d)
    # the r locations are on the specified radius
    r2d = np.ones_like(p2d)*radius

    # define the wrapped trace function
    def trace_function(launch_points):
        traces = run_backward_tracing(br, bt, bp, launch_points=launch_points,
                                      buffer_size=3, context=context, timeout=timeout,
                                      **mapfl_params)
        return traces

    # compute the mapping
    mapping = map_field_lines_in_parallel(trace_function, r2d, t2d, p2d, nproc=nproc)

    return mapping

def expansion_factor(b_files, mapping, trace_radius, pss, tss):
    """
    Calculate the expansion factor of a given set of field line launch points and
    their mapped end points.

    Parameters
    ----------
    b_files : list of str
         Specified the magnetic field (br, bt, bp ordered) files
    mapping : :class:`~mapflpy.globals.Mapping`
        A namedtuple containing mapping results (:class:`mapflpy.globals.Mapping`).
    trace_radius : float
        The radius from which to map.
    pss : ndarray
        phi points used to generate the mapping
    tss : ndarray
        theta points used to generate the mapping

    Returns
    -------
    efl : ndarray
        An array of the expansion factor
    pss : ndarray
        phi points used to generate the expansion factor
    tss : ndarray
        theta points used to generate the expansion factor


    Notes
    -----
    - This function manually handles periodic boundaries in phi
    """

    # for the sake of interp: make sure phi's periodicity is obeyed
    if (pss[0] < 0) or (pss[-1] > 2 * np.pi):
        pss = np.mod(pss, np.pi * 2)
    # make 2d launch point meshes so we can get the initial magnetic field
    tss2d, pss2d = np.meshgrid(tss, pss)
    ones2d_ss = np.ones_like(pss2d)

    # we get [p, t] ordered arrays with the magnetic field values
    # xfl0: radial launch point location
    xfl0 = ones2d_ss * trace_radius
    # xfl1: radial traced point location
    xfl1 = np.transpose(mapping.r)
    # bs0: radial magnetic field launch point
    bs0 = interpolate_positions_from_hdf(b_files[0], xfl0, tss2d, pss2d)
    # bs1: radial magnetic field traced point
    bs1 = interpolate_positions_from_hdf(b_files[0], np.transpose(mapping.r), np.transpose(mapping.t),
                                                np.transpose(mapping.p))

    # logic checks:
    # if we have a magnetic field of 0 in the denominator, set the expansion factor to 0:
    efl_raw = np.where(bs0 == 0, 0, abs((bs0 * xfl0 ** 2) / (bs1 * xfl1 ** 2)))
    # if we make it to the boundary, use our calculated expansion factor. otherwise, = 0:
    efl = np.where(np.transpose(mapping.traced_to_boundary), efl_raw, 0)

    return efl, pss, tss

def compute_q_on_surface(b_files, direction='fwd', nproc=4, trace_radius=1, p_arr=np.asarray([]),
                         t_arr=np.asarray([]), p_range=None, t_range=None, nppts=300, ntpts=150):
    """
    This wrapper calculates the squashing factor for a specified slice on a specified grid.

    Parameters
    ----------
    b_files : list of str
        Specified the magnetic field (br, bt, bp ordered) files
    direction : str
        either "fwd," "bwd," or "fwdbwd" to specify what direction of tracing (last is averaged)
    nproc: int, optional
        The number of processes to spawn. This should be equal to or less than the number of threads that can be used on the machine.
    trace_radius : float, optional
        The radius from which to map. Defaults to 1.
    p_arr : ndarray,  optional
        User-specified array for phi points used to generate the mapping. Must also specify t_arr. Either specify thse or t_range and p_range.
    t_arr : ndarray,  optional
        User-specified array for theta points used to generate the mapping. Must also specify p_arr. Either specify thse or t_range and p_range.
    p_range : list or ndarray of two float, optional
        User-specified start and end point in theta. defaults to [0, 2*np.pi].
    t_range : list or ndarray of two float, optional
        User-specified start and end point in theta. defaults to [0, np.pi].
    nppts   : int, optional
        Number of points desired in phi when p_range is in use
    ntpts   : int, optional
        Number of points desired in theta when t_range is in use

    Returns
    -------
    squashing_factor : :class:`~mapflpy.globals.SquashingFactor`
        A namedtuple containing the q, p, t results (:class:`mapflpy.globals.SquashingFactor`).

    """
    # default ranges for t, p
    if t_range is None:
        t_range = [0, np.pi]
    if p_range is None:
        p_range = [0, 2 * np.pi]

    # creating theta and phi ranges for field lines from
    # user-defined domains (or above default ranges)
    if t_arr.shape[0] == 0:
        tss_specified = np.linspace(t_range[0], t_range[1], ntpts)
        pss_specified = np.linspace(p_range[0], p_range[1], nppts)

        pss = get_half_mesh(pss_specified)

        if (t_range[0] == 0) or (t_range[-1] == np.pi):

            th = get_half_mesh(tss_specified)
            tss = np.copy(th)

            # clip the boundaries
            tss[0] = 0.0
            tss[-1] = np.pi
            clipped = True
        else:
            tss = get_half_mesh(tss_specified)
            clipped = False
    # using the user-defined mesh for t, p
    else:
        th = get_half_mesh(t_arr)
        tss = np.copy(th)
        pss = get_half_mesh(p_arr)
        if (tss[0] < 0) or (tss[-1] > np.pi):
            # clip the boundaries
            tss[0] = 0.0
            tss[-1] = np.pi
            clipped = True
        else:
            clipped = False

    # field-line tracing by direction. starting with forward
    if direction == 'fwd':
        print('fwd mapping')
        mapping = map_pt_forward(*b_files, pss, tss, radius=trace_radius, nproc=nproc)

        # make the expansion factor.
        ef_arr, p_ef, t_ef = expansion_factor(b_files, mapping, trace_radius, pss, tss)

        # make the components of the jacobian
        dtdt, dtdp, dpdt, dpdp = calc_jacobian(mapping, pss, tss)
        # put the expansion factor, jacobian, and field lines together to get q
        q, p, t = calc_q(dtdt, dtdp, dpdt, dpdp, mapping, pss, tss, ef_arr, clipped)

        return SquashingFactor(q, p, t)

    # field-line tracing by direction. now backward
    elif direction == 'bwd':
        print('bwd mapping')

        mapping = map_pt_backward(*b_files, pss, tss, radius=trace_radius, nproc=nproc)

        # make the expansion factor.
        ef_arr, p_ef, t_ef = expansion_factor(b_files, mapping, trace_radius, pss, tss)

        # make the components of the jacobian
        dtdt, dtdp, dpdt, dpdp = calc_jacobian(mapping, pss, tss)

        # put the expansion factor, jacobian, and field lines together to get q
        q, p, t = calc_q(dtdt, dtdp, dpdt, dpdp, mapping, pss, tss, ef_arr, clipped)

        return SquashingFactor(q, p, t)

    # field-line tracing by direction. now take the average of the forward/backward
    elif direction == 'fwdbwd':
        print('fwdbwd mapping')

        mapping_fwd = map_pt_forward(*b_files, pss, tss, radius=trace_radius, nproc=nproc)
        mapping_bwd = map_pt_backward(*b_files, pss, tss, radius=trace_radius, nproc=nproc)

        # make the expansion factor.
        ef_arr_fwd, p_ef_f, t_ef_f = expansion_factor(b_files, mapping_fwd, trace_radius, pss, tss)
        ef_arr_bwd, p_ef_b, t_ef_b = expansion_factor(b_files, mapping_bwd, trace_radius, pss, tss)

        # make the components of the jacobian
        dtdt_fwd, dtdp_fwd, dpdt_fwd, dpdp_fwd = calc_jacobian(mapping_fwd, pss, tss)
        dtdt_bwd, dtdp_bwd, dpdt_bwd, dpdp_bwd = calc_jacobian(mapping_bwd, pss, tss)

        # put the expansion factor, jacobian, and field lines together to get q
        q_fwd, p_fwd, t_fwd = calc_q(dtdt_fwd, dtdp_fwd, dpdt_fwd, dpdp_fwd, mapping_fwd, pss, tss, ef_arr_fwd, clipped)
        q_bwd, p_bwd, t_bwd = calc_q(dtdt_bwd, dtdp_bwd, dpdt_bwd, dpdp_bwd, mapping_bwd, pss, tss, ef_arr_bwd, clipped)

        # reutrn the averaged quantity
        p = 0.5 * (p_fwd + p_bwd)
        t = 0.5 * (t_fwd + t_bwd)
        q = 0.5 * (q_fwd + q_bwd)

        return SquashingFactor(q, p, t)

    else:
        raise Exception("specify a valid direction: fwd, bwd, fwdbwd")

def inter_domain_tracing(br_cor: PathType,
                         bt_cor: PathType,
                         bp_cor: PathType,
                         br_hel: PathType,
                         bt_hel: PathType,
                         bp_hel: PathType,
                         launch_points: Optional[ArrayLike | int] = None,
                         buffer_size: int = DEFAULT_BUFFER_SIZE,
                         maxiter: int = 10,
                         r_interface: float = 30.0,
                         helio_shift: float = 0.0,
                         rtol: float = 1e-5,
                         **mapfl_params
                         ) -> Tuple[list, NDArray[bool], NDArray[bool]]:
    """
    Perform inter-domain tracing using two tracer processes.

    This method sets up two tracer processes (*e.g.*, for different magnetic domains) that run concurrently.
    It coordinates the tracing between these two processes via multiprocessing pipes. Because launch points
    that start in the corona or heliosphere are handled differently, this function wraps the lower-level inter-domain
    tracing methods to trace forward and backwards from launch points in any domain, joins them together, and returns
    all traces.

    Parameters
    ----------
    br_cor, bt_cor, bp_cor : PathType
        Paths to hdf4 or hdf5 coronal magnetic field files (:math:`B_r`, :math:`B_{\\theta}`, :math:`B_{\\phi}`).
    br_hel, bt_hel, bp_hel : PathType
        Paths to hdf4 or hdf5 heliospheric magnetic field files (:math:`B_r`, :math:`B_{\\theta}`, :math:`B_{\\phi}`).
    launch_points : ArrayLike | int | None, optional
        Launch points used by the tracer. Default is None.
    buffer_size : int, optional
        Buffer size for trace geometry. Default is 2000.
    maxiter : int, optional
        Maximum number of iterations for handling boundary recrossing(s). Default is 10.
    r_interface : float, optional
        Radius at which to connect the traces between domains. Default is 30.
    helio_shift : float, optional
        Longitudinal shift angle between the heliospheric domain and the coronal domain in RADIANS.
        This shift is ADDED to the coronal launch point phi positions. Default is 0.0.
    rtol : float, optional
        Relative tolerance for `np.isclose` for checking a trace has hit the interface boundary. Default is 1e-5.
    **mapfl_params
        Additional keyword arguments to be passed to both tracer initializations.

    Returns
    -------
    final_traces : list of ndarray
        A list of numpy arrays representing the concatenated tracing results for each launch point
        such that:

        - The list is :math:`N` elements long, where :math:`N` is the number of launch points.
        - Each element of the list is a numpy array with shape (3, :math:`M_i`) where
          :math:`M_i` is the number of points in the trace for launch point :math:`i`.

    traced_to_boundary : ndarray
        A boolean array of size :math:`N` indicating whether trace :math:`i` trace hit the
        inner "cor" or outer "hel" boundary on both ends.
    boundary_recross : ndarray
        A boolean array of size :math:`N` indicating whether trace :math:`i` recrossed the
        r_interface boundary after initially hitting it (i.e., whether the trace had to be
        iteratively traced back and forth between domains more than once).

    Notes
    -----
    The function uses two separate :class:`mapflpy.tracer.TracerMP` objects – one for the coronal
    domain and one for the heliospheric domain – to avoid sharing ``mapflpy_fortran`` objects between
    domains.

    .. attention::
       These tracer objects are initialized with the same ``**mapfl_params`` to ensure they
       are configured consistently (with the one caveat that the coronal tracer has its
       ``domain_r_max_`` set to the specified radial interface, while the heliospheric tracer has its
       ``domain_r_min_`` set to the same radial interface).

    See Also
    --------
    :func:`_inter_domain_tracing`
    """
    cor_params = mapfl_params.copy()
    hel_params = mapfl_params.copy()
    with ExitStack() as cstack:
        cor_tracer = cstack.enter_context(TracerMP(br_cor, bt_cor, bp_cor, **cor_params))
        hel_tracer = cstack.enter_context(TracerMP(br_hel, bt_hel, bp_hel, **hel_params))
        return _inter_domain_tracing(
            cor_tracer,
            hel_tracer,
            launch_points=launch_points,
            buffer_size=buffer_size,
            maxiter=maxiter,
            r_interface=r_interface,
            helio_shift=helio_shift,
            rtol=rtol,
        )


def _inter_domain_tracing(cor_tracer: TracerMP,
                          hel_tracer: TracerMP,
                          launch_points: Optional[NDArray[float] | int] = None,
                          buffer_size: int = DEFAULT_BUFFER_SIZE,
                          maxiter: int = 10,
                          r_interface: float = 30.0,
                          helio_shift: float = 0.0,
                          rtol: float = 1e-5,
                          ) -> Tuple[list, NDArray[bool], NDArray[bool]]:
    """
    Perform inter-domain tracing using two tracer processes.

    This private method is exposed in the public API so that the internal tracing logic used for
    interdomain tracing can be accessed with user-provided tracer objects (e.g., if users want
    to manage their own tracer contexts or use different tracer configurations for the two domains).
    This method receives two initialized tracer objects (one for each domain) that are run concurrently.

    Parameters
    ----------
    cor_tracer : ~mapflpy.tracer.TracerMP
        Multiprocessing compatible tracer object initialized with coronal magnetic field files.
    hel_tracer : ~mapflpy.tracer.TracerMP
        Multiprocessing compatible tracer object initialized with heliospheric magnetic field files.
    launch_points : ArrayLike | int | None, optional
        Launch points used by the tracer. Default is None.
    buffer_size : int, optional
        Buffer size for trace geometry. Default is 2000.
    maxiter : int, optional
        Maximum number of iterations for handling boundary recrossing(s). Default is 10.
    r_interface : float, optional
        Radius at which to connect the traces between domains. Default is 30.
    helio_shift : float, optional
        Longitudinal shift angle between the heliospheric domain and the coronal domain in RADIANS.
        This shift is ADDED to the coronal launch point phi positions. Default is 0.0.
    rtol : float, optional
        Relative tolerance for :func:`~numpy.isclose` for checking a trace has hit the interface boundary. Default is 1e-5.
    **mapfl_params
        Additional keyword arguments to be passed to both tracer initializations.

    Returns
    -------
    final_traces : list of ndarray
        A list of numpy arrays representing the concatenated tracing results for each launch point
        such that:

        - The list is :math:`N` elements long, where :math:`N` is the number of launch points.
        - Each element of the list is a numpy array with shape (3, :math:`M_i`) where
          :math:`M_i` is the number of points in the trace for launch point :math:`i`.

    traced_to_boundary : ndarray
        A boolean array of size :math:`N` indicating whether trace :math:`i` trace hit the
        inner "cor" or outer "hel" boundary on both ends.
    boundary_recross : ndarray
        A boolean array of size :math:`N` indicating whether trace :math:`i` recrossed the
        r_interface boundary after initially hitting it (i.e., whether the trace had to be
        iteratively traced back and forth between domains more than once).
    """

    cor_tracer['domain_r_max_'] = r_interface
    hel_tracer['domain_r_min_'] = r_interface

    match launch_points:
        case None:
            # if no launch points are provided, use the default launch points
            lp = fetch_default_launch_points()
        case int():
            # if an integer is provided, use that many default launch points
            lp = fetch_default_launch_points(launch_points)
        case _:
            try:
                lp = np.asarray(launch_points, dtype=float).reshape((3, -1))
            except Exception as e:
                raise ValueError(f"Invalid launch points type: {type(launch_points)}. "
                                 "Expected None, int, or Iterable of launch points.") from e
    # prepare the final arrays
    n_lp = lp.shape[1]
    final_traces = [None] * n_lp
    boundary_recross = np.full(n_lp, False)
    traced_to_boundary = np.full(n_lp, False)

    # determine which indexes of the launch points are coronal and which are heliospheric
    inds_coronal = np.where(lp[0, :] <= r_interface)[0]
    inds_helio = np.where(lp[0, :] > r_interface)[0]

    # separate the launch points by domain (length 0 arrays are OK here)
    cor_lp = lp[:, inds_coronal]
    hel_lp = lp[:, inds_helio]

    # CORONAL INTERDOMAIN TRACE
    if len(inds_coronal) > 0:
        inter_cor = partial(_inter_domain_tracing_from_cor,
                            cor_tracer, hel_tracer,
                            lp=cor_lp, maxiter=maxiter, r_interface=r_interface, helio_shift=helio_shift, rtol=rtol, buffer=buffer_size)
        # trace coronal launch points forward/backward from the coronal domain
        traces_cor_fwd, bndry_cor_fwd, recross_cor_fwd = inter_cor(direction='f')
        traces_cor_bwd, bndry_cor_bwd, recross_cor_bwd = inter_cor(direction='b')

        # join the traces, flipping the backwards trace after dropping its first point (first point is the starting point)
        traces_cor = [np.concatenate((np.flip(traces_cor_bwd[i][:, 1:], axis=1),
                                      traces_cor_fwd[i]), axis=1)
                      for i in range(len(traces_cor_bwd))]

        # combine the tracing flags
        recross_cor = recross_cor_bwd & recross_cor_fwd
        bndry_cor = bndry_cor_bwd & bndry_cor_fwd

        # populate the coronal launch points
        for i, trace in zip(inds_coronal, traces_cor):
            final_traces[i] = trace
        boundary_recross[inds_coronal] = recross_cor
        traced_to_boundary[inds_coronal] = bndry_cor

    # HELIOSPHERIC INTERDOMAIN TRACE
    if len(inds_helio) > 0:
        inter_hel = partial(_inter_domain_tracing_from_hel,
                            cor_tracer, hel_tracer,
                            lp=hel_lp, maxiter=maxiter, r_interface=r_interface, helio_shift=helio_shift, rtol=rtol, buffer=buffer_size)
        # trace helonal launch points forward/backward from the heliospheric domain
        traces_hel_fwd, bndry_hel_fwd, recross_hel_fwd = inter_hel(direction='f')
        traces_hel_bwd, bndry_hel_bwd, recross_hel_bwd = inter_hel(direction='b')

        # join the traces, flipping the backwards trace after dropping its first point (first point is the starting point)
        traces_hel = [np.concatenate((np.flip(traces_hel_bwd[i][:, 1:], axis=1),
                                      traces_hel_fwd[i]), axis=1)
                      for i in range(len(traces_hel_bwd))]

        # combine the tracing flags
        recross_hel = recross_hel_bwd & recross_hel_fwd
        bndry_hel = bndry_hel_bwd & bndry_hel_fwd

        # populate the heliospheric launch points
        for i, trace in zip(inds_helio, traces_hel):
            final_traces[i] = trace
        boundary_recross[inds_helio] = recross_hel
        traced_to_boundary[inds_helio] = bndry_hel

    return final_traces, traced_to_boundary, boundary_recross


def _inter_domain_tracing_from_cor(cor_tracer: TracerMP,
                                   hel_tracer: TracerMP,
                                   direction: DirectionType,
                                   lp: Iterable[float],
                                   maxiter: int,
                                   r_interface: float,
                                   helio_shift: float,
                                   rtol: float,
                                   buffer: int
                                   ) -> Tuple[list, NDArray[bool], NDArray[bool]]:
    """
    Perform inter-domain coronal and heliospheric tracing for CORONAL launch points in the specified direction.

    This method receives two tracer processes (one for each domain) that are run concurrently. The function initiates
    tracing in one process, checks for boundary recrossings, and if necessary, alternates the tracing between
    the two processes until the tracing endpoints no longer cross a defined boundary or the maximum number of
    iterations is reached.

    Parameters
    ----------
    cor_reciever : multiprocessing.connection.Connection
        The coronal domain pipe that does the mapfl tracing (see `tracer_listener`).
    hel_reciever : multiprocessing.connection.Connection
        The heliospheric domain pipe that does the mapfl tracing (see `tracer_listener`).
    direction : str
        The direction of the mapfl tracings. This must be either 'f' or 'b' (forwards or backwards). Default is 'f'.
    lp : any
        Launch points for fieldline tracing.
    maxiter : int, optional
        Maximum number of iterations for handling boundary recrossings. Default is 10.
    r_interface : float, optional
        Radius at which to connect the traces between domains. Default is 30.
    helio_shift : float, optional
        Longitudinal shift angle between the heliospheric domain and the coronal domain in RADIANS.
        This shift is ADDED to the coronal launch point phi positions. Default is 0.0.
    rtol : float, optional
        Relative tolerance for `np.isclose` for checking a trace has hit the interface boundary. Default is 1e-5.

    Returns
    -------
    final_traces : list
        A list of numpy arrays representing the concatenated tracing results for each launch point.
    traced_to_boundary : numpy.ndarray
        A boolean array indicating whether this trace hit the inner cor or outer hel boundary.
    boundary_recross : numpy.ndarray
        A boolean array indicating whether a boundary recrossing occurred for each launch point.

    """
    # set the launch points, make a copy so input lp doesn't get vaporized on succesive traces
    cor_lps = copy.deepcopy(lp)
    cor_tracer.set_tracing_direction(direction)
    traces_ = cor_tracer.trace(cor_lps, buffer)

    final_traces = list([arr[:, ~np.isnan(arr).any(axis=0)] for arr in traces_.geometry.T])

    # get the end positions of the traces in r,t,p, shape: (3,n)
    radial_end_pos = np.copy(traces_.end_pos)

    # determine which traces hit the interface boundary
    midboundary_mask = np.isclose(radial_end_pos[0, :], r_interface, rtol=rtol)

    # initialize the array for checking that you went back through
    boundary_recross = np.full_like(midboundary_mask, False)

    # check that you hit a boundary to end the trace
    traced_to_boundary = traces_.traced_to_boundary

    while np.any(midboundary_mask) and maxiter > 0:
        # set the new heliospheric launchpoints using any that hit the interface from the corona
        # these must also be shifted FORWARD in phi by the helio shift value
        hel_lps = shift_phi_lps(radial_end_pos[:, midboundary_mask], helio_shift)
        hel_tracer.set_tracing_direction('f')
        traces_ = hel_tracer.trace(hel_lps, buffer)
        temp_traces = list([arr[:, ~np.isnan(arr).any(axis=0)] for arr in traces_.geometry.T])

        # shift these traces BACK to the coronal/carrington frame
        temp_traces = shift_phi_traces(temp_traces, -helio_shift)

        # add this trace segment neglecting the first point since it duplicates the last point of the previous segment.
        for i, trace in zip(np.where(midboundary_mask)[0], temp_traces):
            final_traces[i] = np.concatenate([final_traces[i], trace[:, 1:]], axis=1)

        # check that you hit a boundary
        traced_to_boundary[midboundary_mask] = traces_.traced_to_boundary

        # update the radial end positions (SHIFTED BACK!)
        radial_end_pos[:,
        midboundary_mask] = shift_phi_lps(np.copy(traces_.end_pos), -helio_shift)

        # update the flag for traces that hit the interface
        midboundary_mask = np.isclose(radial_end_pos[0, :], r_interface, rtol=rtol)

        # now trace through the corona BACKWARDS
        if np.any(midboundary_mask):
            boundary_recross |= midboundary_mask

            # take the subset of launch points and trace.
            cor_lps = radial_end_pos[:, midboundary_mask]
            cor_tracer.set_tracing_direction('b')
            traces_ = cor_tracer.trace(cor_lps, buffer)

            temp_traces = list([arr[:, ~np.isnan(arr).any(axis=0)] for arr in
                                traces_.geometry.T])
            for i, trace in zip(np.where(midboundary_mask)[0], temp_traces):
                final_traces[i] = np.concatenate([final_traces[i], trace[:, 1:]], axis=1)

            # check the trace, update the end positions and the midboundary flag, continue the loop
            traced_to_boundary[midboundary_mask] = traces_.traced_to_boundary
            radial_end_pos[:, midboundary_mask] = traces_.end_pos
            midboundary_mask = np.isclose(radial_end_pos[0, :], r_interface, rtol=rtol)
            boundary_recross |= midboundary_mask

        # if no more work to be done, break the loop
        else:
            break
        maxiter -= 1

    # return the final traces and tracing checks
    return final_traces, traced_to_boundary, boundary_recross


def _inter_domain_tracing_from_hel(cor_tracer: TracerMP,
                                   hel_tracer: TracerMP,
                                   direction: DirectionType,
                                   lp: Iterable[float],
                                   maxiter: int,
                                   r_interface: float,
                                   helio_shift: float,
                                   rtol: float,
                                   buffer: int
                                   ) -> Tuple[list, NDArray[bool], NDArray[bool]]:
    """
    Perform inter-domain coronal and heliospheric tracing for HELIOSPHERIC launch points in the specified direction.

    This method receives two tracer processes (one for each domain) that are run concurrently. The function initiates
    tracing in one process, checks for boundary recrossings, and if necessary, alternates the tracing between
    the two processes until the tracing endpoints no longer cross a defined boundary or the maximum number of
    iterations is reached.

    Parameters
    ----------
    cor_reciever : multiprocessing.connection.Connection
        The coronal domain pipe that does the mapfl tracing (see `tracer_listener`).
    hel_reciever : multiprocessing.connection.Connection
        The heliospheric domain pipe that does the mapfl tracing (see `tracer_listener`).
    direction : str
        The direction of the mapfl tracings. This must be either 'f' or 'b' (forwards or backwards). Default is 'f'.
    lp : any
        Launch points for fieldline tracing.
    maxiter : int, optional
        Maximum number of iterations for handling boundary recrossings. Default is 10.
    r_interface : float, optional
        Radius at which to connect the traces between domains. Default is 30.
    helio_shift : float, optional
        Longitudinal shift angle between the heliospheric domain and the coronal domain in RADIANS.
        This shift is ADDED to the coronal launch point phi positions. Default is 0.0.
    rtol : float, optional
        Relative tolerance for `np.isclose` for checking a trace has hit the interface boundary. Default is 1e-5.

    Returns
    -------
    final_traces : list
        A list of numpy arrays representing the concatenated tracing results for each launch point.
    traced_to_boundary : numpy.ndarray
        A boolean array indicating whether this trace hit the inner cor or outer hel boundary.
    boundary_recross : numpy.ndarray
        A boolean array indicating whether a boundary recrossing occurred for each launch point.

    """
    # set the launch points, and shift them accordingly
    hel_lps = copy.deepcopy(lp)
    hel_lps = shift_phi_lps(hel_lps, helio_shift)
    hel_tracer.set_tracing_direction(direction)
    traces_ = hel_tracer.trace(hel_lps, buffer)

    final_traces = list([arr[:, ~np.isnan(arr).any(axis=0)] for arr in traces_.geometry.T])
    # shift these traces BACK to the coronal/carrington frame
    final_traces = shift_phi_traces(final_traces, -helio_shift)

    # get and shift the end positions of the traces in r,t,p, shape: (3,n)
    radial_end_pos = shift_phi_lps(np.copy(traces_.end_pos), -helio_shift)

    # determine which traces hit the interface boundary
    midboundary_mask = np.isclose(radial_end_pos[0, :], r_interface, rtol=rtol)

    # initialize the array for checking that you went back through
    boundary_recross = np.full_like(midboundary_mask, False)

    # check that you hit a boundary to end the trace
    traced_to_boundary = traces_.traced_to_boundary

    while np.any(midboundary_mask) and maxiter > 0:
        # set the new coronal launchpoints using any that hit the interface from the heliosphere
        cor_lps = radial_end_pos[:, midboundary_mask]
        cor_tracer.set_tracing_direction('b')
        traces_ = cor_tracer.trace(cor_lps, buffer)
        temp_traces = list([arr[:, ~np.isnan(arr).any(axis=0)] for arr in traces_.geometry.T])

        # add this trace segment neglecting the first point since it duplicates the last point of the previous segment.
        for i, trace in zip(np.where(midboundary_mask)[0], temp_traces):
            final_traces[i] = np.concatenate([final_traces[i], trace[:, 1:]], axis=1)

        # check that you hit a boundary
        traced_to_boundary[midboundary_mask] = traces_.traced_to_boundary

        # update the radial end positions
        radial_end_pos[:, midboundary_mask] = np.copy(traces_.end_pos)

        # update the flag for traces that hit the interface
        midboundary_mask = np.isclose(radial_end_pos[0, :], r_interface, rtol=rtol)

        # now trace through the heliosphere FORWARDS
        if np.any(midboundary_mask):
            boundary_recross |= midboundary_mask

            # set the new heliospheric launchpoints using any that hit the interface from the corona
            # these must also be shifted FORWARD in phi by the helio shift value
            hel_lps = shift_phi_lps(radial_end_pos[:, midboundary_mask], helio_shift)

            # take the subset of launch points and trace.
            hel_tracer.set_tracing_direction('f')
            traces_ = hel_tracer.trace(hel_lps, buffer)

            temp_traces = list([arr[:, ~np.isnan(arr).any(axis=0)] for arr in
                                traces_.geometry.T])

            # shift these traces BACK to the coronal/carrington frame
            temp_traces = shift_phi_traces(temp_traces, -helio_shift)

            # add this trace segment neglecting the first point since it duplicates the last point of the previous segment.
            for i, trace in zip(np.where(midboundary_mask)[0], temp_traces):
                final_traces[i] = np.concatenate([final_traces[i], trace[:, 1:]], axis=1)

            # check the trace, update the end positions (SHIFTED BACK!) and the midboundary flag, continue the loop
            traced_to_boundary[midboundary_mask] = traces_.traced_to_boundary
            radial_end_pos[:,
            midboundary_mask] = shift_phi_lps(np.copy(traces_.end_pos), -helio_shift)
            midboundary_mask = np.isclose(radial_end_pos[0, :], r_interface, rtol=rtol)
            boundary_recross |= midboundary_mask

        # if no more work to be done, break the loop
        else:
            break
        maxiter -= 1

    # return the final traces and tracing checks
    return final_traces, traced_to_boundary, boundary_recross

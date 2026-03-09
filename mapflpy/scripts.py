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
import copy
from contextlib import ExitStack
from functools import partial
from typing import Optional, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray, ArrayLike

from mapflpy.globals import DEFAULT_BUFFER_SIZE, Traces, PathType, DirectionType
from mapflpy.tracer import TracerMP
from mapflpy.utils import shift_phi_traces, shift_phi_lps, fetch_default_launch_points

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

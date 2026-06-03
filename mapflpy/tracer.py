"""
Fieldline Tracing Interface for mapflpy (Single and Multiprocessing Modes)

This module provides a Python interface for fieldline tracing using ``mapflpy_fortran``
(a wrapper for the Fortran-based ``mapfl`` tracing engine).

It defines a unified tracing API through the abstract base class :class:`_Tracer`, with two main
concrete implementations:

- :class:`Tracer`: Executes fieldline tracing in the current (main) Python process. Suitable for
  lightweight, single-threaded use cases.
- :class:`TracerMP`: Runs tracing in a background subprocess via :py:mod:`multiprocessing`, allowing
  multiple Tracer instances to coexist across processes without interfering with each
  other's state. This is required due to the global state held by the Fortran library.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import ChainMap
from collections.abc import MutableMapping
from functools import wraps
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Tuple
from weakref import WeakSet

import numpy as np
from numpy.typing import ArrayLike, NDArray
from psi_io import read_hdf_data

from mapflpy.globals import (
    DEFAULT_PARAMS,
    DEFAULT_FIELDS,
    MAGNETIC_FIELD_LABEL,
    DIRECTION,
    DEFAULT_BUFFER_SIZE,
    MAGNETIC_FIELD_PATHS,
    PathType,
    MagneticFieldArrayType,
    DirectionType,
    MagneticFieldLabelType,
    ArrayType,
    Traces, ContextType)
from mapflpy.utils import fetch_default_launch_points, combine_fwd_bwd_traces

__all__ = ["Tracer", "TracerMP",]

_BS0 = np.zeros(3, order='F').astype(np.float64)
"""Sentinel array for :meth:`mapfl.trace` **bs0** parameter."""

_BS1 = np.zeros(3, order='F').astype(np.float64)
"""Sentinel array for :meth:`mapfl.trace` **bs1** parameter."""


def state_modifier(method):
    """
    Decorator for mutator methods that modify internal state.

    This decorator marks the tracer instance as "stale" by setting ``self._stale = True``
    after executing the wrapped method. It is intended to be applied to methods that
    modify the internal parameter dictionary or magnetic field data in a way that
    requires the tracer to be re-initialized (e.g., calling ``run()`` again).

    Parameters
    ----------
    method : Callable
        The instance method that mutates the tracer’s state.

    Returns
    -------
    Callable
        A wrapped version of the method that also sets `_stale = True`.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._stale = True
        return result

    return wrapper


def flush_state(method):
    """
    Decorator to ensure the tracer state is up-to-date before executing a method.

    This decorator checks whether the instance is marked as "stale" (via ``self._stale``).
    If so, it calls ``self._run()`` to reinitialize the Fortran backend with current parameters
    before proceeding to call the decorated method.

    This ensures that the most recent configuration is applied to operations like ``trace()``.

    Parameters
    ----------
    method : Callable
        The instance method that requires a flushed state.

    Returns
    -------
    Callable
        A wrapped version of the method that automatically calls ``self._run()`` if needed.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._stale:
            self._run()
        result = method(self, *args, **kwargs)
        return result

    return wrapper


def check_process_state(method):
    """
    Decorator for TracerMP methods that communicate with the subprocess.

    This decorator ensures the subprocess is alive before invoking the method,
    then waits (with timeout) for a response from the subprocess over the
    :py:func:`~multiprocessing.Pipe`.

    It performs the following actions:

    1. Checks if the subprocess (``self._process``) is still alive.

       - If not, it calls ``self.disconnect()`` and raises a :py:class:`RuntimeError`.

    2. Calls the decorated method (which is expected to send a message via ``self._parent.send(...)``).

    3. Waits up to ``self._timeout`` seconds for a response from the subprocess via ``self._parent.recv()``.

       - If no message is received in time, it raises a :py:class:`RuntimeError`.

       - If a message is received, it returns the result to the caller.

    .. attention::
        This is intended for internal use within the :class:`TracerMP` class only.

    Parameters
    ----------
    method : Callable
        The method being decorated. Must send a command to the subprocess via ``self._parent.send(...)``.

    Returns
    -------
    Callable
        A wrapped method that performs the process state check and handles the response.

    Raises
    ------
    RuntimeError
        If the subprocess is not alive or if no response is received within the timeout.

    Notes
    -----
    The timeout enforcement is primarily implemented due to the fact that certain errors
    that arise in mapflpy_fortran are not passed back to the caller; instead the error is
    written to ``stdout`` and then hangs indefinitely. This decorator ensures that
    such situations are handled gracefully by raising a timeout exception.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Ensure the subprocess is still running
        if not self._process.is_alive():
            self.disconnect()
            raise RuntimeError("The multiprocessing process is not alive. ")
        # Invoke the original method (which sends a command to the child process)
        method(self, *args, **kwargs)

        # Wait for response within the timeout provided
        if self._parent.poll(self._timeout):
            result = self._parent.recv()
            if isinstance(result, Exception):
                raise result
            return result
        else:
            raise RuntimeError("Timeout while waiting for mapflpy_fortran response.")

    return wrapper


# This function is left as a reference or passing large arrays over piped connections
# using multiprocessing and shared memory. However, the current implementation of TracerMP
# handles magnetic field data as file paths, which is more efficient and avoids any
# potential memory issues with large arrays.
def _load_array_to_shared_memory(array: np.ndarray,
                                 name: str = None) -> Tuple[SharedMemory, np.ndarray]:
    if not array.flags['C_CONTIGUOUS']:
        raise ValueError("Array must be Fortran-contiguous to avoid copy. Use np.asfortranarray first.")

    shm = SharedMemory(create=True, size=array.nbytes, name=name)
    shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shm_array[...] = array[...]

    return shm, shm_array


def _mapflpy_trace_listener(pipe):
    """
    Subprocess listener for handling fieldline tracing commands using `mapflpy_fortran`.

    This function is intended to be used in a separate Python process (typically spawned via
    `multiprocessing.Process`) to run fieldline tracing routines in isolation from the main process.
    It receives commands over a :py:func:`~multiprocessing.Pipe`, processes them using the
    `mapflpy_fortran.mapfl` interface, and sends results back over the pipe.

    This allows `TracerMP` to delegate tracing logic to a worker process while avoiding
    cross-process state conflicts and enabling safe multiprocessing.

    The listener responds to messages received via ``pipe.recv()``. Each message must be a tuple
    of the form:
        ``(method_name: str, *args)``

    **Supported Methods**

    - ``"mapfl_id",``:
        Returns the ``id(mapfl)`` from the subprocess.
    - ``"run", iparams: dict, ifields: dict``:
        Updates `mapfl_params` with provided parameters and magnetic field filepaths,
        reads new magnetic field data if needed, and invokes `mapfl.run(...)`.
    - ``"trace", launch_points: ndarray, buffer_size: int``:
        Performs fieldline tracing using `mapfl.trace` for the given set of launch points and
        returns a `Traces` object.
    - ``"break",``:
        Terminates the subprocess listener loop.

    Parameters
    ----------
    pipe : ~multiprocessing.connection.Connection
        A bidirectional pipe endpoint connected to the parent process. Used for receiving commands
        and sending back results.

    Notes
    -----
    - Magnetic field data (Br, Bt, Bp) is loaded lazily and updated only when file paths change.
    - Exceptions in the subprocess are caught and sent back to the parent via the pipe.

    Raises
    ------
    Exception
        Any exception raised in the subprocess will be sent through the pipe and re-raised locally.
    """
    try:
        import mapflpy.fortran.mapflpy_fortran as mapflpy_fortran
        mapfl = mapflpy_fortran.mapfl
        mapfl_params = ChainMap({}, DEFAULT_PARAMS, DEFAULT_FIELDS)
        magnetic_fields = dict(MAGNETIC_FIELD_PATHS)

        def _set_field(key: MagneticFieldLabelType, value: str) -> None:
            # This function is the same as the _set_field method called in Tracer._set_field()
            # For further details on the implementation of this function, see the Tracer
            # class documentation.
            bx, bx_r, bx_t, bx_p = read_hdf_data(value)
            match bx.shape:
                case fshape if fshape == (len(bx_p), len(bx_t), len(bx_r)):
                    mapfl_params[key] = bx.astype('float64').T
                case cshape if cshape == (len(bx_r), len(bx_t), len(bx_p)):
                    mapfl_params[key] = bx.astype('float64')
                case _:
                    raise ValueError(f"Magnetic field array for '{key}' has invalid shape {bx.shape}. "
                                     f"Expected shape (n_r, n_t, n_p) or (n_p, n_t, n_r).")
            for dim in 'rtp':
                mapfl_params[f"{key}_{dim}"] = locals()[f"bx_{dim}"].astype('float64')
                mapfl_params[f"{key}_n{dim}"] = len(mapfl_params[f"{key}_{dim}"])

        def mapfl_id():
            return id(mapfl)

        def run(iparams, ifields):
            mapfl_params.update(**iparams)
            for k, v in ifields.items():
                if magnetic_fields[k] != v:
                    magnetic_fields[k] = v
                    _set_field(k, v)
            return mapfl.run(**mapfl_params)

        def trace(lps, buffer_size):
            # This function is the same as the trace method called in Tracer._trace()
            # For further details on the implementation of this function, see the Tracer
            # class documentation.
            traces = np.full((buffer_size, *lps.shape), np.nan, order='F').astype(np.float64)
            s1 = np.zeros(lps.shape, np.float64, order='F')
            mask = np.full((1, lps.shape[1]), False, order='F')
            integral = np.full((1, lps.shape[1]), 0.0, order='F')
            for i in range(lps.shape[1]):
                trace_args = dict(
                    s0=lps[:, i],
                    s1=s1[:, i],
                    bs0=_BS0,
                    bs1=_BS1,
                    s=integral[:,i],
                    traced_to_r_boundary=mask[:, i],
                    svec=traces[:, :, i],
                    svec_n=buffer_size
                )
                mapfl.trace(**trace_args)
            return Traces(traces, lps, s1, mask[0, :], integral[0,:])

        while True:
            method, *args = pipe.recv()
            match method.lower():
                case "mapfl_id":
                    pipe.send(mapfl_id())
                case "run":
                    pipe.send(run(*args))
                case "trace":
                    pipe.send(trace(*args))
                case "break":
                    break
        pipe.close()
    except Exception as e:
        pipe.send(e)
        pipe.close()
        # raise e


class _Tracer(MutableMapping, ABC):
    """
    Abstract base class for fieldline tracing with :mod:`mapflpy_fortran`.

    This class defines a mapping-like interface to configure tracing parameters
    for the :mod:`mapflpy_fortran` module, a Python interface to the cross-compiled
    ``mapfl`` Fortran code. It supports custom loading of magnetic field data and
    provides common methods for managing the tracing lifecycle.

    .. warning::
       This interface is designed as a base class for concrete implementations such as
       :class:`~mapflpy.tracer.Tracer` (single-process) and
       :class:`~mapflpy.tracer.TracerMP` (multiprocessing-safe).
       Subclasses implement the tracing logic and interaction with
       :func:`mapflpy_fortran.mapfl.run`.

       **Subclasses must implement the following methods**

       - :meth:`__init__`
       - :meth:`_run`
       - :meth:`_trace`
       - :meth:`_mapfl_id`

    Parameters
    ----------
    br : PathType, optional
        HDF filepath for the radial component of the magnetic field.
    bt : PathType, optional
        HDF filepath for the theta component of the magnetic field.
    bp : PathType, optional
        HDF filepath for the phi component of the magnetic field.

    Other Parameters
    ----------------
    **mapfl_params
        Additional parameters forwarded to :func:`mapflpy_fortran.mapfl.run`.
        These override defaults in :data:`~mapflpy.typing.DEFAULT_PARAMS`.

    Attributes
    ----------
    _mapfl_params : ~collections.ChainMap
        Mapping of parameters passed to :meth:`run` (which sets global ``mapfl`` state).
    _stale : bool
        Whether changes to :attr:`_mapfl_params` need to be propagated via :meth:`run`.

    Raises
    ------
    ValueError
        If an invalid value is provided for magnetic field components.
    TypeError
        If values are not of the expected type (``tuple`` or ``str``).
    FileNotFoundError
        If provided file paths do not exist.
    ImportError
        If :mod:`mapflpy_fortran` cannot be imported (shared library missing or not built).
    """

    __hash__ = object.__hash__

    @abstractmethod
    def __init__(self,
                 br: Optional[PathType] = None,
                 bt: Optional[PathType] = None,
                 bp: Optional[PathType] = None,
                 **mapfl_params):
        # Combine user-provided, default, and fallback fields
        self._mapfl_params = ChainMap(mapfl_params, DEFAULT_PARAMS, DEFAULT_FIELDS)

        # Optionally load the magnetic field datasets
        for label, dim in zip(('br', 'bt', 'bp'), (br, bt, bp)):
            if dim is not None:
                self._set_field(label, dim)
        self._stale = True

    def __repr__(self):
        return f"Tracer({self._mapfl_params})"

    def __str__(self):
        return f"Tracer({self._mapfl_params})"

    # -----------------------
    # MutableMapping protocol
    # -----------------------

    def __iter__(self):
        return iter(self._mapfl_params)

    def __len__(self):
        return len(self._mapfl_params)

    def __getitem__(self, __item):
        return self._mapfl_params[__item]

    @state_modifier
    def __setitem__(self, __key, __value):
        self._mapfl_params[__key] = __value

    @state_modifier
    def __delitem__(self, __key):
        del self._mapfl_params[__key]

    @state_modifier
    def clear(self):
        """Reset the mapfl :attr:`params` to their default values."""
        self._mapfl_params.clear()

    @state_modifier
    def update(self, *args, **kwargs):
        """Update mapfl :attr:`params` with the provided key-value pairs."""
        self._mapfl_params.update(*args, **kwargs)

    @state_modifier
    def popitem(self):
        """Remove and return the value of the given mapfl :attr:`params` key."""
        return self._mapfl_params.popitem()

    @state_modifier
    def pop(self, __key):
        """Pop an item from the mapfl :attr:`params` ."""
        return self._mapfl_params.pop(__key)

    # -----------------------------------
    # Magnetic field component properties
    # -----------------------------------

    @property
    def br(self) -> MagneticFieldArrayType:
        """Return the radial component of the magnetic field and its scales.

        .. note::
            :class:`TracerMP` subclass returns only the provided field file path."""
        return self._get_field('br')

    @br.setter
    @state_modifier
    def br(self, value: MagneticFieldArrayType) -> None:
        self._set_field('br', value)

    @property
    def bt(self) -> MagneticFieldArrayType:
        """Return the theta component of the magnetic field and its scales.

        .. note::
            :class:`TracerMP` subclass returns only the provided field file path."""
        return self._get_field('bt')

    @bt.setter
    @state_modifier
    def bt(self, value: MagneticFieldArrayType) -> None:
        self._set_field('bt', value)

    @property
    def bp(self) -> MagneticFieldArrayType:
        """Return the phi component of the magnetic field and its scales.

        .. note::
            :class:`TracerMP` subclass returns only the provided field file path."""
        return self._get_field('bp')

    @bp.setter
    @state_modifier
    def bp(self, value: MagneticFieldArrayType) -> None:
        self._set_field('bp', value)

    # ----------------------------------
    # Core Tracer properties and methods
    # ----------------------------------

    @property
    def params(self) -> MappingProxyType:
        """Return an immutable view of the mapfl :attr:`_mapfl_params` dictionary."""
        return MappingProxyType(self._mapfl_params)

    @property
    def stale(self) -> bool:
        """Indicates whether the parameters have changed since the last :meth:`run` call."""
        return self._stale

    @property
    def mapfl_id(self) -> int:
        """Return the memory ID of the ``mapflpy_fortran`` interface (useful for debugging).
        Dispatches to overridden :meth:`_mapfl_id` method in subclasses."""
        return self._mapfl_id()

    @state_modifier
    def load_fields(self,
                    br: MagneticFieldArrayType,
                    bt: MagneticFieldArrayType,
                    bp: MagneticFieldArrayType
                    ) -> None:
        """
        Load Br, Bt, and Bp fields into the input parameters.

        .. note::
            This method is equivalent to calling `Tracer.br = br, Tracer.bt = bt, and Tracer.bp = bp`.
            The same constraints apply to the input values as for the individual setters, *e.g.* for
            multiprocessing support in :class:`TracerMP`, the values must be paths to HDF files.

        Parameters
        ----------
        br, bt, bp : MagneticFieldArrayType
            Each should be an HDF filepath or tuple of (bx, bx_r, bx_t, bx_p),
            where bx is the magnetic field component and bx_r, bx_t, bx_p are
            the radial, theta, and phi scales respectively.

        Raises
        ------
        FileNotFoundError
            If the provided file paths do not exist.
        TypeError
            If the provided values are not of the expected type (tuple or str).
        """
        self._set_field('br', br)
        self._set_field('bt', bt)
        self._set_field('bp', bp)

    def _set_field(self,
                   key: MagneticFieldLabelType,
                   value: MagneticFieldArrayType
                   ) -> None:
        """
        Internal helper to register a magnetic field and its coordinate axes.

        Parameters
        ----------
        key : MagneticFieldLabelType
            The magnetic field component label.
        value : MagneticFieldArrayType
            A 4-tuple of field data (data, r, t, p), or a path to an HDF file.
        """
        if key not in MAGNETIC_FIELD_LABEL:
            raise ValueError(f"Key '{key}' is not a valid magnetic field component. "
                             f"Valid keys are: {MAGNETIC_FIELD_LABEL}")
        match value:
            case (bx, bx_r, bx_t, bx_p):
                pass
            case filepath if isinstance(filepath, (str, Path, PathLike)):
                if not Path(filepath).exists():
                    raise FileNotFoundError(f"File '{filepath}' does not exist.")
                bx, bx_r, bx_t, bx_p = read_hdf_data(str(filepath))
            case None:
                self._mapfl_params[key] = None
                for dim in 'rtp':
                    self._mapfl_params[f"{key}_{dim}"] = None
                    self._mapfl_params[f"{key}_n{dim}"] = None
                return
            case _:
                raise TypeError(f"Value for '{key}' must be a tuple of (bx, bx_r, bx_t, bx_p) "
                                f"or a path to an HDF file containing these values.")

        # Ensure that the data values array for the field has the correct shape
        # Expected shapes are (n_r, n_t, n_p) or (n_p, n_t, n_r)
        # mapflpy_fortran expects (n_r, n_t, n_p), which is the transpose of the
        # default output from PSI's hdf readers and writers **i.e.** (n_p, n_t, n_r)
        match bx.shape:
            case fshape if fshape == (len(bx_p), len(bx_t), len(bx_r)):
                self._mapfl_params[key] = bx.astype('float64').T
            case cshape if cshape == (len(bx_r), len(bx_t), len(bx_p)):
                self._mapfl_params[key] = bx.astype('float64')
            case _:
                raise ValueError(f"Magnetic field array for '{key}' has invalid shape {bx.shape}. "
                                 f"Expected shape (n_r, n_t, n_p) or (n_p, n_t, n_r).")

        for dim in 'rtp':
            self._mapfl_params[f"{key}_{dim}"] = locals()[f"bx_{dim}"].astype('float64')
            self._mapfl_params[f"{key}_n{dim}"] = len(self._mapfl_params[f"{key}_{dim}"])

    def _get_field(self,
                   key: MagneticFieldLabelType
                   ) -> MagneticFieldArrayType:
        """
        Retrieve a magnetic field component and its coordinates.

        Returns
        -------
        field : MagneticFieldLabelType
            A tuple of (bx, bx_r, bx_t, bx_p) for the specified magnetic field component,
            or a path to an HDF file if using TracerMP.
        """
        if key not in MAGNETIC_FIELD_LABEL:
            raise ValueError(f"Key '{key}' is not a valid magnetic field component. "
                             f"Valid keys are: {MAGNETIC_FIELD_LABEL}")
        return self._mapfl_params[key], *[self._mapfl_params[f"{key}_{dim}"] for dim in 'rtp']

    def _parse_launch_points(self,
                             lps: Optional[ArrayLike],
                             **kwargs
                             ) -> ArrayType:
        """
        Parse or generate launch points for tracing.

        Parameters
        ----------
        lps : ArrayLike | None
            Launch points of shape (3, N) in spherical coordinates (r, t, p).
        kwargs : dict
            If `lps` is None, these are passed to :func:`~mapflpy.utils.fetch_default_launch_points`.

        Returns
        -------
        launch_points : ndarray
            A (3, N) Fortran-ordered array.
        """
        if lps is None:
            return fetch_default_launch_points(**kwargs)
        else:
            launch_points = np.asarray(lps)
            try:
                return launch_points.swapaxes(0, launch_points.shape.index(3)).reshape((3, -1))
            except ValueError:
                raise ValueError("Launch points could not be reshaped into a (3, N) array.")

    @state_modifier
    def set_tracing_direction(self,
                              direction: DirectionType = 'f'
                              ) -> None:
        """Set tracing direction flags.

        Parameters
        ----------
        direction : DirectionType
            'f' for forward tracing, 'b' for backward tracing.
        """
        if direction not in DIRECTION:
            raise ValueError("Direction must be either 'f' (forward) or 'b' (backward).")
        self._mapfl_params['trace_fwd_'] = direction == 'f'
        self._mapfl_params['trace_bwd_'] = direction == 'b'

    def run(self) -> None:
        """
        Initialize or reinitialize the internal Fortran state with current parameters.
        """
        self._run()
        self._stale = False

    @flush_state
    def trace(self,
              launch_points: Optional[ArrayLike] = None,
              buffer_size: int = DEFAULT_BUFFER_SIZE,
              **kwargs
              ) -> Traces:
        """
        Perform fieldline tracing from launch points.

        Parameters
        ----------
        launch_points : Arraylike | None, optional
            Array of shape (3, N) for r, t, p coordinates. If None, uses defaults.
        buffer_size : int
            Maximum number of steps per fieldline.
        **kwargs
            Extra arguments passed to default launch point generator if ``launch_points`` is None.

        Returns
        -------
        Traces
            Structured container with traced fieldline geometry, starting points, ending points,
            and whether the fieldlines reached the radial boundary of the provided mesh.

        Notes
        -----
        The ``launch_points`` should be in Fortran (column-major) order, i.e., shape (3, N).
        where N is the number of launch points, and the first dimension (of size 3) corresponds
        to r, t, p coordinates in the carrington frame, such that:
            - r (radial distance) is in R_sun,
            - t (co-latitude) is in radians,
            - p (longitude) is in radians.
        """
        lps = self._parse_launch_points(launch_points, **kwargs)
        return self._trace(lps, buffer_size)

    def trace_fwd(self, *args, **kwargs):
        """
        Perform forward fieldline tracing from launch points.

        This is a convenience method equivalent to calling ``set_tracing_direction('f')``
        before invoking ``trace``.

        See Also
        --------
        :meth:`trace`
        :meth:`set_tracing_direction`
        """
        self.set_tracing_direction('f')
        return self.trace(*args, **kwargs)

    def trace_bwd(self, *args, **kwargs):
        """
        Perform backward fieldline tracing from launch points.

        This is a convenience method equivalent to calling ``set_tracing_direction('b')``
        before invoking ``trace``.

        See Also
        --------
        :meth:`trace`
        :meth:`set_tracing_direction`
        """
        self.set_tracing_direction('b')
        return self.trace(*args, **kwargs)

    def trace_fbwd(self, *args, **kwargs):
        """
        Perform fieldline tracing in both forward and backward directions.

        This is a convenience method equivalent to calling ``trace_fwd()`` and ``trace_bwd()``
        sequentially, combining the results into a single :py:class:`~mapflpy.typing.Traces` object.

        See Also
        --------
        :meth:`trace`
        :meth:`trace_fwd`
        :meth:`trace_bwd`
        :meth:`set_tracing_direction`
        """
        fwd_traces = self.trace_fwd(*args, **kwargs)
        bwd_traces = self.trace_bwd(*args, **kwargs)
        return combine_fwd_bwd_traces(fwd_traces, bwd_traces)

    # ---------------------------------
    # mapflpy_fortran interface methods
    # ---------------------------------

    @abstractmethod
    def _mapfl_id(self) -> int:
        """Return the memory ID of the Fortran interface (for process isolation validation)."""

    @abstractmethod
    def _run(self) -> None:
        """Invoke :meth:`run` with the current parameter set."""

    @abstractmethod
    def _trace(self,
               lps: ArrayType,
               buff: int
               ) -> Traces:
        """Perform fieldline tracing using `mapfl.trace()`."""


class Tracer(_Tracer):
    """
    Concrete implementation of the :class:`_Tracer` class for single-process fieldline tracing.

    This class provides a direct interface to the ``mapflpy_fortran.mapfl`` object
    for tracing magnetic field lines using in-memory data and parameter control.
    It is not safe for use in multiprocessing contexts due to Fortran global state.

    .. attention::
        - For subprocess-safe tracing (using :py:mod:`multiprocessing`), use :class:`TracerMP`.
        - This class enforces a singleton pattern within a process to avoid conflicts
          with the Fortran global state, as managed by :class:`_Tracer`.

    Parameters
    ----------
    br : PathType, optional
        HDF filepath for the radial component of the magnetic field.
    bt : PathType, optional
        HDF filepath for the theta component of the magnetic field.
    bp : PathType, optional
        HDF filepath for the phi component of the magnetic field.
    **mapfl_params : dict
        Additional tracing parameters passed directly to ``mapflpy_fortran.mapfl.run()``.
        These parameters will override the default parameters defined in
        :data:`~mapflpy.typing.DEFAULT_PARAMS`.

    Attributes
    ----------
    _mapfl_params : ~collections.ChainMap
        Mapping of parameters passed to :meth:`~mapflpy.tracer._Tracer.run` (which sets the global ``mapfl`` state).
    _stale : bool
        Indicates whether changes to :attr:`_mapfl_params` need to be propagated through :meth:`run`.
    _instances : ~weakref.WeakSet
        A weak reference set to enforce singleton constraint for the Tracer class per process.

    Raises
    ------
    RuntimeError
        If multiple instances of Tracer are created in the same process.
    ValueError
        If an invalid value is provided for the magnetic field components.
    TypeError
        If the provided values for magnetic fields are not of the expected type (tuple or str).
    FileNotFoundError
        If the provided file paths for magnetic fields do not exist.
    ImportError
        If the `mapflpy_fortran` module cannot be imported, indicating that the Fortran
        shared library is not available or not built correctly.
    """
    _instances: WeakSet = WeakSet()
    """Weak reference set to track Tracer instances for singleton enforcement."""

    def __init__(self,
                 br: Optional[PathType] = None,
                 bt: Optional[PathType] = None,
                 bp: Optional[PathType] = None,
                 **mapfl_params):
        # Register instance and enforce singleton constraint per process
        Tracer._instances.add(self)
        if len(Tracer._instances) > 1:
            raise RuntimeError("Multiple instances of Tracer within the same process are not supported. "
                               "Use a single instance for tracing, or branch each Tracer into a subprocess.")
        super().__init__(br=br, bt=bt, bp=bp, **mapfl_params)
        import mapflpy.fortran.mapflpy_fortran as mapflpy_fortran
        self._mapfl = mapflpy_fortran.mapfl

    def _mapfl_id(self) -> int:
        """
        Return the memory ID of the Fortran interface (for process isolation validation).

        Returns
        -------
        int
            The memory ID of the ``mapfl`` object.
        """
        return id(self._mapfl)

    def _run(self) -> None:
        """
        Run the ``mapfl.run()`` method to configure the Fortran library with parameters.

        This sets the internal state of the tracing engine. Must be called before tracing
        if any parameters or fields have changed.
        """
        return self._mapfl.run(**self._mapfl_params)

    def _trace(self,
               lps: ArrayLike,
               buff: int) -> Traces:
        """
        Run fieldline tracing from a set of launch points.

        Parameters
        ----------
        lps : ArrayLike
            Launch points in shape (3, N), in Fortran (column-major) order.
            Each column corresponds to a (r, t, p) coordinate.
        buff : int
            Maximum number of steps for each fieldline trace.

        Returns
        -------
        Traces
            Structured container with traced fieldline geometry, starting points, ending points,
            and whether the fieldlines reached the radial boundary of the provided mesh.
        """
        # `traces`, `s1`, `mask`, and `integral` are initialized to Fortran-contiguous arrays
        # and are used to store the traced fieldlines, their end points, boundary masks, and
        # the scalar field integral along the field line (length by default).
        # `mapflpy_fortran` alters these arrays in-place during tracing.
        # Each array must have a dimensionality > 1 so that a numpy view is passed.
        traces = np.full((buff, *lps.shape), np.nan, order='F').astype(np.float64)
        s1 = np.zeros(lps.shape, np.float64, order='F')
        mask = np.full((1, lps.shape[1]), False, order='F')
        integral = np.full((1, lps.shape[1]), 0.0, order='F')

        # Since the parameters _bs0 and _bs1 are discarded after the trace,
        # we can use the globals `_BS0` and `_BS1` as placeholders to avoid having
        # to repeatedly create and destruct arrays.
        for i in range(lps.shape[1]):
            self._mapfl.trace(s0=lps[:, i],
                              s1=s1[:, i],
                              bs0=_BS0,
                              bs1=_BS1,
                              s=integral[:,i],
                              traced_to_r_boundary=mask[:, i],
                              svec=traces[:, :, i],
                              svec_n=buff)
        return Traces(traces, lps, s1, mask[0, :], integral[0,:])


class TracerMP(_Tracer):
    """
    Fieldline tracing wrapper class for ``mapflpy`` with multiprocessing support.

    This class launches a separate subprocess that manages a persistent instance of the
    ``mapflpy_fortran`` fieldline tracing engine. It communicates with the subprocess via a
    :py:func:`~multiprocessing.Pipe` interface. This allows safe use of Fortran code that relies on
    global state from Python without conflict across processes.

    This class is useful for:
    - Avoiding thread-safety issues with Fortran globals.
    - Running multiple :class:`TracerMP` instances in parallel via multiprocessing.

    Notes
    -----
    - This class is context-manager compatible (``with TracerMP(...) as tracer:``).
    - You **must** call :meth:`connect` before tracing (:meth:`__enter__`).
    - Remember to call :meth:`disconnect` when done (:meth:`__exit__`).
    - **Manually adding fields loaded into memory is not supported**; use the
      :attr:`br`, :attr:`bt`, and :attr:`bp` properties or :meth:`load_fields` to set magnetic
      fields. Passing large arrays directly to the subprocess is not efficient and may lead to
      memory issues. Instead, provide paths to HDF files containing the fields.

    Parameters
    ----------
    br : PathType, optional
        Path to the Br magnetic field file.
    bt : PathType, optional
        Path to the Bt magnetic field file.
    bp : PathType, optional
        Path to the Bp magnetic field file.
    timeout : float, optional
        Timeout in seconds for interprocess communication. Default is 30 seconds.
    context : ContextType, optional
        The multiprocessing context to use when spawning the subprocess.
        Default is 'spawn'.
    **mapfl_params : dict
        Additional tracing parameters passed to the subprocess.

    Attributes
    ----------
    _mapfl_params : ChainMap
        Mapping of parameters passed to :meth:`run()` (which sets the global ``mapfl`` state).
    _stale : bool
        Indicates whether changes to :attr:`_mapfl_params` need to be propagated through :meth:`run`.
    _process : ~multiprocessing.Process
        The subprocess running the tracing engine.
    _parent : ~multiprocessing.connection.Connection
        Parent end of the communication pipe.
    _child : ~multiprocessing.connection.Connection
        Child end (passed to the subprocess).
    _timeout : float
        Timeout value used in pipe communication.
    _magnetic_fields : dict
        Dictionary of magnetic field file paths keyed by 'br', 'bt', and 'bp'.

    Raises
    ------
    ValueError
        If an invalid value is provided for the magnetic field components.
    TypeError
        If the provided values for magnetic fields are not of the expected type (tuple or str).
    FileNotFoundError
        If the provided file paths for magnetic fields do not exist.
    ImportError
        If the `mapflpy_fortran` module cannot be imported, indicating that the Fortran
        shared library is not available or not built correctly.
    """

    def __init__(self,
                 br: Optional[PathType] = None,
                 bt: Optional[PathType] = None,
                 bp: Optional[PathType] = None,
                 timeout: Optional[float] = 30,
                 context: Optional[ContextType] = 'spawn',
                 **mapfl_params):
        # Setup communication pipe and subprocess
        ctx = get_context(context)
        self._parent, self._child = ctx.Pipe()
        self._process = ctx.Process(target=_mapflpy_trace_listener, args=(self._child,))
        self._magnetic_fields = dict(MAGNETIC_FIELD_PATHS)
        self._timeout = timeout

        # Initialize parent _Tracer logic
        super().__init__(br=br, bt=bt, bp=bp, **mapfl_params)

        # Drop the fallback/default fields layer in the ChainMap for clarity
        # These fields are handled strictly within the subprocess to avoid passing
        # large arrays directly through the pipe.
        self._mapfl_params = ChainMap(*self._mapfl_params.maps[:-1])

    def __enter__(self):
        """Enter the context and connect to the tracing subprocess."""
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context and terminate the tracing subprocess."""
        return self.disconnect()

    def connect(self):
        """
        Start the tracing subprocess if not already running.

        Returns
        -------
        self : TracerMP
            The connected tracer instance.
        """
        if not self._process.is_alive():
            self._process.start()
        return self

    def disconnect(self):
        """
        Gracefully shut down the tracing subprocess.

        Attempts to send a shutdown message, wait briefly for the process to exit,
        and then force termination if needed.
        """
        if self._process.is_alive():
            try:
                self._parent.send(("break", None))
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.terminate()
            except Exception as e:
                print(f"Error while shutting down TracerMP process: {e}")
        self._parent.close()
        self._child.close()
        self._process.close()

    def _set_field(self,
                   key: MagneticFieldLabelType,
                   value: Optional[PathType]
                   ) -> None:
        """
        Set the magnetic field file path for a given component ('br', 'bt', or 'bp').

        Parameters
        ----------
        key : MagneticFieldLabelType
            The magnetic field component label.
        value : PathType, optional
            Path to the HDF file, or None to clear the field.

        Raises
        ------
        FileNotFoundError
            If the given path does not exist.
        ValueError
            If the input key is not a valid magnetic field label.
        """
        if key not in MAGNETIC_FIELD_LABEL:
            raise ValueError(f"Key '{key}' is not a valid magnetic field component. "
                             f"Valid keys are: {MAGNETIC_FIELD_LABEL}")
        match value:
            case filepath if isinstance(filepath, (str, Path, PathLike)):
                if not Path(filepath).exists():
                    raise FileNotFoundError(f"File '{filepath}' does not exist.")
                self._magnetic_fields[key] = str(filepath)
            case None:
                self._magnetic_fields[key] = ''
            case _:
                raise ValueError(f"Input must be a path to an HDF file or None.")

    def _get_field(self,
                   key: MagneticFieldLabelType
                   ) -> str:
        """
        Retrieve the magnetic field file path for a given component.

        Parameters
        ----------
        key : MagneticFieldLabelType
            The magnetic field component label.

        Returns
        -------
        str
            The file path for the given magnetic field component.

        Raises
        ------
        ValueError
            If the key is not a valid magnetic field label.
        """
        if key not in MAGNETIC_FIELD_LABEL:
            raise ValueError(f"Key '{key}' is not a valid magnetic field component. "
                             f"Valid keys are: {MAGNETIC_FIELD_LABEL}")
        return self._magnetic_fields[key]

    @check_process_state
    def _mapfl_id(self):
        """
        Request the internal ``mapfl`` object ID from the subprocess.

        This sends a message to the child process asking for its ID.

        .. note::
            Return handling and error checking is done in the
            :func:`~mapflpy.tracer.check_process_state` decorator.
        """
        self._parent.send(("mapfl_id",))

    @check_process_state
    def _run(self):
        """
        Send run parameters and magnetic field paths to the subprocess.

        This triggers the `mapfl.run()` call inside the worker process with the
        current parameter and field configuration.

        .. note::
            Return handling and error checking is done in the
            :func:`~mapflpy.tracer.check_process_state` decorator.
        """
        self._parent.send(("run", self._mapfl_params.maps[0], self._magnetic_fields))

    @check_process_state
    def _trace(self,
               lps: NDArray[np.float64],
               buff: int):
        """
        Send trace request to the subprocess.

        .. note::
            Return handling and error checking is done in the
            :func:`~mapflpy.tracer.check_process_state` decorator.

        Parameters
        ----------
        lps : ndarray
            Launch points array in shape (3, N), Fortran-ordered.
        buff : int
            Maximum number of steps in each trace.
        """
        self._parent.send(("trace", lps, buff))

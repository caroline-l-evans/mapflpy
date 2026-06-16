
project(
  'mapflpy',
  'c', 'cpp', 'fortran',
  version: run_command(['tools/pyproject_version.py'], check: true).stdout().strip(),
  license : 'Apache-2.0',
  meson_version : '>=1.1.0',
  default_options : [
    'warning_level=1',
    'buildtype=release'
   ],
)

# ---- Compilers / platform niceties
cc = meson.get_compiler('c')
fc = meson.get_compiler('fortran')

if host_machine.system() == 'darwin'
  if cc.has_link_argument('-Wl,-dead_strip')
    add_project_link_arguments('-Wl,-dead_strip', language : ['c', 'fortran'])
  endif
endif

# Optional quadmath (GCC Fortran)
quadmath_dep = fc.find_library('quadmath', required : false)

# ---- Python & NumPy / F2PY
py = import('python').find_installation(pure : false)
py_dep = py.dependency()

np_inc_dir = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

f2py_inc_dir = run_command(
  py, ['-c', 'import numpy; print(numpy.f2py.get_include())'],
  check : true
).stdout().strip()

# supply absolute include paths via compile args so they work for C and Fortran
numpy_dep = declare_dependency(compile_args : ['-I' + np_inc_dir])
f2py_dep  = declare_dependency(compile_args : ['-I' + f2py_inc_dir])

# fortranobject.c from f2py
fortranobject_c = f2py_inc_dir / 'fortranobject.c'

# ---- HDF5 dependencies
hdf5_c = dependency('hdf5', required : true)

# On Linux, meson resolves pkg-config deps to absolute library paths, which
# drops the -L flag from the link command.  Bare -lhdf5_fortran then fails
# because the conda env's lib/ is never on the search path.  Use find_library
# with an explicit dir (taken from the HDF5 pkg-config libdir) so both macOS
# and Linux get a resolved path, not a bare -l flag.
_hdf5_libdir = hdf5_c.get_variable(pkgconfig : 'libdir', default_value : '')
_hdf5_dirs   = _hdf5_libdir != '' ? [_hdf5_libdir] : []
hdf5f    = cc.find_library('hdf5_fortran',    dirs : _hdf5_dirs, required : true)

# The high-level Fortran library ships under two different names depending on
# the HDF5 build system: 'hdf5_hl_fortran' (CMake builds) and 'hdf5hl_fortran'
# (autotools / conda-forge builds).  Try the CMake name first, then fall back.
hdf5f_hl = cc.find_library('hdf5_hl_fortran', dirs : _hdf5_dirs, required : false)
if not hdf5f_hl.found()
  hdf5f_hl = cc.find_library('hdf5hl_fortran', dirs : _hdf5_dirs, required : true)
endif

# Optional: OpenMP
# omp_dep = dependency('openmp', required : false)
# if omp_dep.found()
#   deps += omp_dep
# endif

# ---- F2PY codegen as a tracked build step
f2py_wrapped = custom_target(
  'mapflpy_fortran_wrappers',
  input  : ['src/mapflpy_fortran.f90'],
  output : ['mapflpy_fortranmodule.c', 'mapflpy_fortran-f2pywrappers2.f90'],
  command: [
    py, '-m', 'numpy.f2py',
    '@INPUT0@',
    '-m', 'mapflpy_fortran',
    '--backend', 'meson',
    '--f2cmap', meson.project_source_root() / '.f2py_f2cmap',
  ],
  capture : false
)

# ---- Build CPython extension, install under mapflpy/fortran
py.extension_module(
  'mapflpy_fortran',
  [
    'src/mapflpy_fortran.f90',
    'src/mapfl/psi_io.f90',
    'src/mapfl/mapfl.f',
    f2py_wrapped[0],
    f2py_wrapped[1],
    fortranobject_c,
  ],
  dependencies : [
    py_dep,
    quadmath_dep,
    numpy_dep,
    f2py_dep,
    hdf5_c,
    hdf5f,
    hdf5f_hl,
  ],
  subdir  : 'mapflpy/fortran',
  install : true,
)

# ---- Generate mapflpy/_version.py
conf = configuration_data()
conf.set('VERSION', meson.project_version())
configure_file(
  input         : 'mapflpy/_version.py.in',     # contains: __version__ = "@VERSION@"
  output        : '_version.py',
  configuration : conf,
  install       : true,
  install_dir   : py.get_install_dir() / 'mapflpy',
)

# ---- Install the pure-Python package tree
install_subdir(
  'mapflpy',
  install_dir : py.get_install_dir(),
  exclude_directories : ['fortran'],
  exclude_files: [
    '_version.py.in',
    '__pycache__',
    '*.pyc',
  ]
)

# Install mapflpy/fortran/__init__.py separately, so that the install_subdir
# above can exclude the entire fortran/ directory.  Without this exclusion,
# any compiled .so left in the source tree (e.g. from a dev install) would
# be bundled into every wheel produced by "python -m build".
install_data(
  'mapflpy/fortran/__init__.py',
  install_dir : py.get_install_dir() / 'mapflpy' / 'fortran',
)

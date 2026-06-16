.. _development:

Development
===========

This guide covers everything needed to develop ``mapflpy`` from a source checkout: setting up an
environment, building the compiled Fortran extension, running the test suite, building the
documentation, and reproducing the continuous-integration (CI) pipeline locally before opening a
pull request.

.. attention::

    We highly recommend using a virtual environment to manage your Python packages and avoid conflicts with other
    projects. For the best results, we recommend using ``conda`` – *via* Miniforge (preferred), Miniconda, or Anaconda
    – to create and manage your virtual environments.

.. note::

    ``mapflpy`` ships a compiled Fortran extension. Unlike a pure-Python package, an editable
    ``pip install -e .`` is **not** sufficient for day-to-day development: any change to the Fortran
    sources (or the ``meson.build`` that compiles them) requires the shared object to be rebuilt and
    reinstalled. The :ref:`make_build.py <tools-make-build>` helper exists to make that loop painless.


Overview of the Workflow
------------------------

Most contributors only ever need two helper scripts, both in the ``tools/`` directory:

- ``tools/make_build.py`` — build (and optionally install/extract) the compiled wheel, then run tests.
- ``tools/make_docs.py`` — build the Sphinx documentation.

Everything else (Nox sessions, the GitHub Actions workflows, the smaller utility scripts) wraps
those same two operations for isolated, reproducible, or multi-version builds. A typical local loop
looks like:

.. code-block:: bash

    # 1. edit Fortran/Python sources ...
    python tools/make_build.py --clean --sdist --extract   # rebuild + reinstall the extension
    python -m pytest                                       # quick test run against the source tree
    python tools/make_docs.py --clean                      # rebuild the docs

Before pushing, reproduce the CI matrix locally with Nox (see :ref:`local-ci`) so failures surface
on your machine rather than in the pipeline.


Setting Up the Development Environment
--------------------------------------

1. Clone the repository
***********************

Over HTTPS:

.. code-block:: bash

    git clone https://github.com/predsci/mapflpy.git

or over SSH:

.. code-block:: bash

    git clone git@github.com:predsci/mapflpy.git


2. Create the virtual environment(s)
************************************

There are two distinct environments used during development, each with a different purpose:

``mapflpy-dev``
    A full-featured environment containing the compiler toolchain, runtime dependencies, and all
    testing, linting, and documentation tools. This is your **interactive** environment — the one
    you activate to edit code, run ``make_build.py``/``make_docs.py``, and execute ``pytest``
    directly. It is defined by ``environment.yml``.

``mapflpy-ci``
    A minimal environment whose only job is to run `Nox <https://nox.thea.codes/>`_. Nox in turn
    creates its *own* throwaway conda environments for each session, so this environment deliberately
    contains nothing but Python and Nox. Use it to reproduce CI locally.

.. code-block:: bash

    cd mapflpy
    conda env create -n mapflpy-dev -f environment.yml
    conda create -n mapflpy-ci python=3.13 nox

.. tip::

    Keeping the two environments separate means the isolated Nox builds cannot accidentally pick up a
    dependency that only exists in your interactive environment — exactly the kind of "works on my
    machine" bug CI is meant to catch.


3. Build the local Fortran shared object
****************************************

.. code-block:: bash

    conda activate mapflpy-dev
    python tools/make_build.py --clean --sdist --extract

Because ``mapflpy`` wraps Fortran, the compiled shared object (``.so`` on Linux, ``.dylib`` on macOS,
``.pyd`` on Windows) must be rebuilt and placed where the package can import it whenever the Fortran
code changes. The command above will:

- ``--clean`` — remove previous ``build/``, ``dist/``, ``.pytest_cache/``, and ``*.egg-info`` artifacts;
- build a wheel (and, with ``--sdist``, a source distribution) via ``python -m build``;
- ``--extract`` — unpack the freshly compiled extension from the wheel back into ``mapflpy/fortran/``
  so the in-tree package imports the up-to-date binary.

.. seealso::

    See :ref:`tools-make-build` for the full list of flags.


Running Tests Locally
---------------------

Once the extension is built and extracted into the source tree, the fastest feedback loop is to run
``pytest`` directly against the working copy:

.. code-block:: bash

    conda activate mapflpy-dev
    python -m pytest

This runs the suite (with coverage, per the ``[tool.pytest.ini_options]`` settings in
``pyproject.toml``) against the *source tree* and the extension you just extracted. It is fast and
ideal for iterating, but it tests the code as laid out in your checkout rather than as an installed,
packaged artifact.

``make_build.py`` will also run ``pytest`` automatically after a successful build if a ``tests/``
directory is present, so a single ``python tools/make_build.py`` doubles as a build-and-test command.

.. important::

    Testing against the source tree is convenient but does **not** validate packaging. A wheel can
    build and import locally yet still be broken once installed on another machine (missing bundled
    libraries, wrong platform tags, etc.). For that, use the isolated Nox builds described in
    :ref:`local-ci`, which install the *repaired wheel* into a clean environment before testing.


Building the Documentation Locally
----------------------------------

.. code-block:: bash

    conda activate mapflpy-dev
    python tools/make_docs.py --clean

This invokes ``sphinx-build`` on ``docs/source`` and writes HTML to ``docs/_build/html``. The
``--clean`` flag first removes the previous ``_build/`` output as well as the auto-generated
``source/autodoc`` and ``source/gallery`` directories, guaranteeing a from-scratch build (important
when API signatures or gallery examples change, since Sphinx otherwise caches them).

The documentation depends on `Sphinx-Gallery <https://sphinx-gallery.github.io/>`_, which *executes*
the example scripts under ``examples/`` at build time. This means the docs build will fail if the
examples fail — making it a useful (if heavyweight) integration check in its own right.

.. seealso::

    See :ref:`tools-make-docs` for the full list of flags, and :ref:`tools-make-intersphinx` for
    refreshing the cross-reference inventories used to link out to NumPy, SciPy, etc.


.. _local-ci:

Reproducing CI Locally with Nox
-------------------------------

The CI pipeline does **not** test the source tree directly. Instead it builds a wheel in an isolated
environment, *repairs* it so that third-party shared libraries (HDF5, the Fortran runtime, ...) are
bundled into the wheel, installs that repaired wheel into a fresh environment, and only then runs the
tests. This proves the package works as a standalone, redistributable artifact. `Nox
<https://nox.thea.codes/>`_ orchestrates these steps, and you can run the exact same sessions on your
own machine.

.. code-block:: bash

    conda activate mapflpy-ci
    nox -s build      # compile the wheel(s) in an isolated conda env
    nox -s repair     # bundle external libs into the wheel (delocate/auditwheel/delvewheel)
    nox -s test       # install the repaired wheel into a clean env and run pytest

To target a single interpreter version instead of the whole matrix, append the version to the
session name:

.. code-block:: bash

    conda activate mapflpy-ci
    nox -s build-3.11
    nox -s repair
    nox -s test-3.11

The *repair* step is platform-specific and is what makes the wheels portable:

============  ==================  ============================================================
Platform      Repair tool         Purpose
============  ==================  ============================================================
Linux         ``auditwheel``      Bundle external ``.so`` files; tag as ``manylinux``.
macOS         ``delocate``        Copy external ``.dylib`` files into the wheel.
Windows       ``delvewheel``      Copy external ``.dll`` files into the wheel.
============  ==================  ============================================================

Nox also exposes convenience sessions that chain the individual steps into a single entry point:

``nox -s build_matrix``
    ``sdist`` → ``build`` → ``repair`` → ``test`` across every supported Python version.

``nox -s build_docs``
    The build/repair/test chain followed by a documentation build against the *installed* wheel.

``nox -s build_qa``
    The build/repair/test chain followed by type checking (``mypy``) and linting (``ruff``).

Standalone quality sessions are available too: ``nox -s lint`` (Ruff), ``nox -s types`` (mypy),
``nox -s sdist`` (source distribution only), and ``nox -s docs`` (docs against an installed wheel).

.. note::

    Nox uses a conda-compatible backend (``conda``, ``mamba``, or ``micromamba``) for the compiled
    sessions because the Fortran toolchain comes from ``conda-forge``. The artifacts it produces are
    written under ``.nox/_artifacts/`` (``wheels/``, ``wheelhouse/``, ``sdist/``, ``docs/``).


.. _tools-reference:

Reference: the ``tools/`` Scripts
----------------------------------

The ``tools/`` directory holds the helper scripts used throughout development. In day-to-day work you
should only need the first two; the remainder are small utilities invoked by the build configuration
and CI.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Script
     - Audience
     - Purpose
   * - ``make_build.py``
     - **Everyday**
     - Build the wheel/sdist, optionally extract the compiled extension into the source tree, and run tests.
   * - ``make_docs.py``
     - **Everyday**
     - Build the Sphinx documentation with optional cleanup.
   * - ``make_intersphinx.py``
     - Occasional
     - Download/refresh the intersphinx inventory (``objects.inv``) files used for cross-project links.
   * - ``pyproject_version.py``
     - Internal
     - Print the project version read from ``pyproject.toml``.
   * - ``python_version.py``
     - Internal
     - Print the ``MAJOR.MINOR`` Python version read from ``.python-version``.
   * - ``python_versions.py``
     - Internal
     - Print the list of supported Python versions read from the trove classifiers.


.. _tools-make-build:

``make_build.py``
*****************

Builds the package wheel (and optionally an sdist), selects the wheel that best matches the current
interpreter and platform, optionally extracts the compiled extension back into the source tree, and
runs the test suite.

.. code-block:: bash

    python tools/make_build.py [options]

============================  ==================================================================
Option                        Effect
============================  ==================================================================
``-c``, ``--clean``           Remove ``build/``, ``dist/``, ``.pytest_cache/`` and ``*.egg-info`` first.
``-s``, ``--sdist``           Also build a source distribution.
``--extract``                 Unpack the compiled extension from the wheel into ``mapflpy/fortran/``.
``--root PATH``               Project root (defaults to the parent of ``tools/``).
``--package-name NAME``       Override the package name (defaults to ``[project].name``).
``--tests PATH``              Test directory to run (defaults to ``tests/`` if present).
``--pytest-args ...``         Extra arguments forwarded to ``pytest`` (use after ``--``).
============================  ==================================================================

.. warning::

    ``--extract`` deliberately copies a compiled binary into your source tree so that the in-place
    package imports the latest build. This is convenient for development but is *not* how the package
    is normally installed — do not commit the extracted artifact, and prefer the Nox builds when you
    need to validate real packaging behavior.


.. _tools-make-docs:

``make_docs.py``
****************

Builds the Sphinx documentation. By default it calls ``sphinx-build`` directly (portable across
platforms); pass ``--use-make`` to drive the ``docs/Makefile`` instead.

.. code-block:: bash

    python tools/make_docs.py [options]

============================  ==================================================================
Option                        Effect
============================  ==================================================================
``-c``, ``--clean``           Remove the previous ``_build/``, ``source/autodoc/``, and ``source/gallery/``.
``--use-make``                Use ``make -C docs html`` instead of calling ``sphinx-build`` directly.
``--sphinx-args ...``         Extra arguments forwarded to ``sphinx-build`` after ``html`` (use after ``--``).
============================  ==================================================================

.. tip::

    To treat warnings as errors locally (as a stricter check), forward the ``-W`` flag:
    ``python tools/make_docs.py -- -W --keep-going``. Documentation warnings most commonly come from
    malformed numpydoc docstrings (for example, unexpected indentation in a ``Notes`` block), so a
    warning-free local build is the easiest way to keep the docs pipeline green.


.. _tools-make-intersphinx:

``make_intersphinx.py``
***********************

Downloads the intersphinx inventory files (``objects.inv``) that let the docs cross-reference other
projects' APIs (Python, NumPy, SciPy, Matplotlib, pytest, ``psi-data-utils``). Run it when adding a
new cross-referenced dependency or refreshing stale inventories.

.. code-block:: bash

    python tools/make_intersphinx.py
    python tools/make_intersphinx.py --target docs/_intersphinx
    python tools/make_intersphinx.py --add pandas=https://pandas.pydata.org/docs/objects.inv


Continuous Integration / Delivery
---------------------------------

CI/CD runs on GitHub Actions and reuses the same Nox sessions described above, so a green local
``nox`` run is a strong predictor of a green pipeline. There are two workflows in
``.github/workflows/``:

``publish.yml`` (*Publish Build*)
    Triggered on pull requests, manual dispatch, and ``v*`` tag pushes. It builds an sdist, builds and
    repairs wheels across a platform matrix (``ubuntu-24.04``, ``ubuntu-22.04-arm``,
    ``macos-15-intel``, ``macos-15``), installs and tests each repaired wheel, and — only on a tag —
    publishes the distributions to GitHub Releases and PyPI.

``docs.yml`` (*Publish Docs*)
    Triggered on pull requests, manual dispatch, and ``v*`` tag pushes. It runs ``nox -s build_docs``
    to build the documentation against the freshly built wheel, uploads the HTML as an artifact, and
    deploys it to the documentation server over ``rsync``.

.. note::

    A new release is cut by pushing a ``v*`` tag (matching the ``[project].version`` in
    ``pyproject.toml``). That single action drives wheel/sdist publication to PyPI **and** the
    documentation deployment. Validate locally with ``nox -s build_qa`` and ``nox -s build_docs``
    before tagging.
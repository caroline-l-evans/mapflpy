.. |psi| image:: https://predsci.com/doc/psi_logo.png
   :target: https://predsci.com
   :alt: Predictive Science Inc.
   :width: 20px

.. |pypi| image:: https://img.shields.io/pypi/v/mapflpy?logo=pypi&logoColor=white
   :target: https://pypi.org/project/mapflpy
   :alt: PyPI

.. |license| image:: https://img.shields.io/pypi/l/mapflpy?logo=apache&logoColor=white
   :target: https://opensource.org/license/apache-2-0/
   :alt: License

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/mapflpy.svg?logo=python&label=python&logoColor=white
   :target: https://pypi.org/project/mapflpy
   :alt: Python Versions

.. |deps| image:: https://img.shields.io/librariesio/github/predsci/mapflpy?logo=Libraries.io&logoColor=white
   :target: https://github.com/predsci/mapflpy/blob/main/pyproject.toml
   :alt: Libraries.io

.. |publish| image:: https://github.com/predsci/mapflpy/actions/workflows/publish.yml/badge.svg?
   :target: https://github.com/predsci/mapflpy/actions/workflows/publish.yml
   :alt: Publish workflow

.. |docs| image:: https://github.com/predsci/mapflpy/actions/workflows/docs.yml/badge.svg?
   :target: https://predsci.com/doc/mapflpy
   :alt: Docs workflow

|pypi|
|license|
|pyversions|
|deps|
|publish|
|docs|

|psi| **MAPFLPY** | *field line tracing for spherical vector fields*
--------------------------------------------------------------------

**mapflpy** is a python package for tracing field lines using the
`MapFL <https://github.com/predsci/MapFL>`_ Fortran tracer developed by
Predictive Science Inc.

The goal of **mapflpy** is to provide fast and accurate tracing capabilities for
spherical vector fields inside a convenient Python interface. **mapflpy** is
designed to work natively with the staggered meshes produced by Predictive Science
Inc.'s codes for simulating the solar corona, and inner heliosphere (*e.g.*
`MAS <https://www.predsci.com/mas>`_ or `POT3D <https://github.com/predsci/POT3D>`_),
but it should be generally compatible with any global vector field that can be
described on a rectilinear grid in spherical coordinates.

To get started with **mapflpy**, visit the
`User Guide <https://predsci.com/doc/mapflpy/guide/>`_ for installation instructions,
an overview of features, and development/contribution guidelines; a gallery of
`examples <https://predsci.com/doc/mapflpy/gallery/>`_ is also available, showcasing
various use cases and functionalities of the package. Please direct any questions or
issues to the `issue tracker <https://github.com/predsci/mapflpy/issues>`_,
or `contact <https://www.predsci.com/portal/contact.php>`_ Predictive Science Inc. directly.

----

`Predictive Science Inc. <https://predsci.com>`_ |
`Repository <https://github.com/predsci/mapflpy>`_ |
`Documentation <https://predsci.com/doc/mapflpy>`_ |
`Distribution <https://pypi.org/project/mapflpy>`_

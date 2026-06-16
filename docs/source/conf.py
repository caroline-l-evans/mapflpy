from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

try:
    # First try to run sphinx_build against installed dist
    # This is primarily included for nox-based doc builds
    import mapflpy
except ImportError:
    # Fallback: add project root to sys.path
    # This is included for local dev builds without install
    sys.path.insert(0, Path(__file__).resolve().parents[2].as_posix())
    import mapflpy

try:
    from pthree import build_node_tree, node_tree_to_dict
except ImportError:
    raise ImportError(
        "The 'pthree' package is required to build the documentation. "
        "Please install it via 'pip install pthree' and try again."
    )

# ------------------------------------------------------------------------------
# Project Information
# ------------------------------------------------------------------------------
project = "mapflpy"
author = "Predictive Science Inc"
copyright = f"{datetime.now():%Y}, {author}"
version = mapflpy.__version__
release = mapflpy.__version__

# ------------------------------------------------------------------------------
# General Configuration
# ------------------------------------------------------------------------------
extensions = []

# --- HTML Theme
_logo = "https://predsci.com/doc/psi_logo.png"
html_favicon = _logo
html_logo = _logo
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "show_prev_next": False,
    "navigation_with_keys": False,
    "show_nav_level": 3,
    "navigation_depth": 5,
    "logo": {
        "text": f"{project} v{version}",
        "image_light": _logo,
        "image_dark": _logo,
    },
    'icon_links': [
        {
            'name': 'PSI Home',
            'url': 'https://www.predsci.com/',
            'icon': 'fa fa-home fa-fw',
            "type": "fontawesome",
        },
        {
            'name': 'Repository',
            'url': 'https://github.com/predsci/mapflpy',
            # 'url': 'https://bitbucket.org/predsci/mapflpy',
            # "icon": "fa-brands fa-bitbucket fa-fw",
            "icon": "fa-brands fa-github fa-fw",
            "type": "fontawesome",
        },
        {
            'name': 'Documentation',
            'url': 'https://predsci.com/doc/mapflpy',
            "icon": "fa fa-file fa-fw",
            "type": "fontawesome",
        },
        {
            'name': 'Contact',
            'url': 'https://www.predsci.com/portal/contact.php',
            'icon': 'fa fa-envelope fa-fw',
            "type": "fontawesome",
        },
    ],
}

# --- Python Syntax
add_module_names = False
python_maximum_signature_line_length = 80

# --- Templating
templates_path = ['_templates', ]

# ------------------------------------------------------------------------------
# Viewcode Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.viewcode")

viewcode_line_numbers = True

# ------------------------------------------------------------------------------
# Autosummary Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.autosummary")

root_package = project
exclude_private = False
exclude_tests = True
exclude_dunder = True
sort_members = True
exclusions = ['\\._abc_impl',
              'mapflpy.fortran',
              '_load_array_to_shared_memory',
              'scripts._inter_domain_tracing_from*',
              '_version',
              ]

node_tree = build_node_tree(root_package,
                            sort_members,
                            exclude_private,
                            exclude_tests,
                            exclude_dunder,
                            exclusions)

autosummary_context = dict(pkgtree=node_tree_to_dict(node_tree))

# ------------------------------------------------------------------------------
# Autodoc Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.autodoc")

autodoc_typehints = "none"
autodoc_member_order = 'bysource'
autodoc_default_options = {
    "show-inheritance": True,
}

# ------------------------------------------------------------------------------
# Numpydoc Configuration
# ------------------------------------------------------------------------------
extensions.append("numpydoc")

numpydoc_xref_param_type = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_xref_ignore = {"optional", "default", "of", "or"}
numpydoc_xref_aliases = {
    "NumberType": "mapflpy.globals.NumberType",
    "PathType": "mapflpy.globals.PathType",
    "ArrayType": "mapflpy.globals.ArrayType",
    "MagneticFieldArrayType": "mapflpy.globals.MagneticFieldArrayType",
    "DirectionType": "mapflpy.globals.DirectionType",
    "MagneticFieldLabelType": "mapflpy.globals.MagneticFieldLabelType",
    "ContextType": "mapflpy.globals.ContextType",
    "Traces": "mapflpy.globals.Traces",
    "Polarity": "mapflpy.globals.Polarity",
    "ndarray": "numpy.ndarray",
    "ChainMap": "collections.ChainMap",
    "Callable": "collections.abc.Callable",
    "MutableMapping": "collections.abc.MutableMapping",
    "ArrayLike": "numpy.typing.ArrayLike",
    "Sequence": "collections.abc.Sequence",
}

# ------------------------------------------------------------------------------
# Intersphinx Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.intersphinx")

DOCS = Path(__file__).resolve().parents[1]
INV = DOCS / "_intersphinx"
intersphinx_cache_limit = 30
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        # (INV / "python-objects.inv").as_posix(),
        None
    ),
    "numpy": (
        "https://numpy.org/doc/stable/",
        # (INV / "numpy-objects.inv").as_posix(),
        None
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference/",
        # (INV / "scipy-objects.inv").as_posix(),
        None
    ),
    "matplotlib": (
        "https://matplotlib.org/stable/",
        # (INV / "matplotlib-objects.inv").as_posix(),
        None
    ),
    "psi_data": (
        "https://predsci.com/doc/psi-data-utils/",
        # (INV / "psi-data-utils-objects.inv").as_posix(),
        None
    ),
    "psi_io": (
        "https://predsci.com/doc/psi-io/",
        # (INV / "psi-io-objects.inv").as_posix(),
        None
    ),
}

# ------------------------------------------------------------------------------
# Sphinx-Gallery Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx_gallery.gen_gallery")

import matplotlib
matplotlib.use("Agg")
os.environ.setdefault('SPHINX_GALLERY_BUILD', '1')

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["gallery"],
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
    "filename_pattern": r"\.py$",
    "plot_gallery": True,
    "run_stale_examples": True,
    "matplotlib_animations": True,
}

# ------------------------------------------------------------------------------
# Sphinx Copy Button Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx_copybutton")

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True


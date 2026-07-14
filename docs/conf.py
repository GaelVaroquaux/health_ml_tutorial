# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "Health ML Tutorial"
copyright = "2026, Gaël Varoquaux"
author = "Gaël Varoquaux"
release = "0.1"

# -- General configuration ---------------------------------------------------
# sphinx_gallery.gen_gallery must be listed before jupyterlite_sphinx so
# that sphinx-gallery's build-finished hook (create_jupyterlite_contents)
# runs before jupyterlite-sphinx's build-finished hook (jupyterlite_build).
# This ensures the notebooks are copied into jupyterlite_contents/ before
# JupyterLite is built. The config-inited ordering is handled by explicit
# priorities inside sphinx-gallery and does not depend on this order.
extensions = [
    "sphinx_gallery.gen_gallery",
    "jupyterlite_sphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
docstitle = "Health ML tutorial"

html_theme_options = {
    "github_url": "https://github.com/GaelVaroquaux/health_ml_tutorial",
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "secondary_sidebar_items": {
        "**": [
            "page-toc",
            # Sphinx-Gallery sidebar components (download and launch buttons)
            # https://sphinx-gallery.github.io/stable/advanced.html#using-sphinx-gallery-sidebar-components
            "sg_download_links",
            "sg_launcher_links",
        ],
        "auto_examples/index": [],
    },
}

# -- Sphinx-Gallery configuration --------------------------------------------
# The first notebook cell is prepended to every converted notebook so that
# the required packages are installed when running inside JupyterLite (Pyodide).
sphinx_gallery_conf = {
    "examples_dirs": "../examples",   # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    # Enable the JupyterLite launch button on every gallery page.
    # sphinx-gallery will automatically copy the generated notebooks into
    # jupyterlite_contents/auto_examples/ and register that directory with
    # jupyterlite-sphinx (via post_configure_jupyterlite_sphinx).
    "jupyterlite": {
        "notebook_modification_function": None,
    },
    # Binder configuration: adds a "launch on Binder" button to every example.
    "binder": {
        "org": "GaelVaroquaux",
        "repo": "health_ml_tutorial",
        "binderhub_url": "https://mybinder.org",
        "branch": "main",
        "dependencies": "../binder/requirements.txt",
        "use_jupyter_lab": True,
    },
    # This cell is inserted at the top of every notebook produced by
    # sphinx-gallery. It installs required packages in the Pyodide environment.
    "first_notebook_cell": (
        "%pip install numpy matplotlib 'scikit-learn<1.6' pandas hazardous"
    ),
}

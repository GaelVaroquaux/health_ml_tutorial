.. Health ML Tutorial documentation master file.

====================
Health ML Tutorial
====================

Materials for a course in Health ML — practical Python examples with
interactive Jupyter notebooks that run entirely in your browser via
`JupyterLite <https://jupyterlite.readthedocs.io>`_.

.. toctree::
   :maxdepth: 1
   :caption: Gallery

   auto_examples/index

Getting started
===============

Each example page has a **Launch** button at the top that opens the
corresponding notebook in JupyterLite — no installation required. You
can edit, run, and experiment with the code directly in your browser.

To run the examples locally, install the dependencies::

    pip install -r requirements.txt

then build the documentation::

    cd docs && make html

and open ``docs/_build/html/index.html`` in your browser.

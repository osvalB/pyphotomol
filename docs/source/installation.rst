Installation
============

Requirements
------------

PyPhotoMol requires Python 3.12 or later. The core package depends on:

* numpy
* pandas
* scipy
* matplotlib
* plotly
* h5py
* kaleido

Install from PyPI
-----------------

Install PyPhotoMol directly from PyPI using pip:

.. code-block:: bash

   pip install pyphotomol

If you need to install the runtime dependencies manually, use:

.. code-block:: bash

   pip install numpy pandas scipy matplotlib plotly h5py kaleido

Verify Installation
-------------------

Verify that the package can be imported:

.. code-block:: python

   import pyphotomol
   print(pyphotomol.__version__)

Install from Source
-------------------

Clone the repository and install the development environment with ``uv``:

.. code-block:: bash

   git clone https://github.com/osvalB/pyphotomol.git
   cd pyphotomol
   uv sync --extra dev

Run Tests
---------

Verify the development installation by running the test suite:

.. code-block:: bash

   uv run pytest

Build Documentation
-------------------

Create the local documentation build with:

.. code-block:: bash

   uv run build_docs.py

The generated HTML documentation is written to ``docs/_build/html/``.

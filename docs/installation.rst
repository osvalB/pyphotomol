Installation
============

Requirements
------------

PyPhotoMol requires Python 3.7 or later and the following packages:

* numpy
* pandas
* scipy
* matplotlib
* plotly
* h5py
* kaleido

Install from pip
-------------------

You can install PyPhotoMol directly from PyPI using pip:

.. code-block:: bash

   pip install numpy pandas scipy matplotlib plotly h5py kaleido
   pip install pyphotomol

Verify Installation
-------------------

By importing the package:

.. code-block:: python

    import pyphotomol
    print(pyphotomol.__version__)


Install from Source - Development
-----------------------------   
Clone the repository and install in development mode (requires `uv`):

.. code-block:: bash

    git clone https://github.com/osvalB/pyphotomol.git
    cd pyphotomol
    uv sync --extra dev

Verify Installation
------------------- 

By running the tests:

.. code-block:: bash

    uv run pytest

By creating the documentation:

.. code-block:: bash

    uv run build_docs.py
"""
PyPhotoMol: A Python package for mass photometry data analysis.

PyPhotoMol provides a comprehensive suite of tools for analyzing mass photometry
data, including data import, histogram analysis, peak detection, Gaussian fitting,
and comprehensive operation logging.

Main Classes
------------
PyPhotoMol : Main class for single-dataset analysis
MPAnalyzer : Class for batch processing multiple datasets

Key Features
------------
- Import from HDF5 and CSV files
- Histogram creation 
- Peak detection 
- Multi-Gaussian fitting
- Mass-contrast calibration

Examples
--------
Basic single-file analysis:

>>> from pyphotomol import PyPhotoMol
>>> model = PyPhotoMol()
>>> model.import_file('data.h5')
>>> model.create_histogram(use_masses=True, window=[0, 1000], bin_width=20)
>>> model.guess_peaks(min_height=5)
>>> model.fit_histogram(
...     peaks_guess=model.peaks_guess,
...     mean_tolerance=200,
...     std_tolerance=300
... )
>>> model.create_fit_table()

Batch processing:

>>> from pyphotomol import MPAnalyzer
>>> batch = MPAnalyzer()
>>> batch.import_files(['file1.h5', 'file2.h5', 'file3.h5'])
>>> batch.apply_to_all('count_binding_events')
>>> batch.apply_to_all('create_histogram', use_masses=True, window=[0, 1000], bin_width=20)
"""

# Import main classes
from .main import PyPhotoMol, MPAnalyzer

# Import specific utility functions for convenience
from .utils.helpers import (
    contrasts_to_mass,
    create_histogram,
    guess_peaks,
    fit_histogram,
    create_fit_table,
    calibrate
)
from .utils.data_import import import_file_h5, import_csv
from .utils.plotting import (
    config_fig,
    plot_histograms_and_fits,
    plot_histogram,
    plot_calibration,
    PlotConfig,
    AxisConfig,
    LayoutConfig,
    LegendConfig
)

# Define what gets imported with "from pyphotomol import *"
__all__ = [
    'PyPhotoMol', 
    'MPAnalyzer',
    # Key helper functions
    'contrasts_to_mass',
    'create_histogram',
    'guess_peaks', 
    'fit_histogram',
    'create_fit_table',
    'calibrate',
    # Data import functions
    'import_file_h5',
    'import_csv',
    # Main plotting functions
    'plot_histograms_and_fits',
    'plot_histogram',
    'config_fig',
    'plot_calibration',
    'PlotConfig',
    'AxisConfig',
    'LayoutConfig',
    'LegendConfig'
]

# Package metadata
__version__ = '0.1.0'
__author__ = 'osvalB'
__email__ = 'oburastero@gmail.com' 
__description__ = 'A Python package for mass photometry data analysis'
Overview
========

Welcome to PyPhotoMol, a Python package for analyzing mass photometry data.

About PyPhotoMol
----------------

PyPhotoMol is the Python package adapted from the original GUI tool *PhotoMol*,
which was developed by the Sample Preparation and Characterisation core Facility (SPC), at EMBL-Hamburg,
available for free use at the eSPC platform and published as a paper:

Niebling, S., Veith, K., Vollmer, B., Lizarrondo, J., Burastero, O., Schiller, J., ... & García-Alai, M. (2022). Biophysical Screening Pipeline for Cryo-EM Grid Preparation of Membrane Proteins. Frontiers in Molecular Biosciences, 535.


* eSPC platform: https://spc.embl-hamburg.de/
* Original PhotoMol paper: https://doi.org/10.3389/fmolb.2022.882288

PyPhotoMol provides a comprehensive suite of tools for analyzing mass photometry
data, including data import, multi-gaussian fitting and plotting functions.

PyPhotoMol GitHub repository: https://github.com/SPC-Facility-EMBL-Hamburg/pyphotomol

Quick Start Guide
=================

This guide will help you get started with PyPhotoMol for analyzing mass photometry data.

Basic Workflow
--------------

The typical workflow for mass photometry analysis involves:

1. **Import data** from HDF5 or CSV files
2. **Create histograms** from the imported data
3. **Detect peaks** in the histograms
4. **Fit a multi-gaussian model** to the peaks
5. **Analyze results** and create summary tables

Simple Example
--------------

Here's a complete example showing the basic workflow:

.. code-block:: python

    from pyphotomol import PyPhotoMol
    
    # Create a new analysis instance
    model = PyPhotoMol()
    
    # Import data from an HDF5 file
    model.import_file('data.h5')
    
    # Count binding and unbinding events
    model.count_binding_events()
    
    # Create a histogram using mass data
    model.create_histogram(use_masses=True, window=[0, 1000], bin_width=10)
    
    # Automatically detect peaks in the histogram
    model.guess_peaks(min_height=10, min_distance=4, prominence=4)
    
    # Fit Gaussian models to the detected peaks
    model.fit_histogram(
        peaks_guess=model.peaks_guess,
        mean_tolerance=200,
        std_tolerance=200
    )
    
    # Create a summary table of the fitting results
    model.create_fit_table()
    
    # Print the results
    print(model.fit_table)
    
    # View the analysis logbook
    model.print_logbook_summary()

Batch Processing
----------------

For analyzing multiple files, use the MPAnalyzer class:

.. code-block:: python

    from pyphotomol import MPAnalyzer
    
    # Create a batch processing instance
    batch = MPAnalyzer()
    
    # Import multiple files
    files = ['file1.h5', 'file2.h5', 'file3.h5']
    batch.import_files(files)
    
    # Apply the same analysis to all files
    batch.apply_to_all('count_binding_events')
    batch.apply_to_all('create_histogram', use_masses=True, window=[0, 1000], bin_width=10)
    batch.apply_to_all('guess_peaks')
    # Extrac peak guesses from the first model
    peaks_guess = batch.models[0].peaks_guess
    batch.apply_to_all('fit_histogram', peaks_guess=peaks_guess, mean_tolerance=200, std_tolerance=300)
    
    # Create fit tables for all models
    batch.apply_to_all('create_fit_table')

    # Get results from all models
    fit_tables = batch.get_properties('fit_table')

Next Steps
----------

* `Example notebooks on GitHub <https://github.com/osvalB/pyPhotoMol/tree/main/example_notebooks>`_

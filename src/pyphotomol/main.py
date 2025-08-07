# Main class to handle the mass photometry data 

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import datetime
import json
from functools import wraps

from .utils.helpers import (
    contrasts_to_mass,
    create_histogram,
    count_binding_events_from_contrasts,
    count_binding_events_from_masses,
    guess_peaks,
    fit_histogram,
    create_fit_table,
    calibrate
)
from .utils.data_import import import_file_h5, import_csv
from .utils.palette import COLOR_PALETTE_9, COLOR_PALETTE_12, COLOR_PALETTE_40, HISTOGRAM_PALETTE

def log_method(func):
    """
    Decorator to automatically handle errors and logging.
    
    This decorator will:
    1. Handle exceptions by logging them to the logbook
    2. Re-raise exceptions after logging
    Note: The actual success logging is handled by each method individually
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        method_name = func.__name__
        
        try:
            # Call the original method (which handles its own success logging)
            result = func(self, *args, **kwargs)
            return result
            
        except Exception as e:
            # Log the error with all available parameters
            parameters = {}
            if args:
                parameters['args'] = args
            if kwargs:
                parameters.update(kwargs)
                
            self._log_error(method_name, parameters=parameters, error_message=str(e))
            # Re-raise the exception
            raise
    
    return wrapper

def log_batch_method(func):
    """
    Decorator to automatically handle errors and logging for MPAnalyzer methods.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        method_name = func.__name__
        
        try:
            # Call the original method (which handles its own success logging)
            result = func(self, *args, **kwargs)
            return result
            
        except Exception as e:
            # Log the batch error with all available parameters
            parameters = {}
            if args:
                parameters['args'] = args
            if kwargs:
                parameters.update(kwargs)
                
            self._log_batch_error(method_name, parameters=parameters, error_message=str(e))
            # Re-raise the exception
            raise
    
    return wrapper

class PyPhotoMol:
    """
    Main class for analyzing mass photometry data.
    
    The PyPhotoMol class provides a comprehensive suite of tools for importing,
    analyzing, and visualizing mass photometry data. It supports data import from
    HDF5 and CSV files, histogram creation and analysis, peak detection, Gaussian
    fitting, and mass-contrast calibration.
    
    All operations are automatically logged to a comprehensive logbook that tracks
    parameters, results, and any errors encountered during analysis.
    
    Attributes
    ----------
    contrasts : np.ndarray or None
        Array of contrast values from imported data
    masses : np.ndarray or None  
        Array of mass values (in kDa) from imported data or converted from contrasts
    histogram_centers : np.ndarray or None
        Bin centers for created histograms
    hist_counts : np.ndarray or None
        Count values for histogram bins
    hist_nbins : int or None
        Number of bins in the histogram
    hist_window : list or None
        [min, max] window used for histogram creation
    bin_width : float or None
        Width of histogram bins
    hist_data_type : str or None
        Type of data used for histogram ('masses' or 'contrasts')
    peaks_guess : np.ndarray or None
        Positions of detected peaks in the histogram
    nbinding : int or None
        Number of binding events detected
    nunbinding : int or None
        Number of unbinding events detected
    fitted_params : np.ndarray or None
        Parameters from Multi-Gaussian fitting
    fitted_data : np.ndarray or None
        Fitted curve data points
    fitted_params_errors : np.ndarray or None
        Error estimates for fitted parameters
    masses_fitted : np.ndarray or None
        Mass values corresponding to fitted peaks
    baseline : float
        Baseline value used for fitting operations (default: 0)
    fit_table : pd.DataFrame or None
        Summary table of fitting results
    calibration_dic : dict or None
        Dictionary containing mass-contrast calibration parameters
    logbook : list
        List of all operations performed, with timestamps and parameters
        
    Examples
    --------
    Basic workflow for mass photometry analysis:
    
    >>> model = PyPhotoMol()
    >>> model.import_file('data.h5')
    >>> model.count_binding_events()
    >>> model.create_histogram(use_masses=True, window=[0, 1000], bin_width=10)
    >>> model.guess_peaks()
    >>> model.fit_histogram(peaks_guess=model.peaks_guess, mean_tolerance=200, std_tolerance=300)
    >>> model.print_logbook_summary()
    
    """

    def __init__(self):
        """
        Initialize a new PyPhotoMol instance.
        
        Creates an empty instance with all data properties set to None
        and initializes an empty logbook for operation tracking.
        """

        self.file = None  # Track the file being analyzed

        # Data properties
        self.contrasts = None
        self.masses = None
        
        # Histogram properties
        self.histogram_centers = None
        self.hist_counts = None
        self.hist_nbins = None
        self.hist_window = None
        self.bin_width = None
        self.hist_data_type = None  # Track whether histogram uses 'masses' or 'contrasts'
        
        # Peak detection properties
        self.peaks_guess = None
        
        # Event counting properties
        self.nbinding = None
        self.nunbinding = None
        
        # Fitting properties
        self.fitted_params = None
        self.fitted_data = None
        self.fitted_params_errors = None
        self.masses_fitted = None
        self.baseline = 0  # Default baseline for fitting
        
        # Results properties
        self.fit_table = None
        
        # Calibration properties
        self.calibration_dic = None
        
        # Logbook properties
        self.logbook = []
        
    def _log_operation(self, method_name, parameters=None, result_summary=None, notes=None):
        """
        Internal method to log successful operations with timestamp and details.
        
        Parameters
        ----------
        method_name : str
            Name of the method that was executed
        parameters : dict, optional
            Parameters passed to the method
        result_summary : dict, optional
            Summary of results produced by the method
        notes : str, optional
            Additional notes about the operation
        """
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'method': method_name,
            'parameters': parameters or {},
            'result_summary': result_summary or {},
            'notes': notes,
            'success': True
        }
        
        self.logbook.append(log_entry)
    
    def _log_error(self, method_name, parameters=None, error_message=None):
        """
        Internal method to log errors that occurred during operations.
        
        Parameters
        ----------
        method_name : str
            Name of the method where the error occurred
        parameters : dict, optional
            Parameters that were passed to the method
        error_message : str, optional
            Description of the error that occurred
        """
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'method': method_name,
            'parameters': parameters or {},
            'error': error_message,
            'success': False
        }
        
        self.logbook.append(log_entry)
    
    def get_logbook(self, as_dataframe=True, save_to_file=None):
        """
        Retrieve the logbook of all operations performed on this instance.
        
        The logbook contains a complete history of all method calls, including
        parameters used, results obtained, timestamps, and any errors encountered.
        This provides full traceability of the analysis workflow.
        
        Parameters
        ----------
        as_dataframe : bool, default True
            If True, return logbook as a pandas DataFrame for easy analysis.
            If False, return as a list of dictionaries.
        save_to_file : str, optional
            If provided, save the logbook to this file path as JSON format.
            
        Returns
        -------
        pandas.DataFrame or list
            Logbook entries containing operation history. DataFrame columns include:
            - timestamp: ISO format timestamp of when operation was performed
            - method: Name of the method that was called
            - parameters: Dictionary of parameters passed to the method
            - result_summary: Summary of results produced (for successful operations)
            - notes: Additional notes about the operation
            - success: Boolean indicating if operation completed successfully
            - error: Error message (only present for failed operations)
            
        Examples
        --------
        Get logbook as DataFrame for analysis:
        
        >>> model = PyPhotoMol()
        >>> model.import_file('data.h5')
        >>> logbook_df = model.get_logbook()
        >>> print(logbook_df[['timestamp', 'method', 'success']])
        
        Save logbook to file:
        
        >>> model.get_logbook(save_to_file='analysis_log.json')
        
        Get raw logbook data:
        
        >>> raw_logbook = model.get_logbook(as_dataframe=False)
        """
        if save_to_file:
            with open(save_to_file, 'w') as f:
                json.dump(self.logbook, f, indent=2)
        
        if as_dataframe:
            if not self.logbook:
                return pd.DataFrame(columns=['timestamp', 'method', 'parameters', 'result_summary', 'notes', 'success'])
            return pd.DataFrame(self.logbook)
        
        return self.logbook
    
    def clear_logbook(self):
        """Clear all logbook entries."""
        self.logbook = []
        self._log_operation('clear_logbook', notes='Logbook cleared')
    
    def print_logbook_summary(self):
        """Print a summary of the logbook."""
        if not self.logbook:
            print("No operations logged yet.")
            return
        
        print(f"\n=== PyPhotoMol Logbook Summary ===")
        print(f"Total operations: {len(self.logbook)}")
        
        successful_ops = [entry for entry in self.logbook if entry.get('success', True)]
        failed_ops = [entry for entry in self.logbook if not entry.get('success', True)]
        
        print(f"Successful operations: {len(successful_ops)}")
        print(f"Failed operations: {len(failed_ops)}")
        
        if self.logbook:
            print(f"First operation: {self.logbook[0]['timestamp']} - {self.logbook[0]['method']}")
            print(f"Last operation: {self.logbook[-1]['timestamp']} - {self.logbook[-1]['method']}")
        
        # Show method frequency
        methods = [entry['method'] for entry in self.logbook]
        method_counts = pd.Series(methods).value_counts()
        print(f"\nMost used methods:")
        for method, count in method_counts.head(5).items():
            print(f"  {method}: {count} times")
        
        if failed_ops:
            print(f"\nFailed operations:")
            for entry in failed_ops[-3:]:  # Show last 3 failures
                print(f"  {entry['timestamp']} - {entry['method']}: {entry.get('error', 'Unknown error')}")

    @log_method
    def import_file(self, file_path):
        """
        Import mass photometry data from HDF5 or CSV files.
        
        This method loads contrast and mass data from supported file formats.
        NaN values are automatically removed from the imported data. The import
        operation is automatically logged with file information and data statistics.
        
        Parameters
        ----------
        file_path : str
            Path to the data file. Supported formats are:
            - '.h5' : HDF5 files with standard mass photometry structure
            - '.csv' : CSV files with contrast and mass columns
            
        Raises
        ------
        ValueError
            If the file format is not supported (not .h5 or .csv)
        FileNotFoundError
            If the specified file does not exist
        KeyError
            If required data columns are missing from the file
            
        Notes
        -----
        After import, the following attributes are populated:
        - self.contrasts : Array of contrast values with NaN removed
        - self.masses : Array of mass values with NaN removed (if available)
        
        The logbook will record:
        - File path and type
        - Number of data points imported
        - Range of contrast and mass values
        
        Examples
        --------
        Import HDF5 data:
        
        >>> model = PyPhotoMol()
        >>> model.import_file('experiment_data.h5')
        >>> print(f"Imported {len(model.contrasts)} contrast measurements")
        
        Import CSV data:
        
        >>> model.import_file('processed_data.csv')
        >>> print(f"Mass range: {model.masses.min():.1f} - {model.masses.max():.1f} kDa")
        """
        if file_path.endswith('.h5'):
            contrasts, masses = import_file_h5(file_path)
        elif file_path.endswith('.csv'):
            contrasts, masses = import_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Use .h5 or .csv files.")

        # Extract the basename from the file path
        self.file = file_path.split('/')[-1] if '/' in file_path else file_path

        # Remove NaN values from contrasts and masses
        if contrasts is not None:
            contrasts = contrasts[~np.isnan(contrasts)] 
            
        if masses is not None:
            masses = masses[~np.isnan(masses)]

        self.contrasts = contrasts
        self.masses = masses

        # Log the operation with detailed results
        self._log_operation(
            'import_file',
            parameters={'file_path': file_path, 'file_type': file_path.split('.')[-1]},
            result_summary={
                'n_contrasts': len(contrasts) if contrasts is not None else 0,
                'n_masses': len(masses) if masses is not None else 0,
                'contrast_range': [float(np.min(contrasts)), float(np.max(contrasts))] if contrasts is not None else None,
                'mass_range': [float(np.min(masses)), float(np.max(masses))] if masses is not None else None
            }
        )

        return None

    @log_method
    def count_binding_events(self):
        """Count binding and unbinding events from the mass data."""

        # Check if we have masses available
        if self.masses is not None:
            n_binding, n_unbinding = count_binding_events_from_masses(self.masses)
            self.nbinding   = n_binding
            self.nunbinding = n_unbinding
        
        elif self.masses is None and self.contrasts is not None:

            n_binding, n_unbinding = count_binding_events_from_contrasts(self.contrasts)
            self.nbinding   = n_binding
            self.nunbinding = n_unbinding

        else:

            raise ValueError("No mass or contrast data available to count binding events.")

        # Log the operation with detailed results
        self._log_operation(
            'count_binding_events',
            result_summary={
                'n_binding': int(n_binding),
                'n_unbinding': int(n_unbinding),
                'binding_ratio': float(n_binding / (n_binding + n_unbinding)) if (n_binding + n_unbinding) > 0 else 0
            }
        )

        return None
    
    @log_method
    def create_histogram(self, use_masses=True, window=[0,2000], bin_width=10):
        """
        Create a histogram from imported contrast or mass data.
        
        This method generates a histogram from the imported data, which is essential
        for subsequent peak detection and fitting operations. The histogram parameters
        can be customized for different types of analysis.
        
        Parameters
        ----------
        use_masses : bool, default True
            If True, create histogram from mass data (requires masses to be available).
            If False, create histogram from contrast data.
        window : list of two floats, default [0, 2000]
            Range for the histogram as [min, max]. Units depend on data type:
            - For masses: typically [0, 2000] kDa
            - For contrasts: typically [-1, 0] (e.g., [-0.8, -0.2])
        bin_width : float, default 10
            Width of histogram bins. Units depend on data type:
            - For masses: typically 10-50 kDa
            - For contrasts: typically 0.0004-0.001
            
        Raises
        ------
        AttributeError
            If no data has been imported yet
        ValueError
            If use_masses=True but no mass data is available
            
        Notes
        -----
        After histogram creation, the following attributes are populated:
        - self.histogram_centers : Bin center positions
        - self.hist_counts : Count values for each bin
        - self.hist_nbins : Number of bins created
        - self.hist_window : Window used for histogram
        - self.bin_width : Bin width used
        - self.hist_data_type : Data type used ('masses' or 'contrasts')
        
        Examples
        --------
        Create mass histogram for protein analysis:
        
        >>> model.create_histogram(use_masses=True, window=[0, 1000], bin_width=20)
        
        Create contrast histogram for calibration:
        
        >>> model.create_histogram(use_masses=False, window=[-0.8, -0.2], bin_width=0.0004)
        
        High-resolution histogram for detailed analysis:
        
        >>> model.create_histogram(use_masses=True, window=[50, 200], bin_width=5)
        """

        # Count binding events if not done yet

        if use_masses and self.masses is not None:
            histogram_centers, hist_counts, hist_nbins = create_histogram(self.masses, window=window, bin_width=bin_width)
            data_type = 'masses'
        else:
            histogram_centers, hist_counts, hist_nbins = create_histogram(self.contrasts, window=window, bin_width=bin_width)
            data_type = 'contrasts'

        self.histogram_centers = histogram_centers
        self.hist_counts = hist_counts
        self.hist_nbins = hist_nbins
        self.hist_window = window
        self.bin_width = bin_width
        self.hist_data_type = data_type  # Store the data type used for histogram

        # Log the operation with detailed results
        self._log_operation(
            'create_histogram',
            parameters={
                'use_masses': use_masses,
                'window': window,
                'bin_width': bin_width
            },
            result_summary={
                'n_bins': int(hist_nbins),
                'total_counts': int(np.sum(hist_counts)),
                'max_count': int(np.max(hist_counts)),
                'data_type': data_type
            }
        )

        return None

    @log_method
    def guess_peaks(self, min_height=10, min_distance=4, prominence=4):
        """Guess peaks in the histogram data.

        The different arguments will be adjusted according to the region of the histogram.
        For example, the given distance will be used for mass data between 0 and 650 kDa,
        between 650 and 1500 kDa, the distance will be multiplied by a factor of 3, 
        and for data above 1500 kDa, the distance will be multiplied by a factor of 8.
        See the `guess_peaks` function in `utils.helpers` for more details.

        Example of min_height, min_distance and prominence for contrasts:
            min_height=10, min_distance=4, prominence=4

        Parameters
        ----------
        min_height : int, default 10
            Minimum height of the peaks.
        min_distance : int, default 4
            Minimum distance between peaks.
        prominence : int, default 4
            Minimum prominence of the peaks.
            
        """
        if self.hist_counts is None:
            raise ValueError("Histogram data not found. Please create a histogram first.")

        peaks = guess_peaks(self.hist_counts, self.histogram_centers, min_height, min_distance, prominence)
        self.peaks_guess = peaks

        # Log the operation with detailed results
        self._log_operation(
            'guess_peaks',
            parameters={
                'min_height': min_height,
                'min_distance': min_distance,
                'prominence': prominence
            },
            result_summary={
                'n_peaks_found': len(peaks),
                'peak_positions': [float(pos) for pos in peaks] if len(peaks) > 0 else []
            }
        )

        return None
    
    @log_method
    def contrasts_to_masses(self,slope=1.0, intercept=0.0):
        """
        Convert contrasts to masses using a linear transformation.
        We assume a calibratio was done using
        f(mass) = slope * contrast + intercept

        Parameters
        ----------
        slope : float, default 1.0
            Slope of the linear transformation.
        intercept : float, default 0.0
            Intercept of the linear transformation.
        """
        if self.contrasts is None:
            raise ValueError("Contrasts data not found. Please import contrasts data first.")

        self.masses = contrasts_to_mass(self.contrasts, slope=slope, intercept=intercept)

        # Log the operation with detailed results
        self._log_operation(
            'contrasts_to_masses',
            parameters={'slope': slope, 'intercept': intercept},
            result_summary={
                'n_masses_converted': len(self.masses),
                'mass_range': [float(np.min(self.masses)), float(np.max(self.masses))]
            }
        )

        return None
    
    @log_method
    def fit_histogram(
            self, 
            peaks_guess,
            mean_tolerance,
            std_tolerance,
            threshold=None,
            baseline=0.0):
        
        """
        Fit the histogram data to the guessed peaks.
        We use a multi-Gaussian fit to the histogram data.
        
        The data type (masses or contrasts) is automatically detected from the histogram
        that was previously created using create_histogram().

        Parameters
        ----------
        peaks_guess : list
            List of guessed peaks.
        mean_tolerance : float
            Tolerance for the mean of the Gaussian fit.
        std_tolerance : float
            Tolerance for the standard deviation of the Gaussian fit.
        threshold : float, optional
            For masses: minimum value that can be observed (in kDa units). Default is 40.
            For contrasts: maximum value that can be observed (should be negative). Default is -0.0024.
            If None, defaults are applied based on detected data type.
        baseline : float, default 0.0
            Baseline value to be subtracted from the fit.

        Examples
        --------
        Fit histogram after creating it:
        
        >>> model.create_histogram(use_masses=True, window=[0, 2000], bin_width=10)
        >>> model.guess_peaks()
        >>> model.fit_histogram(model.peaks_guess, mean_tolerance=100, std_tolerance=200)
        """
        if self.hist_counts is None:
            raise ValueError("Histogram data not found. Please create a histogram first.")
        
        # Automatically detect data type from histogram
        use_masses = (self.hist_data_type == 'masses')
        
        # Set default threshold based on data type if not provided
        if threshold is None:
            if use_masses:
                threshold = 40  # Default for masses in kDa
            else:
                threshold = -0.0024  # Default for contrasts

        if use_masses:
            popt, fit, fit_errors = fit_histogram(
                self.hist_counts, 
                self.histogram_centers, 
                peaks_guess, 
                masses=True,
                mean_tolerance=mean_tolerance,
                std_tolerance=std_tolerance,
                threshold=threshold,
                baseline=baseline)
        else:
            popt, fit, fit_errors = fit_histogram(
                self.hist_counts, 
                self.histogram_centers, 
                peaks_guess, 
                masses=False,
                mean_tolerance=mean_tolerance,
                std_tolerance=std_tolerance,
                threshold=threshold,
                baseline=baseline)
            
        self.fitted_params = popt
        self.fitted_data = fit
        self.fitted_params_errors = fit_errors
        self.baseline = baseline  # Store the baseline used for fitting

        self.masses_fitted = use_masses

        # Log the operation with detailed results
        self._log_operation(
            'fit_histogram',
            parameters={
                'peaks_guess': list(peaks_guess),
                'mean_tolerance': mean_tolerance,
                'std_tolerance': std_tolerance,
                'threshold': threshold,
                'baseline': baseline,
                'data_type_detected': self.hist_data_type
            },
            result_summary={
                'n_peaks_fitted': len(peaks_guess),
                'data_type': self.hist_data_type,
                'fit_quality': 'Parameters fitted successfully'
            }
        )

        # Automatic creation of fit table after fitting
        self.create_fit_table()

        return None

    @log_method
    def create_fit_table(self):
        """
        Create a fit table from the fitted parameters
        """
        if self.fitted_params is None:
            raise ValueError("Fitted parameters not found. Please fit the histogram first.")
        
        if self.histogram_centers is None:
            raise ValueError("Histogram centers not found. Please create a histogram first.")
        
        if self.nbinding is None:
            self.count_binding_events()
        
        fit_table = create_fit_table(
            self.fitted_params, 
            self.fitted_params_errors, 
            self.fitted_data, 
            self.nbinding, 
            self.nunbinding, 
            self.histogram_centers,
            masses=self.masses_fitted,
            include_errors=True)

        self.fit_table = fit_table

        # Log the operation with detailed results
        self._log_operation(
            'create_fit_table',
            result_summary={
                'n_peaks': len(fit_table),
                'table_columns': list(fit_table.columns),
                'data_type': 'masses' if self.masses_fitted else 'contrasts'
            }
        )

        return None
    
    @log_method
    def calibrate(self,calibration_standards):
        """
        Obtain a calibration of the type f(mass) = slope * contrast + intercept
        
        Parameters
        ----------
        calibration_standards : list
            List with the known masses
        """
        if self.fit_table is None:
            raise ValueError("Fit table not found. Please create a fit table first.")
        
        if self.contrasts is None:
            raise ValueError("Contrasts data not found. Please import contrasts data first.")

        self.calibration_dic = calibrate(
            calibration_standards, 
            self.fit_table)

        # Log the operation with detailed results
        self._log_operation(
            'calibrate',
            parameters={'calibration_standards': calibration_standards},
            result_summary={
                'calibration_results': self.calibration_dic,
                'n_standards': len(calibration_standards)
            }
        )

        return None

class MPAnalyzer():
    """
    A class to handle multiple PyPhotoMol instances.
    This is useful for batch processing of multiple files.
    """
    
    def __init__(self):
        self.models = {}
        # Logbook for batch operations
        self.batch_logbook = []

    def _log_batch_operation(self, method_name, parameters=None, result_summary=None, notes=None):
        """Internal method to log batch operations."""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'method': method_name,
            'parameters': parameters or {},
            'result_summary': result_summary or {},
            'notes': notes,
            'success': True,
            'operation_type': 'batch'
        }
        
        self.batch_logbook.append(log_entry)

    def _log_batch_error(self, method_name, parameters=None, error_message=None):
        """Internal method to log batch operation errors."""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'method': method_name,
            'parameters': parameters or {},
            'error': error_message,
            'success': False,
            'operation_type': 'batch'
        }
        
        self.batch_logbook.append(log_entry)

    def get_batch_logbook(self, as_dataframe=True, save_to_file=None):
        """
        Retrieve the batch logbook of all operations performed.
        
        Parameters
        ----------
        as_dataframe : bool, default True
            If True, return as pandas DataFrame, else as list of dicts
        save_to_file : str, optional
            Optional file path to save logbook as JSON
            
        Returns
        -------
        pandas.DataFrame or list
            Batch logbook entries
        """
        if save_to_file:
            with open(save_to_file, 'w') as f:
                json.dump(self.batch_logbook, f, indent=2)
        
        if as_dataframe:
            if not self.batch_logbook:
                return pd.DataFrame(columns=['timestamp', 'method', 'parameters', 'result_summary', 'notes', 'success', 'operation_type'])
            return pd.DataFrame(self.batch_logbook)
        
        return self.batch_logbook

    def get_all_logbooks(self, save_to_file=None):
        """
        Get combined logbooks from all models plus batch operations.
        
        Parameters
        ----------
        save_to_file : str, optional
            Optional file path to save combined logbook as JSON
            
        Returns
        -------
        dict
            Combined logbooks with batch and individual model logs
        """
        combined_logbook = {
            'batch_operations': self.batch_logbook,
            'individual_models': {}
        }
        
        for name, model in self.models.items():
            combined_logbook['individual_models'][name] = model.logbook
        
        if save_to_file:
            with open(save_to_file, 'w') as f:
                json.dump(combined_logbook, f, indent=2)
        
        return combined_logbook

    @log_batch_method
    def import_files(self, files,names=None):
        """
        Load multiple files into PyPhotoMol instances.
        
        Parameters
        ----------
        files : list
            List of file paths to load.
        names : list, optional
            List of names for the PyPhotoMol instances.
        """
        # Convert string to list if a single string is provided
        if isinstance(files, str):
            files = [files]

        if names is None:
            # Extract only the file name without extension
            names = [file.split('/')[-1].split('.')[0] for file in files]

        if isinstance(names, str):
            names = [names]

        for file, name in zip(files, names):
            model = PyPhotoMol()
            model.import_file(file)
            self.models[name] = model

        # Log the batch operation with detailed results
        self._log_batch_operation(
            'import_files',
            parameters={
                'files': files,
                'names': names,
                'n_files': len(files)
            },
            result_summary={
                'models_created': len(files),
                'model_names': names
            }
        )

        return None

    @log_batch_method 
    def apply_to_all(self, method_name, *args, names=None, **kwargs):
        """
        Apply a method to all or selected PyPhotoMol instances.
        
        Parameters
        ----------
        method_name : str
            Name of the method to apply to instances
        *args : tuple
            Positional arguments to pass to the method
        names : list or str, optional
            Names of specific models to apply method to.
            If None (default), applies to all models.
        **kwargs : dict
            Keyword arguments to pass to the method
            
        Examples
        --------
        Count binding events for all models:
        
        >>> pms.apply_to_all('count_binding_events')
        
        Create histograms for specific models only:
        
        >>> pms.apply_to_all('create_histogram', names=['model1', 'model2'], 
        ...                  use_masses=True, window=[0, 2000], bin_width=10)
        
        Guess peaks for a single model:
        
        >>> pms.apply_to_all('guess_peaks', names='model1', 
        ...                  min_height=10, min_distance=4, prominence=4)
        """
        # Determine which models to process
        if names is None:
            # Apply to all models
            target_models = self.models.items()
            target_names = list(self.models.keys())
        else:
            # Apply to selected models
            if isinstance(names, str):
                names = [names]
            
            target_models = []
            for name in names:
                if name in self.models:
                    target_models.append((name, self.models[name]))
                else:
                    raise ValueError(f"Model '{name}' not found in models")
            target_names = names
        
        # Apply method to target models
        successful_models = []
        failed_models = []
        
        for name, model in target_models:
            if hasattr(model, method_name):
                method = getattr(model, method_name)
                if callable(method):
                    try:
                        method(*args, **kwargs)
                        successful_models.append(name)
                    except Exception as e:
                        failed_models.append((name, str(e)))
                        raise RuntimeError(f"Error applying {method_name} to model {name}: {str(e)}")
                else:
                    raise AttributeError(f"{method_name} is not callable for model {name}")
            else:
                raise AttributeError(f"Model {name} does not have method {method_name}")
        
        # Log the batch operation with detailed results
        self._log_batch_operation(
            'apply_to_all',
            parameters={
                'method_name': method_name,
                'target_models': target_names,
                'args': args,
                'kwargs': kwargs
            },
            result_summary={
                'successful_models': successful_models,
                'failed_models': [name for name, _ in failed_models],
                'total_models_processed': len(target_names)
            }
        )
        
        return None

    def get_properties(self, variable):
        """
        Get properties from all PyPhotoMol instances.
        
        Parameters
        ----------
        variable : str
            The property to get from each instance.
            
        Returns
        -------
        list
            List of the specified property from each instance.
            
        Examples
        --------
        Get masses from all models:
        
        >>> masses_list = pms.get_properties('masses')
        
        Get fit tables from all models:
        
        >>> fit_tables = pms.get_properties('fit_table')
        """

        return [getattr(self.models[name], variable) for name in self.models.keys()]

    def create_plotting_config(self, repeat_colors=True):
        """
        Create configuration dataframes for plotting multiple PhotoMol models.
        
        Parameters
        ----------
        repeat_colors : bool, default True
            If True, repeat the same color scheme for each model's peaks.
            If False, use sequential colors across all peaks from all models.
        
        Returns
        -------
            tuple: (legends_df, colors_hist_df, legends_all, color_palette_all, sels_all, colors_hist)
                - legends_df: DataFrame with legends, colors, and selection flags for Gaussian traces
                - colors_hist_df: DataFrame with histogram colors for each model
        """
        
        legends_all = []
        gaussian_sum_idx = []
        fit_tables = []
        cnt1 = 0
        cnt2 = 0
        color_palette_all = []
        
        # Process each model to generate legends
        for i, (name, model) in enumerate(self.models.items()):
            # Skip if no fit data available
            if not hasattr(model, 'fit_table') or model.fit_table is None:
                continue
            
            # Get number of peaks from fit table
            num_peaks = len(model.fit_table)
            
            if num_peaks == 0:
                continue
                
            # Create legends for this model's peaks
            legends = [f"Peak #{cnt2 + j + 1}" for j in range(num_peaks)]
            
            # Handle colors based on repeat_colors setting
            if repeat_colors:
                # Reset color index for each model
                model_color_palette = COLOR_PALETTE_9  # Use standard palette for each model
                model_colors = model_color_palette[:num_peaks]
                
                # Add Gaussian sum color if multiple peaks
                if len(legends) > 1:
                    legends = [f"Gaussian sum ({i + 1})"] + legends
                    gaussian_sum_idx.append(1 + len(legends_all))
                    # Use light gray color for Gaussian sum to distinguish from individual peaks
                    sum_color = '#808080'  # Light gray for Gaussian sum
                    model_colors = [sum_color] + model_colors
                
                color_palette_all.extend(model_colors)
            else:
                # Sequential colors across all models
                # Add Gaussian sum if multiple peaks
                if len(legends) > 1:
                    legends = [f"Gaussian sum ({i + 1})"] + legends
                    gaussian_sum_idx.append(1 + len(legends_all))
            
            cnt2 += num_peaks
            legends_all.extend(legends)
            
            # Store fit table info
            table = model.fit_table.copy()
            if len(self.models) > 1:
                table['File'] = name
            
            if len(table) > 0:
                cnt1 += 1
                fit_tables.append(table)
        
        # Create legends dataframe
        numberOfLegends = len(legends_all)
        
        if numberOfLegends == 0:
            legends_df = pd.DataFrame({
                'legends': ['No fits'],
                'color': ['#000000'],
                'select': [False]
            })
            color_palette_all = ['#000000']
            sels_all = [False]
        else:
            # Handle sequential coloring if not repeat_colors
            if not repeat_colors:
                # Select appropriate color palette based on total number of legends
                if numberOfLegends < 10:
                    color_palette = COLOR_PALETTE_9
                elif numberOfLegends < 13:
                    color_palette = COLOR_PALETTE_12
                elif numberOfLegends < 40:
                    color_palette = COLOR_PALETTE_40
                else:
                    # Fallback to only one colour - lightblue
                    color_palette = ['#ADD8E6'] * numberOfLegends
                
                # Ensure we have enough colors
                color_palette_all = (color_palette * ((numberOfLegends // len(color_palette)) + 1))[:numberOfLegends]
            
            sels_all = [True] * numberOfLegends
            
            legends_df = pd.DataFrame({
                'legends': legends_all,
                'color': color_palette_all,
                'select': sels_all
            })
        
        # Create histogram colors dataframe
        model_names = list(self.models.keys())
        
        if repeat_colors:
            # Use the same histogram color for all models
            histogram_color = '#1f77b4'  # Standard blue
            colors_hist = [histogram_color] * len(model_names)
        else:
            # Use different colors for each model's histogram
            histogram_palette = HISTOGRAM_PALETTE
            colors_hist = (histogram_palette * ((len(model_names) // len(histogram_palette)) + 1))[:len(model_names)]
        
        colors_hist_df = pd.DataFrame({
            'legends': model_names,
            'color': colors_hist
        })
        
        return legends_df, colors_hist_df

    @log_batch_method
    def master_calibration(self,calibration_standards):
        """
        Perform master calibration using known masses.
        It uses information from the fit table of each model to create a master calibration.
        Parameters
        ----------
        calibration_standards : list
            List of known masses for calibration.
        """

        # Extract fit tables from all models
        fit_tables = self.get_properties('fit_table')

        # Check if fit tables are not None and contain data
        fit_tables = [ft for ft in fit_tables if ft is not None and not ft.empty]
        
        if not fit_tables:
            raise ValueError("No fit tables found in any model. Please create fit tables first.")
        
        combined_fit_table = pd.concat(fit_tables, ignore_index=True)

        self.calibration_dic = calibrate(
            calibration_standards, 
            combined_fit_table)
        
        # Log the operation with detailed results
        self._log_batch_operation(
            'master_calibration',
            parameters={'calibration_standards': calibration_standards},
            result_summary={
                'calibration_results': self.calibration_dic,
                'n_standards': len(calibration_standards)
            }
        )

        return None

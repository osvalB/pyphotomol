import pytest
import os
import tempfile
import json
import io
import sys
import numpy as np
import pandas as pd
from pyphotomol import PyPhotoMol, MPAnalyzer

def test_pyphotomol_class_errors():

    """Test the PyPhotoMol class for error handling"""

    pm_instance = PyPhotoMol()

    # Try importing a file with an invalid extension
    with pytest.raises(ValueError):
        pm_instance.import_file("non_existent_file.sarasa")

    # Try guessing peaks without creating a histogram
    with pytest.raises(ValueError):
        pm_instance.guess_peaks()

    # Try contras to masses without contrasts
    with pytest.raises(ValueError):
        pm_instance.contrasts_to_masses()

    # Try fitting histogram 
    with pytest.raises(ValueError):
        pm_instance.fit_histogram(peaks_guess=[1, 2, 3],mean_tolerance=10,std_tolerance=20)

    # Try fitting histogram - with masses loaded
    with pytest.raises(ValueError):
        pm_instance.masses = np.load("test_files/masses_kDa.npy")
        pm_instance.fit_histogram(peaks_guess=[1, 2, 3],mean_tolerance=10,std_tolerance=20)

    # Try fitting histogram - should still fail because no histogram was created
    with pytest.raises(ValueError):
        pm_instance.fit_histogram(peaks_guess=[1, 2, 3],mean_tolerance=10,std_tolerance=20)

    # Try creating a fit table without fitting the histogram centers
    with pytest.raises(ValueError):
        pm_instance.create_fit_table()
    
    # Try creating a fit table without fitted_params
    with pytest.raises(ValueError):
        pm_instance.fitted_params = []
        pm_instance.create_fit_table()

    # Try calibrating without a fit table
    with pytest.raises(ValueError):
        pm_instance.calibrate([480, 166, 66])

    # Try calibration with a fit table but no contrasts data
    with pytest.raises(ValueError):
        pm_instance.fit_table = 'Dummy fit table'
        pm_instance.calibrate([480, 166, 66])

def test_log_method_decorator_error_handling():
    """Test that the log_method decorator properly logs errors and re-raises exceptions (line 955)"""
    
    pm_instance = PyPhotoMol()
    
    # Clear any existing logbook entries
    pm_instance.clear_logbook()
    
    # Verify logbook starts empty (except for the clear_logbook entry)
    initial_logbook_size = len(pm_instance.logbook)
    
    # Try an operation that will fail and trigger the decorator's error handling
    with pytest.raises(ValueError) as exc_info:
        pm_instance.guess_peaks()  # This will fail because no histogram exists
    
    # Verify the exception was re-raised (line 955: raise)
    assert "Histogram data not found" in str(exc_info.value)
    
    # Verify that the error was logged to the logbook
    logbook = pm_instance.get_logbook(as_dataframe=False)
    assert len(logbook) == initial_logbook_size + 1  # Should have one more entry
    
    # Find the error log entry
    error_entry = logbook[-1]  # Last entry should be the error
    
    # Verify the error log contains the expected information
    assert error_entry['method'] == 'guess_peaks'
    assert error_entry['success'] == False
    assert 'Histogram data not found' in error_entry['error']
    assert 'timestamp' in error_entry
    
    # Test with another method that takes parameters to verify parameter logging
    with pytest.raises(ValueError) as exc_info:
        pm_instance.contrasts_to_masses(slope=2.0, intercept=1.0)  # Will fail: no contrasts data
    
    # Verify the exception was re-raised with correct message
    assert "Contrasts data not found" in str(exc_info.value)
    
    # Verify parameter logging in the error entry
    logbook = pm_instance.get_logbook(as_dataframe=False)
    error_entry = logbook[-1]  # Last entry should be the new error
    
    assert error_entry['method'] == 'contrasts_to_masses'
    assert error_entry['success'] == False
    assert error_entry['parameters']['slope'] == 2.0
    assert error_entry['parameters']['intercept'] == 1.0

def test_pyphotomol_class():

    """Test the PyPhotoMol class"""

    pm_instance = PyPhotoMol()

    pm_instance.import_file("test_files/demo.h5")

    # Check that contrasts and masses are not None
    assert pm_instance.contrasts is not None
    assert pm_instance.masses is not None

    # import the csv now
    pm_instance.import_file("test_files/demo.csv")
    # Check that contrasts and masses are not None
    assert pm_instance.contrasts is not None
    assert pm_instance.masses is not None  

    pm_instance.count_binding_events()

    # Check that we have binding and unbinding events
    assert pm_instance.nbinding > 0
    assert pm_instance.nunbinding > 0

    # Create a histogram
    pm_instance.create_histogram(use_masses=True, window=[0, 2000], bin_width=10)
    # Check that histogram centers, counts, and number of bins are not None
    assert pm_instance.histogram_centers is not None
    assert pm_instance.hist_counts is not None
    assert pm_instance.hist_nbins > 0

    # Guess the peaks in the histogram
    pm_instance.guess_peaks(min_height=10, min_distance=4, prominence=4)
    # Check that peaks are found
    assert pm_instance.peaks_guess is not None

    # Fit the histogram to the peaks
    pm_instance.fit_histogram(peaks_guess=pm_instance.peaks_guess,mean_tolerance=100, std_tolerance=200)
    # Check that fitted_params, fitted_data, and fitted_params_errors are not None
    assert pm_instance.fitted_params is not None
    assert pm_instance.fitted_data is not None
    assert pm_instance.fitted_params_errors is not None

    # Create a fit table
    pm_instance.create_fit_table()
    # Check that fit_table is not None
    assert pm_instance.fit_table is not None
    # Check that the fit table has three rows (one per Gaussian)
    assert len(pm_instance.fit_table) == 3


def test_pyphotomol_class_contrasts():

    """Test the PyPhotoMol class"""
    pm_instance = PyPhotoMol()
    pm_instance.import_file("test_files/demo.h5")

    # Check that contrasts and masses are not None
    assert pm_instance.contrasts is not None
    assert pm_instance.masses is not None

    # Convert masses to None
    pm_instance.masses = None

    # Create a histogram
    pm_instance.create_histogram(use_masses=False, window=[-1, 0], bin_width=0.0004)
    # Check that histogram centers, counts, and number of bins are not None
    assert pm_instance.histogram_centers is not None
    assert pm_instance.hist_counts is not None
    assert pm_instance.hist_nbins > 0

    # Guess the peaks in the histogram
    pm_instance.guess_peaks(min_height=10, min_distance=4, prominence=4)
    # Check that peaks are found
    assert pm_instance.peaks_guess is not None
    # Check that we have three peaks
    assert len(pm_instance.peaks_guess) == 3

    # Fit the histogram to the peaks
    pm_instance.fit_histogram(peaks_guess=pm_instance.peaks_guess, mean_tolerance=0.05, std_tolerance=0.1)
    # Check that fitted_params, fitted_data, and fitted_params_errors are not None
    assert pm_instance.fitted_params is not None
    assert pm_instance.fitted_data is not None
    assert pm_instance.fitted_params_errors is not None

    # Create a fit table
    pm_instance.create_fit_table()
    # Check that fit_table is not None
    assert pm_instance.fit_table is not None
    # Check that the fit table has three rows (one per Gaussian)
    assert len(pm_instance.fit_table) == 3

    # Obtain the calibration
    pm_instance.calibrate([480,166,66])
    # Check that the calibration is not None
    assert pm_instance.calibration_dic is not None
    # Check the keys of the calibration dictionary 
    expected_keys = ['standards','exp_points','fit_params','fit_r2']
    assert all(key in pm_instance.calibration_dic for key in expected_keys)

    # Extract the slope and intercept from the calibration
    slope = pm_instance.calibration_dic['fit_params'][0]
    intercept = pm_instance.calibration_dic['fit_params'][1]

    # Check that the slope and intercept are not None
    assert slope is not None
    assert intercept is not None

    # Convert contrasts to masses
    pm_instance.contrasts_to_masses(slope=slope, intercept=intercept)
    # Check that masses are not None
    assert pm_instance.masses is not None


def test_mpanalyzer_class():

    pms = MPAnalyzer()
    files = ["test_files/demo.h5", "test_files/demo.csv"]
    names = ["demo_h5", "demo_csv"]
    pms.import_files(files,names)
    assert len(pms.models) == 2

    # No name import 
    pms = MPAnalyzer()
    files = ["test_files/demo.h5"]
    pms.import_files(files)
    assert len(pms.models) == 1

    # Import with string instead of list
    pms = MPAnalyzer()
    file = "test_files/demo.h5"
    pms.import_files(file, names="demo_h5")
    assert len(pms.models) == 1

    # Get properties
    pms.get_properties('masses')

def test_logbook_functionality():
    """Test the logbook functionality of PyPhotoMol"""
    
    pm_instance = PyPhotoMol()
    
    # Test initial logbook state
    assert pm_instance.logbook == []
    
    # Test importing file and logbook entry
    pm_instance.import_file("test_files/demo.h5")
    assert len(pm_instance.logbook) == 1
    assert pm_instance.logbook[0]['method'] == 'import_file'
    assert pm_instance.logbook[0]['success'] == True
    
    # Test count_binding_events logging
    pm_instance.count_binding_events()
    assert len(pm_instance.logbook) == 2
    assert pm_instance.logbook[1]['method'] == 'count_binding_events'
    
    # Test create_histogram logging
    pm_instance.create_histogram(use_masses=True, window=[0, 2000], bin_width=10)
    assert len(pm_instance.logbook) == 3
    assert pm_instance.logbook[2]['method'] == 'create_histogram'
    
    # Test guess_peaks logging
    pm_instance.guess_peaks(min_height=10, min_distance=4, prominence=4)
    assert len(pm_instance.logbook) == 4
    assert pm_instance.logbook[3]['method'] == 'guess_peaks'
    
    # Test fit_histogram logging (now automatically calls create_fit_table)
    pm_instance.fit_histogram(peaks_guess=pm_instance.peaks_guess, mean_tolerance=100, std_tolerance=200)
    assert len(pm_instance.logbook) == 6  # fit_histogram + create_fit_table
    assert pm_instance.logbook[4]['method'] == 'fit_histogram'
    assert pm_instance.logbook[5]['method'] == 'create_fit_table'
    assert pm_instance.logbook[5]['method'] == 'create_fit_table'
    
    # Test get_logbook as DataFrame
    logbook_df = pm_instance.get_logbook(as_dataframe=True)
    assert len(logbook_df) == 6
    assert 'timestamp' in logbook_df.columns
    assert 'method' in logbook_df.columns
    assert 'success' in logbook_df.columns
    
    # Test get_logbook as list
    logbook_list = pm_instance.get_logbook(as_dataframe=False)
    assert isinstance(logbook_list, list)
    assert len(logbook_list) == 6
    
    # Test clear_logbook
    pm_instance.clear_logbook()
    assert len(pm_instance.logbook) == 1  # Clear operation itself is logged
    assert pm_instance.logbook[0]['method'] == 'clear_logbook'
    
    # Test print_logbook_summary with empty logbook
    pm_instance.logbook = []
    pm_instance.print_logbook_summary()  # Should not raise error
    
    # Test get_logbook with save_to_file
    pm_instance.import_file("test_files/demo.h5")
    test_file = 'test_logbook.json'
    pm_instance.get_logbook(save_to_file=test_file)
    
    # Clean up test file
    if os.path.exists(test_file):
        os.remove(test_file)


def test_error_logging():
    """Test error logging functionality"""
    
    pm_instance = PyPhotoMol()
    
    # Test error in import_file
    try:
        pm_instance.import_file("non_existent_file.h5")
    except:
        pass
    
    # Check that error was logged
    assert len(pm_instance.logbook) == 1
    assert pm_instance.logbook[0]['method'] == 'import_file'
    assert pm_instance.logbook[0]['success'] == False
    assert 'error' in pm_instance.logbook[0]
    
    # Test error in count_binding_events (no contrasts)
    try:
        pm_instance.count_binding_events()
    except:
        pass
    
    # Test error in create_histogram (no data)
    try:
        pm_instance.create_histogram()
    except:
        pass
    
    # Test error in guess_peaks (no histogram)
    try:
        pm_instance.guess_peaks()
    except:
        pass
    
    # Test error in fit_histogram (no histogram)
    try:
        pm_instance.fit_histogram(peaks_guess=[100, 200], mean_tolerance=10, std_tolerance=20)
    except:
        pass


def test_contrasts_to_masses_logging():
    """Test contrasts_to_masses with logging"""
    
    pm_instance = PyPhotoMol()
    pm_instance.import_file("test_files/demo.h5")
    
    # Test contrasts_to_masses logging
    pm_instance.contrasts_to_masses(slope=1.5, intercept=50)
    
    # Find the contrasts_to_masses log entry
    contrasts_log = None
    for entry in pm_instance.logbook:
        if entry['method'] == 'contrasts_to_masses':
            contrasts_log = entry
            break
    
    assert contrasts_log is not None
    assert contrasts_log['success'] == True
    assert contrasts_log['parameters']['slope'] == 1.5
    assert contrasts_log['parameters']['intercept'] == 50


def test_calibrate_logging():
    """Test calibrate method with logging"""
    
    pm_instance = PyPhotoMol()
    pm_instance.import_file("test_files/demo.h5")
    pm_instance.masses = None  # Force to use contrasts
    pm_instance.create_histogram(use_masses=False, window=[-1, 0], bin_width=0.0004)
    pm_instance.guess_peaks(min_height=10, min_distance=4, prominence=4)
    pm_instance.fit_histogram(peaks_guess=pm_instance.peaks_guess, mean_tolerance=0.05, std_tolerance=0.1)
    pm_instance.create_fit_table()
    
    # Test calibrate logging
    calibration_standards = [480, 166, 66]
    pm_instance.calibrate(calibration_standards)
    
    # Find the calibrate log entry
    calibrate_log = None
    for entry in pm_instance.logbook:
        if entry['method'] == 'calibrate':
            calibrate_log = entry
            break
    
    assert calibrate_log is not None
    assert calibrate_log['success'] == True
    assert calibrate_log['parameters']['calibration_standards'] == calibration_standards


def test_mpanalyzer_batch_operations():
    """Test MPAnalyzer batch operations and logging"""
    
    pms = MPAnalyzer()
    
    # Test import_files logging
    files = ["test_files/demo.h5"]
    names = ["demo"]
    pms.import_files(files, names)
    
    assert len(pms.batch_logbook) == 1
    assert pms.batch_logbook[0]['method'] == 'import_files'
    assert pms.batch_logbook[0]['success'] == True
    
    # Test apply_to_all logging
    pms.apply_to_all('count_binding_events')
    assert len(pms.batch_logbook) == 2
    assert pms.batch_logbook[1]['method'] == 'apply_to_all'
    
    # Test apply_to_all with specific names
    pms.apply_to_all('create_histogram', names='demo', use_masses=True, window=[0, 2000], bin_width=10)
    assert len(pms.batch_logbook) == 3
    
    # Test apply_to_all with multiple names as list
    pms.apply_to_all('guess_peaks', names=['demo'], min_height=10, min_distance=4, prominence=4)
    assert len(pms.batch_logbook) == 4


def test_mpanalyzer_logbook_methods():
    """Test MPAnalyzer logbook retrieval methods"""
    
    pms = MPAnalyzer()
    
    # Test get_batch_logbook with empty logbook
    batch_df = pms.get_batch_logbook(as_dataframe=True)
    assert len(batch_df) == 0
    
    batch_list = pms.get_batch_logbook(as_dataframe=False)
    assert isinstance(batch_list, list)
    assert len(batch_list) == 0
    
    # Add some operations
    pms.import_files("test_files/demo.h5", "demo")
    pms.apply_to_all('count_binding_events')
    
    # Test get_batch_logbook with data
    batch_df = pms.get_batch_logbook(as_dataframe=True)
    assert len(batch_df) == 2
    
    # Test get_all_logbooks
    all_logs = pms.get_all_logbooks()
    assert 'batch_operations' in all_logs
    assert 'individual_models' in all_logs
    assert len(all_logs['batch_operations']) == 2
    assert 'demo' in all_logs['individual_models']
    
    # Test get_all_logbooks with save_to_file (lines 840-841)
    test_all_logs_file = 'test_all_logbooks.json'
    all_logs_saved = pms.get_all_logbooks(save_to_file=test_all_logs_file)
    
    # Check that the file was created and content is correct
    assert os.path.exists(test_all_logs_file)
    assert 'batch_operations' in all_logs_saved
    assert 'individual_models' in all_logs_saved
    
    # Test save functionality
    test_file = 'test_batch_logbook.json'
    pms.get_batch_logbook(save_to_file=test_file)
    
    # Check that the file was created
    assert os.path.exists(test_file)

    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists(test_all_logs_file):
        os.remove(test_all_logs_file)
    


def test_mpanalyzer_error_handling():
    """Test MPAnalyzer error handling and logging"""
    
    pms = MPAnalyzer()
    
    # Test apply_to_all with non-existent model
    try:
        pms.apply_to_all('count_binding_events', names='non_existent')
    except ValueError:
        pass
    
    # Check error was logged
    assert len(pms.batch_logbook) == 1
    assert pms.batch_logbook[0]['success'] == False
    
    # Test apply_to_all with non-existent method
    pms.import_files("test_files/demo.h5", "demo")
    try:
        pms.apply_to_all('non_existent_method')
    except AttributeError:
        pass
    
    # Test import_files with error
    try:
        pms.import_files("non_existent_file.h5", "test")
    except:
        pass
    
    # Check that errors are logged
    error_logs = [log for log in pms.batch_logbook if not log['success']]
    assert len(error_logs) >= 1


def test_mpanalyzer_get_properties():
    """Test MPAnalyzer get_properties method"""
    
    pms = MPAnalyzer()
    pms.import_files(["test_files/demo.h5", "test_files/demo.csv"], ["demo1", "demo2"])
    
    # Test get_properties
    masses_list = pms.get_properties('masses')
    assert len(masses_list) == 2
    assert all(mass_array is not None for mass_array in masses_list)
    
    contrasts_list = pms.get_properties('contrasts')
    assert len(contrasts_list) == 2
    assert all(contrast_array is not None for contrast_array in contrasts_list)


def test_create_plotting_config():
    """Test create_plotting_config method"""
    
    pms = MPAnalyzer()
    pms.import_files(["test_files/demo.h5"], ["demo"])
    
    # Prepare the model for plotting config
    pms.apply_to_all('count_binding_events')
    pms.apply_to_all('create_histogram', use_masses=True, window=[0, 2000], bin_width=10)
    pms.apply_to_all('guess_peaks', min_height=10, min_distance=4, prominence=4)
    pms.apply_to_all('fit_histogram', peaks_guess=pms.models['demo'].peaks_guess, mean_tolerance=100, std_tolerance=200)
    pms.apply_to_all('create_fit_table')
    
    # Test create_plotting_config with repeat_colors=True
    legends_df, colors_hist_df = pms.create_plotting_config(repeat_colors=True)
    
    assert isinstance(legends_df, pd.DataFrame)
    assert isinstance(colors_hist_df, pd.DataFrame)
    assert 'legends' in legends_df.columns
    assert 'color' in legends_df.columns
    assert 'select' in legends_df.columns
    assert len(legends_df) > 0
    
    # Test create_plotting_config with repeat_colors=False
    legends_df2, colors_hist_df2 = pms.create_plotting_config(repeat_colors=False)
    
    assert isinstance(legends_df2, pd.DataFrame)
    assert isinstance(colors_hist_df2, pd.DataFrame)
    assert len(legends_df2) > 0
    
    # Test with multiple models
    pms.import_files(["test_files/demo.csv"], ["demo2"])
    pms.apply_to_all('count_binding_events', names='demo2')
    pms.apply_to_all('create_histogram', names='demo2', use_masses=True, window=[0, 2000], bin_width=10)
    pms.apply_to_all('guess_peaks', names='demo2', min_height=10, min_distance=4, prominence=4)
    pms.apply_to_all('fit_histogram', names='demo2', peaks_guess=pms.models['demo2'].peaks_guess, mean_tolerance=100, std_tolerance=200)
    pms.apply_to_all('create_fit_table', names='demo2')
    
    # Test with multiple models
    legends_df3, colors_hist_df3 = pms.create_plotting_config(repeat_colors=True)
    assert len(legends_df3) > len(legends_df)  # Should have more entries with two models
    
    # Test edge case with no fit data
    pms_empty = MPAnalyzer()
    pms_empty.import_files("test_files/demo.h5", "empty")
    legends_df_empty, colors_hist_df_empty = pms_empty.create_plotting_config()
    
    assert len(legends_df_empty) == 1  # Should have "No fits" entry
    assert legends_df_empty.iloc[0]['legends'] == 'No fits'


def test_logbook_edge_cases():
    """Test edge cases in logbook functionality"""
    
    pm_instance = PyPhotoMol()
    
    # Test get_logbook with empty logbook returning DataFrame (line 92)
    empty_df = pm_instance.get_logbook(as_dataframe=True)
    assert isinstance(empty_df, pd.DataFrame)
    assert len(empty_df) == 0
    expected_columns = ['timestamp', 'method', 'parameters', 'result_summary', 'notes', 'success']
    assert list(empty_df.columns) == expected_columns
    
    # Test print_logbook_summary with operations and failed operations (lines 108-131)
    pm_instance.import_file("test_files/demo.h5")
    
    # Create an error to test failed operations display
    try:
        pm_instance.import_file("non_existent_file.h5")
    except:
        pass
    
    # Add more operations to test method frequency display
    pm_instance.count_binding_events()
    pm_instance.create_histogram(use_masses=True, window=[0, 2000], bin_width=10)
    pm_instance.count_binding_events()  # Repeat to test frequency
    
    # This should trigger lines 108-131 including failed operations display
    pm_instance.print_logbook_summary()
    
    # Check that we have both successful and failed operations
    successful_ops = [entry for entry in pm_instance.logbook if entry.get('success', True)]
    failed_ops = [entry for entry in pm_instance.logbook if not entry.get('success', True)]
    assert len(successful_ops) > 0
    assert len(failed_ops) > 0


def test_mpanalyzer_error_edge_cases():
    """Test MPAnalyzer error handling edge cases"""
    
    pms = MPAnalyzer()
    pms.import_files("test_files/demo.h5", "demo")
    
    # Test apply_to_all with method that's not callable (lines 955)
    # Create a non-callable attribute on the model to test this specific case
    pms.models['demo'].non_callable_attr = "this_is_just_a_string"
    
    try:
        pms.apply_to_all('non_callable_attr')  # This exists but is not callable
    except AttributeError as e:
        assert "is not callable" in str(e)
        assert "non_callable_attr" in str(e)
    
    # Test the case where target_names is not defined in error logging (lines 689-693)
    pms_error = MPAnalyzer()
    try:
        # This should fail before target_names is defined
        pms_error.apply_to_all('count_binding_events', names='nonexistent')
    except Exception:
        pass
    
    # Check that error was logged even without target_names being defined
    error_logs = [log for log in pms_error.batch_logbook if not log['success']]
    assert len(error_logs) >= 1
    
    # Test apply_to_all with method that raises an exception during execution (lines 951-553)
    # This tests the RuntimeError case when a method exists and is callable but fails
    pms_runtime_error = MPAnalyzer()
    pms_runtime_error.import_files("test_files/demo.h5", "demo")
    
    try:
        # Try to create a histogram with invalid parameters that will cause an exception
        pms_runtime_error.apply_to_all('create_histogram', use_masses=True, window=[2000, 0], bin_width=-10)
    except RuntimeError as e:
        # Check that the error message follows the expected format
        assert "Error applying create_histogram to model demo:" in str(e)
        # Verify that failed_models list was populated by checking it would be in the error message
        assert len(pms_runtime_error.models) == 1  # Ensure we have a model to test against


def test_plotting_config_edge_cases():
    """Test create_plotting_config edge cases"""
    
    # Test with very large number of peaks to trigger fallback color (lines 831-837)
    pms = MPAnalyzer()
    pms.import_files("test_files/demo.h5", "demo")
    
    # Create a model with many peaks (more than 40)
    pms.models['demo'].fitted_data = np.random.rand(100)  # Dummy fitted data
    
    # Create a large fit table to test color palette fallback
    large_fit_table = pd.DataFrame({
        'Peak': range(50),  # 50 peaks - more than COLOR_PALETTE_40
        'Mean': np.random.rand(50) * 1000,
        'Std': np.random.rand(50) * 100,
        'Area': np.random.rand(50) * 1000
    })
    pms.models['demo'].fit_table = large_fit_table
    
    # Test with repeat_colors=False to trigger the large number fallback (line 831-837)
    legends_df, colors_hist_df = pms.create_plotting_config(repeat_colors=False)
    
    # Should fall back to light blue for all when > 40 peaks
    assert len(legends_df) > 40
    # Check that fallback color (light blue) is used
    assert '#ADD8E6' in legends_df['color'].values
    
    # Test the case where fit_table is empty but exists (line 769)
    pms2 = MPAnalyzer()
    pms2.import_files("test_files/demo.h5", "demo2")
    pms2.models['demo2'].fitted_data = np.random.rand(10)
    pms2.models['demo2'].fit_table = pd.DataFrame()  # Empty fit table
    
    legends_df2, colors_hist_df2 = pms2.create_plotting_config()
    assert len(legends_df2) == 1
    assert legends_df2.iloc[0]['legends'] == 'No fits'
    
    # Test case with no fit_table attribute (line 775)
    pms3 = MPAnalyzer()
    pms3.import_files("test_files/demo.h5", "demo3")
    
    legends_df3, _ = pms3.create_plotting_config()
    assert len(legends_df3) == 1
    assert legends_df3.iloc[0]['legends'] == 'No fits'


def test_plotting_config_multiple_color_palettes():
    """Test create_plotting_config with different numbers of peaks for palette selection"""
    
    pms = MPAnalyzer()
    pms.import_files("test_files/demo.h5", "demo")
    
    # Test with exactly 9 peaks (COLOR_PALETTE_9)
    pms.models['demo'].fitted_data = np.random.rand(100)
    fit_table_9 = pd.DataFrame({
        'Peak': range(9),
        'Mean': np.random.rand(9) * 1000,
        'Std': np.random.rand(9) * 100,
        'Area': np.random.rand(9) * 1000
    })
    pms.models['demo'].fit_table = fit_table_9
    
    legends_df_9, _ = pms.create_plotting_config(repeat_colors=False)
    assert len(legends_df_9) == 10 # 9 peaks plus the gaussian sum
    
    # Test with exactly 12 peaks (COLOR_PALETTE_12)
    fit_table_12 = pd.DataFrame({
        'Peak': range(12),
        'Mean': np.random.rand(12) * 1000,
        'Std': np.random.rand(12) * 100,
        'Area': np.random.rand(12) * 1000
    })
    pms.models['demo'].fit_table = fit_table_12
    
    legends_df_12, _ = pms.create_plotting_config(repeat_colors=False)
    assert len(legends_df_12) == 13 # 12 peaks plus the gaussian sum
    
    # Test with exactly 25 peaks (COLOR_PALETTE_40)
    fit_table_25 = pd.DataFrame({
        'Peak': range(25),
        'Mean': np.random.rand(25) * 1000,
        'Std': np.random.rand(25) * 100,
        'Area': np.random.rand(25) * 1000
    })
    pms.models['demo'].fit_table = fit_table_25
    
    legends_df_25, _ = pms.create_plotting_config(repeat_colors=False)
    assert len(legends_df_25) == 26 # 25 peaks plus the gaussian sum


def test_logbook_save_to_file():
    """Test logbook save_to_file functionality to cover lines 285-286."""
    
    pm_instance = PyPhotoMol()
    pm_instance.import_file("test_files/demo.h5")
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.close()
    
    try:
        # Test save_to_file functionality (lines 285-286)
        logbook_data = pm_instance.get_logbook(save_to_file=temp_file.name)
        
        # Verify file was created and contains correct data
        assert os.path.exists(temp_file.name)
        
        with open(temp_file.name, 'r') as f:
            saved_data = json.load(f)
        
        # Verify saved data matches logbook
        assert len(saved_data) > 0
        assert saved_data[0]['method'] == 'import_file'
        
    finally:
        # Clean up
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def test_empty_logbook_cases():
    """Test empty logbook cases to cover lines 291 and 303-304."""
    
    pm_instance = PyPhotoMol()
    
    # Test get_logbook with empty logbook (line 291)
    empty_df = pm_instance.get_logbook(as_dataframe=True)
    assert isinstance(empty_df, pd.DataFrame)
    assert len(empty_df) == 0
    
    # Test print_logbook_summary with empty logbook (lines 303-304)
    # Capture stdout to verify the print statement
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    pm_instance.print_logbook_summary()
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Verify the output message
    output = captured_output.getvalue()
    assert "No operations logged yet." in output


def test_save_to_file_and_empty_logbook_coverage():
    """Test PyPhotoMols save_to_file functionality to cover remaining lines."""
    
    pms = MPAnalyzer()
    
    # Test empty logbook save
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.close()
    
    try:
        # Test get_all_logbooks with save_to_file when no models exist
        all_logbooks = pms.get_all_logbooks(save_to_file=temp_file.name)
        
        # Verify file was created
        assert os.path.exists(temp_file.name)
        
        with open(temp_file.name, 'r') as f:
            saved_data = json.load(f)
        
        # Verify structure
        assert 'batch_operations' in saved_data
        assert 'individual_models' in saved_data
        assert len(saved_data['individual_models']) == 0
        
    finally:
        # Clean up
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def test_master_calibration():
    """Test master calibration functionality"""
    
    # Test ValueError if there are no fit_tables
    mp_empty = MPAnalyzer()
    mp_empty.import_files(["test_files/demo.h5", "test_files/demo.csv"], ["demo_h5", "demo_csv"])
    mp_empty.apply_to_all('count_binding_events')
    mp_empty.apply_to_all('create_histogram', use_masses=False, window=[-1, 0], bin_width=0.0004)
    mp_empty.apply_to_all('guess_peaks', min_height=10, min_distance=4, prominence=4)
    # No fit_histogram call, so no fit_tables exist
    
    with pytest.raises(ValueError):
        mp_empty.master_calibration([480, 166, 66, 480, 166, 66])

    # Now test the successful calibration
    mp = MPAnalyzer()
    mp.import_files(["test_files/demo.h5", "test_files/demo.csv"], ["demo_h5", "demo_csv"])

    mp.apply_to_all('count_binding_events')
    mp.apply_to_all('create_histogram', use_masses=False, window=[-1, 0], bin_width=0.0004)
    mp.apply_to_all('guess_peaks', min_height=10, min_distance=4, prominence=4)
    mp.apply_to_all('fit_histogram', peaks_guess=mp.models['demo_h5'].peaks_guess, mean_tolerance=0.05, std_tolerance=0.1)
    
    # fit_histogram now automatically creates fit_tables, so no need for explicit create_fit_table call

    # Apply master calibration
    mp.master_calibration([480, 166, 66, 480, 166, 66])
    assert mp.calibration_dic is not None

    # Check that the calibration dictionary has the expected keys
    expected_keys = ['standards', 'exp_points', 'fit_params', 'fit_r2']
    assert all(key in mp.calibration_dic for key in expected_keys)
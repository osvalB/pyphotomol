import pytest
import numpy as np
import pandas as pd
from pyphotomol.utils.helpers import (
    contrasts_to_mass,
    create_histogram,
    count_binding_events_from_masses,
    count_binding_events_from_contrasts,
    guess_peaks,
    multi_gauss,
    truncated_multi_gauss_with_baseline,
    fit_histogram,
    create_fit_table,
    r_squared,
    calibrate,
    evaluate_fitting
)

def test_create_histogram():
    """Test the create_histogram function with a simple array."""
    data = np.random.normal(loc=0, scale=1, size=1000)
    hist_centers, hist_counts, hist_nbins = create_histogram(data, window=[-3, 3], bin_width=0.1)
    
    assert len(hist_centers) == len(hist_counts)
    assert isinstance(hist_centers, np.ndarray)
    assert isinstance(hist_counts, np.ndarray)
    assert hist_nbins > 0

def test_guess_peaks():
    """Test the guess_peaks function with a simple histogram.
    Use sample data present at test_files/masses_kDa.npy."""
    masses = np.load("test_files/masses_kDa.npy")
    hist_centers, hist_counts,  _ = create_histogram(masses)
    peaks = guess_peaks(hist_counts, hist_centers)

    assert isinstance(peaks, np.ndarray)
    assert len(peaks) > 0  # Should find at least one peak

def test_count_binding_events_from_masses():
    """Test the count_binding_events_from_masses function."""
    masses = np.array([66, 146, 480, 66, 146])  # Example masses
    n_binding, n_unbinding = count_binding_events_from_masses(masses)
    
    assert n_binding >= 0
    assert n_unbinding >= 0
    assert n_binding + n_unbinding == len(masses)

def test_count_binding_events_from_contrasts():
    """Test the count_binding_events_from_contrasts function."""
    contrasts = np.array([0.1, 0.2, 0.3, 0.1, 0.2])  # Example contrasts
    n_binding, n_unbinding = count_binding_events_from_contrasts(contrasts)
    
    assert n_binding >= 0
    assert n_unbinding >= 0
    assert n_binding + n_unbinding == len(contrasts)

def test_multi_gauss():
    """Test the multi_gauss function with known parameters."""
    x = np.linspace(0, 10, 100)
    # Single Gaussian: center=5, amplitude=1, std=1
    params = [5, 1, 1]
    y = multi_gauss(x, *params)
    
    assert len(y) == len(x)
    assert isinstance(y, np.ndarray)
    # Peak should be at center
    peak_idx = np.argmax(y)
    assert abs(x[peak_idx] - 5) < 0.1

def test_multi_gauss_multiple():
    """Test multi_gauss with multiple Gaussians."""
    x = np.linspace(0, 10, 100)
    # Two Gaussians
    params = [3, 1, 0.5, 7, 0.8, 0.3]
    y = multi_gauss(x, *params)
    
    assert len(y) == len(x)
    assert isinstance(y, np.ndarray)
    assert np.all(y >= 0)  # Gaussians should be non-negative

def test_truncated_multi_gauss_with_baseline():
    """Test the truncated_multi_gauss_with_baseline function."""
    x = np.linspace(0, 10, 100)
    # Two Gaussians with baseline
    params = [3, 1, 0.5, 7, 0.8, 0.3]  
    # We assume no baseline and no truncation for simplicity
    y = truncated_multi_gauss_with_baseline(x, 0,0,*params)
    
    assert len(y) == len(x)
    assert isinstance(y, np.ndarray)
    assert np.all(y >= 0)  # Should be non-negative

def test_fit_histogram():
    """Test the fit_histogram function with a simple histogram."""
    masses = np.load("test_files/masses_kDa.npy")
    hist_centers, hist_counts,  _ = create_histogram(masses)
    peaks = guess_peaks(hist_counts, hist_centers)
    
    popt, fit, fit_errors = fit_histogram(hist_counts, hist_centers, peaks)
    
    assert isinstance(popt, np.ndarray)
    assert isinstance(fit, np.ndarray)
    assert isinstance(fit_errors, np.ndarray)
    assert len(popt) > 0
    assert len(fit) > 0

    # Test raise ValueError by setting the std threshold to an unreasonable value
    with pytest.raises(ValueError):
        fit_histogram(hist_counts, hist_centers, peaks, std_tolerance=9)

    # Check ValueError if the std_tolerance is not greater than 5
    with pytest.raises(ValueError):
        fit_histogram(hist_counts, hist_centers, peaks, std_tolerance=2)

    # Fit with std_tolerance set as None
    fit_histogram(hist_counts, hist_centers, peaks, std_tolerance=None)
    assert isinstance(popt, np.ndarray)
    assert isinstance(fit, np.ndarray)

def test_fit_histogram_no_peaks():
    """Test the fit_histogram function with no peaks found."""
    masses = np.load("test_files/masses_kDa.npy")
    hist_centers, hist_counts,  _ = create_histogram(masses)
    
    fit = fit_histogram(hist_counts, hist_centers, [])

    assert fit is None, "Fit should be None when no peaks are found"

def test_fit_histogram_with_contrasts():
    """Test the fit_histogram function with contrasts."""
    contrasts = np.load("test_files/contrasts.npy")
    hist_centers, hist_counts, _ = create_histogram(contrasts, window=[-0.2, 0], bin_width=0.0004)
    peaks = guess_peaks(hist_counts, hist_centers)
    
    # Raise an error if threshold is positive
    with pytest.raises(ValueError):
        popt, fit, fit_errors = fit_histogram(hist_counts, hist_centers, peaks, masses=False, threshold=40)

    popt, fit, fit_errors = fit_histogram(hist_counts, hist_centers, peaks, masses=False,threshold=0)
    
    assert isinstance(popt, np.ndarray)
    assert isinstance(fit, np.ndarray)
    assert isinstance(fit_errors, np.ndarray)
    assert len(popt) > 0
    assert len(fit) > 0

    # Test raise ValueError by setting the std threshold to an unreasonable value
    with pytest.raises(ValueError):
        fit_histogram(hist_counts, hist_centers, peaks, masses=False,std_tolerance=0)

def test_create_fit_table():
    """Test the create_fit_table function."""
    masses = np.load("test_files/masses_kDa.npy")
    hist_centers, hist_counts,  _ = create_histogram(masses)
    peaks = guess_peaks(hist_counts, hist_centers)
    
    popt, fit, fit_errors = fit_histogram(hist_counts, hist_centers, peaks)
    n_binding, n_unbinding = count_binding_events_from_masses(masses)
    
    table = create_fit_table(popt, fit_errors, fit, n_binding, n_unbinding,hist_centers)
    assert isinstance(table, pd.DataFrame)

def test_r_squared():
    """Test the r_squared function."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 4.1, 5.0])
    
    r2 = r_squared(y_true, y_pred)
    
    assert isinstance(r2, float)
    assert r2 >= 0  # R-squared should be non-negative

def test_contrasts_to_mass_0():
    """Test the calibrate function with example calibration parameters."""
    contrasts = np.load("test_files/contrasts.npy")

    slope   = -6.10130913e-05 
    intercept =  3.70353065e-04

    expected_masses = np.load("test_files/masses_kDa.npy")

    # Filter only masses between 0 and 2000 kDa
    valid_mask = (expected_masses < 2000) & (expected_masses > 0)
    contrasts = contrasts[valid_mask]
    expected_masses = expected_masses[valid_mask]

    masses = contrasts_to_mass(contrasts, slope, intercept)
    assert len(masses) == len(contrasts)
    assert np.allclose(masses, expected_masses, atol=0.1)

def test_contrasts_to_mass():
    """Test contrast to mass conversion."""
    contrasts = np.array([0.1, 0.2, 0.3])
    masses    = contrasts_to_mass(contrasts,1,1)
    
    assert len(masses) == len(contrasts)
    assert isinstance(masses, np.ndarray)

def test_calibrate():
    """Test the calibrate function with example calibration parameters."""
    contrasts = np.load("test_files/contrasts.npy")
    hist_centers, hist_counts, _ = create_histogram(contrasts, window=[-0.2, 0], bin_width=0.0004)
    peaks = guess_peaks(hist_counts, hist_centers,10, 4, 4, False)
    
    popt, fit, fit_errors = fit_histogram(hist_counts, hist_centers, peaks,masses=False,threshold=0)
    n_binding, n_unbinding = count_binding_events_from_contrasts(contrasts)
    
    table = create_fit_table(popt, fit_errors, fit, n_binding, n_unbinding,hist_centers,masses=False)

    calib_mass = [480, 146, 66]  # Example calibration standards in kDa
    calibration_dic = calibrate(calib_mass, table)
    assert isinstance(calibration_dic, dict)
    assert 'fit_params' in calibration_dic.keys()
    assert 'fit_r2' in calibration_dic.keys()

def test_evaluate_fitting():

    fitted_params = np.array([1.0])
    low_bound     = np.array([0.5])
    high_bound    = np.array([1.5])

    is_valid = evaluate_fitting(fitted_params, low_bound, high_bound)

    assert is_valid
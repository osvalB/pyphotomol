import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pyphotomol.utils.data_import import import_file_h5, import_csv


def test_import_file_h5_basic():
    """Test that import_file_h5 returns numpy arrays when successful."""
    for file in ["test_files/demo.h5", "test_files/demo2.h5"]:
        contrasts, masses_kda = import_file_h5(file)
        assert isinstance(contrasts, np.ndarray)
        assert isinstance(masses_kda, np.ndarray)
        assert len(contrasts) == len(masses_kda)
        assert len(contrasts) > 0  # Ensure we have data
        assert not np.any(np.isnan(contrasts))  # Ensure NaN values are filtered


def test_import_file_h5_per_movie_events():
    """Test import_file_h5 with per_movie_events structure (lines 98-114)."""
    # This tests the specific code path for per_movie_events
    contrasts, masses_kda = import_file_h5("test_files/demo_per_movie_events.h5")
    
    assert isinstance(contrasts, np.ndarray)
    assert isinstance(masses_kda, np.ndarray)
    assert len(contrasts) == len(masses_kda)
    
    # Should have 8 total events: 5 from movie_001 + 3 from movie_002
    # (movie_003, movie_004, movie_005 should be skipped due to errors)
    assert len(contrasts) == 8
    
    # Verify that masses were calculated correctly using contrasts_to_mass function
    from src.pyphotomol.utils.helpers import contrasts_to_mass
    
    # Movie 1: contrasts with gradient=1200, offset=150
    expected_movie1_contrasts = np.array([-0.12, -0.18, -0.25, -0.15, -0.22])
    expected_movie1_masses = contrasts_to_mass(expected_movie1_contrasts, 1200.0, 150.0)
    
    # Movie 2: contrasts with gradient=1000, offset=100
    expected_movie2_contrasts = np.array([-0.08, -0.30, -0.14])
    expected_movie2_masses = contrasts_to_mass(expected_movie2_contrasts, 1000.0, 100.0)
    
    # Check that all expected values are present (order may vary due to concatenation)
    all_expected_contrasts = np.concatenate([expected_movie1_contrasts, expected_movie2_contrasts])
    all_expected_masses = np.concatenate([expected_movie1_masses, expected_movie2_masses])
    
    # Sort for comparison
    sort_idx_actual = np.argsort(contrasts)
    sort_idx_expected = np.argsort(all_expected_contrasts)
    
    np.testing.assert_array_almost_equal(
        contrasts[sort_idx_actual], 
        all_expected_contrasts[sort_idx_expected]
    )
    np.testing.assert_array_almost_equal(
        masses_kda[sort_idx_actual], 
        all_expected_masses[sort_idx_expected]
    )


def test_import_file_h5_calibrated_values():
    """Test import_file_h5 with calibrated_values structure (lines 91-94)."""
    contrasts, masses_kda = import_file_h5("test_files/demo_calibrated_values.h5")
    
    assert isinstance(contrasts, np.ndarray)
    assert isinstance(masses_kda, np.ndarray)
    assert len(contrasts) == len(masses_kda)
    assert len(contrasts) == 6  # Should have all 6 test values
    
    # Verify that masses were calculated from contrasts using calibration
    from src.pyphotomol.utils.helpers import contrasts_to_mass
    expected_contrasts = np.array([-0.10, -0.15, -0.20, -0.12, -0.18, -0.25])
    expected_masses = contrasts_to_mass(expected_contrasts, 1500.0, 200.0)
    
    np.testing.assert_array_almost_equal(contrasts, expected_contrasts)
    np.testing.assert_array_almost_equal(masses_kda, expected_masses)


def test_import_file_h5_with_nans():
    """Test that NaN values are properly filtered out."""
    contrasts, masses_kda = import_file_h5("test_files/demo_with_nans.h5")
    
    assert isinstance(contrasts, np.ndarray)
    assert isinstance(masses_kda, np.ndarray)
    assert len(contrasts) == 4  # Should have 4 non-NaN values
    assert not np.any(np.isnan(contrasts))  # No NaN values should remain
    
    # Check that the correct non-NaN values are preserved
    expected_contrasts = np.array([-0.10, -0.15, -0.20, -0.12])
    np.testing.assert_array_almost_equal(contrasts, expected_contrasts)


def test_import_csv_with_masses():
    """Test that import_csv returns numpy arrays when masses are present."""
    contrasts, masses_kda = import_csv("test_files/demo.csv")
    assert isinstance(contrasts, np.ndarray)
    assert isinstance(masses_kda, np.ndarray)
    assert len(contrasts) == len(masses_kda)
    assert len(contrasts) > 0
    assert not np.any(np.isnan(contrasts))  # Ensure NaN values are filtered

    # Test with another CSV file - different header names
    contrasts, masses_kda = import_csv("test_files/demo2.csv")
    assert isinstance(contrasts, np.ndarray)
    assert isinstance(masses_kda, np.ndarray)
    assert len(contrasts) == len(masses_kda)
    assert len(contrasts) > 0
    assert not np.any(np.isnan(contrasts))  # Ensure NaN values are filtered
    

def test_import_csv_contrasts_only():
    """Test import_csv with contrasts only (no masses)."""
    contrasts, masses_kda = import_csv("test_files/contrasts.csv")
    assert isinstance(contrasts, np.ndarray)
    assert masses_kda is None  # No masses_kDa in this file
    assert len(contrasts) > 0  # Ensure there are contrasts in the file
    assert not np.any(np.isnan(contrasts))


def test_import_csv_with_nan_values():
    """Test that CSV import properly filters NaN values."""
    # Create a temporary CSV with NaN values
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    
    try:
        csv_data = pd.DataFrame({
            'contrasts': [-0.1, np.nan, -0.2, -0.3, np.nan, -0.15],
            'masses_kDa': [100, np.nan, 200, 300, np.nan, 150]
        })
        csv_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        # Test the import
        contrasts, masses_kda = import_csv(temp_file.name)
        
        assert isinstance(contrasts, np.ndarray)
        assert isinstance(masses_kda, np.ndarray)
        assert len(contrasts) == 4  # Should have 4 non-NaN contrast values
        assert not np.any(np.isnan(contrasts))  # No NaN values should remain
        
        # Check that the correct non-NaN values are preserved
        expected_contrasts = np.array([-0.1, -0.2, -0.3, -0.15])
        np.testing.assert_array_equal(contrasts, expected_contrasts)
        
    finally:
        # Clean up
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def test_import_file_h5_error_cases():
    """Test error handling for HDF5 import."""
    # Test with non-existent file
    with pytest.raises((FileNotFoundError, OSError)):
        import_file_h5("non_existent_file.h5")


def test_import_csv_error_cases():
    """Test error handling for CSV import."""
    # Test with non-existent file
    with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
        import_csv("non_existent_file.csv")
    
    # Test with CSV missing contrasts column
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    
    try:
        csv_data = pd.DataFrame({
            'other_column': [1, 2, 3, 4]
        })
        csv_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        with pytest.raises(ValueError):
            import_csv(temp_file.name)
            
    finally:
        # Clean up
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def test_import_file_h5_edge_cases():
    """Test edge cases like empty arrays after processing."""
    # Test file that should result in empty arrays and cause an error
    # This reveals a bug in the original code where empty arrays cause concatenation to fail
    with pytest.raises(ValueError, match="need at least one array to concatenate"):
        import_file_h5("test_files/demo_empty_arrays.h5")

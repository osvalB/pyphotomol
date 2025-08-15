from pyphotomol.utils.plotting import (
    config_fig,
    add_histogram,
    add_labels_to_fig,
    add_values_in_legend,
    add_gaussian_traces,
    add_gaussian_simulation_to_plot,
    plot_histograms_and_fits,
    plot_histogram,
    plot_calibration,
    PlotConfig,
    AxisConfig,
    LayoutConfig,
    LegendConfig
)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pyphotomol import MPAnalyzer, PyPhotoMol

file = "test_files/demo.h5"

fitted_means = np.array([100, 150, 300])
fitted_counts_perc = np.array([10, 20, 30])
font_size = 14

fit_table = pd.DataFrame({
    'Position / kDa': fitted_means,
    'Sigma / kDa': [1, 2, 3],
    'Counts': fitted_counts_perc,
    'Counts / %': fitted_counts_perc,
    'Amplitudes': [0.1, 0.1, 0.3],
    'Position Error / %': [0.01, 0.02, 0.03],
    'Sigma Error / %': [0.01, 0.02, 0.03],
    'Amplitude Error / %': [0.01, 0.02, 0.03]
})

def test_add_histogram():
    """Test the add_histogram function with a simple histogram."""
     # Create a figure and add a mass histogram
    fig = go.Figure()
    masses = np.random.normal(loc=500, scale=100, size=1000)
    bin_info = {"start": 0, "end": 1000, "size": 10}
    hex_color = "#FF5733"   
    fig = add_histogram(fig, masses, bin_info, normalize=True, hex_color=hex_color)

    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Histogram)
    assert fig.data[0].marker.color == hex_color
    assert fig.data[0].xbins['start'] == bin_info['start']
    assert fig.data[0].xbins['end'] == bin_info['end']
    assert fig.data[0].xbins['size'] == bin_info['size']

def test_config_fig():
    """Test the config_fig function to ensure it sets the configuration correctly."""
    fig = go.Figure()
    plot_width = 800
    plot_height = 600
    fig = config_fig(fig,plot_width=plot_width, plot_height=plot_height)

    assert fig._config is not None

def test_add_labels_to_fig():
    """Test the add_labels_to_fig function with a simple figure."""
    fig = go.Figure()

    for sels in [None, [True, True, True]]:
        for contrasts in [True, False]:
            for show_counts in [True, False]:
                for show_values in [True, False]:
                    for stacked in [True, False]:                    
                        
                        fig = go.Figure()
                        fig = add_labels_to_fig(fig, fit_table, contrasts=contrasts,
                                                scaling_factor=1.0, font_size=font_size,
                                                sels=sels, show_values=show_values,
                                                show_counts=show_counts,stacked=stacked)
                        
                        assert len(fig.layout.annotations) == len(fitted_means)

def test_show_values_in_legend():
    """Test the add_masses_to_legend function to ensure it adds mass labels correctly."""
    labels = ["Peak 1", "Peak 2", "Peak 3"]
    
    for contrasts in [True, False]:

        updated_legends = add_values_in_legend(
            labels, 
            fit_table, 
            contrasts=contrasts,
            cst_factor_for_contrast=1000)

        assert len(updated_legends) == len(labels)

def test_add_gaussian_traces():
    """Test the add_gaussian_traces function to ensure it adds Gaussian traces correctly."""
    fig = go.Figure()
    # Import fitted data of three gaussians
    fitted_data = np.load("test_files/fitted_data.npy")

    # For data with sum column, legends should be: [Sum, Gaussian1, Gaussian2, Gaussian3]
    legends = ['Sum', 'Gaussian 1', 'Gaussian 2', 'Gaussian 3']
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
    
    fig = add_gaussian_traces(fig, fitted_data, fit_table,
                              legends=legends, color_palette=colors,
                              sels=[True, True, True, True])
    
    # Check if four traces were added - one per single Gaussian + gaussian sum
    assert len(fig.data) == 4

    # Test selecting only the sum trace
    fig = go.Figure()
    fig = add_gaussian_traces(fig, fitted_data, fit_table,
                              legends=legends, color_palette=colors,
                              sels=[True, False, False, False])
    # Check if only one trace was added
    assert len(fig.data) == 1
    
    # Try a reduced dataset, only x, and one gaussian (no sum column)
    reduced_fitted_data = fitted_data[:,[0,1,-1]]  # Just x, first gaussian and sum column
    
    # Remember that fitted_data always has a sum column at the end!

    # Create a reduced fit_table for just one Gaussian
    reduced_fit_table = fit_table.iloc[:1].copy()  # Only first row
    
    for contrasts in [True, False]:
        for select in [True, False]:
            
            fig2 = go.Figure()
        
            # Add the reduced data to the figure - no sum column, so just one gaussian
            fig2 = add_gaussian_traces(fig2, reduced_fitted_data, reduced_fit_table,
                                    legends=['Gaussian 1'], color_palette=colors[:1],
                                    sels=[select],contrasts=contrasts)
            
            # Check if only one trace was added
            assert len(fig2.data) == np.sum(select)  # Only one trace for the single gaussian

def test_add_gaussian_simulation_to_plot():
    """Test the add_gaussian_simulation_to_plot function."""
    fig = go.Figure()
    mean = 500
    std = 50
    amplitude = 1
    left_bound = 0
    
    fig = add_gaussian_simulation_to_plot(fig, mean, std, amplitude, left_bound)

    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.data[0].mode == 'lines'

def test_plot_histogram_PyPhotoMol_instance():

    photomol = PyPhotoMol()
    photomol.import_file(file)
    photomol.create_histogram(use_masses=True, window=[0, 1000], bin_width=10)

    fig = plot_histogram(photomol,
                         colors_hist='gray')

    assert isinstance(fig, go.Figure)

def test_raise_error_on_invalid_instance():

    """Test that an error is raised when passing an invalid instance to plot_histogram."""
    with pytest.raises(TypeError):
        # Pass a string instead of a PyPhotoMol instance
        plot_histogram("invalid_instance", colors_hist=['gray'])

def test_plot_histogram():

    pms = MPAnalyzer()
    pms.import_files([file])
    pms.apply_to_all('count_binding_events')
    pms.apply_to_all('create_histogram', use_masses=True, window=[0, 1000], bin_width=10)

    # Get histogram data
    _, hist_df = pms.create_plotting_config()

    # Create configuration objects
    plot_config = PlotConfig(
        plot_width=1000,
        plot_height=400,
        plot_type="png",
        font_size=14,
        normalize=False,
        contrasts=False,
        cst_factor_for_contrast=1000
    )
    
    layout_config = LayoutConfig(
        stacked=False,
        show_subplot_titles=False
    )

    fig = plot_histogram(pms,
                        colors_hist=hist_df,
                        plot_config=plot_config,
                        layout_config=layout_config)
    
    assert isinstance(fig, go.Figure)

def test_plot_without_data():

    photomol = PyPhotoMol()

    plot_config = PlotConfig(contrasts=False)
    
    with pytest.raises(ValueError):
        plot_histogram(photomol, colors_hist=['gray'], plot_config=plot_config)

    plot_config = PlotConfig(contrasts=True)
    
    with pytest.raises(ValueError):
        plot_histogram(photomol, colors_hist=['gray'], plot_config=plot_config)

def test_plot_histograms_and_fits():
    """Test the plot_histograms_and_fits function with a simple MPAnalyzer instance."""

    files = [file,file]
    names = ['demo_h5', 'demo_csv']

    # Start the MPAnalyzer class
    pms = MPAnalyzer()

    pms.import_files(files,names)

    pms.apply_to_all('count_binding_events')
    pms.apply_to_all('create_histogram', use_masses=True, window=[0, 2000], bin_width=10)
    pms.apply_to_all('guess_peaks', min_height=10, min_distance=4, prominence=4)

    # Assuming peaks were already guessed
    all_peaks = pms.get_properties('peaks_guess')[0]  # Use first model's peaks as example

    pms.apply_to_all('fit_histogram', 
                    peaks_guess=all_peaks,
                    mean_tolerance=100, 
                    std_tolerance=200)

    # Create a fit table
    pms.apply_to_all('create_fit_table')

    # Plot it
    legends_df, hist_df = pms.create_plotting_config()

    colors_hist = hist_df['color'].tolist()

    fig = plot_histograms_and_fits(pms,
                                   legends_df,
                                   colors_hist)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Ensure some data was added to the figure
    
    # Check stacked histograms with non-shared y-axes
    layout_config = LayoutConfig(stacked=True, shared_yaxes=False)
    
    fig = plot_histograms_and_fits(pms,
                                   legends_df,
                                   colors_hist,
                                   layout_config=layout_config)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Ensure some data was added to the figure

    layout_config = LayoutConfig(stacked=True)
    
    fig = plot_histograms_and_fits(pms,
                                   legends_df,
                                   colors_hist,
                                   layout_config=layout_config)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Ensure some data was added to the figure

    # Set all Sels to False and plot again
    legends_df_copy = legends_df.copy()
    legends_df_copy['select'] = False
    fig = plot_histograms_and_fits(pms,
                                   legends_df_copy,
                                   colors_hist)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Ensure some data was added to the figure

    # Set all Sels to False and plot again
    legends_df_copy = legends_df.copy()
    legends_df_copy['select'] = True
    legend_config = LegendConfig()
    
    fig = plot_histograms_and_fits(pms,
                                   legends_df_copy,
                                   colors_hist,
                                   legend_config=legend_config)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Ensure some data was added to the figure

def test_plot_histograms_and_fits_with_contrasts():
    """Test the plot_histograms_and_fits function with a simple MPAnalyzer instance."""

    files = [file,file]
    names = ['demo_h5', 'demo_csv']

    # Start the MPAnalyzer class
    pms = MPAnalyzer()

    pms.import_files(files,names)

    pms.apply_to_all('count_binding_events')
    pms.apply_to_all('create_histogram', use_masses=False, window=[-1, 0], bin_width=0.0004)

    # Plot without fitted data
    legends_df = pd.DataFrame({
        'legends': ['Model 1', 'Model 2'],
        'color': ['#FF0000', '#00FF00'],  # Dummy colors for traces
        'select': [True, True],
        'show_legend': [True, True]
    })
    colors_hist = ['#0000FF', '#FFFF00']  # Colors for histograms

    plot_config = PlotConfig(contrasts=True)
    
    fig = plot_histograms_and_fits(pms,
                                   legends_df,
                                   colors_hist,
                                   plot_config=plot_config)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Ensure some data was added to the figure

    pms.apply_to_all('guess_peaks', min_height=10, min_distance=4, prominence=4)

    # Assuming peaks were already guessed
    all_peaks = pms.get_properties('peaks_guess')[0]  # Use first model's peaks as example

    for peak_guess in [all_peaks, [all_peaks[0]]]:  # Test with all peaks and just the first one

        print(f"Using peak guess: {peak_guess}")

        pms.apply_to_all('fit_histogram', 
                        peaks_guess=peak_guess,
                        mean_tolerance=0.01, 
                        std_tolerance=0.02,
                        baseline=1 # We need a small baseline because otherwise the fit with one peak fails
                        )

        # Create a fit table
        pms.apply_to_all('create_fit_table')

        # Print the fitted parameters
        fit_params = pms.get_properties('fit_table')
        print(f"Fitted parameters for peak guess {peak_guess}: {fit_params}")

        # Plot it
        legends_df, hist_df = pms.create_plotting_config()

        colors_hist = hist_df['color'].tolist()

        plot_config = PlotConfig(contrasts=True)
        
        fig = plot_histograms_and_fits(pms,
                                       legends_df,
                                       colors_hist,
                                       plot_config=plot_config)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Ensure some data was added to the figure

        # Test plotting without legends_df - default colours should be used
        plot_config = PlotConfig(contrasts=True)
        
        fig = plot_histograms_and_fits(pms,
                                       legends_df=None,
                                       colors_hist=colors_hist,
                                       plot_config=plot_config)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Ensure some data was added to the figure

def test_plot_calibration():
    """Test the plot_calibration function with calibration data."""
    # Test data: mass vs contrast calibration
    mass = np.array([66, 146, 480, 669])
    contrast = np.array([-0.1, -0.2, -0.5, -0.7])
    slope = -0.001
    intercept = 0.02
    
    # Create the plot
    fig = plot_calibration(mass, contrast, slope, intercept)
    
    # Basic checks
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Should have scatter plot and line
    
    # Check scatter plot trace
    scatter_trace = fig.data[0]
    assert isinstance(scatter_trace, go.Scatter)
    assert scatter_trace.mode == 'markers'
    assert scatter_trace.marker.color == '#377EB8'
    assert scatter_trace.marker.size == 8
    assert scatter_trace.marker.opacity == 0.7
    assert np.array_equal(scatter_trace.x, mass)
    assert np.array_equal(scatter_trace.y, contrast)
    
    # Check calibration line trace
    line_trace = fig.data[1]
    assert isinstance(line_trace, go.Scatter)
    assert line_trace.mode == 'lines'
    assert line_trace.line.color == '#ffa500'
    assert line_trace.line.width == 3
    
    # Check line coordinates (should span from min to max mass)
    expected_x1, expected_x2 = np.min(mass), np.max(mass)
    expected_y1 = expected_x1 * slope + intercept
    expected_y2 = expected_x2 * slope + intercept
    assert np.isclose(line_trace.x[0], expected_x1)
    assert np.isclose(line_trace.x[1], expected_x2)
    assert np.isclose(line_trace.y[0], expected_y1)
    assert np.isclose(line_trace.y[1], expected_y2)
    
    # Check layout
    layout = fig.layout
    assert layout.xaxis.title.text == 'Mass (kDa)'
    assert layout.yaxis.title.text == 'Ratiometric contrast'
    assert layout.showlegend == False
    assert layout.font.family == "Roboto"
    assert layout.plot_bgcolor == 'white'
    assert layout.paper_bgcolor == 'white'
    
    # Check axes styling
    assert layout.xaxis.showline == True
    assert layout.xaxis.linecolor == 'black'
    assert layout.xaxis.showgrid == True
    assert layout.xaxis.gridcolor == 'lightgray'
    assert layout.yaxis.showline == True
    assert layout.yaxis.linecolor == 'black'
    assert layout.yaxis.showgrid == True
    assert layout.yaxis.gridcolor == 'lightgray'


def test_plot_calibration_custom_parameters():
    """Test plot_calibration with custom parameters."""
    # Simple test data
    mass = np.array([100, 200])
    contrast = np.array([-0.15, -0.25])
    slope = -0.0008
    intercept = 0.05
    
    # Custom parameters
    custom_width = 1000
    custom_height = 700
    custom_font_size = 16
    custom_plot_type = "jpeg"

    plot_config = PlotConfig(
        plot_width=custom_width,
        plot_height=custom_height,
        plot_type=custom_plot_type,
        font_size=custom_font_size
    )

    fig = plot_calibration(
        mass, contrast, slope, intercept,
        plot_config=plot_config
    )    # Check that custom parameters are applied
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.layout.width == custom_width
    assert fig.layout.xaxis.title.font.size == custom_font_size
    
    # Check config (download settings)
    config = fig._config
    assert config['toImageButtonOptions']['format'] == custom_plot_type
    assert config['toImageButtonOptions']['width'] == custom_width 


def test_plot_calibration_edge_cases():
    """Test plot_calibration with edge cases."""
    # Single data point
    mass = np.array([100])
    contrast = np.array([-0.1])
    slope = -0.001
    intercept = 0.02
    
    fig = plot_calibration(mass, contrast, slope, intercept)
    
    # Should still create both traces
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    
    # Line should be degenerate (same start and end point)
    line_trace = fig.data[1]
    assert line_trace.x[0] == line_trace.x[1] == mass[0]
    expected_y = mass[0] * slope + intercept
    assert np.isclose(line_trace.y[0], expected_y)
    assert np.isclose(line_trace.y[1], expected_y)

def test_plot_histograms_and_fits_stacked_single_gaussian():
    """Test plot_histograms_and_fits with stacked=True and single Gaussian to cover label_sels branch."""
    # Create a MPAnalyzer instance with single Gaussian fits
    
    pms = MPAnalyzer()
    pms.import_files([file], ['demo_single'])
    pms.apply_to_all('count_binding_events')
    pms.apply_to_all('create_histogram', use_masses=True, window=[0, 2000], bin_width=10)
    pms.apply_to_all('guess_peaks', min_height=10, min_distance=4, prominence=4)
    
    # Get first peak only to create a single Gaussian fit
    all_peaks = pms.get_properties('peaks_guess')[0]
    first_peak_only = [all_peaks[0]] if len(all_peaks) > 0 else [500]  # Use first peak or default
    
    pms.apply_to_all('fit_histogram', 
                     peaks_guess=first_peak_only,
                     mean_tolerance=100, 
                     std_tolerance=200)
    
    pms.apply_to_all('create_fit_table')
    
    # Create plotting config
    legends_df, hist_df = pms.create_plotting_config()
    
    colors_hist = hist_df['color'].tolist()
    
    # Test stacked plot with labels - this should cover the single Gaussian case
    layout_config = LayoutConfig(stacked=True)
    legend_config = LegendConfig(add_labels=True, add_percentages=True, line_width=3)
    
    fig = plot_histograms_and_fits(pms,
                                   legends_df,
                                   colors_hist,
                                   layout_config=layout_config,
                                   legend_config=legend_config)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # Verify that the plot has traces and annotations (indicating successful plotting)
    assert len(fig.layout.annotations) > 0  # Should have label annotations


def test_plot_histograms_and_fits_x_range():
    """Test the x_range parameter in PlotConfig for setting custom x-axis limits."""
    
    files = [file, file]
    names = ['demo_h5', 'demo_csv']

    # Start the MPAnalyzer class
    pms = MPAnalyzer()
    pms.import_files(files, names)
    pms.apply_to_all('count_binding_events')
    pms.apply_to_all('create_histogram', use_masses=True, window=[0, 2000], bin_width=10)
    pms.apply_to_all('guess_peaks', min_height=10, min_distance=4, prominence=4)

    # Fit histogram
    all_peaks = pms.get_properties('peaks_guess')[0]
    pms.apply_to_all('fit_histogram', 
                    peaks_guess=all_peaks,
                    mean_tolerance=100, 
                    std_tolerance=200)
    pms.apply_to_all('create_fit_table')

    # Create plotting config
    legends_df, hist_df = pms.create_plotting_config()
    colors_hist = hist_df['color'].tolist()

    # Test custom x-axis range
    plot_config = PlotConfig(x_range=[100, 800])
    fig = plot_histograms_and_fits(pms, legends_df, colors_hist, plot_config=plot_config)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # Verify that x-axis range was set correctly
    assert fig.layout.xaxis.range == (100, 800)
    
    # Test x-axis range with stacked plots to cover line 1116
    plot_config_stacked = PlotConfig(x_range=[200, 1000])
    layout_config_stacked = LayoutConfig(stacked=True)
    fig_stacked = plot_histograms_and_fits(pms, legends_df, colors_hist, 
                                          plot_config=plot_config_stacked,
                                          layout_config=layout_config_stacked)
    
    assert isinstance(fig_stacked, go.Figure)
    assert len(fig_stacked.data) > 0
    # Verify that x-axis range was set correctly for stacked plots
    assert fig_stacked.layout.xaxis.range == (200, 1000)

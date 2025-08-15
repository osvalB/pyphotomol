"""
Plotting utilities for mass photometry data visualization.
"""
import pandas as pd
import numpy  as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Optional

from datetime import datetime
from plotly.subplots import make_subplots

from .helpers import truncated_multi_gauss_with_baseline

from ..main import MPAnalyzer, PyPhotoMol

@dataclass
class PlotConfig:
    """General plot configuration"""
    plot_width: int = 1000
    plot_height: int = 400
    plot_type: str = "png"
    font_size: int = 14
    normalize: bool = False
    contrasts: bool = False
    cst_factor_for_contrast: float = 1
    x_range: Optional[List[float]] = None  # [x_min, x_max] for setting x-axis limits

@dataclass
class AxisConfig:
    """Axis styling configuration"""
    showgrid_x: bool = True
    showgrid_y: bool = True
    n_y_axis_ticks: int = 3
    axis_linewidth: int = 1
    axis_tickwidth: int = 1
    axis_gridwidth: int = 1

@dataclass
class LayoutConfig:
    """Layout and spacing configuration"""
    stacked: bool = False
    show_subplot_titles: bool = False
    vertical_spacing: float = 0.1
    shared_yaxes: bool = True
    extra_padding_y_label: float = 0

@dataclass
class LegendConfig:
    """Legend and labeling configuration"""
    add_masses_to_legend: bool = True
    add_percentage_to_legend: bool = False
    add_labels: bool = True
    add_percentages: bool = True
    line_width: int = 3

__all__ = [
    'PlotConfig',
    'AxisConfig', 
    'LayoutConfig',
    'LegendConfig',
    'config_fig',
    'create_axis_config',
    'add_histogram',
    'add_labels_to_fig',
    'add_values_in_legend',
    'add_gaussian_traces',
    'add_percentages_to_legend',
    'add_gaussian_simulation_to_plot',
    'plot_histograms_and_fits',
    'plot_histogram',
    'plot_calibration',
    'DECIMAL_THRESHOLDS'
]

DECIMAL_THRESHOLDS = [
    (0.001, 4),   # if min_value < 0.001, use 4 decimal places
    (0.01, 3),    # if min_value < 0.01, use 3 decimal places
    (0.1, 2),     # if min_value < 0.1, use 2 decimal places
    (float('inf'), 0)  # else use 0 decimal places
]

def config_fig(fig, 
               plot_width=800, 
               plot_height=600, 
               plot_type="png",
               plot_title_for_download="plot"):
    """
    Configure plotly figure with download options and toolbar settings.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    plot_width : int, default 800
        Width of the plot in pixels
    plot_height : int, default 600
        Height of the plot in pixels
    plot_type : str, default "png"
        Format for downloading the plot (e.g., "png", "jpeg")
    plot_title_for_download : str, default "plot"
        Title for the downloaded plot file
        
    Returns
    -------
    go.Figure
        Configured plotly figure
    """

    # Append the file extension to the title for download
    plot_title_for_download += f".{plot_type}"

    config = {
        'toImageButtonOptions': {
            'format': plot_type,
            'filename': plot_title_for_download,
            'width': plot_width,
            'height': plot_height
        },
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'sendDataToCloud',
            'hoverClosestCartesian',
            'hoverCompareCartesian', 
            'lasso2d',
            'select2d'
        ]
    }
    
    fig.update_layout(
        width=plot_width,
        height=plot_height
    )
    
    fig._config = config
    
    return fig


def create_axis_config(title=None,
                      font_size=14,
                      show_grid=True,
                      axis_linewidth=1,
                      axis_tickwidth=1,
                      axis_gridwidth=1,
                      nticks=None):
    """
    Create a standardized axis configuration dictionary for Plotly figures.
    
    Parameters
    ----------
    title : str or None
        Title for the axis
    font_size : int, default 14
        Font size for axis title and tick labels
    show_grid : bool, default True
        Whether to show grid lines
    axis_linewidth : int, default 1
        Width of the axis line
    axis_tickwidth : int, default 1
        Width of the tick marks
    axis_gridwidth : int, default 1
        Width of the grid lines
    nticks : int or None
        Number of ticks on the axis (optional guideline)
        
    Returns
    -------
    dict
        Axis configuration dictionary for Plotly
    """
    config = dict(
        tickfont=dict(size=font_size),
        showline=True,
        linewidth=axis_linewidth,
        linecolor='black',
        ticks='outside',
        tickwidth=axis_tickwidth,
        tickcolor='black',
        showgrid=show_grid,
        gridwidth=axis_gridwidth,
        gridcolor='lightgray'
    )
    
    if title is not None:
        config['title'] = dict(
            text=title,
            font=dict(size=font_size)
        )
        
    if nticks is not None:
        config['nticks'] = nticks
        
    return config


def add_histogram(fig, 
                  values, 
                  bin_info, 
                  normalize=False,
                  hex_color="#377EB8"):
    
    """
    Add mass histogram trace to plotly figure.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    values : np.ndarray
        Array of mass or contrast values
    bin_info : dict
        Dictionary containing histogram bin information
    normalize : bool
        Whether to normalize the histogram
    hex_color : str
        Hex color code for the histogram bars
        
    Returns
    -------
    go.Figure
        Updated plotly figure with histogram trace
    """
    histnorm = "probability" if normalize else ""
    
    xbins_dict = {
        'start': bin_info['start'],
        'end': bin_info['end'], 
        'size': bin_info['size']
    }
    
    fig.add_trace(
        go.Histogram(
            x=values,
            marker_color=hex_color,
            opacity=0.5,
            xbins=xbins_dict,
            histnorm=histnorm,
            showlegend=False
        )
    )
    
    return fig



def add_labels_to_fig(fig,
                     fit_table,
                     contrasts=False,
                     scaling_factor=1.0,
                     font_size=14,
                     sels=None,
                     stacked=False,
                     max_y=None,
                     show_values=True,
                     show_counts=True,
                     cst_factor_for_contrast=1000,
                     subplot_row=None):
    """
    Add peak labels to plotly figure.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    fit_table : pd.DataFrame
        DataFrame containing fit results
    contrasts : bool
        Whether data represents contrasts
    scaling_factor : float
        Factor to scale y values
    font_size : int
        Font size for axes
    sels : list or None
        Boolean list for selecting which peaks to label
    stacked : bool
        Whether this is a stacked plot
    max_y : float or None
        Maximum y value for label positioning, required if the y-axis range is shared
    show_values : bool
        Whether to include mass / contrast values in labels
    show_counts : bool
        Whether to include count percentages in labels
    cst_factor_for_contrast : float
        Factor to convert contrasts for display
    subplot_row : int or None
        Row number for stacked subplots (1-indexed), None for single plot
        
    Returns
    -------
    go.Figure
        Updated plotly figure with peak labels
    """
    fitted_means = fit_table.iloc[:, 0]
    fitted_counts_perc = fit_table.iloc[:, 3]
    fitted_amp = fit_table.iloc[:, 4]
    
    if sels is None:
        sels = [True] * len(fitted_means)
    
    if contrasts:
        fitted_means = fitted_means * cst_factor_for_contrast
    
    fitted_amp = fitted_amp * scaling_factor
    
    # Calculate y-shift for label positioning
    y_shift = np.max(fitted_amp) * 0.02
    
    # Initialize empty labels
    labels = [""] * len(fitted_means)
    
    # Build labels
    if show_values:

        if contrasts:

            # Find the minimum value in fitted_means, with more decimal places
            min_value = np.min(np.abs(fitted_means))

            # Find appropriate decimal places
            decimal_places = next(places for threshold, places in DECIMAL_THRESHOLDS if min_value < threshold)

            # Format labels with determined decimal places
            labels = [f" {np.round(mean, decimal_places)}" for mean in fitted_means]

        else:
            # 0 decimal places for masses
            labels = [f" {round(mean)} kDa" for mean in fitted_means]

    if show_counts:
        if show_values:
            labels = [f"{label} ({round(perc)} % counts )" 
                     for label, perc in zip(labels, fitted_counts_perc)]
        else:
            labels = [f" ({round(perc)} % counts )" 
                     for perc in fitted_counts_perc]
    
    fitted_means_array = fitted_means.values
    fitted_amp_array = fitted_amp.values
    sels_array = np.array(sels, dtype=bool)
    
    selected_means = fitted_means_array[sels_array]
    selected_amplitudes = (fitted_amp_array + y_shift)[sels_array]
    selected_labels = np.array(labels)[sels_array]
    
    # Heuristic method to avoid overlapping labels
    if len(selected_means) > 1:

        extra_y_shift = 0.2 * max_y if max_y is not None else 0

        i = 0
        # Verify if two labels are too close by comparing their x and y positions
        for mean, amplitude in zip(selected_means, selected_amplitudes):
            
            other_means      = np.delete(selected_means, i)
            other_amplitudes = np.delete(selected_amplitudes, i)

            if max_y is None:

                condition1 = [np.abs((a-amplitude)/np.max([a, amplitude])) < 0.25 for a in other_amplitudes]
                condition1 = np.array(condition1,dtype=bool)
            # We check the different in respect to the total amplitude
            # This is useful for stacked plots where the y-axis range is shared
            else:

                condition1 = [np.abs((a-amplitude)/max_y) < 0.3 for a in other_amplitudes]
                condition1 = np.array(condition1,dtype=bool)

            interval_to_check = [mean / 1.6, 1.6 * mean]

            # Second condition, x-position is between mean / 1.6 and 1.6 * mean
            condition2 = (other_means > interval_to_check[0]) & (other_means < interval_to_check[1])

            # If both conditions are met, just shift the y position of the current label to avoid overlap
            both_conditions = condition1 & condition2

            # Insert False at the position index 'i' to keep the same length as selected_means and selected_amplitudes
            both_conditions = np.insert(both_conditions, i, False)

            # Iterate over the both_conditions to find overlaps
            for j, boolean in enumerate(both_conditions):
                if boolean:

                    i_or_j = i if selected_amplitudes[i] > selected_amplitudes[j] else j

                    selected_amplitudes[i_or_j] += 0.06 * selected_amplitudes[i_or_j] + extra_y_shift

                    j_mean_is_lower = selected_means[j] < selected_means[i]

                    selected_means[j] -= (0.05 * selected_means[j]) * j_mean_is_lower
                    selected_means[i] += (0.05 * selected_means[i]) * j_mean_is_lower
                    
                    selected_means[j] += (0.05 * selected_means[j]) * (not j_mean_is_lower)
                    selected_means[i] -= (0.05 * selected_means[i]) * (not j_mean_is_lower)

            i += 1

    # Add text annotations to figure
    for mean, amplitude, label in zip(selected_means, selected_amplitudes, selected_labels):
        if stacked and subplot_row is not None:
            # For stacked plots, add annotation with row/col specification
            fig.add_annotation(
                x=mean,
                y=amplitude,
                text=label,
                showarrow=False,
                font=dict(
                    family="Roboto",
                    size=font_size - 1,
                    color="black"
                ),
                xanchor="left",
                yanchor="bottom",
                row=subplot_row,
                col=1
            )
        else:
            # For single plots, add annotation normally
            fig.add_annotation(
                x=mean,
                y=amplitude,
                text=label,
                showarrow=False,
                font=dict(
                    family="Roboto",
                    size=font_size - 1,
                    color="black"
                ),
                xanchor="left",
                yanchor="bottom"
            )
    
    return fig


def add_values_in_legend(legends, 
                        fit_table, 
                        contrasts=False,
                        cst_factor_for_contrast=1000):
    """
    Add mass / contrast values to legend labels.
    
    Parameters
    ----------
    legends : list
        List of legend labels
    fit_table : pd.DataFrame
        DataFrame with fit table data
    contrasts : bool
        Whether data represents contrasts
    cst_factor_for_contrast : float
        Factor to convert contrasts for display
        
    Returns
    -------
    list
        Updated legend labels with mass values
    """
    fitted_means = fit_table.iloc[:, 0]
    
    if contrasts:
        fitted_means = fitted_means * cst_factor_for_contrast
    
    updated_legends = []
    for legend, mean in zip(legends, fitted_means):
       
        if contrasts:
            # Choose the number of decimal places according to the value
            min_abs_mean = np.min(np.abs(fitted_means))
            
            # Find appropriate decimal places
            decimal_places = next(places for threshold, places in DECIMAL_THRESHOLDS if min_abs_mean < threshold)
            # Format the legend with determined decimal places
            updated_legends.append(f"{legend} : {round(mean, decimal_places)}")

        else:

            updated_legends.append(f"{legend} : {round(mean)} kDa")
    
    return updated_legends


def add_gaussian_traces(fig,
                       arr_fitted_values,
                       fitted_params_table,
                       legends,
                       color_palette,
                       sels,
                       baseline=0,
                       scaling_factor=1.0,
                       show_values_in_legend=True,
                       show_perc_in_legend=True,
                       contrasts=False,
                       cst_factor_for_contrast=1000,
                       line_width=3,
                       show_legend_vec=[]):
    """
    Add individual Gaussian traces to plotly figure.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    arr_fitted_values : np.ndarray
        Array of fitted values (x, individual Gaussians, sum)
    fitted_params_table : pd.DataFrame
        DataFrame with fit table results
    legends : list
        List of legend labels - Same length as columns in arr_fitted_values 
    color_palette : list
        List of hex color codes
    sels : list
        Boolean list for selecting which traces to show
    baseline : float
        Baseline value for filtering
    scaling_factor : float
        Factor to scale y values
    show_values_in_legend : bool
        Whether to add masses / contrast values to legend labels
    show_perc_in_legend : bool
        Whether to add percentages to legend labels
    contrasts : bool
        Whether data represents contrasts
    add_labels : bool
        Whether to add peak labels
    add_percentages : bool
        Whether to add percentage labels
    stacked : bool
        Whether this is a stacked plot
    cst_factor_for_contrast : float
        Factor to convert contrasts for display
    subplot_row : int or None
        Row number for stacked subplots (1-indexed), None for single plot
    line_width : int, default 3
        Width of the Gaussian trace lines
    show_legend_vec : list, default []
        Boolean list for showing Gaussian traces in legend
    Returns
    -------
    go.Figure
        Updated plotly figure with Gaussian traces
    """
    # Convert to numpy array if not already
    arr_fitted_values = np.array(arr_fitted_values)
    
    # Convert to all True if show_legend_vec is empty
    if not show_legend_vec:
        show_legend_vec = [True] * len(sels)

    # Convert contrasts if needed
    if contrasts:
        arr_fitted_values[:, 0] = arr_fitted_values[:, 0] * cst_factor_for_contrast
    
    # Handle Gaussian sum trace if more than 3 columns (x, individual gaussians, sum)
    if arr_fitted_values.shape[1] > 3:
        # Add Gaussian sum trace if selected
        if sels[0]:
            # Extract x and sum columns (first and last)
            x_values = arr_fitted_values[:, 0]
            y_values = arr_fitted_values[:, -1]
            
            # Filter and scale
            mask = (y_values > baseline + 0.05)
            filtered_x = x_values[mask]
            filtered_y = y_values[mask] * scaling_factor
            
            fig.add_trace(
                go.Scatter(
                    x=filtered_x,
                    y=filtered_y,
                    mode='lines',
                    line=dict(color=color_palette[0], width=line_width),
                    name=legends[0],
                    showlegend=show_legend_vec[0] 
                )
            )
        
        # Remove first elements for individual Gaussians
        sels = sels[1:]
        
        proceed = sum(sels) > 0
        
        if proceed:
            legends = legends[1:]
            color_palette = color_palette[1:]
            show_legend_vec = show_legend_vec[1:]
        else:
            return fig
    
    # Update legends with masses and percentages
    if show_values_in_legend:
        legends = add_values_in_legend(legends, fitted_params_table, contrasts, cst_factor_for_contrast)
    
    if show_perc_in_legend:
        legends = add_percentages_to_legend(legends, fitted_params_table)
    
    # Process individual Gaussian traces
    # Extract individual Gaussians (exclude first column x and last column sum)
    x_values = arr_fitted_values[:, 0]
    gaussian_columns = arr_fitted_values[:, 1:-1]
    
    # Create DataFrame for easier melting
    gaussian_data = pd.DataFrame(gaussian_columns)
    gaussian_data.insert(0, 'x', x_values)
    
    # Set column names
    gaussian_cols = ['x'] + legends[:gaussian_columns.shape[1]]
    gaussian_data.columns = gaussian_cols
    
    # Melt the dataframe
    melted_data = pd.melt(gaussian_data, id_vars=['x'], var_name='variable', value_name='value')
    
    # Filter data
    melted_data = melted_data[melted_data['value'] > baseline + 0.05]
    melted_data['variable'] = melted_data['variable'].astype('category')
    
    # Add individual Gaussian traces
    counter_color_palette = 0
    
    for var in melted_data['variable'].unique():
        if counter_color_palette < len(sels) and sels[counter_color_palette]:
            hex_color = color_palette[counter_color_palette]
            temp_df = melted_data[melted_data['variable'] == var].copy()
            temp_df['value'] = temp_df['value'] * scaling_factor
            
            fig.add_trace(
                go.Scatter(
                    x=temp_df['x'],
                    y=temp_df['value'],
                    mode='lines',
                    line=dict(color=hex_color, width=line_width),
                    name=var,
                    showlegend=show_legend_vec[counter_color_palette]
                )
            )
        
        counter_color_palette += 1
    
    return fig


def add_percentages_to_legend(legends, fit_table):
    """
    Add percentage values to legend labels.
    
    Parameters
    ----------
    legends : list
        List of legend labels
    fit_table : pd.DataFrame
        DataFrame with fit table data
        
    Returns
    -------
    list
        Updated legend labels with percentage values
    """
    fitted_counts_perc = fit_table.iloc[:, 3]
    
    updated_legends = []
    for legend, perc in zip(legends, fitted_counts_perc):
        updated_legends.append(f"{legend} ({round(perc)} % counts )")
    
    return updated_legends


def add_gaussian_simulation_to_plot(fig,
                          mean,
                          std,
                          amplitude,
                          left_bound):
    """
    Add simulated truncated Gaussian trace to plotly figure.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    mean : float
        Mean of the Gaussian distribution
    std : float
        Standard deviation of the Gaussian distribution
    amplitude : float
        Amplitude of the Gaussian distribution
    left_bound : float
        Left boundary for truncation
        
    Returns
    -------
    go.Figure
        Updated plotly figure with simulation trace
    """
    # Create x sequence from max(left_bound, mean-std*2.5) to mean+std*2.5
    x_start = max(left_bound, mean - std * 2.5)
    x_end = mean + std * 2.5
    x_seq = np.linspace(x_start, x_end, 100)
    
    # Calculate truncated Gaussian values
    y = truncated_multi_gauss_with_baseline(x_seq, left_bound,0,mean, std, amplitude)
    
    # Add trace to figure
    fig.add_trace(
        go.Scatter(
            x=x_seq,
            y=y,
            mode='lines',
            line=dict(
                color='black',
                width=3
            ),
            name='Simulation'
        )
    )
    
    return fig

def extract_max_y(analyzer,
                  plot_config: PlotConfig = None):

    """
    Extract the maximum Y value from all models in the analyzer for histogram scaling.
    This function iterates through all models in the analyzer, retrieves the mass or contrast values,
    filters them based on the histogram window, and calculates the maximum Y value for scaling purposes.

    Parameters
    ----------
    analyzer : MPAnalyzer
        MPAnalyzer instance containing multiple PyPhotoMol models
    plot_config : PlotConfig, optional
        Plot configuration object that specifies whether to use contrasts or masses,
        and whether to normalize the histogram counts.
    Returns
    -------
    float
        Maximum Y value for histogram scaling
    """

    plot_config = plot_config or PlotConfig()

    photomol_models = list(analyzer.models.values())

    max_y = 0
    
    # Process each model
    for photomol in photomol_models:

        # Raise an error if no masses are available and the user wants to plot them
        if not plot_config.contrasts and (photomol.masses is None or len(photomol.masses) == 0):
            raise ValueError(f"No masses available for model {photomol.file}. "
                            "Please ensure the model has been processed correctly.")

        # Raise an error if no contrasts are available and the user wants to plot them
        if plot_config.contrasts and (photomol.contrasts is None or len(photomol.contrasts) == 0):
            raise ValueError(f"No contrasts available for model {photomol.file}. "
                            "Please ensure the model has been processed correctly.")

        # Get mass/contrast values from model and filter by histogram window
        values = photomol.contrasts.copy() if plot_config.contrasts else photomol.masses.copy()
        
        # Filter values based on histogram window
        mask = (values >= photomol.hist_window[0]) & (values <= photomol.hist_window[1])
        values = values[mask]
        
        # Calculate scaling factor
        scaling_factor = 1.0 / len(values) if plot_config.normalize else 1.0
        
        # Extract the max value from photomol.hist_counts - useful for scaling the Y range
        max_y = np.max((max_y, photomol.hist_counts.max() * scaling_factor))

    # Convert to float for consistency
    max_y = float(max_y)

    return max_y

def plot_histograms_and_fits(analyzer,
                             legends_df=None,
                             colors_hist=None,
                             plot_config: PlotConfig = None,
                             axis_config: AxisConfig = None,
                             layout_config: LayoutConfig = None,
                             legend_config: LegendConfig = None):
    """
    Create a comprehensive plot of PhotoMol fit data with histograms and Gaussian traces.
    
    Parameters
    ----------
    analyzer : pyphotomol.MPAnalyzer or pyphotomol.PyPhotoMol
        MPAnalyzer instance containing multiple PyPhotoMol models - or a single PyPhotoMol instance
    legends_df : pd.DataFrame, optional
        DataFrame containing legends, colors, and selections with columns ['legends', 'color', 'select', 'show_legend']
        This dataframe affects the fitted curves only, not the histograms.
    colors_hist : list, str, or pd.DataFrame, optional
        List of colors for histograms (one per model)
        If a string, it will be used for all histograms.
        If a DataFrame, it should have a column 'color' with hex color codes.
    plot_config : PlotConfig, optional
        General plot configuration (dimensions, format, contrasts, etc.)
    axis_config : AxisConfig, optional
        Axis styling configuration (grid, line widths, etc.)
    layout_config : LayoutConfig, optional
        Layout configuration (stacked, spacing, etc.)
    legend_config : LegendConfig, optional
        Legend and labeling configuration

    Returns
    -------
    go.Figure
        Configured plotly figure object

    Examples
    --------
    Simple plot with default settings:
    
    >>> fig = plot_histograms_and_fits(analyzer, colors_hist=['blue', 'red'])
    >>> fig.show()
    
    Customized plot with configuration objects:
    
    >>> plot_config = PlotConfig(plot_width=800, contrasts=True, x_range=[0, 500])
    >>> layout_config = LayoutConfig(stacked=True, vertical_spacing=0.05)
    >>> fig = plot_histograms_and_fits(analyzer, plot_config=plot_config, 
    ...                               layout_config=layout_config)
    >>> fig.show()
    
    Plot with custom x-axis limits:
    
    >>> plot_config = PlotConfig(x_range=[100, 800])  # Zoom to 100-800 kDa range
    >>> fig = plot_histograms_and_fits(analyzer, plot_config=plot_config)
    >>> fig.show()
    """
    
    # Set defaults for configuration objects
    plot_config = plot_config or PlotConfig()
    axis_config = axis_config or AxisConfig()
    layout_config = layout_config or LayoutConfig()
    legend_config = legend_config or LegendConfig()
    
    # check if analyzer is an instance of MPAnalyzer
    if not isinstance(analyzer, MPAnalyzer):
        # check if it is a PhotoMol instance
        if isinstance(analyzer, PyPhotoMol):
            # Convert single PhotoMol instance to MPAnalyzer with one model
            analyzer_temp = MPAnalyzer()
            analyzer_temp.models[analyzer.file] = analyzer
            analyzer = analyzer_temp

        else:
            raise TypeError("analyzer must be an instance of MPAnalyzer or PyPhotoMol")

    # Extract data from legends_df, if it is not None
    if legends_df is not None:
        legends_all = legends_df['legends'].tolist()
        color_palette_all = legends_df['color'].tolist()
        sels_all = legends_df['select'].tolist()
        show_legend_all = legends_df['show_legend'].tolist()

    # If colors_hist is a dataframe and it has a column called 'color',
    # then extract the colors from that column
    if isinstance(colors_hist, pd.DataFrame) and 'color' in colors_hist.columns:
        colors_hist = colors_hist['color'].tolist()

    # If colors_hist is a string, use that colour for all histograms
    if isinstance(colors_hist, str):
        colors_hist = [colors_hist] * len(analyzer.models)

    photomol_models = list(analyzer.models.values())
    subplot_titles = list(analyzer.models.keys()) if layout_config.show_subplot_titles else None
    # Set axis labels based on data type
    y_label = "Normalised counts" if plot_config.normalize else "Counts"

    # Set the title to be used for the x-axis. Mass or contrasts
    # If the cst factor is 1, we do not need to mention it    
    if plot_config.cst_factor_for_contrast == 1:
        x_title = "Ratiometric contrast" if plot_config.contrasts else "Mass (kDa)"
    else:
        x_title = f"Ratiometric contrast * {plot_config.cst_factor_for_contrast}" if plot_config.contrasts else "Mass (kDa)"

    # Configure axes using helper function
    y_axis = create_axis_config(
        title=y_label,
        font_size=plot_config.font_size,
        show_grid=axis_config.showgrid_y,
        axis_linewidth=axis_config.axis_linewidth,
        axis_tickwidth=axis_config.axis_tickwidth,
        axis_gridwidth=axis_config.axis_gridwidth,
        nticks=axis_config.n_y_axis_ticks
    )
    
    x_axis = create_axis_config(
        title=x_title,
        font_size=plot_config.font_size,
        show_grid=axis_config.showgrid_x,
        axis_linewidth=axis_config.axis_linewidth,
        axis_tickwidth=axis_config.axis_tickwidth,
        axis_gridwidth=axis_config.axis_gridwidth
    )
    
    # Apply x-axis range if specified
    if plot_config.x_range is not None:
        x_axis['range'] = plot_config.x_range
            
    # Extract the maximum y-value of the plots
    max_y = extract_max_y(analyzer, plot_config)

    # Initialize figure(s)
    if layout_config.stacked:
        figs = []
        # Collect label information for stacked plots
        label_data = []
    else:
        fig = go.Figure()
    
    id_start  = 0
    model_cnt = 0
    
    # Process each model
    for photomol in photomol_models:
        model_cnt += 1
        
        # Create individual figure for stacked plots
        if layout_config.stacked:
            current_fig = go.Figure()
            figs.append(current_fig)
        else:
            current_fig = fig

        # Get mass/contrast values from model and filter by histogram window
        values = photomol.contrasts.copy() if plot_config.contrasts else photomol.masses.copy()
        
        # Filter values based on histogram window
        mask = (values >= photomol.hist_window[0]) & (values <= photomol.hist_window[1])
        values = values[mask]

        # Apply contrast factor if needed
        if plot_config.contrasts:
            values = values * plot_config.cst_factor_for_contrast
        
        # Calculate scaling factor
        scaling_factor = 1.0 / len(values) if plot_config.normalize else 1.0
        baseline = photomol.baseline * scaling_factor  # Use individual photomol baseline
        
        # Get bin information and add histogram
        bin_info = {
            'start': photomol.hist_window[0],
            'end': photomol.hist_window[1],
            'size': photomol.bin_width
        }
        
        # Apply contrast factor to bin_info if needed
        if plot_config.contrasts:
            bin_info['start'] = bin_info['start'] * plot_config.cst_factor_for_contrast
            bin_info['end'] = bin_info['end'] * plot_config.cst_factor_for_contrast
            bin_info['size'] = bin_info['size'] * plot_config.cst_factor_for_contrast
        
        # Default to light blue if no colors_hist provided
        color_hist = "#377EB8" if colors_hist is None else colors_hist[model_cnt - 1]
        
        current_fig = add_histogram(
            current_fig, values, bin_info, plot_config.normalize, color_hist
        )

        # Skip if no fit data available
        if not hasattr(photomol, 'fitted_data') or photomol.fitted_data is None:
            continue
        
        # Process fit data
        fitted_data = photomol.fitted_data.copy()
        
        # Determine trace indices for this model
        # Calculate how many traces this model contributes
        if fitted_data.shape[1] <= 3:
            # Only one gaussian
            num_traces = 1
            id_end = id_start + num_traces - 1
        else:
            # x + individual gaussians + sum
            num_traces = fitted_data.shape[1] - 2  # Exclude x and sum
            id_end = id_start + num_traces  # Include sum trace
        
        # If we do not have the legends_df,
        #  set all to True and use light black color as default for the individual gaussians and red for the sum
        # The legend for the gaussian sum is the first element in the legends_all
        if legends_df is None:

            color_palette =  ['#B0B0B0'] * (num_traces)  # light gray for individual Gaussians
            legends = [f"Gaussian {i+1}" for i in range(num_traces)]
            
            if fitted_data.shape[1] > 3:
                color_palette.insert(0, '#FF7F0E')  # light orange for the Gaussian sum trace
                legends.insert(0, 'Gaussian Sum')  # First legend is for the sum trace

            show_legend = [True] * len(legends)
            sels = [True] * len(legends)

        else:

            # Extract relevant colors, legends, and selections
            color_palette = color_palette_all[id_start:id_end + 1]
            legends = legends_all[id_start:id_end + 1]
            sels = sels_all[id_start:id_end + 1]
            show_legend = show_legend_all[id_start:id_end + 1]

        # Update starting index for next model
        id_start = id_end + 1
        
        # Skip if no traces are selected
        if sum(sels) == 0:
            continue
        
        # Store original sels for label collection (before modification in add_gaussian_traces)
        original_sels = sels.copy()
        
        # Add Gaussian traces
        current_fig = add_gaussian_traces(
            current_fig,
            fitted_data,
            photomol.fit_table,
            legends,
            color_palette,
            sels,
            baseline=baseline,
            scaling_factor=scaling_factor,
            show_values_in_legend=legend_config.add_masses_to_legend,
            show_perc_in_legend=legend_config.add_percentage_to_legend,
            contrasts=plot_config.contrasts,
            cst_factor_for_contrast=plot_config.cst_factor_for_contrast,
            line_width=legend_config.line_width,
            show_legend_vec=show_legend
        )

        # Determine correct sels for labels based on fitted data structure
        if fitted_data.shape[1] > 3:
            # Multiple Gaussians: skip the sum trace (first element) for labels
            label_sels = original_sels[1:] if len(original_sels) > 1 else []
        else:
            # Single Gaussian: use all selections
            label_sels = original_sels

        # If the figure is not stacked (subplots style), add the labels here
        if not layout_config.stacked and (legend_config.add_labels or legend_config.add_percentages):
            current_fig = add_labels_to_fig(
                current_fig,
                photomol.fit_table,
                contrasts=plot_config.contrasts,
                scaling_factor=scaling_factor,
                font_size=plot_config.font_size,
                sels=label_sels,  # Use original sels for labels
                stacked=False,
                max_y=max_y,
                show_values=legend_config.add_labels,
                show_counts=legend_config.add_percentages,
                cst_factor_for_contrast=plot_config.cst_factor_for_contrast
            )

        # If the figure is stacked, collect label information for later use
        if layout_config.stacked and (legend_config.add_labels or legend_config.add_percentages):
            
            label_data.append({
                'fit_table': photomol.fit_table,
                'contrasts': plot_config.contrasts,
                'scaling_factor': scaling_factor,
                'font_size': plot_config.font_size,
                'sels': label_sels,  # Use adjusted sels for labels
                'show_values': legend_config.add_labels,
                'show_counts': legend_config.add_percentages,
                'cst_factor_for_contrast': plot_config.cst_factor_for_contrast,
                'subplot_row': model_cnt
            })
    
    # Finalize figure configuration
    if not layout_config.stacked:
        # Single plot with overlay
        fig.update_layout(
            barmode="overlay",
            xaxis=x_axis,
            yaxis=y_axis,
            font_family="Roboto",
            legend=dict(font=dict(size=plot_config.font_size)),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        final_fig = fig
    else:
        # Create subplot with all individual figures
        num_models = len(photomol_models)
        
        final_fig = make_subplots(
            rows=num_models,
            cols=1,
            subplot_titles=subplot_titles,
            shared_xaxes=True,
            shared_yaxes='all' if layout_config.shared_yaxes else False,
            vertical_spacing=layout_config.vertical_spacing
        )

        # Update the font size of the subplot titles
        final_fig.update_annotations(font_size=plot_config.font_size)

        # Update the x-axis only once, because they are shared
        final_fig.update_xaxes(**x_axis)

        # If y-axis are shared, set the y-axis range globally
        if layout_config.shared_yaxes and max_y is not None:
            y_axis['title'] = None  # No title for shared y-axis - we will add it later as an annotation
            final_fig.update_yaxes(**y_axis)
            final_fig.update_yaxes(range=[0, max_y * 1.3])
            final_fig.update_yaxes(matches='y')

        # Add all traces from individual figures to subplot
        for i, subplot_fig in enumerate(figs):
            for trace in subplot_fig.data:
                final_fig.add_trace(trace, row=i+1, col=1)
        
        # Update subplot layout with white background and axis styling
        final_fig.update_layout(
            font_family="Roboto",
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=plot_config.font_size-1))
        )
        
        for i in range(num_models):
      
            # Remove the title if i is different from num_models -1
            if i < num_models-1:
                final_fig.update_xaxes(title_text=None, showticklabels=False, row=i+1, col=1)

            # If y-axis are not shared configure them individually
            if not layout_config.shared_yaxes:

                # Configure y-axis using helper function
                y_config = create_axis_config(
                    title=None,  # No title - it will be added later as an annotation
                    font_size=plot_config.font_size,
                    show_grid=axis_config.showgrid_y,
                    axis_linewidth=axis_config.axis_linewidth,
                    axis_tickwidth=axis_config.axis_tickwidth,
                    axis_gridwidth=axis_config.axis_gridwidth,
                    nticks=axis_config.n_y_axis_ticks 
                )
                y_config['automargin'] = True
                
                final_fig.update_yaxes(**y_config, row=i+1, col=1)
        
        # Add labels to stacked plots
        for label_info in label_data:
            final_fig = add_labels_to_fig(
                final_fig,
                label_info['fit_table'],
                contrasts=label_info['contrasts'],
                scaling_factor=label_info['scaling_factor'],
                font_size=label_info['font_size'],
                sels=label_info['sels'],
                stacked=True,
                max_y=max_y if layout_config.shared_yaxes else None,
                show_values=label_info['show_values'],
                show_counts=label_info['show_counts'],
                cst_factor_for_contrast=label_info['cst_factor_for_contrast'],
                subplot_row=label_info['subplot_row']
            )

        tick_str_len = len(str(int(max_y*1.3)))
        # Estimate width per character (roughly 7-8px for axis font size ~12-14)
        padding_px = 35  # additional spacing

        char_width_px = (plot_config.font_size / 12) * 8

        # Calculate left margin and annotation position
        left_margin = tick_str_len * char_width_px + padding_px
        x_offset = - (left_margin / plot_config.plot_width)  # scale to figure width (600 px default)

        # Add centered y-axis title for stacked plots
        final_fig.add_annotation(
            text=y_label,
            xref="paper", yref="paper",
            x=x_offset-layout_config.extra_padding_y_label, y=0.5,
            showarrow=False,
            textangle=-90,
            font=dict(size=plot_config.font_size),
            xanchor="center",
            yanchor="middle"
        )
        
    # Apply final configuration
    plot_title = f"MP_plot-{datetime.now().strftime('%Y-%m-%d')}"
    final_fig = config_fig(
        final_fig,
        plot_config.plot_width,
        plot_config.plot_height,
        plot_config.plot_type,
        plot_title
    )
    
    return final_fig


def plot_calibration(mass, 
                     contrast, 
                     slope, 
                     intercept,
                     plot_config: PlotConfig = None,
                     axis_config: AxisConfig = None):
    """
    Create a scatter plot of mass vs contrast with calibration line.
    
    This function creates a visualization showing the relationship between
    mass and ratiometric contrast, with a fitted calibration line overlaid.
    This is useful for visualizing calibration quality and outliers.
    
    Parameters
    ----------
    mass : array-like
        Array of mass values in kDa
    contrast : array-like  
        Array of corresponding ratiometric contrast values
    slope : float
        Slope of the calibration line (contrast = slope * mass + intercept)
    intercept : float
        Intercept of the calibration line
    plot_config : PlotConfig, optional
        General plot configuration (dimensions, format, axis size, etc.)
    axis_config : AxisConfig, optional
        Axis styling configuration (grid, line widths, etc.)
        
    Returns
    -------
    go.Figure
        Plotly figure object containing the mass vs contrast calibration plot
        
    Examples
    --------
    Plot mass vs contrast calibration:
    
    >>> import numpy as np
    >>> from pyphotomol.utils.plotting import plot_calibration, PlotConfig, AxisConfig
    >>> 
    >>> # Simulated calibration data
    >>> mass = np.array([66, 146, 480])
    >>> contrast = np.array([-0.1, -0.2, -0.5])
    >>> slope = -0.001
    >>> intercept = 0.02
    >>> 
    >>> # Simple plot with defaults
    >>> fig = plot_calibration(mass, contrast, slope, intercept)
    >>> fig.show()
    >>> 
    >>> # Customized plot
    >>> plot_config = PlotConfig(plot_width=600, plot_height=400, font_size=12)
    >>> axis_config = AxisConfig(showgrid_x=False, n_y_axis_ticks=6)
    >>> fig = plot_calibration(mass, contrast, slope, intercept, 
    ...                       plot_config=plot_config, axis_config=axis_config)
    >>> fig.show()
    """
    
    # Set defaults
    plot_config = plot_config or PlotConfig(plot_width=800, plot_height=600)
    axis_config = axis_config or AxisConfig(n_y_axis_ticks=4)
    
    # Create figure
    fig = go.Figure()
    
    # Create DataFrame for data points
    df = pd.DataFrame({'mass': mass, 'contrast': contrast})
    
    # Add scatter plot for data points
    fig.add_trace(
        go.Scatter(
            x=df['mass'],
            y=df['contrast'],
            mode='markers',
            marker=dict(
                color='#377EB8',
                size=8,
                opacity=0.7
            ),
            name='Data Points',
            showlegend=False
        )
    )
    
    # Calculate calibration line endpoints
    x1 = np.min(mass)
    y1 = x1 * slope + intercept
    x2 = np.max(mass)
    y2 = x2 * slope + intercept
    
    # Create DataFrame for prediction line
    df_pred = pd.DataFrame({'x': [x1, x2], 'y': [y1, y2]})
    
    # Add calibration line
    fig.add_trace(
        go.Scatter(
            x=df_pred['x'],
            y=df_pred['y'],
            mode='lines',
            line=dict(
                color='#ffa500',
                width=3
            ),
            name='Calibration Line',
            showlegend=False
        )
    )
    
    # Configure axes using helper function
    x_config = create_axis_config(
        title='Mass (kDa)',
        font_size=plot_config.font_size,
        show_grid=axis_config.showgrid_x,
        axis_linewidth=axis_config.axis_linewidth,
        axis_tickwidth=axis_config.axis_tickwidth,
        axis_gridwidth=axis_config.axis_gridwidth
    )
    
    y_config = create_axis_config(
        title='Ratiometric contrast',
        font_size=plot_config.font_size,
        show_grid=axis_config.showgrid_y,
        axis_linewidth=axis_config.axis_linewidth,
        axis_tickwidth=axis_config.axis_tickwidth,
        axis_gridwidth=axis_config.axis_gridwidth,
        nticks=axis_config.n_y_axis_ticks
    )
    
    fig.update_layout(
        xaxis=x_config,
        yaxis=y_config,
        showlegend=False,
        font=dict(family="Roboto"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Apply configuration
    plot_title = f"calibrationMassVsContrast-{datetime.now().strftime('%Y-%m-%d')}"
    fig = config_fig(fig, plot_config.plot_width, plot_config.plot_height, 
                     plot_config.plot_type, plot_title)
    
    return fig


def plot_histogram(analyzer,
                   colors_hist=None,
                   plot_config: PlotConfig = None,
                   axis_config: AxisConfig = None,
                   layout_config: LayoutConfig = None):
    """
    Create a plot with only histograms from PhotoMol data (wrapper around plot_histograms_and_fits).
    
    This function is a simplified wrapper that creates histogram-only plots without
    requiring fitted data or legend configuration.
    
    Parameters
    ----------
    analyzer : pyphotomol.MPAnalyzer or pyphotomol.PyPhotoMol
        MPAnalyzer instance containing multiple PyPhotoMol models or a single PyPhotoMol instance
    colors_hist : list, optional
        List of colors for histograms (one per model)
    plot_config : PlotConfig, optional
        General plot configuration (dimensions, format, contrasts, etc.)
    axis_config : AxisConfig, optional
        Axis styling configuration (grid, line widths, etc.)
    layout_config : LayoutConfig, optional
        Layout configuration (stacked, spacing, etc.)
        
    Returns
    -------
    go.Figure
        Configured plotly figure object with histograms only
        
    Examples
    --------
    Create a simple histogram plot:
    
    >>> fig = plot_histogram(analyzer, ['#FF5733', '#33C3FF'])
    >>> fig.show()
    
    Create stacked normalized histograms:
    
    >>> plot_config = PlotConfig(normalize=True)
    >>> layout_config = LayoutConfig(stacked=True)
    >>> fig = plot_histogram(analyzer, ['blue', 'red'], 
    ...                      plot_config=plot_config, layout_config=layout_config)
    >>> fig.show()
    """
    
    # Set defaults
    plot_config = plot_config or PlotConfig()
    axis_config = axis_config or AxisConfig()
    layout_config = layout_config or LayoutConfig()
    
    # Create a legend config that disables all fit-related features
    legend_config = LegendConfig(
        add_masses_to_legend=False,
        add_percentage_to_legend=False,
        add_labels=False,
        add_percentages=False
    )
    
    # Call plot_histograms_and_fits with legends_df=None to get histograms only
    return plot_histograms_and_fits(
        analyzer=analyzer,
        legends_df=None,  # No fitted traces, histograms only
        colors_hist=colors_hist,
        plot_config=plot_config,
        axis_config=axis_config,
        layout_config=layout_config,
        legend_config=legend_config
    )



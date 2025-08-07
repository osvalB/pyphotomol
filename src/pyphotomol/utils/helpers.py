import numpy  as np
import pandas as pd
import sys
from scipy.signal import find_peaks 
from scipy.optimize import curve_fit

__all__ = [
    'contrasts_to_mass',
    'create_histogram',
    'count_binding_events_from_masses',
    'count_binding_events_from_contrasts',
    'guess_peaks',
    'multi_gauss',
    'truncated_multi_gauss_with_baseline',
    'fit_histogram',
    'create_fit_table',
    'r_squared',
    'calibrate',
    'evaluate_fitting'
]

def contrasts_to_mass(contrasts, slope, intercept):
    """
    Function to convert masses from contrasts using known calibration parameters.

    Caution! slope and intercept are based on f(mass) = contrast !!!!
    In other words, contrast = slope*mass + intercept
    
    Parameters
    ----------
    contrasts : np.ndarray
        Contrasts to convert.
    slope : float
        Slope of the calibration line.
    intercept : float
        Intercept of the calibration line.
        
    Returns
    -------
    np.ndarray
        Converted masses in kDa.
    """
    
    intercept_inverse = -intercept / slope
    slope_inverse = 1 / slope

    masses_kDa = np.polyval(np.array([slope_inverse, intercept_inverse]), contrasts)

    return masses_kDa

def create_histogram(vector, window=[0,2000], bin_width=10):
    """
    Creates an histogram of the provided vector within a specified window and bin width.
    
    Parameters
    ----------
    vector : np.ndarray
        The data to create the histogram from.
    window : list, default [0, 2000]
        The range of values to include in the histogram [min, max].
    bin_width : float, default 10
        The width of each bin in the histogram.

    Returns
    -------
    histogram_centers : np.ndarray
        The x-coordinates of the histogram bins.
    hist_counts : np.ndarray
        The counts of values in each bin (Y-axis of the histogram).
    hist_nbins : int
        The number of bins in the histogram.
    
    Examples
    --------
    For contrast data:
    
    >>> centers, counts, nbins = create_histogram(contrasts, window=[-1, 0], bin_width=0.0004)
    
    For mass data:
    
    >>> centers, counts, nbins = create_histogram(masses, window=[0, 2000], bin_width=10)
    """

    # Determine number of bins based on bin_width
    nbins = (np.max(window) - np.min(window)) // bin_width
    nbins = int(nbins)

    # Create histogram
    hist_counts, hist_bins = np.histogram(vector, range=window, bins=nbins)
    histogram_centers = (hist_bins[1:] + hist_bins[:-1]) / 2.0
    
    # Write parameters to instance
    bin_width    = bin_width
    hist_nbins   = nbins

    return histogram_centers, hist_counts, hist_nbins

def count_binding_events_from_masses(masses_kDa):
    """
    Count binding events based on the provided mass data.
    
    Parameters
    ----------
    masses_kDa : np.ndarray
        Array of masses in kDa.
        
    Returns
    -------
    n_binding : int
        Number of binding events.
    n_unbinding : int
        Number of unbinding events.
    """
    
    # Count binding events
    n_binding = np.sum(masses_kDa > 0)
    
    # Count unbinding events
    n_unbinding = np.sum(masses_kDa < 0)

    return n_binding, n_unbinding

def count_binding_events_from_contrasts(contrasts):
    """
    Count binding events based on the provided contrast data.
    
    Parameters
    ----------
    contrasts : np.ndarray
        Array of contrasts.
        
    Returns
    -------
    n_binding : int
        Number of binding events (negative contrasts).
    n_unbinding : int
        Number of unbinding events (positive contrasts).
    """
    
    # Count binding events
    n_binding = np.sum(contrasts < 0)
    
    # Count unbinding events
    n_unbinding = np.sum(contrasts > 0)

    return n_binding, n_unbinding

def guess_peaks(x,histogram_centers,height=14, distance=4,prominence=8,masses=True):
    """
    Try to find peaks in the histogram data.
    
    Automatically finds peaks in histogram data with adaptive parameters based on
    the data range. For mass data, different distance thresholds are used for
    different mass ranges to account for peak spacing variations.
    
    Parameters
    ----------
    x : np.ndarray
        The histogram counts data to find peaks in.
    histogram_centers : np.ndarray
        The centers of the histogram bins corresponding to x values.
    height : int, default 14
        Minimum height of peaks.
    distance : int, default 4
        Minimum distance between peaks (will be scaled for different mass ranges).
    prominence : int, default 8
        Minimum prominence of peaks.
    masses : bool, default True
        If True, find peaks in mass data; if False, find peaks in contrast data.
    
    Returns
    -------
    np.ndarray
        Histogram centers of the found peaks.

    Examples
    --------
    For contrast data:
    
    >>> peaks = guess_peaks(hist_counts, hist_centers, height=10, distance=4, prominence=4, masses=False)
    
    For mass data:
    
    >>> peaks = guess_peaks(hist_counts, hist_centers, height=14, distance=4, prominence=8, masses=True)
    """

    # Heuristics logic to find peaks in different ranges of the histogram  - can be improved...
    # This is based on the assumption that the mass data has different characteristics in different ranges
    if masses:

        sel1     = histogram_centers < 650
        sel2     = np.logical_and(histogram_centers >= 650, histogram_centers  < 1500)
        sel3     = np.logical_and(histogram_centers >= 1500, histogram_centers < 10000)
        
        pks1     = find_peaks(x[sel1], height=height,   distance=distance,   prominence=prominence)[0]
        pks2     = find_peaks(x[sel2], height=height*2, distance=distance*3, prominence=prominence)[0]
        pks3     = find_peaks(x[sel3], height=height*5, distance=distance*8, prominence=prominence*2)[0]

        pks = ( pks1, pks2+len(x[sel1]), pks3 + len(x[sel1]) + len(x[sel2]) )
        pks = np.concatenate(pks)

    # Heuristic for contrasts - can be improved...
    else:

        sel1     = histogram_centers <   -0.1
        sel2     = histogram_centers >=  -0.1
        
        pks1     = find_peaks(x[sel1], height=height*3,   distance=distance*20,   prominence=prominence)[0]
        pks2     = find_peaks(x[sel2], height=height, distance=distance, prominence=prominence)[0]

        pks = ( pks1, pks2+len(x[sel1]))
        pks = np.concatenate(pks)

    return histogram_centers[pks]

def multi_gauss(x,*params):
    """
    Multiple gaussian function
    Inputs x values and gaussian parameters
    Outputs y values
    
    Parameters
    ----------
    x : np.ndarray
        The x-coordinates where the Gaussian function is evaluated.
    *params : tuple
        Parameters for the Gaussian functions, where each set of parameters consists of:
        - Center (mean) of the Gaussian
        - Amplitude (height) of the Gaussian
        - Standard deviation (width) of the Gaussian
        
    Returns
    -------
    np.ndarray
        The y-coordinates of the multiple-Gaussian function evaluated at x.
    
    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> params = [5, 1, 0.5, 7, 0.5, 0.2]  # Two Gaussians
    >>> y = multi_gauss(x, *params)
    """

    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        std = params[i+2]

        gaussian = amp*np.exp(-(x-ctr)**2 / (2*(std**2)))

        y = y + gaussian

    return y

def truncated_multi_gauss_with_baseline(x,lower_limit_of_resolution,baseline,*params):
    """
    Multiple gaussian function with a baseline and a lower limit of resolution.
    Inputs x values, gaussian parameters, lower limit of resolution, and baseline.
    Outputs y values.
    
    Parameters
    ----------
    x : np.ndarray
        The x-coordinates where the Gaussian function is evaluated.
    lower_limit_of_resolution : float
        The lower limit of resolution below which the function is zero
    baseline : float
        The baseline value to be added to the Gaussian function.
    *params : tuple
        Parameters for the Gaussian functions, where each set of parameters consists of:
        - Center (mean) of the Gaussian
        - Amplitude (height) of the Gaussian
        - Standard deviation (width) of the Gaussian
        
    Returns
    -------
    np.ndarray
        The y-coordinates of the multiple-Gaussian function evaluated at x,
        truncated below the lower limit of resolution and with a baseline added.
    """

    #lower_limit_of_resolution' is the lower limit of a mass that can be observed, i.e. 30 for refeyn model 1
    values_have_sense = np.greater(np.abs(x),lower_limit_of_resolution)
    y = multi_gauss(x,*params)
    y = y * values_have_sense

    return y + baseline

def fit_histogram(hist_counts, 
                  hist_centers,
                  guess_positions=[66,148,480], 
                  mean_tolerance=100, 
                  std_tolerance=200,
                  threshold=40,
                  baseline=0,
                  masses=True):
        
    """
    Fit a histogram with multiple truncated gaussians.
    
    Parameters
    ----------
    hist_counts : np.ndarray
        The counts of values in each bin of the histogram.
    hist_centers : np.ndarray
        The centers of the histogram bins.
    guess_positions : list, default [66,148,480]
        Initial guesses for the positions of the peaks.
    mean_tolerance : int, default 100
        Tolerance for the peak positions.
    std_tolerance : int, default 200
        Maximum standard deviation for the peaks.
    threshold : int, default 40 for masses in kDa units
        For masses, minimum value that can be observed. 
        For contrasts, it is be the max value that can be observed. It should be a negative value.
    baseline : float, default 0
        Baseline value to be added to the fit.
    masses : bool, default True
        If True, the fit is for mass data; if False, it is for contrast data.
        
    Returns
    -------
    popt : np.ndarray
        Optimized parameters for the fit.
    fit : np.ndarray
        Fitted values for the histogram. The first column is the x-coordinates, 
        followed by the individual Gaussian fits and the total fit.
    fit_error : np.ndarray
        Errors of the fitted parameters.

    Examples
    --------
    For contrast data:
    
    >>> popt, fit, errors = fit_histogram(counts, centers, mean_tolerance=0.05, std_tolerance=0.1)
    """

    fit          = None

    if len(guess_positions) == 0:
        return None
    
    else:

        # Get amplitude for each guess position
        guess_amp = []
        for pos in guess_positions:
            ind = np.argmin(np.abs(hist_centers - pos))
            guess_amp.append(hist_counts[ind])

        if masses:

            # Raise an error if the std_tolerance is not greater than 5
            if std_tolerance <= 5:
                raise ValueError("Standard deviation tolerance must be greater than 5 for mass data.")

            fit_guess    = np.column_stack((np.array(guess_positions), np.array(guess_amp), np.array([5]*len(guess_positions)))).flatten()
        else:

            # Raise an error if the std_tolerance is not greater than 0.0001
            if std_tolerance <= 0.0001:
                raise ValueError("Standard deviation tolerance must be greater than 0.0001 for contrast data.")

            # Convert guess positions to positive values for the fitting
            guess_positions = np.abs(guess_positions)
            fit_guess    = np.column_stack((np.array(guess_positions), np.array(guess_amp), np.array([0.0001]*len(guess_positions)))).flatten()
        
        lower_bounds = np.column_stack((np.array(guess_positions) - mean_tolerance , np.array([0]*len(guess_positions)), np.array([0]*len(guess_positions)))).flatten()
        upper_bounds = np.column_stack((np.array(guess_positions) + mean_tolerance , np.array([np.max(hist_counts)*1.3]*len(guess_positions)), np.array([std_tolerance]*len(guess_positions)))).flatten()

        bounds       = (tuple(lower_bounds), tuple(upper_bounds))
    
    if masses:

        values_have_sense = np.greater(np.abs(hist_centers),threshold) # We use np.abs to allow fitting negative masses also...

    else:

        # Raise an error if threshold is positive
        if threshold > 0:
            raise ValueError("Threshold for contrasts can not be positive. "
            "Set it to zero (no threshold) or to a negative value.")

        values_have_sense = np.less(hist_centers,threshold)

        # Convert contrasts to positive values for the fitting
        hist_centers = hist_centers * -1
        threshold    = threshold    * -1

    # Restrict data range for the fitting
    hist_counts = hist_counts * values_have_sense

    def fitting_helper_func(x,*params):
        return truncated_multi_gauss_with_baseline(x,threshold,baseline,*params) 

    func = fitting_helper_func

    # Do the fitting
    popt, pcov = curve_fit(func, hist_centers, hist_counts, p0=fit_guess, bounds=bounds)  #, method='dogbox', maxfev=1E5)
    # Create fit and individual gaussians for plotting
    # Finer grid
    x_grid = np.linspace(np.min(hist_centers), np.max(hist_centers), 800)
    single_gauss = []
    for i in range(0, len(popt), 3):
        ctr = popt[i]
        amp = popt[i+1]
        wid = popt[i+2]
        single_gauss.append(func(x_grid, ctr, amp, wid))

    # Sum of all
    fit_sum = func(x_grid, *popt)

    # Create one array for all
    fit = np.column_stack((x_grid, np.array(single_gauss).T, fit_sum))
    # Errors
    popt_error = np.sqrt(np.diag(pcov))

    # Convert to relative errors in percent
    popt_error = np.abs(np.round(popt_error / popt * 100, 2))

    # Evaluate the fitting and raise an error if the fitting is not valid
    evaluation = evaluate_fitting(popt, lower_bounds, upper_bounds)

    if not evaluation:
        raise ValueError("Fitting is not valid, because the fitted parameters are not within the specified bounds. Please check the initial guesses, or tolerances.")

    # If we fited the contrasts, we need to convert the means to negative values
    # and the fitted x-axis 
    if not masses:
        popt[::3] = -popt[::3] 
        # Convert the fit x-axis to negative values, leave the first column as is (x-coordinates)
        fit[:, 0] = fit[:, 0] * -1

    return popt, fit, popt_error

def create_fit_table(popt,popt_error,fit,
                     n_binding,n_unbinding,hist_centers,
                     masses=True,include_errors=True):

    """
    Generate a pandas DataFrame that summarizes fit results
    
    Parameters
    ----------
    popt : np.ndarray
        Optimized parameters from the fit.
    popt_error : np.ndarray
        Errors of the fitted parameters.
    fit : np.ndarray
        Fitted values for the histogram.
    n_binding : int
        Number of binding events.
    n_unbinding : int
        Number of unbinding events.
    hist_centers : np.ndarray
        The centers of the histogram bins.
    masses : bool, default True
        If True, the fit is for mass data; if False, it is for contrast data.
    include_errors : bool, default True
        If True, include errors in the fit table.
        
    Returns
    -------
    fit_table : pd.DataFrame
        DataFrame containing the fit results.
    """

    # Create lists with fitting parameters
    # These are later used to create a pandas DataFrame
    list_pos, list_sigma, list_ampl, list_counts = [], [], [], []

    # Create list for the fitted parameters errors
    list_pos_error, list_sigma_error, list_ampl_error = [], [], []

    # Loop over entries in optimized parameters
    for i in range(int(len(popt)/3)):

        list_pos.append(popt[3*i])
        list_ampl.append(popt[3*i+1])
        list_sigma.append(popt[3*i+2])

        list_pos_error.append(popt_error[3*i])
        list_sigma_error.append(popt_error[3*i+2])
        list_ampl_error.append(popt_error[3*i+1])

        # Calculate the area under the curve (AUC) for each Gaussian
        x = fit[:,0] * -1 if not masses else fit[:,0]  # Convert to negative values for contrasts

        auc = np.trapezoid(y=fit[:,i+1], x=x) / np.diff(hist_centers)[0]

        list_counts.append(auc)

    # Create Pandas Dataframe
    total_counts = n_binding 

    ## Add the counts from the unbinding data, if the user also fitted the unbinding data
    # Only add unbinding counts if we are dealing with masses
    if masses:

        has_negative_elements = any(element < 0 for element in list_pos)

        total_counts = total_counts + n_unbinding * has_negative_elements

    units = 'kDa' if masses else 'contrasts'

    fit_table = pd.DataFrame(
        data = {'Position / ' + units: list_pos,
               'Sigma / ' + units: list_sigma,
               'Counts' : list_counts,
               'Counts / %': np.round(np.array(list_counts)/total_counts*100),
               'Amplitudes' : list_ampl
               }
            )

    if include_errors:
        fit_table['Position Error / %' ] = list_pos_error
        fit_table['Sigma Error / %' ] = list_sigma_error
        fit_table['Amplitude Error / %'] = list_ampl_error

    return fit_table

def r_squared(data, fit):
    """
    Compute the R-squared value for the fit.
    
    Parameters
    ----------
    data : np.ndarray
        The original data.
    fit : np.ndarray
        The fitted values.
        
    Returns
    -------
    float
        The R-squared value.
    """
    mean_data = np.mean(data)
    ss_tot = np.sum(np.power(fit - mean_data, 2))
    ss_res = np.sum(np.power(data - fit, 2))
    return 1 - (ss_res/ss_tot)

def calibrate(calib_floats,fit_table):
    """ 
    Calibration based on contrasts histogram
    
    Parameters
    ----------
    calib_floats : list
        List of calibration standards in kDa (e.g. [66, 146, 480]).
    fit_table : pd.DataFrame
        DataFrame containing the fit results. Created by create_fit_table.
        
    Returns
    -------
    calibration_dic : dict
        Dictionary containing the calibration results:
        - 'standards': Calibration standards used.
        - 'exp_points': Expected points from the fit.
        - 'fit_params': Parameters of the fit.
        - 'fit_r2': R-squared value of the fit.
    """

    calib_stand = np.array(calib_floats)

    # Calibration points from fitting
    calib_points = fit_table['Position / contrasts']

    # First order polynomial fit
    params = np.polyfit(calib_stand, calib_points, 1) # x-axis are the masses (calib_stand), y-axis are the contrasts
    # returns slope, intercept
    # Calculate R2
    calc = np.polyval(params, calib_stand)
    r2   = r_squared(calib_points, calc)

    calibration_dic = {
        'standards':  calib_stand,
        'exp_points': calib_points,
        'fit_params': params,
        'fit_r2': r2}

    return calibration_dic

def evaluate_fitting(fitted_values,
                     low_bounds,
                     high_bounds,
                     tolerance=0.02):

    """
    Evaluate the quality of a constrained fitting by checking 
    if the fitted values are within the specified bounds.

    Parameters
    ----------
    fitted_values : np.ndarray
        The values obtained from the fitting process.
    low_bounds : np.ndarray
        The lower bounds for the fitted values.
    high_bounds : np.ndarray
        The upper bounds for the fitted values.

    Returns
    -------
    bool
        True if all fitted values are within the specified bounds
        False otherwise.
    """

    # Check if all fitted means are within the specified tolerance of the initial guess
    diff_to_lower = fitted_values - low_bounds
    diff_to_upper = high_bounds   - fitted_values

    # Find if the difference between the fitted and the boundaries is too small

    close_values1 = np.abs(diff_to_lower / fitted_values) < tolerance
    close_values2 = np.abs(diff_to_upper / high_bounds)   < tolerance

    # Check if any of the fitted values are within the bounds
    return np.logical_not(np.any(close_values1) or np.any(close_values2))



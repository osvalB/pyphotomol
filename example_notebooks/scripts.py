def display_figure_static(fig,format="png", width=800, height=600, show_interactive=False):

    """
    Display a Plotly figure in both interactive and static formats.
    
    This function is useful for Jupyter notebooks that need to render properly
    on GitHub, where interactive Plotly figures may not display.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object to display
    format : str, default "png"
        Image format for static display ("png", "svg", "jpeg")
    show_interactive : bool, default True
        Whether to also show the interactive version
        
    Examples
    --------
    >>> fig = plot_photomols_fit(pms, legends_df, colors_hist)
    >>> display_figure_static(fig, width=1000, height=700)
    """
    try:
        # Try to import display functionality
        from IPython.display import Image, display
        
        # Show interactive version first (for local Jupyter)
        if show_interactive:
            fig.show()
        
        # Display static version (for GitHub compatibility)
        static_image = fig.to_image(format=format, width=width, height=height,scale=2)
        display(Image(static_image))
        
    except ImportError:
        # Fallback to regular show if not in Jupyter
        print("IPython not available, showing interactive figure only")
        fig.show()
    except Exception as e:
        print(f"Error creating static image: {e}")
        print("Showing interactive figure only")
        fig.show()

def create_notebook_6_files(Kdim=8.35e-9,monomer_mass=80,concentrations_nM=[1,2,4,8,16,32,64]):

    """
    Generate synthetic mass photometry data files for testing.
    We simulate a simple 2M ⇌ D equilibrium system
    with varying total monomer concentrations and generate mass data.
    The data is saved as CSV files in the 'test_files' directory.
    Each file contains a column 'masses_kDa' with the simulated mass data.

    Parameters
    ----------
    Kdim : float, default 8.35e-9
        Dissociation constant in Molar units
    monomer_mass : float, default 80
        Mass of the monomer in kDa
    concentrations_nM : list of float, default [1,2,4,8,16,32,64]
        List of total monomer concentrations in nM
    """

    import numpy as np
    import os
    import pandas as pd

    Kd = Kdim  # Kd value from the paper in Molar units

    total_monomer_concentration_base = np.array(concentrations_nM)  # in nM - base values for filenames
    total_monomer_concentration = total_monomer_concentration_base * 1e-9  # convert to Molar units

    # Add ±3% random error in total_monomer_concentration
    error_factor = 1 + np.random.uniform(-0.03, 0.03, size=total_monomer_concentration.shape)
    total_monomer_concentration = total_monomer_concentration * error_factor

    def calculate_monomer_concentration(Kd, total_monomer_conc):
        """
        Calculate monomer concentration [M] for 2M ⇌ D equilibrium.

        Parameters:
            Kd (float): Dissociation constant (in M)
            total_monomer_conc (array-like): Total monomer concentrations [M]_total (in M)
            
        Returns:
            monomer_conc (np.array): Monomer concentrations [M] (in M)
        """
        free_monomer = (Kd / 4) * (np.sqrt(1 + (8 * total_monomer_conc / Kd)) - 1)

        return free_monomer

    # Generate a synthetic mass photometry dataset
    def simulate_mass_photometry_counts(mean,std,n):

        """
        Simulate mass photometry counts based on a normal distribution.
        """
        return np.random.normal(mean, std, n)

    # For each concentration, generate counts for monomers and dimers
    # The number of counts is proportional to the fraction of monomers and dimers

    def generate_mass_photometry_data(total_monomer_concentration, Kd, monomer_mass):

        masses = []

        monomer_conc = calculate_monomer_concentration(Kd, total_monomer_concentration)
        dimer_conc   = (total_monomer_concentration - monomer_conc) / 2

        assert total_monomer_concentration == monomer_conc + 2 * dimer_conc, "Total concentration should equal monomer + 2*dimer"

        monomer_fraction = monomer_conc  / (total_monomer_concentration)
        dimer_fraction   = dimer_conc*2  / (total_monomer_concentration)

        assert np.isclose(monomer_fraction + dimer_fraction, 1), f"Fractions should sum to 1. We got {monomer_fraction + dimer_fraction}"

        dimer_mass = 2 * monomer_mass  # Assuming dimer mass is twice the monomer mass

        monomer_std = 0.16 * monomer_mass
        dimer_std   = 0.16 * dimer_mass

        # Counts are directly proportional to the concentration
        total_counts   = 1800  # Total counts for each concentration
        monomer_counts = total_counts * monomer_conc / (monomer_conc + dimer_conc)
        dimer_counts   = total_counts * dimer_conc  / (monomer_conc + dimer_conc)
            
        # Simulate counts for monomers and dimers
        monomer_counts_i = simulate_mass_photometry_counts(monomer_mass, monomer_std, int(monomer_counts))
        dimer_counts_i   = simulate_mass_photometry_counts(dimer_mass, dimer_std, int(dimer_counts))

        # Combine the counts
        masses.extend(monomer_counts_i)
        masses.extend(dimer_counts_i)

        masses = np.array(masses)

        # Shuffle the masses to simulate random order
        np.random.shuffle(masses)

        return masses

    masses_per_concentration = []
    concentration_labels = []
    
    for i, mc in enumerate(total_monomer_concentration):
        
        # Generate mass data for each concentration
        mass_data = generate_mass_photometry_data(mc, Kd, monomer_mass)
        masses_per_concentration.append(mass_data)
        
        # Use base concentration (without error) for filename (more readable and predictable)
        conc_nM = total_monomer_concentration_base[i]  # Use base value without error
        concentration_labels.append(f"{conc_nM}nM")

    # Export in the test_files folder - as csv with column called 'masses_Kda'
    test_files = 'test_files'
    if not os.path.exists(test_files):
        os.makedirs(test_files)


    # Remove all files with the pattern 'masses_monomer_*.csv' before generating new ones
    for file in os.listdir(test_files):
        if file.startswith('masses_monomer_') and file.endswith('.csv'):
            os.remove(os.path.join(test_files, file))

    for i, (mass_data, conc_label) in enumerate(zip(masses_per_concentration, concentration_labels)):
        df = pd.DataFrame(mass_data, columns=['masses_kDa'])

        df.to_csv(f'{test_files}/masses_monomer_{conc_label}.csv', index=False)
        print(f"Generated file: {test_files}/masses_monomer_{conc_label}.csv")

    return None

def create_notebook_7_files(
        mass_A=150, 
        mass_B=30, 
        Kd=1e-9,
        A_concentration_molar=5e-9,
        B_concentrations_nM=[0.125, 0.25, 0.5, 1, 2, 4, 8],
        detection_limit=40):

    """
    Generate synthetic mass photometry data files for testing.
    We simulate a simple A+B ⇌ AB equilibrium system (complex formation)
    with varying total B concentrations and generate mass data.
    We assume that B is under the limit of detection, so we only see A and AB.
    The data is saved as CSV files in the 'test_files' directory.
    Each file contains a column 'masses_kDa' with the simulated mass data.

    Parameters
    ----------
    mass_A : float, default 30
        Mass of the A protein in kDa
    mass_B : float, default 150
        Mass of the B ligand in kDa
    Kd : float, default 1.5e-9
        Equilibrium dissociation constant in Molar units
    A_concentration_molar : float, default 5e-9
        Concentration of A in Molar units
    detection_limit : float, default 40
        Detection limit for mass photometry in kDa
    """
    
    import numpy as np
    import os
    import pandas as pd

    def calculate_complex_concentration(Kd, total_A_conc, total_B_conc):
        """
        Calculate complex concentration [M] for A + B ⇌ AB equilibrium.

        Parameters:
            Kd (float): Dissociation constant (in M)
            total_A_conc (array-like): Total A concentrations [M]_total (in M)
            total_B_conc (array-like): Total B concentrations [M]_total (in M)
            
        Returns:
            complex_conc (np.array): Complex concentrations [M] (in M)
        """
        # Using the formula for complex formation

        AB = 0.5* ( (Kd + total_A_conc + total_B_conc) - np.sqrt((Kd + total_A_conc + total_B_conc)**2 - 4 * total_A_conc * total_B_conc))

        return AB 
    
    B_concentrations_nM = np.array(B_concentrations_nM)  # in nM - base values for filenames

    B_concentrations_molar = B_concentrations_nM.copy()*1e-9  # Working concentrations

    # Add ±2% random error in B_concentrations
    error_factor = 1 + np.random.uniform(-0.02, 0.02, size=B_concentrations_molar.shape)
    B_concentrations_molar = B_concentrations_molar * error_factor

    # Add ±2% random error in A_concentration
    error_factor_A = 1 + np.random.uniform(-0.02, 0.02)
    A_concentration_molar = A_concentration_molar * error_factor_A

    # Generate a synthetic mass photometry dataset
    def simulate_mass_photometry_counts(mean, std, n):
        """
        Simulate mass photometry counts based on a normal distribution.
        """
        return np.random.normal(mean, std, n)
    
    # For each B concentration, generate counts for A, B and AB
    # The counts for B will be filtered out later 

    def generate_mass_photometry_data(A_conc, B_conc, Kd, mass_A, mass_B):
        """
        Generate mass photometry data for given A and B concentrations.
        """
        complex_conc = calculate_complex_concentration(Kd, A_conc, B_conc)
        
        # Calculate fractions
        A_free = A_conc - complex_conc
        B_free = B_conc - complex_conc
        free_A_fraction = A_free / (A_free + complex_conc)
        bound_A_fraction = complex_conc / A_conc
       
        free_B_fraction  = B_free / B_conc
        bound_B_fraction = complex_conc / B_conc

        # Assert the sum of concentrations from A_free and complex_conc equals total A concentration
        assert np.isclose(A_free + complex_conc, A_conc), f"A free and complex concentration should sum to total A concentration. We got {A_free + complex_conc} != {A_conc}"

        assert np.isclose(free_A_fraction + bound_A_fraction, 1), f"Fractions should sum to 1. We got {free_A_fraction + bound_A_fraction}"
        
        # Check the sum of B_fraction and AB_fraction
        assert np.isclose(free_B_fraction + bound_B_fraction, 1), f"B fraction and AB fraction should sum to 1. We got {free_B_fraction + bound_B_fraction}"

        # Simulate counts for A, B and AB - counts are directly proportional to the concentration
        total_counts = 3200  # Total counts for each concentration
        A_counts     = total_counts * A_free / (A_free + complex_conc + B_free)
        AB_counts    = total_counts * complex_conc / (A_free + complex_conc + B_free)
        B_counts     = total_counts * B_free / (A_free + complex_conc + B_free)

        std_factor = 0.08  # Standard deviation factor for mass photometry

        A_std = std_factor * mass_A
        AB_std = std_factor * (mass_A + mass_B)  # Assuming AB mass is the sum of A and B masses
        B_std = std_factor * mass_B

        # Simulate counts for A, B and AB
        A_counts = simulate_mass_photometry_counts(mass_A, A_std, int(A_counts))
        AB_counts = simulate_mass_photometry_counts(mass_A + mass_B, AB_std, int(AB_counts))
        B_counts = simulate_mass_photometry_counts(mass_B, B_std, int(B_counts))

        A_counts = A_counts[A_counts >= detection_limit]
        AB_counts = AB_counts[AB_counts >= detection_limit]
        B_counts = B_counts[B_counts >= detection_limit]

        masses = np.concatenate([A_counts,B_counts,AB_counts])

        # Shuffle the masses to simulate random order
        np.random.shuffle(masses)

        return masses
    
    masses_per_concentration = []
    concentration_labels = []

    for i, B_conc in enumerate(B_concentrations_molar):
        # Generate mass data for each B concentration
        masses = generate_mass_photometry_data(A_concentration_molar, B_conc, Kd, mass_A, mass_B)
        masses_per_concentration.append(masses)
        
        # Use base concentration (without error) for filename (more readable and predictable)
        B_conc_nM = B_concentrations_nM[i]  # Use base value without error
        concentration_labels.append(f"B_{B_conc_nM:.3f}nM")

    # Export in the test_files folder - as csv with column called 'masses_kDa'
    test_files = 'test_files'
    if not os.path.exists(test_files):
        os.makedirs(test_files)

    # Remove all files with the pattern 'masses_A_B*.csv' before generating new ones
    for file in os.listdir(test_files):
        if file.startswith('masses_A_B') and file.endswith('.csv'):
            os.remove(os.path.join(test_files, file))

    for i, (mass_data, conc_label) in enumerate(zip(masses_per_concentration, concentration_labels)):
        df = pd.DataFrame(mass_data, columns=['masses_kDa'])

        df.to_csv(f'{test_files}/masses_A_{conc_label}.csv', index=False)
        print(f"Generated file: {test_files}/masses_A_{conc_label}.csv")

    return None

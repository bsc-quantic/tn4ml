import os
import argparse
from typing import Collection, List, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import auc

from tn4ml.eval import *

from utils import *


def load_anomaly_scores(signal_name: str, 
                        initializers_strings: Collection[str],
                        latent_spaces: Collection[int],
                        bond_dim: dict,
                        embedding: str,
                        nruns: int,
                        save_dir: str,
                        train_scaling: str,
                        ):
    """
    Load anomaly scores for different initializers and bond dimensions.

    Parameters
    ----------
    signal_name : str
        Name of the signal for which anomaly scores are obtained
    initializers_strings : Collection[str]
        List of initializer names
    latent_spaces : Collection[int]
        List of latent space dimensions to compare
    bond_dim : dict
        Dictionary containing bond dimensions for each latent space - e.g. {'4': [2, 4], '8': [2, 4, 8]}
    embedding : str
        Embedding name for the model
    nruns : int
        Number of runs to average over
    save_dir : str
        Directory to save the plot
    train_scaling : str
        Training size and scaling to consider
    
    Returns
    -------
    tpr_per_init : dict
        Dictionary containing true positive rates for each initializer and bond dimension
    tpr_per_init_err : dict
        Dictionary containing statistical errors for true positive rates - from multiple runs
    fpr_per_init : dict
        Dictionary containing false positive rates for each initializer and bond dimension
    fpr_per_init_err : dict
        Dictionary containing statistical errors for false positive rates - from multiple runs
    auc_per_init : dict
        Dictionary containing area under the curve values for each initializer and bond dimension
    auc_per_init_err : dict
        Dictionary containing statistical errors for area under the curve values - from multiple runs
    fpr_per_tpr_8_per_init : dict
        Dictionary containing false positive rates for TPR = 0.8 for each initializer and bond dimension
    fpr_per_tpr_8_per_init_err : dict
        Dictionary containing statistical errors for false positive rates for TPR = 0.8 - from multiple runs
    fpr_per_tpr_6_per_init : dict
        Dictionary containing false positive rates for TPR = 0.6 for each initializer and bond dimension
    fpr_per_tpr_6_per_init_err : dict
        Dictionary containing statistical errors for false positive rates for TPR = 0.6 - from multiple runs
    """

    # Initialize dictionaries to store results
    tpr_per_init = {}; tpr_per_init_err = {}
    fpr_per_init = {}; fpr_per_init_err = {}
    auc_per_init = {}; auc_per_init_err = {}
    fpr_per_tpr_8_per_init = {}; fpr_per_tpr_8_per_init_err = {}
    fpr_per_tpr_6_per_init = {}; fpr_per_tpr_6_per_init_err = {}

    for initializer in initializers_strings:
        for i, lat in enumerate(latent_spaces):
            for j, bond in enumerate(bond_dim[str(lat)]):
                loss_qcd_runs=[]; loss_sig_runs=[]
                tpr_data = []; fpr_data = []; auc_data = []
                fpr_per_tpr_8_data = []; fpr_per_tpr_6_data = []
                for run in range(1, nruns+1):
                    path = f'{save_dir}/{initializer}/{train_scaling}/{embedding}/lat{lat}/bond{bond}/run_{run}/fidelity_scores_{signal_name}.h5'
                    if not os.path.exists(path):
                        continue
                    else:
                        with h5py.File(path, 'r') as file:
                            loss_qcd = file['loss_qcd'][:]
                            loss_sig = file['loss_sig'][:]
                            
                            loss_qcd = np.power(loss_qcd, 2)
                            loss_sig = np.power(loss_sig, 2)
                            
                            loss_qcd_runs.append(loss_qcd)
                            loss_sig_runs.append(loss_sig)

                            fpr, tpr = get_roc_curve_data(loss_sig, loss_qcd, anomaly_det=True)
                            tpr_data.append(tpr)
                            fpr_data.append(fpr)
                            # Get auc
                            auc_value = auc(fpr, tpr)
                            auc_data.append(auc_value)

                            # Get fpr per tpr = {0.8, 0.6}
                            fpr_per_tpr_8 = get_FPR_for_fixed_TPR(0.8, np.array(fpr), np.array(tpr), tolerance=0.01)
                            fpr_per_tpr_6 = get_FPR_for_fixed_TPR(0.6, np.array(fpr), np.array(tpr), tolerance=0.01)
                            fpr_per_tpr_8_data.append(fpr_per_tpr_8)
                            fpr_per_tpr_6_data.append(fpr_per_tpr_6)

                loss_qcd = get_mean_and_error(np.array(loss_qcd_runs))
                loss_sig = get_mean_and_error(np.array(loss_sig_runs))

                if np.isnan(loss_qcd[0]).sum() > 0:
                    print(f'{Colors.RED.value}{path}: NaNs{Colors.RESET.value}')
                    continue

                # Get mean error for tpr, fpr
                tpr_mean_error = get_mean_and_error(np.array(tpr_data))
                tpr_per_init[f'init={initializer},bond={bond},lat={lat},s={signal_name}'] = tpr_mean_error[0]
                tpr_per_init_err[f'init={initializer},bond={bond},lat={lat},s={signal_name}'] = tpr_mean_error[1]

                fpr_mean_error = get_mean_and_error(1./np.array(fpr_data))
                fpr_per_init[f'init={initializer},bond={bond},lat={lat},s={signal_name}'] = fpr_mean_error[0]
                fpr_per_init_err[f'init={initializer},bond={bond},lat={lat},s={signal_name}'] = fpr_mean_error[1]

                # AUC mean error
                auc_mean_error = get_mean_and_error(np.array(auc_data))
                auc_per_init[f'init={initializer},bond={bond},lat={lat},s={signal_name}'] = auc_mean_error[0]
                auc_per_init_err[f'init={initializer},bond={bond},lat={lat},s={signal_name}'] = auc_mean_error[1]

                # fpr per tpr = 0.8
                fpr_per_tpr_8_mean_error = get_mean_and_error(1./np.array(fpr_per_tpr_8_data))
                fpr_per_tpr_8_per_init[f'init={initializer},bond={bond},lat={lat},s={signal_name}'] = fpr_per_tpr_8_mean_error[0]
                fpr_per_tpr_8_per_init_err[f'init={initializer},bond={bond},lat={lat},s={signal_name}'] = fpr_per_tpr_8_mean_error[1]

                # fpr per tpr = 0.6
                fpr_per_tpr_6_mean_error = get_mean_and_error(1./np.array(fpr_per_tpr_6_data))
                fpr_per_tpr_6_per_init[f'init={initializer},bond={bond},lat={lat},s={signal_name}'] = fpr_per_tpr_6_mean_error[0]
                fpr_per_tpr_6_per_init_err[f'init={initializer},bond={bond},lat={lat},s={signal_name}'] = fpr_per_tpr_6_mean_error[1]
                    
    return tpr_per_init, tpr_per_init_err, \
            fpr_per_init, fpr_per_init_err, \
            auc_per_init, auc_per_init_err, \
            fpr_per_tpr_8_per_init, fpr_per_tpr_8_per_init_err,\
            fpr_per_tpr_6_per_init, fpr_per_tpr_6_per_init_err


def plot_losses_per_initializer(latent: int, 
                                bond_dims: Collection[int],
                                initializers_strings: Collection[str],
                                embedding: str,
                                save_dir: str,
                                N_epochs: int = 1000,
                                train_size: str = "10k",
                                minmax: str = "minmax-11",
                                nruns: int = 5):
    """Create a subplot grid with training loss plots for all initializers for fixed embedding.

    Parameters
    ----------
    latent : int
        Latent space dimension
    bond_dims : Collection[int]
        List of bond dimensions to compare
    initializers_strings : Collection[str]
        List of initializer names to compare
    embedding : str
        Embedding name for the model
    save_dir : str
        Directory to save the plot
    N_epochs : int, optional
        Number of epochs for training, default 1000
    train_size : str, optional
        Training size to consider, default "10k"
    minmax : str, optional
        Min-Max scaling option, default "minmax-11"
    nruns : int, optional
        Number of runs to average over, default 5
    
    Returns
    -------
    None

    Notes
    -----
    - The function creates a grid of subplots, each showing the training loss for a different initializer.
    - Each subplot contains multiple lines representing different bond dimensions - corresponding to different colors in a color palette.
    - The function is assuming the data files are structured in a specific way, but you can change it to fit your needs.


    Example
    -------
    >>> plot_losses_per_initializer(
    ...     latent = 4,
    ...     bond_dims = [2, 4, 8, 16],
    ...     initializers_strings = ['unitary_canonize', 'randn_std', 'randn_1e-2', 'gramschmidt_n_std'],
    ...     embedding = 'laguerre_2',
    ...     save_dir = './results',
    ...     N = 1000,
    ...     train_size = '10k',
    ...     minmax = 'minmax-11',
    ...     nruns = 10
    ... )

    """
    
    # Calculate grid dimensions
    n_initializers = len(initializers_strings)
    n_cols = min(2, n_initializers)  # Maximum 4 columns
    n_rows = (n_initializers + n_cols - 1) // n_cols  # Ceiling division
    
    # Color maps for different bond dimensions
    color_maps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlOrBr', 'GnBu', 'PuRd']
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Flatten axes array for easy indexing if multiple rows/columns
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Convert to list for single subplot case
    
    # Plot for each initializer
    for i, initializer in enumerate(initializers_strings):
        ax = axes[i]
        
        # Plot for each bond dimension
        for j, bond in enumerate(bond_dims):
            # Create color spectrum for this bond dimension
            cmap = plt.cm.get_cmap(color_maps[j % len(color_maps)])
            
            # Process each run
            for run in range(1, nruns+1):
                # Load loss data
                file_path = f'{save_dir}/{initializer}/{train_size}_{minmax}/{embedding}/lat{latent}/bond{bond}/run_{run}/loss.npy'
                
                try:
                    # Load and pad with NaN
                    loss_data = np.load(file_path)
                    loss = np.full(N_epochs, np.nan)
                    loss[:len(loss_data)] = loss_data
                    
                    # Plot with color from the spectrum
                    color_intensity = 0.3 + 0.6 * (run/nruns)
                    color = cmap(color_intensity)
                    
                    # Plot data
                    epochs = np.arange(1, N_epochs+1)
                    ax.plot(epochs, loss, '-', color=color, linewidth=3, alpha=0.7)
                    
                except FileNotFoundError:
                    print(f"{Colors.RED.value}File not found: {file_path}{Colors.RESET.value}")
                    continue
            
            # Add label for this bond dimension (just once per bond)
            ax.plot([], [], '-', color=cmap(0.5), label=f'$\chi$ = {bond}')
        
        # Style this subplot
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('Epochs', fontsize=15)
        ax.set_ylabel('Training Loss', fontsize=15)

        # Create a label for the initializer
        if initializer == 'gramschmidt_n_std':
            initializer = f'Gram-Schmidt (normal - std=0.1667)'
        elif initializer == 'randn_std':
            initializer = 'Random (normal - std=0.1667)'
        elif initializer == 'randn_1e-2':
            initializer = 'Random (normal - std=0.01)'
        elif initializer == 'unitary':
            initializer = 'Unitary'

        ax.set_title(f'{initializer}', fontsize=17)
        ax.set_xlim(0, N_epochs)
        
        # set font size for ticks
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
    
    axes[0].legend(loc='upper right', frameon=True, fontsize=14)
    # Hide any unused subplots
    for j in range(i+1, n_rows*n_cols):
        if j < len(axes):
            axes[j].axis('off')
    
    # Hide y-axis labels for all but the first column
    for ax in axes:
        ax.label_outer()
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(f'{save_dir}/plots/{train_size}_{minmax}/{embedding}/lat{latent}/', exist_ok=True)

    plt.savefig(f'{save_dir}/plots/{train_size}_{minmax}/{embedding}/lat{latent}/train_loss.pdf')


def compare_anomaly_scores_per_embedding(embeddings: Collection[str], 
                                        initializer: str,
                                        latent: int,
                                        bond_dims: Collection[int], 
                                        signal_name: str,
                                        save_dir: str,
                                        train_size_scaling: Collection[str],
                                        nruns: int = 10):
    """
    Create a single plot with subplots for different embeddings, showing QCD and BSM 
    anomaly scores distributions for each bond dimension with consistent coloring.
    
    Parameters
    ----------
    embeddings : Collection[str]
        List of embedding names to compare
    initializer : str
        Initializer name for the model
    latent : int
        Latent space dimension
    bond_dims : Collection[int]
        List of bond dimensions to compare
    signal_name : str
        Name of the signal to be used in the plot
    save_dir : str
        Directory to save the plot
    train_size_scaling : Collection[str]
        List of training sizes and scaling to consider for each embedding
    nruns : int, optional
        Number of runs to average over, default 10
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    axs : List[matplotlib.axes.Axes]
        List of axes objects for each subplot

    Notes
    -----
    - The function assumes that the data files are structured in a specific way,
        with paths containing the bond dimension and run number.
    - The function uses a consistent color scheme for different bond dimensions across all subplots.
    - The function handles multiple runs by averaging the results and plotting the distributions.
    

    Example
    -------
    >>> compare_anomaly_scores_per_embedding(
    ...     embeddings = ['laguerre_2', 'legendre_2', 'hermite_2'],
    ...     initializer = 'unitary',
    ...     latent = 4,
    ...     bond_dims = [2, 4, 8, 16],
    ...     signal_name = 'AtoHZ_to_ZZZ_35',
    ...     save_dir = './results',
    ...     train_size_scaling = ['10k_minmax01', '10k_minmax-11', '10k_minmax-11'],
    ...     nruns = 10
    ... )
    """
    
    # Set up color maps for different bond dimensions - using brighter colors
    color_maps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlOrBr', 'GnBu', 'PuRd']
    
    # Create figure with subplots (one per embedding)
    fig, axs = plt.subplots(1, len(embeddings), figsize=(6*len(embeddings), 5), sharey=True)
    
    # Handle case of single embedding
    if len(embeddings) == 1:
        axs = [axs]
    
    # Create a color dictionary with brighter colors (using 0.8 instead of 0.6)
    bond_colors = {bond: plt.cm.get_cmap(color_maps[i % len(color_maps)])(0.8) 
                   for i, bond in enumerate(bond_dims)}
    
    # Keep track of handles for the bond dimension legend
    bond_handles = []
    
    # Process each embedding
    for e, embedding in enumerate(embeddings):
        train_scaling = train_size_scaling[e]
        ax = axs[e]
        
        # Process each bond dimension
        for bond in bond_dims:
            loss_qcd_runs = []
            loss_sig_runs = []
            
            # Process multiple runs
            for run in range(1, nruns+1):
                path = f'{save_dir}/{initializer}/{train_scaling}/{embedding}/lat{latent}/bond{bond}/run_{run}/fidelity_scores_{signal_name}.h5'
                
                if not os.path.exists(path):
                    continue
                
                # Load and process data
                with h5py.File(path, 'r') as file:
                    loss_qcd = file['loss_qcd'][:]
                    loss_sig = file['loss_sig'][:]
                    
                    # Square the losses (as in original code)
                    loss_qcd = np.power(loss_qcd, 2)
                    loss_sig = np.power(loss_sig, 2)
                    
                    loss_qcd_runs.append(loss_qcd)
                    loss_sig_runs.append(loss_sig)
            
            # Skip if no valid runs found
            if not loss_qcd_runs or not loss_sig_runs:
                continue
                
            # Compute mean and error
            loss_qcd = get_mean_and_error(np.array(loss_qcd_runs))
            loss_sig = get_mean_and_error(np.array(loss_sig_runs))
            
            # Skip if NaNs detected
            if np.isnan(loss_qcd[0]).sum() > 0:
                print(f'{Colors.RED.value}NaNs detected for {embedding}, bond={bond}{Colors.RESET.value}')
                continue
            
            # Plot QCD distribution
            qcd_line = ax.hist(loss_qcd[0], bins=100, fill=False, 
                              color=bond_colors[bond], histtype='step',
                              linewidth=2, alpha=1.0, density=True)
            
            # Plot BSM distribution
            bsm_line = ax.hist(loss_sig[0], bins=100, fill=True, 
                              color=bond_colors[bond], histtype='step',
                              linewidth=2, alpha=0.4, linestyle='--', density=True)
            
            # Save handles for bond dimension legend (only once per bond dimension)
            if e == 0:
                bond_handles.append(plt.Line2D([0], [0], color=bond_colors[bond], linewidth=2, 
                                          label=f'χ = {bond}'))
        
        # Style the subplot
        if embedding == 'legendre_2':
            embedding_label = r'Legendre'
        elif embedding == 'fourier_2':
            embedding_label = r'Fourier'
        elif embedding == 'hermite_2':
            embedding_label = r'Hermite'
        elif embedding == 'laguerre_2_old_2':
            embedding_label = r'Laguerre'
        else:
            embedding_label = embedding
            
        ax.set_title(f'{embedding_label}', fontsize=25)
        ax.set_xlabel(r'Anomaly Score', fontsize=20)
        ax.set_yscale('log')
        ax.set_xlim(-0.01, 0.6)
        
        # Only add y-label to first subplot
        if e == 0:
            ax.set_ylabel('Probability Density', fontsize=20)
    
    # Create style legend handles that accurately represent the visualization
    style_handles = [
        mpatches.Patch(edgecolor='black', facecolor='none', linewidth=2, label='QCD (Background)'),
        mpatches.Patch(facecolor='gray', alpha=0.4, label='BSM (Signal)')
    ]
    # set tick size
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
    
    # Add bond dimension legend at the figure level
    fig.legend(handles=bond_handles, loc='upper left', bbox_to_anchor=(0.05, 0.87), 
               frameon=True, fontsize=10, title="Bond Dimension")
    
    # Add the QCD/BSM legend to the figure
    fig.legend(handles=style_handles, loc='upper right', bbox_to_anchor=(0.35, 0.87), 
               frameon=True, fontsize=10, title="Distribution Type")
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for the legends
    
    # Create save directory if it doesn't exist
    os.makedirs(f'{save_dir}/plots/comparisons/', exist_ok=True)
    
    # Save figure
    plt.savefig(f'{save_dir}/plots/comparisons/embedding_comparison_{initializer}_lat{latent}_{signal_name}.pdf',
               bbox_inches='tight')
    
    return fig, axs

def compare_ROCs_per_bond(latent: int,
                          bond_dim: Collection[int], 
                          initializer: str,
                          initializer_string: str, 
                          tpr_per_init: dict, tpr_per_init_err: dict, 
                          fpr_per_init: dict, fpr_per_init_err: dict, 
                          auc_per_init: dict, auc_per_init_err: dict,
                          save_dir: str,
                          embedding: str,
                          train_scaling: str, 
                          signal_name: str):
    """
    Create ROC curve plot for a single initializer with consistent coloring for bond dimensions.
    
    Parameters
    ----------
    latent : int
        Latent space dimension
    bond_dim : Collection[int]
        List of bond dimensions to compare
    initializer : str
        Initializer of the model
    initializer_string : str
        String representation of the initializer
    tpr_per_init : dict
        Dictionary containing true positive rates for each bond dimension
    tpr_per_init_err : dict
        Dictionary containing statistical errors for true positive rates - from multiple runs
    fpr_per_init : dict
        Dictionary containing false positive rates for each bond dimension
    fpr_per_init_err : dict
        Dictionary containing statistical errors for false positive rates - from multiple runs
    auc_per_init : dict
        Dictionary containing area under the curve values for each bond dimension
    auc_per_init_err : dict
        Dictionary containing statistical errors for area under the curve values - from multiple runs
    save_dir : str
        Directory to save the plot
    embedding : str
        Embedding name for the model
    train_scaling : str
        Training size and scaling to consider
    signal_name : str
        Name of the signal to be used in the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    ax : matplotlib.axes.Axes
        The axes object for the plot
    """
    # Set up color maps for different bond dimensions - using brighter colors
    color_maps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlOrBr', 'GnBu', 'PuRd']
    
    # Create a color dictionary with brighter colors for consistent coloring
    bond_colors = {bond: plt.cm.get_cmap(color_maps[i % len(color_maps)])(0.8) 
                   for i, bond in enumerate(bond_dim)}
    
    # Create a single figure
    fig, ax = plt.subplots(figsize=(7, 7))
    
    for j, bond in enumerate(bond_dim):
        key = f'init={initializer},bond={bond},lat={latent},s={signal_name}'
        
        if key not in tpr_per_init.keys():
            print(f'{Colors.RED.value}{key} not found{Colors.RESET.value}')
            continue
            
        tpr = tpr_per_init[key]
        #tpr_err = tpr_per_init_err[key]
        fpr = fpr_per_init[key]
        fpr_err = fpr_per_init_err[key]
        auc_value = auc_per_init[key]
        auc_err = auc_per_init_err[key]
            
        if signal_name == 'RSGraviton_WW_NA_35':  # uncertainties are bigger for G_NA
            band_ind = np.where(tpr > 0.6)[0]
        else:
            band_ind = np.where(tpr > 0.35)[0]

        # Use consistent color based on bond dimension
        ax.plot(tpr, fpr, 
              label=r'$\chi$ = %s (%.2f)$\pm$(%.2f)'% (bond, auc_value*100., auc_err*100.), 
              linewidth=2, 
              color=bond_colors[bond])  # Use consistent color

        # Error calculation in log space
        log_fpr = np.log10(fpr)
        rel_err = fpr_err/fpr  # Relative error
        log_err = (0.434) * rel_err  # Convert to log10 error (0.434 = 1/ln(10))

        #Calculate bounds in log space, then convert back
        log_upper = log_fpr - log_err
        log_lower = log_fpr + log_err
        fpr_upper = 10**log_upper # convert back to linear scale
        fpr_lower = 10**log_lower # convert back to linear scale
        
        # Add error bands with matching color
        ax.fill_between(tpr[band_ind], fpr_lower[band_ind], fpr_upper[band_ind], 
                      alpha=0.2, color=bond_colors[bond])  # Match fill color
    
    # Add vertical dotted lines at TPR = 0.6 and TPR = 0.8
    ax.axvline(x=0.6, color='black', linestyle=':', linewidth=1.5, alpha=0.3)
    ax.axvline(x=0.8, color='black', linestyle=':', linewidth=1.5, alpha=0.3)
    
    # Style the plot
    #ax.set_title(f'Latent = {latent}, {initializer_string}', fontsize=16)
    ax.legend(loc='lower left', fontsize=12)
    ax.set_yscale('log')
    ax.set_xlabel('TPR', fontsize=20)
    ax.set_ylabel('FPR$^{-1}$', fontsize=20)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(f'{save_dir}/plots/{train_scaling}/{embedding}/lat{latent}', exist_ok=True)
    
    # Save figure
    plt.savefig(f'{save_dir}/plots/{train_scaling}/{embedding}/lat{latent}/roc_curve_{signal_name}.pdf')
    
    return fig, ax


def compare_ROC_by_signal(signal_names: Collection[str],
                          signal_labels: Collection[str],
                          latent: int,
                          bond_dim: Collection[int], 
                          initializer: str,
                          initializer_string: str, 
                          tpr_per_init: dict, tpr_per_init_err: dict, 
                          fpr_per_init: dict, fpr_per_init_err: dict, 
                          auc_per_init: dict, auc_per_init_err: dict,
                          save_dir: str,
                          embedding: str,
                          train_scaling: str):
    """
    Compare ROC curves for different signal types with fixed model parameters.

    Parameters
    ----------
    signal_names : Collection[str]
        List of signal names to compare
    signal_labels : Collection[str]
        List of signal labels for the legend
    latent : int
        Latent space dimension
    bond_dim : Collection[int]
        List of bond dimensions to compare
    initializer : str
        Initializer of the model
    initializer_string : str
        String representation of the initializer
    tpr_per_init : dict
        Dictionary containing true positive rates for each signal
    tpr_per_init_err : dict
        Dictionary containing statistical errors for true positive rates - from multiple runs
    fpr_per_init : dict
        Dictionary containing false positive rates for each signal
    fpr_per_init_err : dict
        Dictionary containing statistical errors for false positive rates - from multiple runs
    auc_per_init : dict
        Dictionary containing area under the curve values for each signal
    auc_per_init_err : dict
        Dictionary containing statistical errors for area under the curve values - from multiple runs
    save_dir : str
        Directory to save the plot
    embedding : str
        Embedding name for the model
    train_scaling : str
        Training size and scaling to consider
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    ax : matplotlib.axes.Axes
        The axes object for the plot
    
    Example
    -------
    >>> compare_ROC_by_signal(
    ...     signal_names = ['RSGraviton_WW_NA_35', 'AtoHZ_to_ZZZ_35', 'RSGraviton_WW_BR_15'],
    ...     signal_labels = [r'Narrow $G \rightarrow WW$', r'$A \rightarrow HZ \rightarrow ZZZ$', r'Broad $G \rightarrow WW$'],
    ...     latent = 4,
    ...     bond_dim = [2, 4, 8, 16],
    ...     initializer = 'unitary',
    ...     initializer_string = 'Unitary',
    ...     tpr_per_init = tpr_per_init,
    ...     tpr_per_init_err = tpr_per_init_err,
    ...     fpr_per_init = fpr_per_init,
    ...     fpr_per_init_err = fpr_per_init_err,
    ...     auc_per_init = auc_per_init,
    ...     auc_per_init_err = auc_per_init_err,
    ...     save_dir = './results',
    ...     embedding = 'laguerre_2',
    ...     train_scaling = '10k_minmax-11'
    ... )

    """
    palette = ['#4CA64C', '#FF5733', '#8A2BE2']

    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Store handles and data for legend
    handles = []
    auc_info = []
    signal_info = []
    
    for i, signal_name in enumerate(signal_names):
        key = f'init={initializer},bond={bond_dim},lat={latent},s={signal_name}'
        
        if key not in tpr_per_init.keys():
            print(f'{Colors.RED.value}{key} not found{Colors.RESET.value}')
            continue
            
        tpr = tpr_per_init[key]
        tpr_err = tpr_per_init_err[key]
        fpr = fpr_per_init[key]
        fpr_err = fpr_per_init_err[key]
        auc_value = auc_per_init[key]
        auc_err = auc_per_init_err[key]
            
        # Determine where to show error bands
        band_ind = np.where(tpr > 0.35)[0]
        if 'RSGraviton_WW_NA' in signal_name:
            band_ind = np.where(tpr > 0.6)[0]
            
        # Plot the ROC curve
        line, = ax.plot(tpr, fpr, linewidth=2, color=palette[i])
        
        # Store data for legend
        handles.append(line)
        auc_info.append(f"{auc_value*100:.2f}±{auc_err*100:.2f}")
        signal_info.append(signal_labels[i])
        
        # Error calculation and bands
        log_fpr = np.log10(fpr)
        rel_err = fpr_err/fpr
        log_err = 0.434 * rel_err
        
        log_upper = log_fpr - log_err
        log_lower = log_fpr + log_err
        fpr_upper = 10**log_upper
        fpr_lower = 10**log_lower
        
        ax.fill_between(tpr[band_ind], fpr_lower[band_ind], fpr_upper[band_ind], alpha=0.2, color=palette[i])
    
    # Create legend with AUC values first
    legend_labels = []
    for auc, signal in zip(auc_info, signal_info):
        legend_labels.append(f"{auc}   {signal}")
    
    # Add legend with AUC first
    legend = ax.legend(handles, legend_labels, loc='lower left', frameon=True, fontsize=12)
    
    # Add the column headers with AUC first
    legend.set_title(f"    AUC               BSM Scenario", prop={'size': 12})
    
    # Add vertical dotted lines at TPR = 0.6 and TPR = 0.8
    ax.axvline(x=0.6, color='black', linestyle=':', linewidth=1.5, alpha=0.3)
    ax.axvline(x=0.8, color='black', linestyle=':', linewidth=1.5, alpha=0.3)
    
    # Style the plot
    ax.set_yscale('log')
    ax.set_xlabel('TPR', fontsize=20)
    ax.set_ylabel('FPR$^{-1}$', fontsize=20)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    #ax.set_yticks(fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory and save
    os.makedirs(f'{save_dir}/plots/{train_scaling}/{embedding}/lat{latent}', exist_ok=True)
    plt.savefig(f'{save_dir}/plots/{train_scaling}/{embedding}/lat{latent}/roc_curve_compare_signals.pdf')
    
    return fig, ax

def compare_ROC_by_latent(latent_spaces: Collection[int],
                            bond_dims: dict,
                            initializer: str, 
                            initializer_string: str,
                            signal_name: str,
                            signal_label: str,
                            tpr_per_init: dict, tpr_per_init_err: dict, 
                            fpr_per_init: dict, fpr_per_init_err: dict, 
                            auc_per_init: dict, auc_per_init_err: dict,
                            save_dir: str,
                            embedding: str,
                            train_scaling: str):
    """
    Compare ROC curves for one signal type across different latent spaces, each with a specific bond dimension.
    
    Parameters
    ----------
    latent_spaces : Collection[int]
        List of latent space dimensions to compare
    bond_dims : dict
        Dictionary mapping latent space dimensions to bond dimensions
    initializer : str
        Initializer of the model
    initializer_string : str
        String representation of the initializer
    signal_name : str
        Name of the signal anomaly scores are calculated for
    signal_label : str
        Label for the signal to be used in the plot
    tpr_per_init : dict
        Dictionary containing true positive rates for each latent space
    tpr_per_init_err : dict
        Dictionary containing statistical errors for true positive rates - from multiple runs
    fpr_per_init : dict
        Dictionary containing false positive rates for each latent space
    fpr_per_init_err : dict
        Dictionary containing statistical errors for false positive rates - from multiple runs
    auc_per_init : dict
        Dictionary containing area under the curve values for each latent space
    auc_per_init_err : dict
        Dictionary containing statistical errors for area under the curve values - from multiple runs
    save_dir : str
        Directory to save the plot
    embedding : str
        Embedding name for the model
    train_scaling : str
        Training size and scaling used in training
    
    """
    # Colors for different latent spaces
    palette = ['#E69F00',  # Muted orange
               '#CC6677',  # Muted red
               '#88CCEE',  # Muted blue
               '#000000',  # Black
               '#44AA99',  # Muted teal
               '#AA4499']  # Muted purple
    
    if len(latent_spaces) > len(palette):
        # Generate more colors if needed
        from matplotlib.cm import get_cmap
        palette = get_cmap('tab10').colors
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Store handles and data for legend
    handles = []
    auc_info = []
    config_info = []
    
    for i, latent in enumerate(latent_spaces):
        # Get corresponding bond dimension for this latent space
        bond_dim = bond_dims[str(latent)]
        
        key = f'init={initializer},bond={bond_dim},lat={latent},s={signal_name}'
        
        if key not in tpr_per_init.keys():
            print(f'{key} not found')
            continue
            
        tpr = tpr_per_init[key]
        tpr_err = tpr_per_init_err[key]
        fpr = fpr_per_init[key]
        fpr_err = fpr_per_init_err[key]
        auc_value = auc_per_init[key]
        auc_err = auc_per_init_err[key]
        
        if auc_value < 0.5:
            auc_value = 1 - auc_value
            
        # Determine where to show error bands
        band_ind = np.where(tpr > 0.35)[0]
        if 'RSGraviton_WW_NA' in signal_name:
            band_ind = np.where(tpr > 0.6)[0]
            
        # Plot the ROC curve
        line, = ax.plot(tpr, fpr, linewidth=2, color=palette[i % len(palette)])
        
        # Store data for legend
        handles.append(line)
        auc_info.append(f"{auc_value*100:.2f}±{auc_err*100:.2f}")
        config_info.append(f"lat = {latent}, χ = {bond_dim}")
        
        # Error calculation and bands
        log_fpr = np.log10(fpr)
        rel_err = fpr_err/fpr
        log_err = 0.434 * rel_err
        
        log_upper = log_fpr - log_err
        log_lower = log_fpr + log_err
        fpr_upper = 10**log_upper
        fpr_lower = 10**log_lower
        
        ax.fill_between(tpr[band_ind], fpr_lower[band_ind], fpr_upper[band_ind], 
                        alpha=0.2, color=palette[i % len(palette)])
    
    # Create legend with AUC values first
    legend_labels = []
    for auc, config in zip(auc_info, config_info):
        legend_labels.append(f"{auc}   {config}")
    
    # Add legend with AUC first
    legend = ax.legend(handles, legend_labels, loc='lower left', frameon=True, fontsize=12)
    
    # Add the column headers with AUC first
    legend.set_title(f"    AUC               Configuration", prop={'size': 12})
    
    # Add vertical dotted lines at TPR = 0.6 and TPR = 0.8
    ax.axvline(x=0.6, color='black', linestyle=':', linewidth=1.5, alpha=0.3)
    ax.axvline(x=0.8, color='black', linestyle=':', linewidth=1.5, alpha=0.3)
    
    # Style the plot
    ax.set_yscale('log')
    ax.set_xlabel('TPR', fontsize=20)
    ax.set_ylabel('FPR$^{-1}$', fontsize=20)
    #ax.set_title(f'Signal: {signal_label}, {initializer_string}', fontsize=16)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory and save
    os.makedirs(f'{save_dir}/plots/{train_scaling}/{embedding}', exist_ok=True)
    plt.savefig(f'{save_dir}/plots/{train_scaling}/{embedding}/roc_curve_latent_comparison_{signal_name}.pdf')
    
    return fig, ax
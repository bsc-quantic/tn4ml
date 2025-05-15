import os
import tarfile
import wget
from enum import Enum
from pathlib import Path
import h5py
import numpy as np
import joblib
import jax
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tn4ml.embeddings import TrigonometricEmbedding, Embedding, embed

class Colors(Enum):
    """ANSI color codes for terminal text styling"""
    RESET = "\033[0m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    ORANGE = "\033[38;2;255;165;0m"
    PINK = "\033[38;2;255;105;180m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def _download_data(data_url: str,
                   data_dir: str = "."):
    
    """
    Downloads the jet data if it does not already exist.
    
    Parameters
    ----------
    data_url : str
        URL to the data file
    data_dir : str, optional
        Directory to save the data, default '.'
    
    Returns
    -------
    None
    """

    if not data_dir.is_dir():
        os.makedirs(data_dir, exist_ok=True)
    
    data_file_path = wget.download(data_url, out=str(data_dir))

    data_tar = tarfile.open(data_file_path, "r:gz")
    data_tar.extractall(str(data_dir))
    data_tar.close()
    os.remove(data_file_path)

def _ensure_data_exists(data_dir: str = "data", latent: int = None) -> dict:
    """
    Check if the data directory exists and download all data if it doesn't.
    Only checks directory existence, not individual files.
    
    Parameters
    ----------
    data_dir : str, optional
        Base directory where data should be stored, default "data"
    latent : int, optional
        Latent space dimension used for subdirectory, default None
        
    Returns
    -------
    dict
        Dictionary with paths to all data files
    """
    # Create data directory path with latent dimension subfolder
    base_dir = Path(data_dir)
    if latent is not None:
        data_dir_path = base_dir / f"latent{latent}"
    else:
        data_dir_path = base_dir    
    
    # ONLY check if the directory exists, not individual files
    if not data_dir_path.is_dir() or not any(data_dir_path.iterdir()):
        print(f"{Colors.BLUE.value}Data directory {data_dir_path} does not exist. Downloading complete dataset...{Colors.RESET.value}" + "\n")
        os.makedirs(data_dir_path, exist_ok=True)
        
        archive_url = "https://zenodo.org/records/7673769/files/QML_paper_data.tar.gz"
        try:
            _download_data(archive_url, base_dir)  # Download to base dir
            print(f"{Colors.BLUE.value}Archive downloaded and extracted successfully.{Colors.RESET.value}" + "\n")
        except Exception as e:
            print(f"{Colors.RED.value}Failed to download archive: {e}{Colors.RESET.value}" + "\n")
    else:
        print(f"{Colors.YELLOW.value}Data directory {data_dir_path} already exists. Assuming all data is present.{Colors.RESET.value}" + "\n")

    return

def load_train_data(read_file: str, 
                    train_size:int = 10000,
                    apply_minmax: bool = False, 
                    apply_standardization: bool = False,
                    feature_range: tuple = (0, 1), 
                    shuffle_seed: int = 42,
                    save_dir: str = '.',
                    prefix: str = 'train_qcd'):
    """
    Load and preprocess training data from a given file.
    
    Parameters
    ----------
    read_file : str
        Path to the file containing the training data
    train_size : int, optional
        Number of training samples to load, default 10000
    apply_minmax : bool, optional
        Whether to apply Min-Max scaling, default False
    apply_standardization : bool, optional
        Whether to apply standardization, default False
    feature_range : tuple, optional
        The desired range for Min-Max scaling, default (0, 1)
    shuffle_seed : int, optional
        Seed for random shuffling, default 42
    save_dir : str, optional
        Directory to save scalers, default '.'
    prefix : str, optional
        Prefix for scaler filenames, default 'train_qcd'
        
    Returns
    -------
    np.ndarray
        Preprocessed training data
    dict
        Dictionary of fitted scalers
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Read and prepare data
    with h5py.File(read_file, "r") as file:
        data = file["latent_space"]
        data = np.concatenate([data[:, 0, :], data[:, 1, :]], axis=-1)
        print(f'{Colors.BLUE.value}Input data shape: {data.shape}{Colors.RESET.value}' + '\n')
        
        # Shuffle data
        np.random.seed(shuffle_seed)
        np.random.shuffle(data)
    
    data_train = data[:train_size]
    scalers = {}
    
    # Apply transformations
    if apply_standardization:
        scaler = StandardScaler()
        data_train = scaler.fit_transform(data_train)
        scalers['standard'] = scaler
        scaler_path = os.path.join(save_dir, f'scaler_standard_{prefix}.pkl')
        joblib.dump(scaler, scaler_path)
        
    if apply_minmax:
        min_max_scaler = MinMaxScaler(feature_range=feature_range)
        data_train = min_max_scaler.fit_transform(data_train)
        scalers['minmax'] = min_max_scaler
        scaler_path = os.path.join(save_dir, f'scaler_minmax_{prefix}.pkl')
        joblib.dump(min_max_scaler, scaler_path)
        
    return data_train, scalers

def load_test_data(read_path: str,
                   dataset_type: str = "qcd",
                   scaler: StandardScaler = None,
                   min_max_scaler: MinMaxScaler = None, 
                   test_size: int = 10000,
                   shuffle_seed: int = 42):
    """
    Load and preprocess test data from a given file.
    
    Parameters
    ----------
    read_path : str
        Path to the file/directory containing test data
    dataset_type : str, optional
        Type of dataset ('qcd' or 'signal'), default 'qcd'
    scaler : StandardScaler, optional
        Fitted StandardScaler to apply, default None
    min_max_scaler : MinMaxScaler, optional
        Fitted MinMaxScaler to apply, default None
    test_size : int, optional
        Number of test samples to use, default 10000
    shuffle_seed : int, optional
        Seed for random shuffling, default 42
        
    Returns
    -------
    np.ndarray
        Preprocessed test data
    """
    # Determine file path based on dataset type
    if dataset_type == "qcd":
        file_path = os.path.join(read_path, 'latentrep_QCD_sig_testclustering.h5')
    else:
        file_path = read_path  # For signal, use the path directly
    
    # Load and prepare data
    with h5py.File(file_path, "r") as file:
        data = file["latent_space"]
        # Concatenate the two latent space components
        data_test = np.concatenate([data[:, 0, :], data[:, 1, :]], axis=-1)
        
        # Shuffle data
        np.random.seed(shuffle_seed)
        np.random.shuffle(data_test)
        data_test = data_test[:test_size]
    
    print(f'{Colors.BLUE.value}Input test {dataset_type} shape: {data_test.shape}{Colors.RESET.value}' + '\n')
    
    # Apply transformations if provided
    if scaler is not None:
        data_test = scaler.transform(data_test)
    if min_max_scaler is not None:
        data_test = min_max_scaler.transform(data_test)
        
    return data_test

def calc_fidelity_batch(points, model, embedding: Embedding = TrigonometricEmbedding(), batch_size: int = 1000):
    """
    Calculate fidelity scores for data points in batches using vectorized operations.
    
    Parameters
    ----------
    points : np.ndarray
        Input data points
    model : TensorNetwork
        Trained tensor network model
    embedding : Embedding, optional
        Embedding function, default TrigonometricEmbedding()
    batch_size : int, optional
        Batch size for processing, default 1000
        
    Returns
    -------
    np.ndarray
        Array of fidelity scores
    """
    # Define a function that processes a single point
    def single_point_fidelity(point):
        input_mps = embed(point, embedding)
        p_mps = input_mps.H & model
        return abs(p_mps^all)
    
    # Vectorize the function for batch processing
    batch_fidelity = jax.vmap(single_point_fidelity)
    
    # Process in batches to avoid memory issues
    n_samples = len(points)
    n_batches = (n_samples + batch_size - 1) // batch_size
    results = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch = points[start_idx:end_idx]
        batch_results = batch_fidelity(batch)
        results.append(batch_results)
    
    return np.concatenate(results, axis=0)

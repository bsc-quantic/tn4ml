import os
import json
import argparse
from time import time
import joblib

import numpy as np
import jax
import jax.numpy as jnp
import optax
import h5py
import matplotlib.pyplot as plt

import tn4ml
from tn4ml.util import EarlyStopping, TrainingType
from tn4ml.metrics import NegLogLikelihood
from tn4ml.models.mps import MPS_initialize
from tn4ml.models.model import load_model
from tn4ml.initializers import (
    gramschmidt, 
    rand_unitary
)
from tn4ml.embeddings import (
    FourierEmbedding,
    LegendreEmbedding,
    LaguerreEmbedding,
    HermiteEmbedding,
)

from utils import (
    load_test_data,
    calc_fidelity_batch,
    _ensure_data_exists,
    Colors,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="read arguments for training of TN model")
    parser.add_argument(
        "-save_dir", dest="save_dir", type=str, help="path to directory for saving results", default="results/"
    )
    parser.add_argument(
        "-load_dir", dest="load_dir", type=str, help="path to directory for loading the data"
    )

    # data params
    parser.add_argument("-feature_range", dest="feature_range", type=float, nargs=2, default=[0, 1], help="Feature range for scaling")
    parser.add_argument("-seed", dest="seed", type=int, help="Seed for random number generator")
    parser.add_argument("-standardization", dest="standardization", type=str, default="yes", choices=["yes", "no"], help="Standardization of data")
    parser.add_argument("-minmax", dest="minmax", type=str, default="yes", choices=["yes", "no"], help="Minmax scaling of data")
    parser.add_argument("-embedding", dest="embedding", type=str, help="Embedding type for input data")
    parser.add_argument("-test_size", dest="test_size", type=int, help="Test size")
    parser.add_argument("-signal_name", dest="signal_name", type=str, default='RSGraviton_WW_NA_35', help="Name of signal")
    
    # MPS params
    parser.add_argument("-bond_dim", dest="bond_dim", type=int, default=5, help="Bond dimension")
    parser.add_argument("-initializer", dest="initializer", type=str, help="Type of MPS initialization")

    # testing params
    parser.add_argument("-batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("-run", dest="run", type=int, help="Number of training repetitions")

    parser.add_argument("-latent", dest="latent", type=int, help="Latent space dimension")

    args = parser.parse_args()
    params = vars(args)

    # Get paths to all data files, downloading them if necessary
    print(Colors.YELLOW.value + "Checking data folder..." + Colors.RESET.value + "\n", end="")

    _ensure_data_exists(args.load_dir, args.latent)

    print(Colors.BLUE.value + "Importing data... " + Colors.RESET.value + "\n", end="")

    if args.standardization == 'yes':
        save_dir = args.save_dir + "/" + args.initializer + "/10k_standard/" + str(args.embedding) + "/lat" + str(args.latent) + "/bond" + str(args.bond_dim) + '/run_' + str(args.run)
    elif args.minmax == 'yes':
        if tuple(args.feature_range) == (-1, 1):
            save_dir = args.save_dir + "/" + args.initializer + "/10k_minmax-11/" + str(args.embedding) + "/lat" + str(args.latent) + "/bond" + str(args.bond_dim) + '/run_' + str(args.run)
        else:
            save_dir = args.save_dir + "/" + args.initializer + "/10k_minmax01/" + str(args.embedding) + "/lat" + str(args.latent) + "/bond" + str(args.bond_dim) + '/run_' + str(args.run)
    else:
        save_dir = args.save_dir + "/" + args.initializer + "/10k/" + str(args.embedding) + "/lat" + str(args.latent) + "/bond" + str(args.bond_dim) + '/run_' + str(args.run)

    # set standardization and minmax to bool
    if args.standardization == 'yes':
        standardization = True
    else:
        standardization = False

    if args.minmax == 'yes':
        minmax = True
    else:
        minmax = False
    
    # check result dir
    if not os.path.exists(save_dir):
        # Create a new directory because it does not exist
        os.makedirs(save_dir)

    if args.seed is not None:
        # Use specified seed for reproducibility
        seed = args.seed
        print(Colors.YELLOW.value + f"Using specified seed: {seed}" + Colors.RESET.value + "\n", end="")
    else:
        # Generate random seed for exploration
        seed = int.from_bytes(os.urandom(4), "big")
        print(Colors.YELLOW.value + f"Using random seed: {seed}" + Colors.RESET.value + "\n", end="")
    
    # Set random seed
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    # Set JAX to use 64-bit precision
    # This is important for numerical stability in some cases
    jax.config.update("jax_enable_x64", True)

    # Parse embedding string to get type and degree
    embedding_string = args.embedding
    try:
        embedding_type, degree_str = embedding_string.split('_', 1)
        degree = int(degree_str)
    except ValueError:
        raise ValueError(Colors.RED.value + f"Invalid embedding format: {embedding_string}. Expected format: 'name_degree' (e.g., 'fourier_2')" + Colors.RESET.value + "\n", end="")

    # Initialize embedding based on type and degree
    if embedding_type == 'fourier':
        phys_dim = degree * 2  # Each frequency component adds 2 dimensions (sin and cos)
        embedding = FourierEmbedding(p=degree)
    elif embedding_type == 'legendre':
        phys_dim = degree + 1  # Legendre polynomials from degree 0 to degree
        embedding = LegendreEmbedding(degree=degree)
    elif embedding_type == 'laguerre':
        phys_dim = degree + 1  # Laguerre polynomials from degree 0 to degree
        embedding = LaguerreEmbedding(degree=degree)
    elif embedding_type == 'hermite':
        phys_dim = degree + 1  # Hermite polynomials from degree 0 to degree
        embedding = HermiteEmbedding(degree=degree)
    else:
        raise ValueError(Colors.RED.value + f"Invalid embedding type: {embedding_type}. Supported types: fourier, legendre, laguerre, hermite" + Colors.RESET.value + "\n", end="")
    
    # Set the standard deviation for the initializer
    # This is a heuristic value based on the bond dimension and physical dimension from the paper https://arxiv.org/abs/2310.20498
    std = np.power(float(phys_dim*args.bond_dim), -1)

    # Define the possible initializers
    initializers = {
            "gramschmidt_n_std": gramschmidt('normal', std, dtype=jnp.float64),
            "randn_std": tn4ml.initializers.randn(std),
            "randn_1e-2": tn4ml.initializers.randn(1e-2),
            "unitary": rand_unitary(),
    }

    # Check if the initializer is valid
    if args.initializer not in initializers.keys():
        raise ValueError(Colors.RED.value + f"Invalid initializer: {args.initializer}. Supported initializers: {', '.join(initializers.keys())}" + Colors.RESET.value + "\n", end="")
    
    print(Colors.BLUE.value + "Loading the model..." + Colors.RESET.value + "\n", end="")
    # Save model
    model_name = "model"
    model = load_model(model_name, save_dir)
    
    # EVALUATION
    print(Colors.BLUE.value + "Evaluating model..." + Colors.RESET.value + "\n", end="")

    # Load scalers
    prefix = 'train_qcd'
    if args.standardization == 'yes':
        scaler_path = os.path.join(save_dir, f'scaler_standard_{prefix}.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = joblib.load(f)
        else:
            raise FileNotFoundError(Colors.RED.value + f"Scaler file not found: {scaler_path}" + Colors.RESET.value + "\n", end="")
    else:
        scaler = None

    if args.minmax == 'yes':
        scaler_path = os.path.join(save_dir, f'scaler_minmax_{prefix}.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                min_max_scaler = joblib.load(f)
        else:
            raise FileNotFoundError(Colors.RED.value + f"Scaler file not found: {scaler_path}" + Colors.RESET.value + "\n", end="")
    else:
        min_max_scaler = None

    # Load test data
    read_data_dir = f"{args.load_dir}/latent{args.latent}"
    qcd_test_scaled = load_test_data(f'{read_data_dir}', dataset_type='qcd', scaler=scaler, min_max_scaler=min_max_scaler, test_size=args.test_size, shuffle_seed=seed)

    sig_test_scaled = load_test_data(f'{read_data_dir}/latentrep_{args.signal_name}.h5', dataset_type='signal', scaler=scaler, min_max_scaler=min_max_scaler, test_size=args.test_size, shuffle_seed=seed)

    # Calculate Fidelity - AD score
    print(Colors.YELLOW.value + "Calculating fidelity scores..." + Colors.RESET.value + "\n", end="")

    fid_qcd = calc_fidelity_batch(qcd_test_scaled, model, embedding=embedding, batch_size=args.batch_size)

    fid_sig = calc_fidelity_batch(sig_test_scaled, model, embedding=embedding, batch_size=args.batch_size)

    # Save anomaly scores
    print(Colors.BLUE.value + "Saving fidelity scores..." + Colors.RESET.value + "\n", end="")
    with h5py.File(f'{save_dir}/fidelity_scores_{args.signal_name}.h5', 'w') as file:
        file.create_dataset('loss_qcd', data=fid_qcd)
        file.create_dataset('loss_sig', data=fid_sig)

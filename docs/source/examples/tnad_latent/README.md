# Tensor Network for Anomaly Detection in Latent Space of Proton-Proton Collision Events at the LHC

Implementation of the anomaly detection pipeline from the paper: [2506.00102](https://arxiv.org/abs/2506.00102v1).\


Install `tn4ml` directly from GitHub:
```bash
git clone https://github.com/bsc-quantic/tn4ml.git
```
\

Install an additional package for data handling:
  - `h5py`

### (optional) Download dataset
The dimensionality of the dataset is reduced by passing it through autoencoder. If you are interested more in the autoenoder's architecture, please refer to [[\*]](https://www.nature.com/articles/s42005-024-01811-6).\
Reduced dataset can be downloaded from `Zenodo` :
[record/7673769](https://zenodo.org/record/7673769)\
Description of filenames:
  - `latentrep_QCD_sig.h5`: train dataset (QCD - background)
  - `latentrep_QCD_sig_testclustering.h5`: test dataset (QCD - background)
  - `latentrep_RSGraviton_WW_NA_35.h5`: test dataset (Signal $\mathrm{NA \ G \rightarrow WW}$)
Or it can be download directly in the `pipeline.py`.

### Run training and evaluation pipeline
Data Parameters
- `save_dir` (str): Path to directory for saving results (`default = "results/"`)
- `load_dir` (str): Path to directory for loading the data
- `feature_range` (list): Feature range for scaling (`default = [0, 1]`)
- `seed` (int): Seed for random number generator
- `standardization` (str): Standardization of data (`"yes"` or `"no"`, `default = "yes"`)
- `minmax` (str): Minmax scaling of data (`"yes"` or `"no"`, `default = "yes"`)
- `embedding` (str): Embedding type for input data (e.g., `"legendre_4"`, `"fourier_2"`, `"hermite_3"`)
- `test_size` (int): Number of samples for testing
- `train_size` (int): Number of samples for training
- `signal_name` (str): Name of signal dataset (`default = "RSGraviton_WW_NA_35"`)
- `latent (int)`: Latent space dimension

MPS Parameters
- `bond_dim` (int): Bond dimension of MPS (`default = 5`)
- `initializer` (str): Type of MPS initialization
- `shape_method` (str): Method for distributing bond dimensions (`default = "even"`)

Training Parameters
- `lr` (float): Learning rate (`default = 1e-3`)
- `min_delta` (float): Minimum improvement required for early stopping (`default = 0`)
- `patience (int)`: Number of epochs with no improvement before early stopping (`default = 20`)
- `epochs (int)`: Maximum number of training epochs (`default = 100`)
- `batch_size` (int): Number of samples per training batch (`default = 32`)
- `run` (int): Number of training repetitions with different seeds

```python
python pipeline.py -save_dir results/ \
                   -load_dir QML_paper_data \
                   -feature_range 0 1 \
                   -minmax yes \
                   -embedding laguerre_2 \
                   -test_size 5000 \
                   -train_size 10000 \
                   -bond_dim 8 \
                   -initializer unitary \
                   -lr 0.001 \
                   -patience 25 \
                   -epochs 100 \
                   -batch_size 128 \
                   -run 1 \
                   -latent 4
```

### Run evaluation only
Data Parameters
- `save_dir` (str): Path to directory for saving results (`default = "results/"`)
- `load_dir` (str): Path to directory for loading the data
- `feature_range` (list): Feature range for scaling (`default = [0, 1]`)
- `seed` (int): Seed for random number generator
- `standardization` (str): Standardization of data (`"yes"` or `"no"`, `default = "yes"`)
- `minmax` (str): Minmax scaling of data (`"yes"` or `"no"`, `default = "yes"`)
- `embedding` (str): Embedding type for input data (e.g., `"legendre_4"`, `"fourier_2"`, `"hermite_3"`)
- `test_size` (int): Number of samples for testing
- `signal_name` (str): Name of signal dataset (`default = "RSGraviton_WW_NA_35"`, `options: "RSGraviton_WW_BR_15", "AtoHZ_to_ZZZ_35"`)
- `latent (int)`: Latent space dimension

MPS Parameters
- `bond_dim` (int): Bond dimension of MPS (`default = 5`)
- `initializer` (str): Type of MPS initialization

Training Parameters
- `batch_size` (int): Number of samples per training batch (`default = 32`)
- `run` (int): Number of training repetitions with different seeds

```python
python evaluation.py -save_dir results/ \
                   -load_dir QML_paper_data \
                   -feature_range 0 1 \
                   -minmax yes \
                   -embedding laguerre_2 \
                   -test_size 5000 \
                   -bond_dim 8 \
                   -initializer unitary \
                   -batch_size 128 \
                   -run 1 \
                   -latent 4
```
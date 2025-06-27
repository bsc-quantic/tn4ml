# Tensor Network for Anomaly Detection in Latent Space of Proton-Proton Collision Events at the LHC

Implementation of breast classification task from the paper: [2502.13090](https://arxiv.org/abs/2502.13090).

Install `tn4ml` directly from GitHub:
```bash
git clone https://github.com/bsc-quantic/tn4ml.git
```


Install an additional package for data handling:
  - `h5py`

### Download dataset
The dataset can be download from [kaggle.com/breast-cancer/data](https://www.kaggle.com/datasets/rahmasleam/breast-cancer/data).

### Run training and evaluation pipeline
Data Parameters
- `save_dir` (str): Path to directory for saving results (`default = "results"`)
- `load_dir` (str): Path to directory for loading the data

MPS Parameters
- `bond_dims` (int): Bond dimensions of per each MPS (`default = [2, 4, 8, 16, 32]`)

Training Parameters
- `lr` (float): Learning rate (`default = 1e-3`)
- `min_delta` (float): Minimum improvement required for early stopping (`default = 0`)
- `patience (int)`: Number of epochs with no improvement before early stopping (`default = 20`)
- `epochs (int)`: Maximum number of training epochs (`default = 100`)
- `batch_size` (int): Number of samples per training batch (`default = 32`)
- `test_batch_size` (int): Number of samples per training batch (`default = 64`)

```python
python breast_class.py -save_dir results \
                   -device cpu\
                   -load_dir data \
                   -bond_dims 2 5 10 50\
                   -lr 0.001 \
                   -patience 25 \
                   -epochs 100 \
                   -batch_size 32 \
                   -test_batch_size 32 \
```
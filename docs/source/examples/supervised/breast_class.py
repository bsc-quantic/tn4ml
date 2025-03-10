import os
from time import time
import json
import argparse
import jax.numpy as jnp
import quimb.tensor as qtn
import numpy as np
import pandas as pd
from jax.nn.initializers import *
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split

from tn4ml.util import *
from tn4ml.initializers import *
from tn4ml.models.mps import *
from tn4ml.models.model import *
from tn4ml.embeddings import *
from tn4ml.metrics import *

import warnings
warnings.filterwarnings("ignore", message="Couldn't import `kahypar`")
warnings.filterwarnings("ignore", message="OMP: Info #276: omp_set_nested routine deprecated")


def crossentropy_loss(*args, **kwargs):
    return OptaxWrapper(optax.softmax_cross_entropy)(*args, **kwargs).mean()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="read arguments for training MPS on data distribution")
    
    parser.add_argument("-device", dest="device", type=str, default="cpu")
    parser.add_argument("-load_dir", dest="load_dir", type=str)
    parser.add_argument("-save_dir", dest="save_dir", type=str, default="results")

    parser.add_argument("-bond_dims", dest="bond_dims", type=int, nargs='+', default=[2, 4, 8, 16, 32])

    parser.add_argument("-lr", dest="lr", type=float, default=1e-3)
    parser.add_argument("-min_delta", dest="min_delta", type=float, default=0)
    parser.add_argument("-patience", dest="patience", type=int, default=10)
    parser.add_argument("-epochs", dest="epochs", type=int, default=100)
    parser.add_argument("-batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("-test_batch_size", dest="test_batch_size", type=int, default=64)

    args = parser.parse_args()
    params = vars(args)

    jax.config.update("jax_platform_name", args.device)
    print("Available devices:", jax.devices())
    print("Current device:", jax.default_device())

    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_default_matmul_precision', 'highest')

    # ------ LOAD DATASET ------
    n_classes = 2 # binary classification

    # load data
    data = pd.read_csv(f'{args.load_dir}/breast-cancer.csv')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X = data.drop(['id', 'diagnosis'], axis=1)
    y = data['diagnosis'].to_numpy()

    feature_min = X.min()
    feature_max = X.max()

    X_normalized = (X - feature_min) / (feature_max - feature_min)
    X_numpy = X_normalized.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_numpy, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print('Train data shape: ', X_train.shape)
    print('Validation data shape: ', X_valid.shape)
    print('Test data shape: ', X_test.shape)

    y_train = integer_to_one_hot(y_train, n_classes)
    y_valid = integer_to_one_hot(y_valid, n_classes)
    y_test = integer_to_one_hot(y_test, n_classes)

    # ------ MODEL ------

    # define model hyperparameters
    key = jax.random.key(42)
    L = X_train.shape[1]
    class_index = int(L//2)
    shape_method='noteven' # default method
    compress = False # connected with shape method
    embedding = polynomial(degree=2, n=1, include_bias=True)
    phys_dim = 3
    initializer = randn(1e-2, dtype=jnp.float64)
    
    for bond_dim in args.bond_dims:
        print('Initializing model with bond dimension: ', bond_dim)
        model = MPS_initialize(L=L,
                            initializer=initializer,
                            key=key,
                            shape_method=shape_method,
                            compress=compress,
                            cyclic=False,
                            phys_dim=phys_dim,
                            bond_dim=bond_dim,
                            class_index=class_index,
                            canonical_center=class_index,
                            class_dim=n_classes,
                            add_identity=True,
                            boundary='obc')

        # define training parameters
        optimizer = optax.adam
        strategy = 'global'
        loss = crossentropy_loss
        train_type = 1 # 1 for supervised learning
        learning_rate = args.lr

        # configure model
        model.configure(optimizer=optimizer, strategy=strategy, loss=loss, train_type=train_type, learning_rate=learning_rate)

        earlystop = EarlyStopping(min_delta=args.min_delta, patience=args.patience, monitor='loss', mode='min')
        
        # ------ TRAIN ------
        print('Training model')
        run_start = time()
        history = model.train(X_train, 
                            targets=y_train,
                            val_inputs=X_valid,
                            val_targets=y_valid,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            embedding = embedding,
                            normalize=True,
                            dtype=jnp.float64,
                            cache=False,
                            earlystop=earlystop,
                            canonize=(True, class_index),
                            display_val_acc = True,
                            eval_metric = crossentropy_loss,
                            val_batch_size = args.batch_size
                            )

        run_end = time()
        run_time = run_end - run_start
        print(f"Training time: {run_time}")
        
        # ------ SAVE loss and model -------
        save_dir = args.save_dir + '/bond_' + str(bond_dim)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # plot loss
        plt.figure()
        plt.plot(range(len(history['loss'])), history['loss'], label='train')
        plt.legend()
        plt.savefig(save_dir + '/loss.pdf')

        # save loss
        np.save(save_dir + '/loss.npy', history['loss'])

        # accuracy
        acc = model.accuracy(X_test, y_test, embedding=embedding, batch_size=args.test_batch_size, dtype=jnp.float64)

        # save model
        model.save('model', save_dir, tn=True)

        # ------ EVALUATION ------
        print('Evaluating model')

        y_pred = model.forward(X_test, embedding, batch_size=args.test_batch_size, normalize=True, dtype=jnp.float64)
        predicted = jnp.argmax(y_pred, axis=-1)
        true = jnp.argmax(y_test, axis=-1)[:len(predicted)]
        
        tn, fp, fn, tp = confusion_matrix(true, predicted).ravel()
        specificity = tn / (tn + fp)

        sensitivity = recall_score(true, predicted)  # default is recall for positive class
        precision = precision_score(true, predicted)
        F_measure = f1_score(true, predicted)

        print('Sensitivity=%.3f'%sensitivity) # as the same as recall
        print('Specificity=%.3f'%specificity)
        print('Precision=%.3f'%precision)
        print('F-measure=%.3f'%F_measure)

        params = {
                'embedding': 'polynomial(degree=2, n=1, include_bias=True)',
                'initializer': 'randn(std=1e-2)',
                'shape_method': shape_method,
                'bond_dim': bond_dim,
                'strategy': strategy,
                'base_lr': args.lr,
                'min_delta': args.min_delta,
                'patience': args.patience,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'save_dir': args.save_dir,
                'test_batch_size': args.test_batch_size,
                'acc': str(acc),
                'train_time': str(run_time),
                'sensitivity': str(sensitivity),
                'specificity': str(specificity),
                'precision': str(precision),
                'F_measure': str(F_measure)
            }

        # save parameters
        with open(os.path.join(save_dir,("parameters.txt")), "w") as f:
            f.write("Parameters: ")
            json.dump(params, f)
            f.write("\n")
        f.close()
        
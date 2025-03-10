import os
import json
import seaborn as sns
import argparse
import jax.numpy as jnp
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from jax.nn.initializers import *
from flax.training.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from tn4ml.initializers import *
from tn4ml.models.smpo import *
from tn4ml.models.model import *
from tn4ml.embeddings import *
from tn4ml.metrics import *
from tn4ml.eval import *

import warnings
warnings.filterwarnings("ignore", message="Couldn't import `kahypar`")
warnings.filterwarnings("ignore", message="OMP: Info #276: omp_set_nested routine deprecated")

def zigzag_order(data):
    data = np.squeeze(data)
    # Reshape the array to (N, -1) where N is the number of images, and flatten each image
    data_zigzag = data.reshape(data.shape[0], -1)
    return data_zigzag

def resize_images(images):
        resized_images = tf.image.resize(images, [14, 14], method=tf.image.ResizeMethod.AREA)
        return resized_images.numpy()

def loss_combined(*args, **kwargs):
        error = LogQuadNorm
        reg = LogReLUFrobNorm
        return CombinedLoss(*args, **kwargs, error=error, reg=lambda P: alpha*reg(P))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read arguments for training TN for anomaly detection")
    
    parser.add_argument("-train_size", dest="train_size", type=int, default=5923)
    parser.add_argument("-normal_class", dest="normal_class", type=int, default=0)

    parser.add_argument("-shape_method", dest="shape_method", type=str, default="even")
    parser.add_argument("-bond_dims", dest="bond_dims", type=int, nargs='+')
    parser.add_argument("-spacings", dest="spacings", type=int, nargs='+')
    parser.add_argument("-alpha", dest="alpha", type=float, default=0.4)

    parser.add_argument("-initializers", dest="initializers", type=str, nargs='+', default="[orthogonal]")
    parser.add_argument("-embedding", dest="embedding", type=str, default="trigonometric")

    parser.add_argument("-lr", dest="lr", type=float, default=1e-3)
    parser.add_argument("-min_delta", dest="min_delta", type=float, default=0)
    parser.add_argument("-patience", dest="patience", type=int, default=10)
    parser.add_argument("-epochs", dest="epochs", type=int, default=100)
    parser.add_argument("-batch_size", dest="batch_size", type=int, default=32)
    
    parser.add_argument("-save_dir", dest="save_dir", type=str, default="results/")
    parser.add_argument("-metric_loss", dest="metric_loss", type=str, default="LogQuadNorm")
    parser.add_argument("-test_batch_size", dest="test_batch_size", type=int, default=64)
    parser.add_argument("-time_limit", dest="time_limit", type=int)

    args = parser.parse_args()
    params = vars(args)

    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_default_matmul_precision', 'highest')


    # ------ LOAD DATASET -------
    train, test = mnist.load_data()
    data = {"X": dict(train=train[0], test=test[0]), "y": dict(train=train[1], test=test[1])}

    normal_class = args.normal_class

   # training data
    X = {
    "normal": data["X"]["train"][data["y"]["train"] == normal_class]/255.0,
    "anomaly": data["X"]["train"][data["y"]["train"] != normal_class]/255.0,
    }

    # test data
    X_test = {
    "normal": data["X"]["test"][data["y"]["test"] == normal_class]/255.0,
    "anomaly": data["X"]["test"][data["y"]["test"] != normal_class]/255.0,
    }
    
    # resize images
    X_resized = {
        "normal": resize_images(np.expand_dims(X["normal"], axis=-1)),
        "anomaly": resize_images(np.expand_dims(X["anomaly"], axis=-1)),
    }

    X_test_resized = {
        "normal": resize_images(np.expand_dims(X_test["normal"], axis=-1)),
        "anomaly": resize_images(np.expand_dims(X_test["anomaly"], axis=-1)),
    }

    # reshape data
    train_normal = zigzag_order(X_resized["normal"])
    test_normal = zigzag_order(X_test_resized["normal"])

    train_anomaly = zigzag_order(X_resized["anomaly"])
    test_anomaly = zigzag_order(X_test_resized["anomaly"])

    # ------ MODEL -------

    # Create SMPO model
    key = jax.random.key(42) # change seed if you are running multiple experiments
    L = train_normal.shape[1]
    print('Number of tensors: ', L)
    alpha = args.alpha
    
    # define initializers - define strings you want to use as keys
    initializers = {
            "glorot_n": jax.nn.initializers.glorot_normal(),
            "he_n": jax.nn.initializers.he_normal(),
            "orthogonal": jax.nn.initializers.orthogonal(),
            "gramschmidt_n_1e-1": gramschmidt('normal', 1e-1, dtype=jnp.float64),
            "randn_1e-1": randn(1e-1),
            "unitary": rand_unitary(),
    }

    # define embedding
    embedding_string = args.embedding
    if embedding_string == 'trigonometric':
        phys_dim = 2
        embedding = trigonometric()
    elif embedding_string == 'fourier':
        phys_dim = 3
        embedding = fourier(p=3)
    elif embedding_string == 'poly_2':
        phys_dim = 3
        embedding = polynomial(degree=2)
    elif embedding_string == 'poly_3':
        phys_dim = 4
        embedding = polynomial(degree=3)
    else:
        raise ValueError("Invalid embedding")

    # compress bond dimensions if shape_method is even
    if args.shape_method == 'even':
        compress = True
    else:
        compress = False
    
    for bond_dim in list(args.bond_dims):
        for spacing in list(args.spacings):
            for init_string in list(args.initializers):
                initializer = initializers[init_string]
                bond_dim = int(bond_dim)
                spacing = int(spacing)

                # initialize model
                print('Initializing model with bond dimension: ', bond_dim, ' and spacing: ', spacing)
                model = SMPO_initialize(L=L,
                                        initializer=initializer,
                                        key=key,
                                        shape_method=args.shape_method,
                                        spacing=spacing,
                                        bond_dim=bond_dim,
                                        phys_dim=(phys_dim, phys_dim),
                                        cyclic=False,
                                        add_identity=True,
                                        boundary='obc',
                                        compress=compress)

                # define training parameters
                optimizer = optax.adam
                strategy = 'global'
                loss = loss_combined
                train_type = 0 # 0 for unsupervised
                learning_rate = args.lr

                # configure model
                model.configure(optimizer=optimizer, strategy=strategy, loss=loss, train_type=train_type, learning_rate=learning_rate)

                earlystop = EarlyStopping(min_delta=args.min_delta, patience=args.patience, mode='min', monitor='loss')
                
                # ------ TRAIN ------
                print('Training model')
                history = model.train(
                                    train_normal,
                                    epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    embedding = embedding,
                                    normalize=True,
                                    dtype=jnp.float64,
                                    cache=True,
                                    earlystop=earlystop,
                                    time_limit=args.time_limit
                                    )
                            
                # ------ SAVE loss and model -------
                print('Saving model')
                save_dir = args.save_dir + '/normal_class_' + str(normal_class) + '/' + init_string + '/bond_' + str(bond_dim) + '/spacing_' + str(spacing)+'/' + embedding_string
                
                if (history["unfinished"]):
                    save_dir = save_dir+'/unfinished'
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # plot loss
                plt.figure()
                plt.plot(range(len(history['loss'])), history['loss'], label='train')
                plt.legend()
                plt.savefig(save_dir + '/loss.pdf')

                # save loss
                np.save(save_dir + '/loss.npy', history['loss'])

                # save model
                model.save('model', save_dir)

                params = {
                        'train_size': args.train_size,
                        'embedding': embedding_string,
                        'initializer': init_string,
                        'shape_method': args.shape_method,
                        'bond_dim': bond_dim,
                        'spacing': spacing,
                        'alpha': args.alpha,
                        'strategy': strategy,
                        'lr': args.lr,
                        'min_delta': args.min_delta,
                        'patience': args.patience,
                        'epochs': args.epochs,
                        'batch_size': args.batch_size,
                        'save_dir': args.save_dir,
                        'metric_loss': args.metric_loss,
                        'test_batch_size': args.test_batch_size
                    }

                # save parameters
                with open(os.path.join(save_dir,("parameters.txt")), "w") as f:
                    f.write("Parameters: ")
                    json.dump(params, f)
                    f.write("\n")
                f.close()

                # ------ EVALUATION -------
                print('Evaluating model')

                # define metric loss
                if args.metric_loss == 'LogQuadNorm':
                    metric_loss = LogQuadNorm

                # evaluate model on normal and anomaly data
                anomaly_score = model.evaluate(test_anomaly, evaluate_type=0, return_list=True, dtype=jnp.float64, embedding=embedding, batch_size=args.test_batch_size, metric = metric_loss)
                normal_score = model.evaluate(test_normal, evaluate_type=0, return_list=True, dtype=jnp.float64, embedding=embedding, batch_size=args.test_batch_size, metric = metric_loss)

                # save scores
                np.save(save_dir + '/anomaly_score.npy', anomaly_score)
                np.save(save_dir + '/normal_score.npy', normal_score)

                fpr, tpr = get_roc_curve_data(normal_score, anomaly_score, anomaly_det=True)
                auc_value = auc(fpr, tpr)

                # save roc data
                np.save(save_dir + '/fpr_values.npy', fpr)
                np.save(save_dir + '/tpr_values.npy', tpr)

                # plot anomaly scores
                sns.set(style='whitegrid')
                plt.figure(figsize=(8,7))
                sns.histplot(anomaly_score, bins=100, kde=True, color='skyblue', label='anomaly')
                sns.histplot(normal_score, bins=100, kde=True, color='red', label='normal')
                plt.title('Normal class ' + str(normal_class) + ' vs Anomaly')
                plt.xlabel('Score')
                plt.ylabel('Frequency')
                legend = plt.legend(loc="upper right", fontsize='medium', frameon=True)
                frame = legend.get_frame()
                frame.set_edgecolor('black')  # Set the edge color of the legend box
                frame.set_facecolor('gainsboro')  # Set the background color of the legend box
                frame.set_linewidth(0.5)
                plt.savefig(save_dir + '/anomaly_score.pdf')

                # plo roc curve
                sns.set(style='whitegrid')
                plt.figure(figsize=(8,7))
                plt.plot(fpr, tpr, label='AUC = %0.3f' % auc_value, color='darkblue')
                plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC curve')
                legend = plt.legend(loc="lower right", fontsize='large', frameon=True)
                frame = legend.get_frame()
                frame.set_edgecolor('black')  # Set the edge color of the legend box
                frame.set_facecolor('gainsboro')  # Set the background color of the legend box
                frame.set_linewidth(0.5)
                plt.savefig(save_dir + '/roc_curve.pdf')
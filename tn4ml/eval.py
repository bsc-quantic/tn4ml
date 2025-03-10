import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from typing import Collection


def plot_loss(history: dict, validation: bool = True, figsize: tuple =(5, 5), save_path: str = None, legend_args: dict = {}):
    """
    Plot the loss of the model during training and validation.

    Parameters
    ----------
    history: dict
        History object from the model training.
    validation: bool
        Whether to plot the validation loss.
    figsize: tuple
        Size of the figure.
    save_path: str
        Path to save the plot.

    Returns
    -------
    Displays the plot.
    """
    plt.figure(figsize=figsize)
    plt.plot(range(len(history['loss'])), history['loss'], label='train')
    if validation:
        plt.plot(range(len(history['val_loss'])), history['val_loss'], label='validation')
    plt.legend(legend_args)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    if save_path:
        plt.savefig(save_path + '.pdf', format='pdf', dpi=300)
    else:
        plt.show()
    plt.close()

def plot_accuracy(history: dict, figsize: tuple =(5, 5), save_path: str = None, legend_args: dict = {}):
    """
    Plot the accuracy of the model during training and validation.

    Parameters
    ----------
    history: dict
        History object from the model training.
    validation: bool
        Whether to plot the validation accuracy.
    figsize: tuple
        Size of the figure.
    save_path: str
        Path to save the plot.
    legend_args: dict
        Arguments for the legend.

    Returns
    -------
        Displays or saves the plot.
    """
    plt.figure(figsize=figsize)
    plt.plot(range(len(history['val_acc'])), history['val_acc'], label='validation')
    plt.legend(legend_args)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    if save_path:
        plt.savefig(save_path + '.pdf', format='pdf', dpi=300)
    else:
        plt.show()
    plt.close()

def get_roc_curve_data(y_true: np.ndarray, y_scores: np.ndarray, anomaly_det: bool = False):
    """
    Calculate the ROC curve data from normal and anomaly scores. Use it when both y_true and y_scores are not binary.

    Parameters
    ----------
    y_true: :class:`numpy.ndarray`
        True or normal scores.
    y_scores: :class:`numpy.ndarray`
        Predicted scores or anomaly scores.
    anomaly: bool
        Whether the scores are anomaly scores or
    
    Returns
    -------
    fpr_loss: :class:`numpy.ndarray`
        False positive rate values.
    tpr_loss: :class:`numpy.ndarray`
        True positive rate values.
    """
    if anomaly_det:
        true_val = np.concatenate((np.ones(y_scores.shape[0]), np.zeros(y_true.shape[0])))
        pred_val = np.concatenate((y_scores, y_true))
    else:
        true_val = y_true
        pred_val = y_scores
    
    fpr, tpr, _ = roc_curve(true_val, pred_val, drop_intermediate=False)
    return fpr, tpr

def get_precision_recall_curve_data(y_true: np.ndarray, y_scores: np.ndarray, anomaly_det: bool = False):
    """
    Calculate the ROC curve data from normal and anomaly scores. Use it when both y_true and y_scores are not binary.

    Parameters
    ----------
    y_true: :class:`numpy.ndarray`
        True or normal scores.
    y_scores: :class:`numpy.ndarray`
        Predicted scores or anomaly scores.
    
    Returns
    -------
    fpr_loss: :class:`numpy.ndarray`
        False positive rate values.
    tpr_loss: :class:`numpy.ndarray`
        True positive rate values.
    """
    if anomaly_det:
        true_val = np.concatenate((np.ones(y_scores.shape[0]), np.zeros(y_true.shape[0])))
        pred_val = np.concatenate((y_scores, y_true))
    else:
        true_val = y_true
        pred_val = y_scores
    precision, recall, _ = precision_recall_curve(true_val, pred_val, drop_intermediate=False)
    return precision, recall


def get_FPR_for_fixed_TPR(tpr_window, fpr, tpr, tolerance):
    """
    Calculate the FPR for a fixed TPR value.

    Parameters
    ----------

    tpr_window: float
        Fixed TPR value.
    fpr: :class:`numpy.ndarray`
        False positive rate values.
    tpr: :class:`numpy.ndarray`
        True positive rate values.
    tolerance: float
        Tolerance value for the fixed TPR value.
    
    Returns
    -------
    fpr: float
        FPR value for the fixed TPR value.
    """
    position = np.where((tpr>=tpr_window-tpr_window*tolerance) & (tpr<=tpr_window+tpr_window*tolerance))[0]
    return np.mean(fpr[position])

def get_TPR_for_fixed_FPR(fpr_window, fpr, tpr, tolerance):
    """
    Calculate the TPR for a fixed FPR value.
    
    Parameters
    ----------
    
    fpr_window: float
        Fixed FPR value.
    fpr: :class:`numpy.ndarray`
        False positive rate values.
    tpr: :class:`numpy.ndarray`
        True positive rate values.
    tolerance: float
        Tolerance value for the fixed FPR value.
    
    Returns
    -------
    tpr: float
        TPR value for the fixed FPR value.
    """

    position = np.where((fpr>=fpr_window-fpr_window*tolerance) & (fpr<=fpr_window+fpr_window*tolerance))[0]
    return np.mean(tpr[position])

def get_mean_and_error(data):
    """
    Calculate the mean and standard deviation of the input data.
    
    Parameters
    ----------
    data: :class:`numpy.ndarray`
        Input data to calculate the mean and standard deviation.
    
    Returns
    -------
    mean: :class:`numpy.ndarray`
        Mean of the input data.
    std: :class:`numpy.ndarray`
        Standard deviation of the input data.
    """

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

def plot_ROC_curve_from_metrics(y_true: np.ndarray, y_scores: np.ndarray, title: str = "ROC Curve", save_path: str = None):
    """
    Calculates TPR and FPR from input metrics and plots the ROC curve.

    Parameters
    ----------
    y_true: :class:`numpy.ndarray`
        List or array of true binary labels (0 or 1).   
    y_scores: :class:`numpy.ndarray`
        List or array of predicted scores or probabilities.
    title: str (Optional) 
        Title for the plot. Defaults to "ROC Curve".
    save_path: str (Optional)
        Path and name to save the plot.

    Returns
    ------
        Displays or saves the plot.
    """
    # Calculate FPR, TPR, and thresholds
    fpr, tpr = get_roc_curve_data(y_true, y_scores)
    
    # Calculate the AUC
    auc_value = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {auc_value:.2f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + '.pdf', format='pdf', dpi=300)
    else:
        plt.show()
    plt.close()

def plot_ROC_curve_from_data(fpr: np.ndarray, tpr: np.ndarray, title: str = "ROC Curve", save_path: str = None):
    """
    Plots the ROC curve from input FPR and TPR values.

    Parameters
    ----------
    fpr_loss: :class:`numpy.ndarray`
        False positive rate values.
    tpr_loss: :class:`numpy.ndarray`
        True positive rate values.
    title: str (Optional) 
        Title for the plot. Defaults to "ROC Curve".
    save_path: str (Optional)
        Path and name to save the plot. Example: `./ROC_curve.pdf`

    Returns
    ------
        Displays or saves the plot.
    """
    # Calculate the AUC
    auc_value = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {auc_value:.2f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + '.pdf', format='pdf', dpi=300)
    else:
        plt.show()
    plt.close()

def plot_PR_curve(y_true: np.ndarray, y_scores: np.ndarray, title: str = "Precision-Recall Curve", save_path: str = None):
    """
    Calculates precision and recall from input metrics and plots the Precision-Recall curve.

    Parameters
    ----------
    y_true: :class:`numpy.ndarray`
        List or array of true binary labels (0 or 1).   
    y_scores: :class:`numpy.ndarray`
        List or array of predicted scores or probabilities.
    title: str (Optional) 
        Title for the plot. Defaults to "Precision-Recall Curve".
    save_path: str (Optional)
        Path and name to save the plot.

    Returns
    ------
        Displays or saves the plot.
    """
        
    # Calculate FPR, TPR, and thresholds
    precision, recall = get_precision_recall_curve_data(y_true, y_scores)
    
    # Calculate the AUC_PR
    if not np.all((y_true == 0) | (y_true == 1)):
        label = 'PR Curve'
    else:
        auc_pr = average_precision_score(y_true, y_scores)
        label = f"PR Curve (AUC = {auc_pr:.2f})"

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=label)
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + '.pdf', format='pdf', dpi=300)
    else:
        plt.show()
    plt.close()

def compare_AUC(save_dir: str = '.',
                bond_dims: Collection[int] = None,
                spacings: Collection[int] = None,
                initializers: Collection[str] = None,
                embedding: str = 'trigonometric',
                nruns: int = 0,
                fig_size: tuple = (6, 5),
                labels: dict = None,
                anomaly_det: bool = False):
    """
    Example of code to compare the TPR values for fixed FPR for different values of hyperparameters, when spacing parameter is fixed.
    - code for generating plots from the paper "tn4ml: Tensor Network Training and Customization for Machine Learning"

    This works with the results saved in the directory structure as follows::

        root_dir/initializer_string/bond_' + str(bond_dim) + '/spacing_' + str(spacing) + '/' + embedding_string+'/run_' + str(nrun)

    Example::

        root_dir/randn_1e-1/bond_10/spacing_2/trigonometric/run_1
    
    Parameters
    ----------
    save_dir: str
        Directory where the results are saved.
    bond_dims: list[int]
        List of bond dimensions.
    spacings: list[int]
        List of spacing values. If model is :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`, then spacing is required.
    initializers: list[str]
        List of initializers.
    embedding: str
        List of embeddings.
    nruns: int
        Number of runs for each model. Assumes that the model is run at least 2 times.
    fig_size: tuple
        Size of the figure.
    labels: dict
        Dictionary containing the labels for the bond dimensions
        Example::
            
            LABELS = {'5': (r'bond = 5', 'o', '#016c59'),
                    '10': (r'bond = 10','X', '#7a5195'),
                    '30': (r'bond = 30', 'v', '#67a9cf'),
                    '50': (r'bond = 50', 'd', '#ffa600')}

    Returns
    -------
        Displays or saves the plot.
    """
    
    for spacing in spacings:
        plt.figure(figsize=fig_size)
        auc_per_bond_data = {}; auc_per_bond_err = {}
        for bond_dim in bond_dims:
            auc_per_init_data = []; auc_per_init_err = []
            for init in initializers:
                auc_data=[]
                for j in range(1, nruns+1):
                    if nruns == 1:
                        dir_name = save_dir + '/' + init + '/bond_' + str(bond_dim) + '/spacing_' + str(spacing)+'/'+ embedding
                    else:
                        dir_name = save_dir + '/' + init + '/bond_' + str(bond_dim) + '/spacing_' + str(spacing)+'/'+ embedding +'/run_'+str(j)
                            
                    fpr, tpr = get_roc_curve_data(np.load(dir_name + '/normal_score.npy'), np.load(dir_name + '/anomaly_score.npy'), anomaly_det=anomaly_det)
                    auc_data.append(auc(fpr, tpr))
                mean_error = get_mean_and_error(np.array(auc_data))
                auc_per_init_data.append(mean_error[0])
                auc_per_init_err.append(mean_error[1])
            auc_per_bond_data[bond_dim] = auc_per_init_data
            auc_per_bond_err[bond_dim] = auc_per_init_err
        
        for bond_dim in bond_dims:
            data = auc_per_bond_data[bond_dim]
            data_err = auc_per_bond_err[bond_dim]
            plt.errorbar(list(range(len(initializers))), data, yerr=data_err, label=labels[str(bond_dim)][0],
                        linestyle='None', marker=labels[str(bond_dim)][1], capsize=3, color=labels[str(bond_dim)][2])
                
        plt.title(f'S = {spacing}')
        plt.ylabel('AUC')
        plt.yticks(fontsize=12)
        plt.xticks(range(len(initializers)), initializers, fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.legend(fancybox=True, frameon=True, prop={"size":10}, loc='best')

        if save_dir:
            if not os.path.exists(f'{save_dir}/results/plots/AUC'):
                os.makedirs(f'{save_dir}/results/plots/AUC')
            plt.savefig(f'{save_dir}/results/plots/AUC/spacing_{spacing}.pdf')
        else:
            plt.show()
        plt.close()

def compare_TPR_per_FPR(save_dir: str = '.',
                FPR_fixed: float = 0.1,
                bond_dims: Collection[int] = None,
                spacings: Collection[int] = None,
                initializers: Collection[str] = None,
                embedding: str = 'trigonometric',
                nruns: int = 0,
                fig_size: tuple = (6, 5),
                labels: dict = None,
                anomaly_det: bool = False):
    """
    Example of code to compare the TPR values for fixed FPR for different values of hyperparameters, when spacing parameter is fixed.

    This works with the results saved in the directory structure as follows::

        root_dir/initializer_string/bond_<bond_dim>/spacing_<spacing>/<embedding_string>/run_<nrun>

    Example::

        root_dir/randn_1e-1/bond_10/spacing_2/trigonometric/run_1

    Parameters
    ----------
    save_dir : str
        Directory where the results are saved.
    FPR_fixed : float
        Fixed FPR value.
    bond_dims : list[int]
        List of bond dimensions.
    spacings : list[int]
        List of spacing values. If model is :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`, then spacing is required.
    initializers : list[str]
        List of initializers.
    embedding : str
        Embedding method used.
    nruns : int
        Number of runs for each model. Assumes that the model is run at least 2 times.
    fig_size : tuple
        Size of the figure.
    labels : dict
        Dictionary containing the labels for the bond dimensions.

        Example::

            LABELS = {'5': (r'bond = 5', 'o', '#016c59'),
                    '10': (r'bond = 10', 'X', '#7a5195'),
                    '30': (r'bond = 30', 'v', '#67a9cf'),
                    '50': (r'bond = 50', 'd', '#ffa600')}

    Returns
    -------
    None
        Displays or saves the plot.
    """
    
    for spacing in spacings:
        plt.figure(figsize=fig_size)
        tpr_per_bond_data = {}; tpr_per_bond_err = {}
        for bond_dim in bond_dims:
            tpr_per_init_data = []; tpr_per_init_err = []
            for init in initializers:
                tpr_data=[]
                for j in range(1, nruns+1):
                    if nruns == 1:
                        dir_name = save_dir + '/' + init + '/bond_' + str(bond_dim) + '/spacing_' + str(spacing)+'/'+ embedding
                    else:
                        dir_name = save_dir + '/' + init + '/bond_' + str(bond_dim) + '/spacing_' + str(spacing)+'/'+ embedding +'/run_'+str(j)
                            
                    fpr, tpr = get_roc_curve_data(np.load(dir_name + '/normal_score.npy'), np.load(dir_name + '/anomaly_score.npy'), anomaly_det=anomaly_det)
                    tpr_per_fpr = get_TPR_for_fixed_FPR(FPR_fixed, np.array(fpr), np.array(tpr), tolerance=0.01)
                    tpr_data.append(tpr_per_fpr)
                mean_error = get_mean_and_error(np.array(tpr_data))
                tpr_per_init_data.append(mean_error[0])
                tpr_per_init_err.append(mean_error[1])
            tpr_per_bond_data[bond_dim] = tpr_per_init_data
            tpr_per_bond_err[bond_dim] = tpr_per_init_err
        
        for bond_dim in bond_dims:
            data = tpr_per_bond_data[bond_dim]
            data_err = tpr_per_bond_err[bond_dim]
            plt.errorbar(list(range(len(initializers))), data, yerr=data_err, label=labels[str(bond_dim)][0],
                        linestyle='None', marker=labels[str(bond_dim)][1], capsize=3, color=labels[str(bond_dim)][2])
                
        plt.title(f'S = {spacing}, FPR = {FPR_fixed}')
        plt.ylabel('TPR')
        plt.yticks(fontsize=12)
        plt.xticks(range(len(initializers)), initializers, fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.legend(fancybox=True, frameon=True, prop={"size":10}, loc='best')

        if save_dir:
            if not os.path.exists(f'{save_dir}/results/plots/TPR'):
                os.makedirs(f'{save_dir}/results/plots/TPR')
            plt.savefig(f'{save_dir}/results/plots/TPR/spacing_{spacing}_FPR_{FPR_fixed}.pdf')
        else:
            plt.show()
        plt.close()

def compare_FPR_per_TPR(save_dir: str = '.',
                TPR_fixed: float = 0.95,
                bond_dims: Collection[int] = None,
                spacings: Collection[int] = None,
                initializers: Collection[str] = None,
                embedding: str = 'trigonometric',
                nruns: int = 0,
                fig_size: tuple = (6, 5),
                labels: dict = None,
                anomaly_det: bool = False):
    """
    Example of code to compare the FPR values for fixed TPR for different values of hyperparameters, when spacing parameter is fixed.
    - code for generating plots from the paper "tn4ml: Tensor Network Training and Customization for Machine Learning"

    This works with the results saved in the directory structure as follows::
        
        root_dir/initializer_string/bond_' + str(bond_dim) + '/spacing_' + str(spacing) + '/' + embedding_string+'/run_' + str(nrun)

    Example::
        
        root_dir/randn_1e-1/bond_10/spacing_2/trigonometric/run_1
    
    Parameters
    ----------
    save_dir: str
        Directory where the results are saved.
    TPR_fixed: float
        Fixed TPR value.
    bond_dims: list[int]
        List of bond dimensions.
    spacings: list[int]
        List of spacing values. If model is :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`, then spacing is required.
    initializers: list[str]
        List of initializers.
    embedding: str
        List of embeddings.
    nruns: int
        Number of runs for each model. Assumes that the model is run at least 2 times.
    fig_size: tuple
        Size of the figure.
    labels: dict
        Dictionary containing the labels for the bond dimensions
        Example:: 
            
            LABELS = {'5': (r'bond = 5', 'o', '#016c59'),
                    '10': (r'bond = 10','X', '#7a5195'),
                    '30': (r'bond = 30', 'v', '#67a9cf'),
                    '50': (r'bond = 50', 'd', '#ffa600')}

    Returns
    -------
        Displays or saves the plot.
    """
    
    for spacing in spacings:
        plt.figure(figsize=fig_size)
        tpr_per_bond_data = {}; tpr_per_bond_err = {}
        for bond_dim in bond_dims:
            tpr_per_init_data = []; tpr_per_init_err = []
            for init in initializers:
                tpr_data=[]
                for j in range(1, nruns+1):
                    if nruns == 1:
                        dir_name = save_dir + '/' + init + '/bond_' + str(bond_dim) + '/spacing_' + str(spacing)+'/'+ embedding
                    else:
                        dir_name = save_dir + '/' + init + '/bond_' + str(bond_dim) + '/spacing_' + str(spacing)+'/'+ embedding +'/run_'+str(j)
                            
                    fpr, tpr = get_roc_curve_data(np.load(dir_name + '/normal_score.npy'), np.load(dir_name + '/anomaly_score.npy'), anomaly_det=anomaly_det)
                    tpr_per_fpr = get_FPR_for_fixed_TPR(TPR_fixed, np.array(fpr), np.array(tpr), tolerance=0.01)
                    tpr_data.append(tpr_per_fpr)
                mean_error = get_mean_and_error(np.array(tpr_data))
                tpr_per_init_data.append(mean_error[0])
                tpr_per_init_err.append(mean_error[1])
            tpr_per_bond_data[bond_dim] = tpr_per_init_data
            tpr_per_bond_err[bond_dim] = tpr_per_init_err
        
        for bond_dim in bond_dims:
            data = tpr_per_bond_data[bond_dim]
            data_err = tpr_per_bond_err[bond_dim]
            plt.errorbar(list(range(len(initializers))), data, yerr=data_err, label=labels[str(bond_dim)][0],
                        linestyle='None', marker=labels[str(bond_dim)][1], capsize=3, color=labels[str(bond_dim)][2])
                
        plt.title(f'S = {spacing}, FPR = {TPR_fixed}')
        plt.ylabel('FPR')
        plt.yticks(fontsize=12)
        plt.xticks(range(len(initializers)), initializers, fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.legend(fancybox=True, frameon=True, prop={"size":10}, loc='best')

        if save_dir:
            if not os.path.exists(f'{save_dir}/results/plots/FPR'):
                os.makedirs(f'{save_dir}/results/plots/FPR')
            plt.savefig(f'{save_dir}/results/plots/FPR/spacing_{spacing}_TPR_{TPR_fixed}.pdf')
        else:
            plt.show()
        plt.close()
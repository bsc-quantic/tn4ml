import warnings

from .initializers import (
    ones,
    zeros,
    gramschmidt,
    identity,
    randn,
    rand_unitary
)
from .embeddings import (
    Embedding,
    TrigonometricEmbedding, 
    FourierEmbedding, 
    PolynomialEmbedding, 
    LinearComplementEmbedding, 
    GaussianRBFEmbedding, 
    JaxArraysEmbedding,
    PatchAmplitudeEmbedding,
    PatchEmbedding,
    embed
)

from .metrics import (
    NegLogLikelihood,
    MeanSquaredError,
    TransformedSquaredNorm,
    NoReg,
    LogFrobNorm,
    LogPowFrobNorm,
    LogReLUFrobNorm,
    QuadFrobNorm,
    LogQuadNorm,
    QuadNorm,
    SemiSupervisedLoss,
    SemiSupervisedNLL,
    Softmax,
    CrossEntropySoftmax,
    OptaxWrapper,
    CrossEntropyWeighted,
    CombinedLoss
)

from .strategy import (
    Strategy,
    Sweeps,
    Global
)

from .util import (
    gramschmidt_row,
    gramschmidt_col,
    return_digits,
    zigzag_order,
    integer_to_one_hot,
    pad_image_alternately,
    divide_into_patches,
    from_dense_to_mps,
    from_mps_to_dense,
    TrainingType
)

from .eval import (
    plot_loss,
    plot_accuracy,
    get_roc_curve_data,
    get_precision_recall_curve_data,
    get_FPR_for_fixed_TPR,
    get_TPR_for_fixed_FPR,
    get_mean_and_error,
    plot_ROC_curve_from_metrics,
    plot_ROC_curve_from_data,
    plot_PR_curve,
    compare_AUC,
    compare_TPR_per_FPR,
    compare_FPR_per_TPR
)
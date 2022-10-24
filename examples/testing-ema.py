from sklearn.metrics import roc_curve, auc, confusion_matrix
import quimb as qu
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tnad.embeddings as e
from autoray import do

def error_logquad(P, phi):
    mps = qu.tensor.tensor_network_apply_op_vec(P, phi)
    return do("power", do("add", do("log", mps.H & mps ^ all), -1.0), 2)

def get_roc_data(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    return fpr_loss, tpr_loss

P = qu.load_from_disk('Trained_SMPO\trained_P_embeddings.pkl')
# P = qu.load_from_disk('Trained_SMPO\trained_P_adam.pkl')
# P = qu.load_from_disk('Trained_SMPO\trained_P_10epoch_jofreparams.pkl')
# P = qu.load_from_disk('Trained_SMPO\trained_P_10epoch.pkl')

print(P.norm())
anomaly_score = error_logquad

_ , (test_X, test_y) = mnist.load_data()
test_data=test_X[test_y!=1][:5000] #anomaly
valid_data = test_X[test_y==1][:5000] #normal

# preprocessing - anomalies
anomalies = []
for sample in test_data:
    sample_tf = tf.constant(sample)
    sample_tf = tf.reshape(sample_tf, [1, 28, 28, 1])
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
       strides=(2, 2), padding='same')
    sample = max_pool_2d(sample_tf).numpy().reshape(14, 14)
    anomalies.append(sample/255)
    
# preprocessing - normal
normal_data = []
for sample in valid_data:
    sample_tf = tf.constant(sample)
    sample_tf = tf.reshape(sample_tf, [1, 28, 28, 1])
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
       strides=(2, 2), padding='same')
    sample = max_pool_2d(sample_tf).numpy().reshape(14, 14)
    normal_data.append(sample/255)

import tnad.FeatureMap as fm

anomalies_adscore=[]
for sample in anomalies:
    phi = e.embed(sample.flatten(), e.trigonometric(k=1))
    # phi = fm.embed(sample.flatten(), fm.trigonometric)[0]
    l = anomaly_score(P, phi)
    # loss
    anomalies_adscore.append(l)

normal_adscore=[]
for sample in normal_data:
    phi = e.embed(sample.flatten(), e.trigonometric(k=1))
    # phi = fm.embed(sample.flatten(), fm.trigonometric)[0]
    l = anomaly_score(P, phi)
    # loss
    normal_adscore.append(l)

fpr, tpr = get_roc_data(np.array(normal_adscore), np.array(anomalies_adscore))
print('normal score: ')
print(fpr)
print('anomalies score: ')
print(tpr)
auc_value = auc(fpr, tpr)

print(auc_value)

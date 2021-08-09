import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm, trange
import pickle

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

from sys import stderr
def eprint(s):  stderr.write(s)

def exists(pathname):
    return os.path.exists(pathname)

def mkdir(dir_names):
    for d in dir_names:
        if not os.path.exists(d):
            os.mkdir(d)

@tf.function
def normalize(img, label):
    return tf.cast(img, tf.float32) / 255., label

@tf.function
def get_x(img, label):
    return img

@tf.function
def get_y(img, label):
    return label

# get auc from scores
def get_score(score, labels):
    return roc_auc_score(labels, score)

def plot_roc_curve(score, labels, filename):
    fpr, tpr, _ = roc_curve(labels, score)
    plt.plot(fpr, tpr)
    plt.savefig(filename)
    plt.close()

def save_roc_curve(score, labels, filename):
    fpr, tpr, _ = roc_curve(labels, score)
    pickle.dump((fpr,tpr), open(filename,'wb'))

def get_stats(preds, labels):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    return (acc, prec, rec)

# get optimal threshold -- will result in the best performance
def opt_thresh(loss_norm, loss_anom):
        q = len(loss_norm) / (len(loss_norm) + len(loss_anom))
        return np.quantile(np.concatenate((loss_norm, loss_anom)), q)

# lp norm loss
def lp_loss(y_true, y_pred, order):
    err = abs(y_true - y_pred)
    return np.linalg.norm(err, ord=order, axis=1)/err.shape[1]

# lp norm loss of top q elements
def lp_loss_q(y_true, y_pred, order):
    err = abs(y_true - y_pred)
    q = 0.5
    thresh = np.expand_dims(np.quantile(err,q,axis=1), axis=1)
    err = np.where(err > thresh, err, np.zeros_like(err))
    return np.linalg.norm(err, ord=order, axis=1)/err.shape[1]

# helper function for anomaly detection
def detect(model, data, thresh, loss_fn):
    recons = model.predict(data)
    loss = loss_fn(flatten_ds(data), flatten_ds(recons))

    return tf.math.greater(loss, thresh).numpy()

# flatten dataset by batch
def flatten_ds(ds):
    assert tf.rank(ds) == 4
    s = ds.shape

    return tf.reshape(ds, (s[0], s[1]*s[2]*s[3]) )

# plot the gradients
def plot_grads(left, right, n=10, filename=None, lcap=None, rcap=None):
    fig, axs = plt.subplots(nrows=n, ncols=2, squeeze=True, figsize=(8, 4*n))
    for i,(l,r) in enumerate(zip(left,right)):
        if i == n:  break

        if lcap is not None:
            axs[i,0].set_title(str(lcap))

        j = axs[i,0].imshow(l[...,0], cmap="gray")
        fig.colorbar(j, ax=axs[i,0])
        axs[i,0].axis('off')

        if rcap is not None:
            axs[i,1].set_title(str(rcap))

        j = axs[i,1].imshow(r[...,0], cmap="gray")
        fig.colorbar(j, ax=axs[i,1])
        axs[i,1].axis('off')

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)
    plt.close()

# plot results
def plot_results(imgs, captions, filename, n=10):
    assert len(imgs) == len(captions)
    m = len(imgs)

    fig, axs = plt.subplots(nrows=n, ncols=m, squeeze=True, figsize=(4*m, 4*n))

    for i,(icol,cap) in enumerate(zip(imgs,captions)):
        for j,img in enumerate(icol):
            if j == n:  break

            axs[j,i].set_title(str(cap))
            if img.shape[-1] == 1:
                a = axs[j,i].imshow(img[...,0], cmap="gray")
                #fig.colorbar(a, ax=axs[j,i])
            else:
                a = axs[j,i].imshow(img)
            axs[j,i].axis('off')

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)
    plt.close()
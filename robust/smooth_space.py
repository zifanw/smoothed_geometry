import keras
import tensorflow as tf
import numpy as np

EPS = 1e-12


def BetaLoss(inputs, logit, alpha=0.3, scale=1e6):
    W = compute_W(inputs, logit)

    def get_smooth_loss(y_pred):
        A = tf.matrix_diag(y_pred)  # (B, C, C)
        p = tf.expand_dims(y_pred, 2)  # (B, C, 1)
        A = A - tf.matmul(p, tf.matrix_transpose(p))  # A-pp^T
        A = A + tf.eye(int(A.shape[1]), int(A.shape[2]), batch_shape=[1]) * EPS
        e, U = tf.linalg.eigh(A)
        e = tf.clip_by_value(e, clip_value_min=1e-20, clip_value_max=np.inf)
        Sigma = tf.matrix_diag(tf.sqrt(e))
        L = tf.matmul(U, Sigma)
        B = tf.matmul(W, L)  # (B, K, C) x (B, C, C) --> (B, K, C)
        B_T = tf.matrix_transpose(B)  # (B, C, K)
        C = tf.matmul(B_T, B)  # (B, C, C)
        C = C + tf.eye(int(C.shape[1]), int(C.shape[2]), batch_shape=[1]) * EPS
        H_e, _ = tf.linalg.eigh(C)
        ave_beta = tf.reduce_mean(
            H_e[:, -1]
        )  # Eigenvalues. Shape is [..., N]. Sorted in non-decreasing order.
        return ave_beta * scale

    def smoothCE_loss(y_true, y_pred):
        # y_pred: B, C
        target_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        smoothness = get_smooth_loss(y_pred)
        return alpha * smoothness + (1 - alpha) * target_loss

    return smoothCE_loss


def compute_W(inputs, logit):
    C = logit.shape[1]
    ws = []
    for c in range(C):
        w = tf.gradients(ys=logit[:, c], xs=[inputs])[0]
        w = tf.reduce_mean(w, axis=-1)
        w = tf.keras.backend.batch_flatten(w)
        w = tf.expand_dims(w, -1)
        ws.append(w)
    return tf.concat(ws, axis=-1)

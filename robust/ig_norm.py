import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from keras.optimizers import Adam, SGD

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_ig_norm_model(model,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        batch_size,
                        epochs,
                        epsilon,
                        ord=2,
                        beta=1.,
                        nbiter=30,
                        approx_factor=5,
                        opt=None,
                        stepsize=None,
                        loss=None):
    """
    Return a loss function to do adversarial training.

    Parameters
    ----------
    model : keras.Model or compatible
        No need to compile before calling this method.
    x_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    x_test : ndarray
        Validation features
    y_test : ndarray
        Validation labels
    batch_size : int
        Batch size to use when training
    epochs : int
        Number of epochs to train for
    epsilon : int
        Adversarial perturbation norm-bound
    ord : int (default: 2)
        Lp-norm to use. Should be either np.inf or 2.
    beta : float (default: 1.0)
        Weight for adversarial loss term. Should usually be 1.0.
    nbiter : int (default: 50)
        Number of iterations of PGD for each update
    stepsize : float or None (default: None)
        Step size for each iteration of PGD. If None,
        then epsilon/nbiter is used.
    loss : keras.losses.LossFunction or None (default: None)
        Loss function to use.
        If None, defaults to categorical_crossentropy.

    Returns
    -------
    None
        The weights of the model are updated in-place.
    """
    if stepsize is None:
        stepsize = epsilon / nbiter

    if loss is None:
        loss = keras.losses.categorical_crossentropy

    input_shape = x_train.shape[1:]
    bt_in_shape = (batch_size, ) + input_shape
    num_features = np.prod(input_shape, dtype=np.int32)
    num_classes = y_train.shape[1]

    adv_input = K.placeholder(shape=(None, ) + input_shape)
    y_true = K.placeholder(dtype=y_train.dtype, shape=(None, num_classes))
    model.y_true = y_true
    y_pred = model.call(model.input)
    y_adv = model.call(adv_input)

    ## Create IG attack operations
    def IG_operation():
        model.IG = 0
        model.IG_approx = 0
        m = 50
        for k in range(1, m + 1):
            z = model.input + float(k) / m * (adv_input - model.input)
            output = model.call(z)
            softmax_output = tf.nn.softmax(output)
            loss = tf.reduce_sum(-tf.multiply(
                model.y_true,
                tf.log(tf.clip_by_value(softmax_output, 1e-10, 1.0))),
                                 axis=1)
            grad = tf.gradients(loss, z)[0]
            model.IG += grad
            if k % approx_factor == 0:
                model.IG_approx += grad

        model.IG *= (adv_input - model.input) / float(m)
        model.IG_approx *= (adv_input - model.input) / float(m / approx_factor)
        model.IG_regularization_term = tf.norm(tf.reshape(model.IG,
                                                          shape=[-1]),
                                               ord=1)
        model.IG_regularization_term_approx = tf.norm(tf.reshape(
            model.IG_approx, shape=[-1]),
                                                      ord=1)

    IG_operation()

    ## Create loss functions
    # ce_loss = K.sum(model.losses)
    ce_loss = K.mean(loss(y_true, y_pred))

    if approx_factor > 0:
        loss_ig = model.IG_regularization_term_approx

    else:
        loss_ig = model.IG_regularization_term

    loss_t = ce_loss + beta * loss_ig
    ce_loss_fn = K.function([model.input, model.y_true], [ce_loss])
    ig_loss_fn = K.function([model.input, adv_input, model.y_true], [loss_ig])
    total_loss_fn = K.function([model.input, adv_input, model.y_true],
                               [loss_t])

    ## Create Attacker
    pgd = LinfPGDAttack(model, adv_input, epsilon, nbiter, stepsize, loss_ig)

    accuracy_t = keras.metrics.categorical_accuracy(y_true, y_pred)
    acc_fn = K.function([model.input, y_true], [accuracy_t])

    ## Create update operations
    opt = Adam() if opt is None else opt
    params = []
    for layer in model.layers:
        if not hasattr(layer, 'trainable') or layer.trainable:
            params += layer.trainable_weights
    updates = opt.get_updates(K.sum(loss_t), params)
    train_fn = K.function([model.input, adv_input, y_true],
                          [loss_t, accuracy_t],
                          updates=updates)

    ## Start training
    best_acc = 0
    for i in range(epochs):

        print('Epoch {}/{}'.format(i + 1, epochs))
        pb = Progbar(1 + len(x_train) // batch_size)
        shuf = np.random.choice(len(x_train),
                                size=(len(x_train, )),
                                replace=False)
        for j in range(len(x_train) // batch_size):
            start = time()
            cx = x_train[shuf[j:j + batch_size]]
            cy = y_train[shuf[j:j + batch_size]]
            cx_adv = pgd.perturb(cx, cy)
            loss_train, acc_tr = train_fn([cx, cx_adv, cy])
            ig_loss_train = ig_loss_fn([cx, cx_adv, cy])[0]
            pb.add(1, [('loss', loss_train), ('ig_loss', ig_loss_train),
                       ('acc', acc_tr)])
            end = time()
            print("\ntraining time: ", end - start)
        if np.mean(acc_tr) > best_acc:

            print(
                "The acc increased from {0:.3f} to {1:.3f}, new weight is saved"
                .format(best_acc, np.mean(acc_tr)))
            best_acc = np.mean(acc_tr)

        accuracy_test = 0.
        loss_test = 0.
        for j in range(len(x_test) // batch_size):
            cx = x_test[j:j + batch_size]
            cx_adv = cx
            cy = y_test[j:j + batch_size]
            den = len(x_test) // batch_size
            accuracy_test += float(acc_fn([cx, cy])[0][0]) / den
            loss_test += ce_loss_fn([cx, cy])[0] / den
        pb.add(1, [('val_loss', loss_test), ('val_acc', accuracy_test)])
        # print(accuracy_test, best_val_acc)
        model.save_weights("res_ignorm_Linf8_relu.h5")


class LinfPGDAttack:
    def __init__(self,
                 model,
                 adv_input,
                 epsilon,
                 k,
                 a,
                 loss_func,
                 random_start=True):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
        self.model = model
        self.adv_input = adv_input
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_func = loss_func
        self.grad = tf.gradients(self.loss_func, self.adv_input)[0]
        self.grad_fn = K.function(
            [self.model.input, self.adv_input, self.model.y_true], [self.grad])

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon,
                                          x_nat.shape)
        else:
            x = np.copy(x_nat)

        for i in range(self.k):
            grad = self.grad_fn([x_nat, x, y])[0]
            x += self.a * np.sign(grad)
            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0.0, 1.0)  # ensure valid pixel range

        return x

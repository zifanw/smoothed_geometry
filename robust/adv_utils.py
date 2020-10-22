import tensorflow as tf
import keras
from keras.optimizers import Adam

try:
    import cleverhans
except:
    import os
    try:
        os.system('pip install cleverhans')
        print("Complete installing the missing package.")
    except:
        raise RuntimeError(
            "Cannot install cleverhans. Please try to run the following script first \n pip install cleverhans"
        )
import cleverhans.attacks as attacks
from cleverhans.attacks import ProjectedGradientDescent as PGD
from cleverhans.utils_keras import KerasModelWrapper

from keras.datasets import mnist
import numpy as np


class AdversarialGenerator(object):
    def __init__(self, model):
        self.model = model
        self.sess = tf.Session()
        keras.backend.set_session(self.sess)

    def get_keras_model(self):
        return self.model

    def prep_generator(self, param_dict=None):
        """Subclass should overwite this method."""
        if 'ord' in param_dict:
            if param_dict['ord'] not in [np.inf, 2]:
                raise ValueError(
                    "Not a supported norm choice. ord must be np.inf or 2")
        self.params = {
            'eps':
            0.3 if 'eps' not in param_dict else param_dict['eps'],
            'ord':
            np.inf if 'ord' not in param_dict else param_dict['ord'],
            'clip_min':
            0 if 'clip_min' not in param_dict else param_dict['clip_min'],
            'clip_max':
            1. if 'clip_max' not in param_dict else param_dict['clip_max'],
            'sanity_checks':
            False if 'sanity_checks' not in param_dict else
            param_dict['sanity_checks'],
            'nb_iter':
            200 if 'nb_iter' not in param_dict else param_dict['nb_iter'],
        }
        self.wrap = KerasModelWrapper(self.model)
        self.attacker = PGD(self.wrap, sess=self.sess)

    def __call__(self, x):
        return self.model(x)

    def generate(self):
        return self.attacker.generate(self.model.layers[0].get_input_at(0),
                                      **self.params)

    def compile(self, **kwargs):
        self.sess.run(tf.global_variables_initializer())
        self.model.compile(**kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def fit_generator(self, **kwargs):
        self.model.fit_generator(**kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, **kwargs):
        return self.model.evaluate(X, y, **kwargs)

    def evaluate_generator(self, **kwargs):
        return self.model.evaluate_generator(**kwargs)

    def summary(self):
        self.model.summary()

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)


class AdversarialTraining(object):
    """Keras (Tensorflow) Implementation of Adversarial Training 
    """
    def __init__(self, robust_model):
        """__Constructor__
        
        Arguments:
            robust_model {AdversarialGenerator} -- An generaotor wrapper for keras model 
                that produces symbolic tensor of adversarial examples. 
                For details, see hw4_adv_attack.py
        """
        self.robust_model = robust_model

    def adversarial_loss(self, alpha=0.5):
        """Define the adversarial loss 

            The modified loss function to train a robust model.
            You need to implement this function.

            Hint: to compute the adversarial example x', just call 
            >>> x' = self.robust_model.generate()
            >>> x_adv = tf.stop_gradient(x_adv)

        Keyword Arguments:
            alpha {float} -- Penality of adversarial examples (default: {0.5})
        
        Returns:
            tf.Tensor -- Adversarial loss
        """
        def adv_loss(y, preds):
            """ Adversarial loss
            
            Arguments:
                y {tf.Tensor} -- Groundtruth. Shape: (N,)
                preds {tf.Tensor} -- Prediction. Shape: (N, num_class)
            
            Returns:
                tf.Tensor -- adversarial loss
            """
            # Cross-entropy on the legitimate examples
            cross_ent = keras.losses.sparse_categorical_crossentropy(y, preds)

            # Generate adversarial examples
            x_adv = self.robust_model.generate()
            # Consider the attack to be constant
            x_adv = tf.stop_gradient(x_adv)

            # Cross-entropy on the adversarial examples
            preds_adv = self.robust_model(x_adv)
            cross_ent_adv = keras.losses.sparse_categorical_crossentropy(
                y, preds_adv)

            return alpha * cross_ent + (1 - alpha) * cross_ent_adv

        return adv_loss

    def adversarial_acc_metric(self):
        """ Evaluate accuracy on adversarial examples
        
        Returns:
            tf.Tensor -- adversarial accuracy
        """
        def adv_acc(y, _):
            x_adv = self.robust_model.generate()
            x_adv = tf.stop_gradient(x_adv)
            preds_adv = self.robust_model(x_adv)
            return keras.metrics.sparse_categorical_accuracy(y, preds_adv)

        return adv_acc

    def compile(self,
                alpha=0.5,
                learning_rate=0.01,
                epsilon=0.01,
                summary=True):
        """Compile the keras model
        
        Keyword Arguments:
            alpha {float} -- adversarial loss trade-off (default: {0.5})
        """
        loss = self.adversarial_loss(alpha)
        adv_metric = self.adversarial_acc_metric()
        self.robust_model.compile(
            optimizer=Adam(lr=learning_rate, epsilon=epsilon),
            loss=loss,
            metrics=[keras.metrics.sparse_categorical_accuracy, adv_metric])
        self.robust_model(self.robust_model.model.layers[0].get_input_at(0))
        if summary:
            self.robust_model.summary()

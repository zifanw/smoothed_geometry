import numpy as np
import tensorflow as tf
from tensorflow import fft, ifft


class IGWrapper(object):
    """IGWrapper Wrapper for adding saliency operations to the keras model

    Implementation adapted from https://github.com/amiratag/InterpretationFragility
    """
    def __init__(self, NET, referece_image=None):
        self.NET = NET
        self.reference_image = referece_image
        self.create_saliency_ops()

    def create_saliency_ops(self):
        w = self.NET.input.get_shape()[1].value
        h = self.NET.input.get_shape()[2].value
        c = self.NET.input.get_shape()[3].value
        if self.reference_image is None:
            self.reference_image = np.zeros((w, h, c))
        num_classes = self.NET.output.get_shape()[-1].value
        self.NET.label_ph = tf.placeholder(tf.int32, shape=(), name="label_ph")
        self.NET.reference_image = tf.placeholder(tf.float32,
                                                  shape=(w, h, c),
                                                  name="reference_image")
        sum_logits = tf.reduce_sum(
            self.NET.output *
            tf.expand_dims(tf.one_hot(self.NET.label_ph, num_classes), 0))
        parallel_gradients = tf.gradients(sum_logits, self.NET.input)[0]
        average_gradients = tf.reduce_mean(parallel_gradients, 0)
        difference_multiplied = average_gradients * (self.NET.input[-1] -
                                                     self.NET.reference_image)
        saliency_unnormalized = tf.reduce_sum(tf.abs(difference_multiplied),
                                              -1)
        self.NET.saliency = w * h * tf.divide(
            saliency_unnormalized, tf.reduce_sum(saliency_unnormalized))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable
        self.NET.saliency_flatten = tf.reshape(self.NET.saliency, [w * h])


class SMWrapper(object):
    def __init__(self, NET):
        self.NET = NET
        self.create_saliency_ops()

    def create_saliency_ops(self):

        logit = self.NET.layers[-1].input
        w = self.NET.input.get_shape()[1].value
        h = self.NET.input.get_shape()[2].value
        num_classes = self.NET.output.get_shape()[-1].value
        self.NET.label_ph = tf.placeholder(tf.int32, shape=())
        gradient = tf.gradients(
            tf.reduce_sum(logit[0] *
                          tf.one_hot(self.NET.label_ph, num_classes)),
            self.NET.input)[0]
        gradient = gradient * self.NET.input[-1]
        saliency_unnormalized = tf.reduce_sum(tf.abs(gradient), -1)
        self.NET.saliency = w * h * tf.divide(
            saliency_unnormalized, tf.reduce_sum(saliency_unnormalized))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable
        self.NET.saliency_flatten = tf.reshape(self.NET.saliency, [w * h])


class BallWrapper(object):
    """Noise Wrapper for Smooth and Uniform Gradient operations to the keras model

    Implementation adapted from https://github.com/amiratag/InterpretationFragility
    """
    def __init__(self, NET, referece_image=None):
        self.NET = NET
        self.reference_image = referece_image
        self.create_saliency_ops()

    def create_saliency_ops(self):
        w = self.NET.input.get_shape()[1].value
        h = self.NET.input.get_shape()[2].value
        c = self.NET.input.get_shape()[3].value
        if self.reference_image is None:
            self.reference_image = np.zeros((w, h, c))
        num_classes = self.NET.output.get_shape()[-1].value
        self.NET.label_ph = tf.placeholder(tf.int32, shape=(), name="label_ph")
        sum_logits = tf.reduce_sum(
            self.NET.output *
            tf.expand_dims(tf.one_hot(self.NET.label_ph, num_classes), 0))
        parallel_gradients = tf.gradients(sum_logits, self.NET.input)[0]
        average_gradients = tf.reduce_mean(parallel_gradients, 0)
        difference_multiplied = average_gradients * self.NET.input[-1]
        saliency_unnormalized = tf.reduce_sum(tf.abs(difference_multiplied),
                                              -1)
        self.NET.saliency = w * h * tf.divide(
            saliency_unnormalized, tf.reduce_sum(saliency_unnormalized))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable
        self.NET.saliency_flatten = tf.reshape(self.NET.saliency, [w * h])

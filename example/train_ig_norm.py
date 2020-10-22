# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append("../")

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from models.keras_model import res_20
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from robust.ig_norm import train_ig_norm_model
from keras.optimizers import Adam, SGD
from time import time
tf.reset_default_graph()

# %%


def get_session(number=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


#
sess = get_session()
K.set_session(sess)
batch_size = 16
nb_epochs = 50

model = res_20(input_shape=(32, 32, 3), activation='relu')
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train / 255.
x_test = x_test / 255.

train_ig_norm_model(model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    batch_size,
                    epochs=50,
                    epsilon=8 / 255.,
                    ord=np.inf,
                    beta=0.1,
                    nbiter=7,
                    approx_factor=10,
                    opt=Adam(),
                    stepsize=2 / 255.,
                    loss=None)
# %%

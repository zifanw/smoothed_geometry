# %%
import keras
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append("../")

from time import time
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
from keras.optimizers import Adam, SGD
import numpy as np
import gzip
import pickle as pkl
from models.keras_model import res_20
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import argparse
from robust.smooth_space import BetaLoss
from time import time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training',
                    type=int,
                    default=1,
                    help='training or testing')
parser.add_argument('--weight_file',
                    type=str,
                    default="weight/res_smooth_relu",
                    help='the weight file')
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--scale', type=float, default=0.3)


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self.epoch_time_start)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def get_session(number=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


args = parser.parse_args()
# %%
sess = get_session()
K.set_session(sess)
batch_size = 32
nb_epochs = 50

model = res_20(input_shape=(32, 32, 3), activation='relu')

smooth_loss = BetaLoss(model.input,
                       model.layers[-1].input,
                       alpha=args.alpha,
                       scale=args.scale)

model.compile(loss=smooth_loss, optimizer=Adam(), metrics=['accuracy'])

weights_file = args.weight_file + "_" + str(int(10 * args.alpha)) + ".h5"
if args.training:
    early_stopper = EarlyStopping(monitor='acc', min_delta=1e-4, patience=20)
    model_checkpoint = ModelCheckpoint(weights_file,
                                       monitor="acc",
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=1)
    time_callback = TimeHistory()
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [model_checkpoint, lr_reducer, lr_scheduler, time_callback]

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=nb_epochs,
              callbacks=callbacks,
              shuffle=True,
              validation_data=(x_test, y_test),
              verbose=1)

    times = time_callback.times
    print("Training time per epoch: ", times)
    model.save_weights(weights_file)
else:
    _, (x_test, y_test) = cifar10.load_data()
    y_test = keras.utils.to_categorical(y_test, 10)
    model.load_weights(weights_file)
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test Accuracy: ", score[1])

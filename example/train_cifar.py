import keras
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append("../")

import keras.backend as K
from keras.datasets import cifar10
from keras.optimizers import Adam
import numpy as np

from models.keras_model import res_20
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training',
                    type=int,
                    default=1,
                    help='training or testing')
parser.add_argument('--weight_file',
                    type=str,
                    default=None,
                    help='the weight file')


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


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


args = parser.parse_args()
tf.reset_default_graph()
sess = get_session()
K.set_session(sess)
batch_size = 35
nb_epochs = 50

model = res_20(input_shape=(32, 32, 3), activation='relu')
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr_schedule(0)),
              metrics=['accuracy'])

if args.weight_file is None:
    weights_file = "weight/res_nat_relu.h5"
else:
    weights_file = args.weight_file

if args.training:
    early_stopper = EarlyStopping(monitor='vacc', min_delta=1e-4, patience=20)
    model_checkpoint = ModelCheckpoint(weights_file,
                                       monitor="acc",
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=1)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [model_checkpoint, lr_reducer, lr_scheduler]

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
else:
    _, (x_test, y_test) = cifar10.load_data()
    y_test = keras.utils.to_categorical(y_test, 10)
    model.load_weights(weights_file)
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test Accuracy: ", score[1])

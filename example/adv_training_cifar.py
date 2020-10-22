from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append("../")

import logging
import numpy as np
import tensorflow as tf
from robust.adv_utils import AdversarialGenerator, AdversarialTraining
import keras
from models.keras_model import res_20
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam, SGD
from keras.datasets import cifar10


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


logging.getLogger('tensorflow').disabled = True
# Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

im_shape = x_train[0].shape

model = res_20(input_shape=(32, 32, 3), activation='relu')
robust_model = AdversarialGenerator(model)
param_dict = {
    'ord': 2,  ## Lp distance : np.inf or 2
    'eps': 0.25 * 255,  ## Maximum allowed perturbation
    'clip_min': 0.0,  ## Valid data upper-bound
    'clip_max': 255.  ## Valid data lower-bound
}

robust_model.prep_generator(param_dict)
robust_trainer = AdversarialTraining(robust_model)

early_stopper = EarlyStopping(monitor='acc', min_delta=1e-4, patience=20)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
weights_file = "weight/robust_resnet.h5"
model_checkpoint = ModelCheckpoint(weights_file,
                                   monitor="acc",
                                   save_best_only=True,
                                   save_weights_only=True,
                                   verbose=1)
callbacks = [model_checkpoint, lr_reducer, lr_scheduler]
# Create classifier wrapper
robust_trainer.compile(alpha=0.5)
robust_trainer.robust_model.fit(x_train,
                                y_train,
                                validation_data=(x_test, y_test),
                                callbacks=callbacks,
                                epochs=50,
                                shuffle=True,
                                batch_size=32)
_, acc, adv_acc = robust_trainer.robust_model.evaluate(x_test,
                                                       y_test,
                                                       batch_size=32,
                                                       verbose=0)
print('Test accuracy on legitimate examples: %0.4f' % acc)
print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

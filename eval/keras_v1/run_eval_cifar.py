from eval_attr import Eval
import keras
import tensorflow as tf
import keras.backend as K
from keras_model import res_20
import numpy as np
import pickle
import argparse
from keras.datasets import cifar10

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='run attacks to verify networks')
parser.add_argument('--weight_file',
                    type=str,
                    default="weight/res_nat_relu.h5")
parser.add_argument('--attr_fn', type=str, default='saliency')
parser.add_argument('--k_top', type=int, default=1000)
parser.add_argument('--num_steps', type=int, default=50)
parser.add_argument('--epsilon', type=float, default=8.0)
parser.add_argument('--max_iters', type=int, default=300)
parser.add_argument('--measure', type=str, default='mass_center')
parser.add_argument('--step_size', type=float, default=0.5)
parser.add_argument('--attack_method', type=str, default='mass_center')
parser.add_argument('--model_arch', type=str, default='resnet')
parser.add_argument('--pixel_max', type=float, default=255.0)
parser.add_argument('--saving_prefix', type=str, default='result/test_report_')

args = parser.parse_args()

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
sess = tf.Session(config=config_gpu)
K.set_session(sess)

model = res_20((32, 32, 3), num_classes=10, activation='relu')
model.load_weights(args.weight_file)

soft_model = res_20((32, 32, 3), num_classes=10, activation='softplus')
soft_model.load_weights(args.weight_file)

(_, _), (x_test, _) = cifar10.load_data()
x_test = x_test.astype(np.float)
x_test /= (255. / args.pixel_max)
print(x_test.max())
pred = np.argmax(model.predict(x_test), axis=-1)
# print(model.predict(x_test)[0])

evaluate = Eval(sess, model, soft_model=soft_model, verbose=1)
evaluate.prep_eval(attr_function=args.attr_fn,
                   reference_image=np.zeros_like(x_test[0]))

target_map = np.load("target_map.npy")

param_dict = {
    'k_top': args.k_top,
    'num_steps': args.num_steps,
    'epsilon': args.epsilon,
    'max_iters': args.max_iters,
    'measure': args.measure,
    'step_size': args.step_size,
    'attack_method': args.attack_method,
    'target_map': target_map,
    'pixel_max': args.pixel_max,
}

param_dict['epsilon'] = param_dict['epsilon'] / (255. /
                                                 param_dict['pixel_max'])
param_dict['step_size'] = param_dict['step_size'] / (255. /
                                                     param_dict['pixel_max'])

print(args.attr_fn, "   ", param_dict['epsilon'], "   ", args.weight_file)
print("True Epsilon", "   ", param_dict['epsilon'] * 255)
evaluation_result = evaluate.run_attack(x_test[:500],
                                        pred[:500],
                                        mean_image=None,
                                        param_dict=param_dict,
                                        analysis=True)
save_path = args.saving_prefix + args.model_arch + "_" + args.attr_fn + "_" + "Linf_" + str(
    int(args.epsilon)) + "_" + args.attack_method + ".pkl"
with open(save_path, 'wb') as f:
    pickle.dump(evaluation_result, f, pickle.HIGHEST_PROTOCOL)
    f.close()
print("Result is saved to " + "\"" + save_path + "\"")

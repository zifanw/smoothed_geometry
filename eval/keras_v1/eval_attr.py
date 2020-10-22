import sys
sys.path.append("../")
sys.path.append("../../")
import numpy as np
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from utils import IntegratedGradientsAttack
from utils import SimpleGradientAttack
from utils import SmoothGradientsAttack
from utils import UniGradientsAttack
from wrappers import IGWrapper, SMWrapper, BallWrapper
from tqdm import trange


class Eval(object):
    def __init__(self,
                 sess,
                 model,
                 soft_model=None,
                 verbose=1,
                 reset_cycle=1000):
        self.sess = sess
        self.model = model
        if soft_model is None:
            self.soft_model = model
        else:
            self.soft_model = soft_model
        self.__well_prep = False
        self.verbose = 1
        self.reset_cycle = reset_cycle
        if self.reset_cycle > 0:
            self.model.save("temp.h5")
            self.soft_model.save("temp_soft.h5")

    def reset_graph(self):
        tf.reset_default_graph()
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_gpu)
        K.set_session(self.sess)
        self.model = load_model("temp.h5")
        self.soft_model = load_model("temp_soft.h5")
        self.prep_eval(attr_function=self.attr_function)

    def prep_eval(self,
                  attr_function='saliency',
                  sigma=0.4,
                  reference_image=None):
        if attr_function not in [
                'saliency', 'integrated', 'smooth', 'laplace', 'uniform'
        ]:
            raise ValueError("method must be saliency, integrated or smooth")

        if attr_function == 'saliency':
            self.model_wrapper = SMWrapper(self.model)
            self.soft_model_wrapper = SMWrapper(self.soft_model)
        elif attr_function == 'laplace':
            self.model_wrapper = LaplaceWrapper(self.model, sigma=sigma)
            self.soft_model_wrapper = LaplaceWrapper(self.soft_model,
                                                     sigma=sigma)
        elif attr_function == 'integrated':
            self.model_wrapper = IGWrapper(self.model, reference_image)
            self.soft_model_wrapper = IGWrapper(self.soft_model,
                                                reference_image)
        elif attr_function in ['smooth', 'uniform']:
            self.model_wrapper = BallWrapper(self.model)
            self.soft_model_wrapper = BallWrapper(self.soft_model)

        self.__well_prep = True
        self.attr_function = attr_function

    def run_attack(self,
                   features,
                   labels,
                   mean_image=None,
                   param_dict=None,
                   analysis=True):

        # module = self._init_attack(features[0], labels[0], mean_image,
        #                            param_dict)
        if self.verbose == 1:
            generator = trange(len(features))
        else:
            generator = range(len(features))

        result = []
        result_adv_images = []
        module = self._init_attack(features[0], labels[0], mean_image,
                                   param_dict)
        for i in generator:
            if i > 0 and i % self.reset_cycle == 0:
                self.reset_graph()
                module = self._init_attack(features[i], labels[i], mean_image,
                                           param_dict)
            module.update_new_image(features[i], labels[i],
                                    param_dict['target_map'])
            output = module.iterative_attack(param_dict['attack_method'],
                                             epsilon=param_dict['epsilon'],
                                             alpha=param_dict['step_size'],
                                             iters=param_dict['max_iters'],
                                             measure=param_dict['measure'])
            if output is not None:
                result.append(output)
                if len(output) > 4:
                    result_adv_images.append(module.perturbed_image)

        if analysis:
            return self.anylyze(result, result_adv_images)
        return result, result_adv_images

    def _init_attack(self, test_image, test_label, mean_image, param_dict):
        if not self.__well_prep:
            raise RuntimeError("Please prepare the wrapper first.")

        param_dict = self.check_param(param_dict)
        if mean_image is None:
            mean_image = np.zeros_like(test_image)

        if self.attr_function in ['saliency', 'laplace']:
            module = SimpleGradientAttack(mean_image,
                                          self.sess,
                                          test_image,
                                          test_label,
                                          NET=self.model_wrapper.NET,
                                          NET2=self.soft_model_wrapper.NET,
                                          k_top=param_dict['k_top'],
                                          pixel_max=param_dict['pixel_max'],
                                          target_map=param_dict['target_map'])

        elif self.attr_function == 'integrated':
            module = IntegratedGradientsAttack(
                self.sess,
                mean_image,
                test_image,
                test_label,
                NET=self.model_wrapper.NET,
                NET2=self.soft_model_wrapper.NET,
                k_top=param_dict['k_top'],
                num_steps=param_dict['num_steps'],
                reference_image=self.model_wrapper.reference_image,
                pixel_max=param_dict['pixel_max'],
                target_map=param_dict['target_map'])

        elif self.attr_function == 'smooth':
            module = SmoothGradientsAttack(
                self.sess,
                mean_image,
                test_image,
                test_label,
                NET=self.model_wrapper.NET,
                NET2=self.soft_model_wrapper.NET,
                k_top=param_dict['k_top'],
                num_steps=param_dict['num_steps'],
                reference_image=self.model_wrapper.reference_image,
                pixel_max=param_dict['pixel_max'],
                target_map=param_dict['target_map'])
        elif self.attr_function == 'uniform':
            module = UniGradientsAttack(
                self.sess,
                mean_image,
                test_image,
                test_label,
                NET=self.model_wrapper.NET,
                NET2=self.soft_model_wrapper.NET,
                k_top=param_dict['k_top'],
                num_steps=param_dict['num_steps'],
                reference_image=self.model_wrapper.reference_image,
                pixel_max=param_dict['pixel_max'],
                target_map=param_dict['target_map'])

        return module

    @staticmethod
    def anylyze(stat, adv_imgs):
        success_rate = 0
        total_intersection = []
        total_correlation = []
        total_center_dislocation = []
        total_confidence = []
        total_perturb_size = []
        total_cosine = []
        success_rate = 0
        idx = []
        for i, s in enumerate(stat):
            if len(s) == 6:
                total_intersection.append(s[0])
                total_correlation.append(s[1])
                total_center_dislocation.append(s[2])
                total_confidence.append(s[3])
                total_perturb_size.append(s[4])
                total_cosine.append(s[5])
                idx.append(i)
                success_rate += 1
        if len(adv_imgs) > 0:
            result = dict()
            result['index'] = idx
            result['intersection'] = np.array(total_intersection)
            result['correlation'] = np.array(total_correlation)
            result['center_dislocation'] = np.array(total_center_dislocation)
            result['confidence'] = np.array(total_confidence)
            result['perturb_size'] = np.array(total_perturb_size)
            result['success_rate'] = success_rate / len(stat)
            result['cosine_distance'] = np.array(total_cosine)
            result['adv_imgs'] = adv_imgs
            return result
        else:
            return None

    @staticmethod
    def get_session():
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        return tf.Session(config=config_gpu)

    @staticmethod
    def check_param(param_dict):
        if param_dict is None:
            raise ValueError("Please provide param_dict")
        else:
            if 'k_top' not in param_dict:
                param_dict['k_top'] = 1000
            if 'num_steps' not in param_dict:
                param_dict['num_steps'] = 50
            if 'epsilon' not in param_dict:
                param_dict['epsilon'] = 8
            if 'max_iters' not in param_dict:
                param_dict['max_iters'] = 300
            if 'measure' not in param_dict:
                param_dict['measure'] = 'topK'
            if 'step_size' not in param_dict:
                param_dict['step_size'] = 0.1 * param_dict['epsilon']
            if 'attack_method' not in param_dict:
                param_dict['attack_method'] = 'topK'
            if 'target_map' not in param_dict:
                param_dict['target_map'] = None

        return param_dict

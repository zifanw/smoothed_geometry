import numpy as np
import tensorflow as tf
import random
import _pickle as pkl
import matplotlib.pyplot as plt
from pylab import rcParams
import scipy
import scipy.stats as stats
from tensorflow.python.ops import gen_nn_ops
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

MEAN_IMAGE = np.zeros((1, 227, 227, 3)).astype(np.float32)
MEAN_IMAGE[:, :, :, 0] = 103.939
MEAN_IMAGE[:, :, :, 1] = 116.779
MEAN_IMAGE[:, :, :, 2] = 123.68
EPSILON = 1e-12
MIN_INPUT = -MEAN_IMAGE
MAX_INPUT = 255 * np.ones_like(MEAN_IMAGE).astype(np.float32) - MEAN_IMAGE


def dataReader():
    X = np.zeros((100, 227, 227, 3))
    y = np.zeros(100)
    for num in range(4):
        with open(
                "./ImagenetValidationSamples/imagenet_sample_{}.pkl".format(
                    num), "rb") as inputs:
            dic_temp = pkl.load(inputs)
            X[num * 20:num * 20 + 20] = dic_temp["X"]
            y[num * 20:num * 20 + 20] = dic_temp["y"]
            labels = dic_temp["labels"]
    return X, y.astype(int), labels


class SimpleGradientAttack(object):
    def __init__(self,
                 mean_image,
                 sess,
                 test_image,
                 original_label,
                 NET,
                 NET2=None,
                 k_top=1000,
                 target_map=None,
                 pixel_max=255.):
        """
        Args:            
            mean_image: The mean image of the data set(The assumption is that the images are mean subtracted)
            sess: Session containing model(and surrogate model's) graphs
            test_image: Mean subtracted test image
            original_label: True label of the image
            NET: Original neural network. It's assumed that NET.saliency is the saliency map tensor and 
            NET.saliency_flatten is its flatten version.
            NET2: Surrogate neural network with the same structure and weights of the orignal network but
            with activations replaced by softplus function
            (necessary only when the activation function of the original function
            does not have second order gradients, ex: ReLU). It's assumed that NET.saliency is the 
            saliency map tensor and NET2.saliency_flatten is its flatten version.
            k_top: the topK parameter of the attack (refer to the original paper)
            pixel_max: the maximum pixel value in the image.
        """
        self.pixel_max = pixel_max
        if len(test_image.shape) != 3:
            raise ValueError("Invalid Test Image Dimensions")
        if NET.input.get_shape()[-3]!=test_image.shape[-3] or NET.input.get_shape()[-2]!=test_image.shape[-2] or\
        NET.input.get_shape()[-1]!=test_image.shape[-1]:
            raise ValueError(
                "Model's input dimensions is not Compatible with the provided test image!"
            )
        if self.check_prediction(sess, original_label, test_image, NET):
            return
        self.sess = sess
        self.target_map = target_map
        self.create_extra_ops(NET, test_image.shape[-3], test_image.shape[-2],
                              k_top)
        if NET2 is None:
            NET2 = NET
        else:
            self.create_extra_ops(NET2, test_image.shape[-3],
                                  test_image.shape[-2], k_top)
            if NET2.input.get_shape()[-3]!=test_image.shape[-3] or NET2.input.get_shape()[-2]!=test_image.shape[-2] or\
            NET2.input.get_shape()[-1]!=test_image.shape[-1]:
                raise ValueError(
                    "Surrogate model's input dimensions is not Compatible with the provided test image!"
                )
        self.NET = NET
        self.NET2 = NET2
        self.test_image = test_image
        self.original_label = original_label
        self.mean_image = mean_image
        self.k_top = k_top
        w, h, c = self.mean_image.shape
        self.topk_ph = tf.placeholder(tf.float32,
                                      shape=[w * h],
                                      name='topk_ph')
        self.mass_center_ph = tf.placeholder(tf.float32,
                                             shape=[2],
                                             name='mass_center_ph')
        self.target_map_ph = tf.placeholder(tf.float32,
                                            shape=[w, h],
                                            name='target_map_ph')

        self.original_output = self.NET.predict(test_image[None, :])
        _, num_class = self.original_output.shape

        self.original_output_ph = tf.placeholder(
            tf.float32, shape=[None, num_class],
            name='original_output_ph')  # only for the manipulation attack

        self.beta_0_ph = tf.placeholder(tf.float32, name='beta_0')
        self.beta_1_ph = tf.placeholder(tf.float32, name='beta_1')
        self.create_attack_ops(NET2, test_image.shape[-3],
                               test_image.shape[-2])
        self.update_new_image(test_image, original_label)

    def update_new_image(self, test_image, original_label, target_map=None):
        w, h, c = test_image.shape
        self.test_image = test_image
        self.original_label = original_label
        assert self.check_prediction(self.sess, original_label, test_image,
                                     self.NET) == False
        if target_map is not None:
            self.target_map = target_map
            self.original_output = self.NET2.predict(test_image[None, :])
        self.saliency1, self.topK = self.run_model(
            self.sess, [self.NET.saliency, self.NET.top_idx], self.test_image,
            self.NET)
        self.saliency1_flatten = np.reshape(
            self.saliency1, [test_image.shape[-3] * test_image.shape[-2]])
        elem1 = np.argsort(np.reshape(self.saliency1, [w * h]))[-self.k_top:]
        self.elements1 = np.zeros(w * h)
        self.elements1[elem1] = 1
        self.original_topk = self.elements1
        self.mass_center1 = self.run_model(self.sess, self.NET.mass_center,
                                           self.test_image,
                                           self.NET).astype(int)
        self.original_mass_center = self.mass_center1

    def check_prediction(self, sess, original_label, image, NET):
        """ If the network's prediction is incorrect in the first place, attacking has no meaning."""
        predicted_scores = sess.run(
            NET.output,
            feed_dict={NET.input: image if len(image.shape) == 4 else [image]})
        if np.argmax(predicted_scores, 1) != original_label:
            print("Network's Prediction is Already Incorrect!")
            return True
        else:
            self.original_confidence = np.max(predicted_scores)
            return False

    def create_extra_ops(self, NET, w, h, k_top):

        top_val, NET.top_idx = tf.nn.top_k(NET.saliency_flatten, k_top)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        NET.mass_center = tf.stack([
            tf.reduce_sum(NET.saliency * x_mesh) / (w * h),
            tf.reduce_sum(NET.saliency * y_mesh) / (w * h)
        ])

    def create_attack_ops(self, NET, w, h):
        topK_loss = tf.reduce_sum((NET.saliency_flatten * self.topk_ph))
        self.topK_direction = -tf.gradients(topK_loss, NET.input)[0]

        mass_center_loss = -tf.reduce_sum(
            (NET.mass_center - self.mass_center_ph)**2)
        self.mass_center_direction = -tf.gradients(mass_center_loss,
                                                   NET.input)[0]
        if self.target_map is not None:
            target_dis = tf.keras.losses.MSE(self.target_map_ph, NET.saliency)
            output_dis = tf.keras.losses.MSE(self.original_output_ph,
                                             NET.output)
            target_loss = tf.reduce_mean(
                target_dis) * self.beta_0_ph + self.beta_1_ph * tf.reduce_mean(
                    output_dis)
            self.debug = target_loss

            self.target_direction = -tf.gradients(target_loss, NET.input)[0]

    def run_model(self, sess, operation, feed, NET):

        if len(feed.shape) == 3:
            if hasattr(self, "original_topk") and hasattr(
                    self, "original_mass_center"):
                if hasattr(self, "use_target") and self.use_target:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: [feed],
                                        NET.label_ph: self.original_label,
                                        self.topk_ph: self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center,
                                        self.beta_0_ph: self.beta_0,
                                        self.beta_1_ph: self.beta_1,
                                        self.original_output_ph:
                                        self.original_output,
                                        self.target_map_ph: self.target_map
                                    })
                else:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: [feed],
                                        NET.label_ph:
                                        self.original_label,
                                        self.topk_ph:
                                        self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center
                                    })
            else:
                return sess.run(operation,
                                feed_dict={
                                    NET.input: [feed],
                                    NET.label_ph: self.original_label,
                                })
        elif len(feed.shape) == 4:
            if hasattr(self, "original_topk") and hasattr(
                    self, "original_mass_center"):
                if hasattr(self, "use_target") and self.use_target:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: feed,
                                        NET.label_ph: self.original_label,
                                        self.topk_ph: self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center,
                                        self.beta_0_ph: self.beta_0,
                                        self.beta_1_ph: self.beta_1,
                                        self.original_output_ph:
                                        self.original_output,
                                        self.target_map_ph: self.target_map
                                    })
                else:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input:
                                        feed,
                                        NET.label_ph:
                                        self.original_label,
                                        self.topk_ph:
                                        self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center
                                    })
            else:
                return sess.run(operation,
                                feed_dict={
                                    NET.input: feed,
                                    NET.label_ph: self.original_label,
                                })
        else:
            raise RuntimeError("Input image shape invalid!")

    def give_simple_perturbation(self, attack_method, in_image):
        w, h, c = self.test_image.shape
        if attack_method == "random":
            perturbation = np.random.normal(size=(w, h, c))
        elif attack_method == "topK":
            perturbation = self.run_model(self.sess, self.topK_direction,
                                          in_image, self.NET2)
            perturbation = np.reshape(perturbation, [w, h, c])
        elif attack_method == "mass_center":
            perturbation = self.run_model(self.sess,
                                          self.mass_center_direction, in_image,
                                          self.NET2)
            perturbation = np.reshape(perturbation, [w, h, c])
        elif attack_method == "target":
            self.use_target = True
            if self.target_map is None:
                raise ValueError("No target region determined!")
            else:
                perturbation = self.run_model(self.sess, self.target_direction,
                                              in_image, self.NET2)
                debug = self.run_model(self.sess, self.debug, in_image,
                                       self.NET2)
                print("MSE: ", debug)
                perturbation = np.reshape(perturbation, [w, h, c])
        return np.sign(perturbation)

    def apply_perturb(self, in_image, pert, alpha, bound=8 / 255, ord=np.inf):
        if self.mean_image is None:
            self.mean_image = np.zeros_like(in_image)
        # out_image = self.test_image + np.clip(
        #     in_image + alpha * np.sign(pert) - self.test_image, -bound, bound)
        d = in_image + alpha * np.sign(pert) - self.test_image
        d_norm = np.linalg.norm(d.flatten(), ord=ord)
        if d_norm > bound:
            proj_ratio = bound / np.linalg.norm(d.flatten(), ord=ord)
        else:
            proj_ratio = 1
        out_image = self.test_image + d * proj_ratio
        out_image = np.clip(out_image, -self.mean_image,
                            self.pixel_max - self.mean_image)
        return out_image

    def check_measure(self, test_image_pert, measure):

        prob = self.run_model(self.sess, self.NET.output, test_image_pert,
                              self.NET)
        if np.argmax(prob, 1) == self.original_label:
            if measure == "intersection":
                top2 = self.run_model(self.sess, self.NET.top_idx,
                                      test_image_pert, self.NET)
                criterion = float(len(np.intersect1d(self.topK,
                                                     top2))) / self.k_top
            elif measure == "correlation":
                saliency2_flatten = self.run_model(self.sess,
                                                   self.NET.saliency_flatten,
                                                   test_image_pert, self.NET)
                criterion = scipy.stats.spearmanr(self.saliency1_flatten,
                                                  saliency2_flatten)[0]
            elif measure == "mass_center":
                center2 = self.run_model(self.sess, self.NET.mass_center,
                                         test_image_pert, self.NET).astype(int)
                criterion = -np.linalg.norm(self.mass_center1 - center2)
            elif measure == "cosine":
                saliency2_flatten = self.run_model(self.sess,
                                                   self.NET.saliency_flatten,
                                                   test_image_pert, self.NET)

                criterion = scipy.spatial.distance.cosine(
                    self.saliency1_flatten, saliency2_flatten)
            else:
                raise ValueError("Invalid measure!")
            return criterion
        else:
            return 1.

    def iterative_attack(self,
                         attack_method,
                         epsilon,
                         iters=100,
                         alpha=1,
                         beta_0=1e11,
                         beta_1=1e6,
                         measure="intersection",
                         target=None):
        """
        Args:
            attack_method: One of "mass_center", "topK" or "random"
            epsilon: Allowed maximum $ell_infty$ of perturbations, eg:8
            iters: number of maximum allowed attack iterations
            alpha: perturbation size in each iteration of the attack
            measure: measure for success of the attack (one of "correlation", "mass_center" or "intersection")
            beta_0: parameter for manipulate (target) attack 
            beta_1: parameter for manipulate (target) attack 
        Returns:
            intersection: The portion of the top K salient pixels in the original picture that are in the 
            top K salient pixels of the perturbed image devided
            correlation: The rank correlation between saliency maps of original and perturbed image
            center_dislocation: The L2 distance between saliency map mass centers in original and perturbed images
            confidence: The prediction confidence of the perturbed image
        """
        self.beta_0 = beta_0
        self.beta_1 = beta_1

        w, h, c = self.test_image.shape
        test_image_pert = self.test_image.copy()
        min_criterion = 1.
        perturb_size = 0.
        last_image = None
        for counter in range(iters):
            pert = self.give_simple_perturbation(attack_method,
                                                 test_image_pert)
            test_image_pert = self.apply_perturb(test_image_pert, pert, alpha,
                                                 epsilon)
            criterion = self.check_measure(test_image_pert, measure)

            if criterion < min_criterion:
                min_criterion = criterion
                self.perturbed_image = test_image_pert.copy()
                perturb_size = np.max(
                    np.abs(self.test_image - self.perturbed_image))
            else:
                pass

        if criterion == 1.:
            return None
        predicted_scores = self.run_model(self.sess, self.NET.output,
                                          self.perturbed_image, self.NET)
        confidence = np.max(predicted_scores)
        self.saliency2, self.top2, self.mass_center2= self.run_model\
        (self.sess, [self.NET.saliency, self.NET.top_idx, self.NET.mass_center], self.perturbed_image, self.NET)
        correlation = scipy.stats.spearmanr(
            self.saliency1_flatten, np.reshape(self.saliency2, [w * h]))[0]
        intersection = float(len(np.intersect1d(self.topK,
                                                self.top2))) / self.k_top
        center_dislocation = np.linalg.norm(self.mass_center1 -
                                            self.mass_center2.astype(int))
        cos_distance = scipy.spatial.distance.cosine(
            self.saliency1_flatten, np.reshape(self.saliency2, [w * h]))
        return intersection, correlation, center_dislocation, confidence, perturb_size, cos_distance


class IntegratedGradientsAttack(object):
    def __init__(self,
                 sess,
                 mean_image,
                 test_image,
                 original_label,
                 NET,
                 NET2=None,
                 k_top=1000,
                 num_steps=100,
                 reference_image=None,
                 target_map=None,
                 pixel_max=255.):
        """
        Args:
            mean_image: The mean image of the data set(The assumption is that the images are mean subtracted)
            sess: Session containing model(and surrogate model's) graphs
            test_image: Mean subtracted test image
            original_label: True label of the image
            NET: Original neural network. It's assumed that NET.saliency is the saliency map tensor and 
            NET.saliency_flatten is its flatten version.
            NET2: Surrogate neural network with the same structure and weights of the orignal network but
            with activations replaced by softplus function
            (necessary only when the activation function of the original function
            does not have second order gradients, ex: ReLU). It's assumed that NET.saliency is the 
            saliency map tensor and NET2.saliency_flatten is its flatten version.
            k_top: the topK parameter of the attack (refer to the original paper)
            num_steps: Number of steps in Integrated Gradients Algorithm
            reference_image: Mean subtracted reference image of Integrated Gradients Algorithm
            pixel_max: the maximum pixel value in the image.
        """
        self.pixel_max = pixel_max
        if len(test_image.shape) != 3:
            raise ValueError("Invalid Test Image Dimensions")
        if sum([
                NET.input.get_shape()[-i] != test_image.shape[-i]
                for i in [1, 2, 3]
        ]):
            raise ValueError(
                "Model's input dimensions is not Compatible with the provided test image!"
            )
        if self.check_prediction(sess, original_label, test_image, NET):
            return
        self.sess = sess
        self.target_map = target_map
        self.create_extra_ops(NET, test_image.shape[-3], test_image.shape[-2],
                              k_top)
        if NET2 is None:
            NET2 = NET
        else:
            self.create_extra_ops(NET2, test_image.shape[-3],
                                  test_image.shape[-2], k_top)
            if sum([
                    NET2.input.get_shape()[-i] != test_image.shape[-i]
                    for i in [1, 2, 3]
            ]):
                raise ValueError(
                    "Surrogate model's input dimensions is not Compatible with the provided test image!"
                )
        self.NET = NET
        self.NET2 = NET2
        self.test_image = test_image
        self.original_label = original_label
        self.mean_image = mean_image
        self.k_top = k_top
        self.num_steps = num_steps
        self.reference_image = np.zeros_like(
            test_image) if reference_image is None else reference_image

        w, h, c = self.mean_image.shape
        self.topk_ph = tf.placeholder(tf.float32,
                                      shape=[w * h],
                                      name='topk_ph')
        self.mass_center_ph = tf.placeholder(tf.float32,
                                             shape=[2],
                                             name='mass_center_ph')
        self.target_map_ph = tf.placeholder(tf.float32,
                                            shape=[w, h],
                                            name='target_map_ph')
        self.beta_0_ph = tf.placeholder(tf.float32, name='beta_0')
        self.beta_1_ph = tf.placeholder(tf.float32, name='beta_1')
        self.original_output = self.NET.predict(test_image[None, :])
        _, num_class = self.original_output.shape

        self.original_output_ph = tf.placeholder(
            tf.float32, shape=[None, num_class],
            name='original_output_ph')  # only for the manipulation attack

        self.create_attack_ops(self.NET2, test_image.shape[-3],
                               test_image.shape[-2])

    def check_prediction(self, sess, original_label, image, NET):
        """ If the network's prediction is incorrect in the first place, attacking has no meaning."""
        predicted_scores = sess.run(
            NET.output,
            feed_dict={NET.input: image if len(image.shape) == 4 else [image]})
        if np.argmax(predicted_scores, 1) != original_label:
            print("Network's Prediction is Already Incorrect!")
            return True
        else:
            self.original_confidence = np.max(predicted_scores)
            return False

    def update_new_image(self, test_image, original_label, target_map=None):
        w, h, c = test_image.shape
        self.test_image = test_image
        self.original_label = original_label
        assert self.check_prediction(self.sess, original_label, test_image,
                                     self.NET) == False
        if target_map is not None:
            self.target_map = target_map
            self.original_output = self.NET2.predict(test_image[None, :])
        counterfactuals = self.create_counterfactuals(test_image)
        self.saliency1, self.topK = self.run_model(
            self.sess, [self.NET.saliency, self.NET.top_idx], counterfactuals,
            self.NET)
        self.saliency1_flatten = np.reshape(
            self.saliency1, [test_image.shape[-3] * test_image.shape[-2]])
        elem1 = np.argsort(np.reshape(self.saliency1, [w * h]))[-self.k_top:]
        self.elements1 = np.zeros(w * h)
        self.elements1[elem1] = 1
        self.original_topk = self.elements1

        self.mass_center1 = self.run_model(self.sess, self.NET.mass_center,
                                           counterfactuals,
                                           self.NET).astype(int)
        self.original_mass_center = self.mass_center1

    def create_extra_ops(self, NET, w, h, k_top):

        top_val, NET.top_idx = tf.nn.top_k(NET.saliency_flatten, k_top)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        NET.mass_center = tf.stack([
            tf.reduce_sum(NET.saliency * x_mesh) / (w * h),
            tf.reduce_sum(NET.saliency * y_mesh) / (w * h)
        ])

    def create_attack_ops(self, NET, w, h):
        topK_loss = tf.reduce_sum((NET.saliency_flatten * self.topk_ph))
        self.debug = topK_loss
        NET.topK_direction = -tf.gradients(topK_loss, NET.input)[0]

        mass_center_loss = -tf.reduce_sum(
            (NET.mass_center - self.mass_center_ph)**2)
        NET.mass_center_direction = -tf.gradients(mass_center_loss,
                                                  NET.input)[0]
        if self.target_map is not None:
            target_dis = tf.keras.losses.MSE(self.target_map_ph, NET.saliency)
            output_dis = tf.keras.losses.MSE(self.original_output_ph,
                                             NET.output)
            target_loss = tf.reduce_mean(
                target_dis) * self.beta_0_ph + self.beta_1_ph * tf.reduce_mean(
                    output_dis)
            self.debug = target_loss

            self.target_direction = -tf.gradients(target_loss, NET.input)[0]

    def create_counterfactuals(self, in_image):

        ref_subtracted = in_image - self.reference_image
        counterfactuals = np.array([(float(i+1)/self.num_steps) * ref_subtracted + self.reference_image\
                                    for i in range(self.num_steps)])
        return np.array(counterfactuals)

    def run_model(self, sess, operation, feed, NET):

        if len(feed.shape) == 3:
            if hasattr(self, "original_topk") and hasattr(
                    self, "original_mass_center"):
                if hasattr(self, "use_target") and self.use_target:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: [feed],
                                        NET.label_ph: self.original_label,
                                        self.topk_ph: self.original_topk,
                                        NET.reference_image:
                                        self.reference_image,
                                        self.mass_center_ph:
                                        self.original_mass_center,
                                        self.beta_0_ph: self.beta_0,
                                        self.beta_1_ph: self.beta_1,
                                        self.original_output_ph:
                                        self.original_output,
                                        self.target_map_ph: self.target_map
                                    })
                else:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: [feed],
                                        NET.label_ph: self.original_label,
                                        NET.reference_image:
                                        self.reference_image,
                                        self.topk_ph: self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center,
                                        self.target_map_ph: self.target_map
                                    })
            else:
                return sess.run(operation,
                                feed_dict={
                                    NET.input: [feed],
                                    NET.reference_image: self.reference_image,
                                    NET.label_ph: self.original_label,
                                    self.target_map_ph: self.target_map
                                })
        elif len(feed.shape) == 4:
            if hasattr(self, "original_topk") and hasattr(
                    self, "original_mass_center"):
                if hasattr(self, "use_target") and self.use_target:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: feed,
                                        NET.label_ph: self.original_label,
                                        NET.reference_image:
                                        self.reference_image,
                                        self.topk_ph: self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center,
                                        self.beta_0_ph: self.beta_0,
                                        self.beta_1_ph: self.beta_1,
                                        self.original_output_ph:
                                        self.original_output,
                                        self.target_map_ph: self.target_map
                                    })
                else:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: feed,
                                        NET.label_ph: self.original_label,
                                        NET.reference_image:
                                        self.reference_image,
                                        self.topk_ph: self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center,
                                        self.target_map_ph: self.target_map
                                    })
            else:
                return sess.run(operation,
                                feed_dict={
                                    NET.input: feed,
                                    NET.reference_image: self.reference_image,
                                    NET.label_ph: self.original_label,
                                })
        else:
            raise RuntimeError("Input image shape invalid!")

    def give_simple_perturbation(self, attack_method, in_image):

        counterfactuals = self.create_counterfactuals(in_image)
        w, h, c = self.test_image.shape
        if attack_method == "random":
            perturbation = np.random.normal(size=(self.num_steps, w, h, c))
        elif attack_method == "topK":
            perturbation = self.run_model(self.sess, self.NET2.topK_direction,
                                          counterfactuals, self.NET2)
            perturbation = np.reshape(perturbation, [self.num_steps, w, h, c])
        elif attack_method == "mass_center":
            perturbation = self.run_model(self.sess,
                                          self.NET2.mass_center_direction,
                                          counterfactuals, self.NET2)
            perturbation = np.reshape(perturbation, [self.num_steps, w, h, c])
        elif attack_method == "target":
            self.use_target = True
            if self.target_map is None:
                raise ValueError("No target region determined!")
            else:
                perturbation = self.run_model(self.sess, self.target_direction,
                                              counterfactuals, self.NET2)
                perturbation = np.reshape(perturbation,
                                          [self.num_steps, w, h, c])
        perturbation_summed = np.sum(np.array([float(i+1)/self.num_steps*perturbation[i]\
                                               for i in range(self.num_steps)]),0)
        return np.sign(perturbation_summed)

    def apply_perturb(self, in_image, pert, alpha, bound=8 / 255, ord=np.inf):
        if self.mean_image is None:
            self.mean_image = np.zeros_like(in_image)
        # out_image = self.test_image + np.clip(
        #     in_image + alpha * np.sign(pert) - self.test_image, -bound, bound)
        d = in_image + alpha * np.sign(pert) - self.test_image
        d_norm = np.linalg.norm(d.flatten(), ord=ord)
        if d_norm > bound:
            proj_ratio = bound / np.linalg.norm(d.flatten(), ord=ord)
        else:
            proj_ratio = 1
        out_image = self.test_image + d * proj_ratio
        out_image = np.clip(out_image, -self.mean_image,
                            self.pixel_max - self.mean_image)
        return out_image

    def check_measure(self, test_image_pert, measure):

        prob = self.run_model(self.sess, self.NET.output, test_image_pert,
                              self.NET)
        if np.argmax(prob, 1) == self.original_label:
            counterfactuals = self.create_counterfactuals(test_image_pert)
            if measure == "intersection":
                top2 = self.run_model(self.sess, self.NET.top_idx,
                                      counterfactuals, self.NET)
                criterion = float(len(np.intersect1d(self.topK,
                                                     top2))) / self.k_top
            elif measure == "correlation":
                saliency2_flatten = self.run_model(self.sess,
                                                   self.NET.saliency_flatten,
                                                   counterfactuals, self.NET)
                criterion = scipy.stats.spearmanr(self.saliency1_flatten,
                                                  saliency2_flatten)[0]
            elif measure == "mass_center":
                center2 = self.run_model(self.sess, self.NET.mass_center,
                                         counterfactuals, self.NET).astype(int)
                criterion = -np.linalg.norm(self.mass_center1 - center2)
            elif measure == "cosine":
                saliency2_flatten = self.run_model(self.sess,
                                                   self.NET.saliency_flatten,
                                                   test_image_pert, self.NET)

                criterion = scipy.spatial.distance.cosine(
                    self.saliency1_flatten, saliency2_flatten)
            else:
                raise ValueError("Invalid measure!")
            return criterion
        else:
            return 1

    def iterative_attack(self,
                         attack_method,
                         epsilon,
                         iters=100,
                         alpha=1,
                         beta_0=1e11,
                         beta_1=1e6,
                         measure="intersection"):
        """
        Args:
            attack_method: One of "mass_center", "topK" or "random"
            epsilon: set of allowed maximum $ell_infty$ of perturbations, eg:[2,4]
            iters: number of maximum allowed attack iterations
            alpha: perturbation size in each iteration of the attack
            measure: measure for success of the attack (one of "correlation", "mass_center" or "intersection")
        Returns:
            intersection: The portion of the top K salient pixels in the original picture that are in the 
            top K salient pixels of the perturbed image devided
            correlation: The rank correlation between saliency maps of original and perturbed image
            center_dislocation: The L2 distance between saliency map mass centers in original and perturbed images
            confidence: The prediction confidence of the perturbed image
        """
        self.beta_0 = beta_0
        self.beta_1 = beta_1

        w, h, c = self.test_image.shape
        test_image_pert = self.test_image.copy()
        min_criterion = 1.
        for counter in range(iters):
            # if counter % int(iters / 5) == 0:
            #     print("Iteration : {}".format(counter))
            pert = self.give_simple_perturbation(attack_method,
                                                 test_image_pert)
            # print(pert.sum())
            test_image_pert = self.apply_perturb(test_image_pert, pert, alpha,
                                                 epsilon)
            criterion = self.check_measure(test_image_pert, measure)

            if criterion < min_criterion:
                # print("attack")
                min_criterion = criterion
                self.perturbed_image = test_image_pert.copy()
                perturb_size = np.max(
                    np.abs(self.test_image - self.perturbed_image))
            else:
                # print("labels is changed")
                pass
        if min_criterion == 1.:
            # print(
            #     "The attack was not successfull for maximum allowed perturbation size equal to {}"
            #     .format(epsilon))
            # return 1., 1., self.original_confidence, 0.
            return None

        # print(
        #     '''For maximum allowed perturbation size equal to {}, the resulting perturbation size was equal to {}
        # '''.format(epsilon,
        #            np.max(np.abs(self.test_image - self.perturbed_image))))
        predicted_scores = self.run_model(self.sess, self.NET.output,
                                          self.perturbed_image, self.NET)
        confidence = np.max(predicted_scores)
        counterfactuals = self.create_counterfactuals(self.perturbed_image)
        self.saliency2, self.top2, self.mass_center2= self.run_model\
        (self.sess, [self.NET.saliency, self.NET.top_idx, self.NET.mass_center], counterfactuals, self.NET)
        correlation = scipy.stats.spearmanr(
            self.saliency1_flatten, np.reshape(self.saliency2, [w * h]))[0]
        intersection = float(len(np.intersect1d(self.topK,
                                                self.top2))) / self.k_top
        center_dislocation = np.linalg.norm(self.mass_center1 -
                                            self.mass_center2.astype(int))
        cos_distance = scipy.spatial.distance.cosine(
            self.saliency1_flatten, np.reshape(self.saliency2, [w * h]))
        return intersection, correlation, center_dislocation, confidence, perturb_size, cos_distance


class SmoothGradientsAttack(object):
    def __init__(self,
                 sess,
                 mean_image,
                 test_image,
                 original_label,
                 NET,
                 NET2=None,
                 k_top=1000,
                 num_steps=100,
                 reference_image=None,
                 target_map=None,
                 pixel_max=255.):
        """
        Args:
            mean_image: The mean image of the data set(The assumption is that the images are mean subtracted)
            sess: Session containing model(and surrogate model's) graphs
            test_image: Mean subtracted test image
            original_label: True label of the image
            NET: Original neural network. It's assumed that NET.saliency is the saliency map tensor and 
            NET.saliency_flatten is its flatten version.
            NET2: Surrogate neural network with the same structure and weights of the orignal network but
            with activations replaced by softplus function
            (necessary only when the activation function of the original function
            does not have second order gradients, ex: ReLU). It's assumed that NET.saliency is the 
            saliency map tensor and NET2.saliency_flatten is its flatten version.
            k_top: the topK parameter of the attack (refer to the original paper)
            num_steps: Number of steps in Integrated Gradients Algorithm
            reference_image: not used
            pixel_max: maximum pixel value in the input image
        """
        self.pixel_max = pixel_max
        if len(test_image.shape) != 3:
            raise ValueError("Invalid Test Image Dimensions")
        if sum([
                NET.input.get_shape()[-i] != test_image.shape[-i]
                for i in [1, 2, 3]
        ]):
            raise ValueError(
                "Model's input dimensions is not Compatible with the provided test image!"
            )
        if self.check_prediction(sess, original_label, test_image, NET):
            return
        self.sess = sess
        self.target_map = target_map
        self.create_extra_ops(NET, test_image.shape[-3], test_image.shape[-2],
                              k_top)
        if NET2 is None:
            NET2 = NET
        else:
            self.create_extra_ops(NET2, test_image.shape[-3],
                                  test_image.shape[-2], k_top)
            if sum([
                    NET2.input.get_shape()[-i] != test_image.shape[-i]
                    for i in [1, 2, 3]
            ]):
                raise ValueError(
                    "Surrogate model's input dimensions is not Compatible with the provided test image!"
                )
        self.NET = NET
        self.NET2 = NET2
        self.test_image = test_image
        self.original_label = original_label
        self.mean_image = mean_image
        self.k_top = k_top
        self.num_steps = num_steps
        self.reference_image = np.zeros_like(
            test_image) if reference_image is None else reference_image

        w, h, c = self.mean_image.shape
        self.topk_ph = tf.placeholder(tf.float32,
                                      shape=[w * h],
                                      name='topk_ph')
        self.mass_center_ph = tf.placeholder(tf.float32,
                                             shape=[2],
                                             name='mass_center_ph')
        self.target_map_ph = tf.placeholder(tf.float32,
                                            shape=[w, h],
                                            name='target_map_ph')
        self.beta_0_ph = tf.placeholder(tf.float32, name='beta_0')
        self.beta_1_ph = tf.placeholder(tf.float32, name='beta_1')
        self.original_output = self.NET.predict(test_image[None, :])
        _, num_class = self.original_output.shape

        self.original_output_ph = tf.placeholder(
            tf.float32, shape=[None, num_class],
            name='original_output_ph')  # only for the manipulation attack
        self.create_attack_ops(self.NET2, test_image.shape[-3],
                               test_image.shape[-2])

        self.update_new_image(test_image, original_label)

    def update_new_image(self, test_image, original_label, target_map=None):
        w, h, c = test_image.shape
        self.test_image = test_image
        self.original_label = original_label
        assert self.check_prediction(self.sess, original_label, test_image,
                                     self.NET) == False
        if target_map is not None:
            self.target_map = target_map
            self.original_output = self.NET2.predict(test_image[None, :])
        counterfactuals = self.create_counterfactuals(test_image)
        self.saliency1, self.topK = self.run_model(
            self.sess, [self.NET.saliency, self.NET.top_idx], counterfactuals,
            self.NET)
        self.saliency1_flatten = np.reshape(
            self.saliency1, [test_image.shape[-3] * test_image.shape[-2]])
        elem1 = np.argsort(np.reshape(self.saliency1, [w * h]))[-self.k_top:]
        self.elements1 = np.zeros(w * h)
        self.elements1[elem1] = 1
        self.original_topk = self.elements1

        self.mass_center1 = self.run_model(self.sess, self.NET.mass_center,
                                           counterfactuals,
                                           self.NET).astype(int)
        self.original_mass_center = self.mass_center1

    def check_prediction(self, sess, original_label, image, NET):
        """ If the network's prediction is incorrect in the first place, attacking has no meaning."""
        predicted_scores = sess.run(
            NET.output,
            feed_dict={NET.input: image if len(image.shape) == 4 else [image]})
        if np.argmax(predicted_scores, 1) != original_label:
            print("Network's Prediction is Already Incorrect!")
            print("Pred: ", np.argmax(predicted_scores, 1))
            print("Label: ", original_label)
            return True
        else:
            self.original_confidence = np.max(predicted_scores)
            return False

    def create_extra_ops(self, NET, w, h, k_top):

        top_val, NET.top_idx = tf.nn.top_k(NET.saliency_flatten, k_top)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        NET.mass_center = tf.stack([
            tf.reduce_sum(NET.saliency * x_mesh) / (w * h),
            tf.reduce_sum(NET.saliency * y_mesh) / (w * h)
        ])

    def create_attack_ops(self, NET, w, h):
        topK_loss = tf.reduce_sum((NET.saliency_flatten * self.topk_ph))
        self.debug = topK_loss
        NET.topK_direction = -tf.gradients(topK_loss, NET.input)[0]

        mass_center_loss = -tf.reduce_sum(
            (NET.mass_center - self.mass_center_ph)**2)
        NET.mass_center_direction = -tf.gradients(mass_center_loss,
                                                  NET.input)[0]
        if self.target_map is not None:
            target_dis = tf.keras.losses.MSE(self.target_map_ph, NET.saliency)
            output_dis = tf.keras.losses.MSE(self.original_output_ph,
                                             NET.output)
            target_loss = tf.reduce_mean(
                target_dis) * self.beta_0_ph + self.beta_1_ph * tf.reduce_mean(
                    output_dis)
            self.debug = target_loss

            self.target_direction = -tf.gradients(target_loss, NET.input)[0]

    def create_counterfactuals(self, in_image, noise_ratio=0.1):
        counterfactuals = np.array([
            in_image + np.random.normal(scale=0.1 *
                                        (in_image.max() - in_image.min()),
                                        size=in_image.shape)
            for _ in range(self.num_steps)
        ])

        return np.array(counterfactuals)

    def run_model(self, sess, operation, feed, NET):

        if len(feed.shape) == 3:
            if hasattr(self, "original_topk") and hasattr(
                    self, "original_mass_center"):
                if hasattr(self, "use_target") and self.use_target:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: [feed],
                                        NET.label_ph: self.original_label,
                                        self.topk_ph: self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center,
                                        self.beta_0_ph: self.beta_0,
                                        self.beta_1_ph: self.beta_1,
                                        self.original_output_ph:
                                        self.original_output,
                                        self.target_map_ph: self.target_map
                                    })
                else:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: [feed],
                                        NET.label_ph: self.original_label,
                                        self.topk_ph: self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center,
                                        self.target_map_ph: self.target_map
                                    })
            else:
                return sess.run(operation,
                                feed_dict={
                                    NET.input: [feed],
                                    NET.label_ph: self.original_label,
                                    self.target_map_ph: self.target_map
                                })
        elif len(feed.shape) == 4:
            if hasattr(self, "original_topk") and hasattr(
                    self, "original_mass_center"):
                if hasattr(self, "use_target") and self.use_target:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: feed,
                                        NET.label_ph: self.original_label,
                                        self.topk_ph: self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center,
                                        self.beta_0_ph: self.beta_0,
                                        self.beta_1_ph: self.beta_1,
                                        self.original_output_ph:
                                        self.original_output,
                                        self.target_map_ph: self.target_map
                                    })
                else:
                    return sess.run(operation,
                                    feed_dict={
                                        NET.input: feed,
                                        NET.label_ph: self.original_label,
                                        self.topk_ph: self.original_topk,
                                        self.mass_center_ph:
                                        self.original_mass_center,
                                        self.target_map_ph: self.target_map
                                    })
            else:
                return sess.run(operation,
                                feed_dict={
                                    NET.input: feed,
                                    NET.label_ph: self.original_label,
                                })
        else:
            raise RuntimeError("Input image shape invalid!")

    def give_simple_perturbation(self, attack_method, in_image):

        counterfactuals = self.create_counterfactuals(in_image)
        w, h, c = self.test_image.shape
        if attack_method == "random":
            perturbation = np.random.normal(size=(self.num_steps, w, h, c))
        elif attack_method == "topK":
            perturbation = self.run_model(self.sess, self.NET2.topK_direction,
                                          counterfactuals, self.NET2)
            perturbation = np.reshape(perturbation, [self.num_steps, w, h, c])
        elif attack_method == "mass_center":
            perturbation = self.run_model(self.sess,
                                          self.NET2.mass_center_direction,
                                          counterfactuals, self.NET2)
            perturbation = np.reshape(perturbation, [self.num_steps, w, h, c])
        elif attack_method == "target":
            if self.target_map is None:
                raise ValueError("No target region determined!")
            else:
                perturbation = self.run_model(self.sess, self.target_direction,
                                              counterfactuals, self.NET2)
                perturbation = np.reshape(perturbation,
                                          [self.num_steps, w, h, c])
        perturbation_summed = np.mean(perturbation, 0)
        return np.sign(perturbation_summed)

    def apply_perturb(self, in_image, pert, alpha, bound=8 / 255, ord=np.inf):
        if self.mean_image is None:
            self.mean_image = np.zeros_like(in_image)
        # out_image = self.test_image + np.clip(
        #     in_image + alpha * np.sign(pert) - self.test_image, -bound, bound)
        d = in_image + alpha * pert - self.test_image
        d_norm = np.linalg.norm(d.flatten(), ord=ord)
        if d_norm > bound:
            proj_ratio = bound / np.linalg.norm(d.flatten(), ord=ord)
        else:
            proj_ratio = 1
        out_image = self.test_image + d * proj_ratio
        out_image = np.clip(out_image, -self.mean_image,
                            self.pixel_max - self.mean_image)
        return out_image

    def check_measure(self, test_image_pert, measure):

        prob = self.run_model(self.sess, self.NET.output, test_image_pert,
                              self.NET)
        if np.argmax(prob, 1) == self.original_label:
            counterfactuals = self.create_counterfactuals(test_image_pert)
            if measure == "intersection":
                top2 = self.run_model(self.sess, self.NET.top_idx,
                                      counterfactuals, self.NET)
                criterion = float(len(np.intersect1d(self.topK,
                                                     top2))) / self.k_top
            elif measure == "correlation":
                saliency2_flatten = self.run_model(self.sess,
                                                   self.NET.saliency_flatten,
                                                   counterfactuals, self.NET)
                criterion = scipy.stats.spearmanr(self.saliency1_flatten,
                                                  saliency2_flatten)[0]
            elif measure == "mass_center":
                center2 = self.run_model(self.sess, self.NET.mass_center,
                                         counterfactuals, self.NET).astype(int)
                criterion = -np.linalg.norm(self.mass_center1 - center2)
            elif measure == "cosine":
                saliency2_flatten = self.run_model(self.sess,
                                                   self.NET.saliency_flatten,
                                                   test_image_pert, self.NET)

                criterion = scipy.spatial.distance.cosine(
                    self.saliency1_flatten, saliency2_flatten)
            else:
                raise ValueError("Invalid measure!")
            return criterion
        else:
            return 1.

    def iterative_attack(self,
                         attack_method,
                         epsilon,
                         iters=100,
                         alpha=1,
                         beta_0=1e11,
                         beta_1=1e6,
                         measure="intersection"):
        """
        Args:
            attack_method: One of "mass_center", "topK" or "random"
            epsilon: set of allowed maximum $ell_infty$ of perturbations, eg:[2,4]
            iters: number of maximum allowed attack iterations
            alpha: perturbation size in each iteration of the attack
            measure: measure for success of the attack (one of "correlation", "mass_center" or "intersection")
        Returns:
            intersection: The portion of the top K salient pixels in the original picture that are in the 
            top K salient pixels of the perturbed image devided
            correlation: The rank correlation between saliency maps of original and perturbed image
            center_dislocation: The L2 distance between saliency map mass centers in original and perturbed images
            confidence: The prediction confidence of the perturbed image
        """

        w, h, c = self.test_image.shape
        test_image_pert = self.test_image.copy()
        self.original = self.test_image.copy()
        if attack_method == 'target':
            self.use_target = True
        else:
            self.use_target = False
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        min_criterion = 1.
        last_image = None
        for counter in range(iters):
            pert = self.give_simple_perturbation(attack_method,
                                                 test_image_pert)
            test_image_pert = self.apply_perturb(test_image_pert, pert, alpha,
                                                 epsilon)
            criterion = self.check_measure(test_image_pert, measure)

            if criterion < min_criterion:
                min_criterion = criterion
                self.perturbed_image = test_image_pert.copy()
                perturb_size = np.max(
                    np.abs(self.test_image - self.perturbed_image))
            else:
                pass

        if criterion == 1.:
            return None

        predicted_scores = self.run_model(self.sess, self.NET.output,
                                          self.perturbed_image, self.NET)
        confidence = np.max(predicted_scores)
        counterfactuals = self.create_counterfactuals(self.perturbed_image)
        self.saliency2, self.top2, self.mass_center2= self.run_model\
        (self.sess, [self.NET.saliency, self.NET.top_idx, self.NET.mass_center], counterfactuals, self.NET)
        correlation = scipy.stats.spearmanr(
            self.saliency1_flatten, np.reshape(self.saliency2, [w * h]))[0]
        intersection = float(len(np.intersect1d(self.topK,
                                                self.top2))) / self.k_top
        center_dislocation = np.linalg.norm(self.mass_center1 -
                                            self.mass_center2.astype(int))
        cos_distance = scipy.spatial.distance.cosine(
            self.saliency1_flatten, np.reshape(self.saliency2, [w * h]))
        return intersection, correlation, center_dislocation, confidence, perturb_size, cos_distance


class UniGradientsAttack(SmoothGradientsAttack):
    def __init__(self,
                 sess,
                 mean_image,
                 test_image,
                 original_label,
                 NET,
                 NET2=None,
                 k_top=1000,
                 num_steps=100,
                 radii=4,
                 reference_image=None,
                 target_map=None,
                 pixel_max=255.):
        self.radii = radii / (255. / pixel_max)
        super(UniGradientsAttack,
              self).__init__(sess,
                             mean_image,
                             test_image,
                             original_label,
                             NET,
                             NET2=NET2,
                             k_top=1000,
                             num_steps=num_steps,
                             reference_image=reference_image,
                             target_map=target_map,
                             pixel_max=255.)

    def create_counterfactuals(self, in_image):

        counterfactuals = np.array([
            in_image +
            np.random.uniform(-1, 1, size=in_image.shape) * self.radii
            for _ in range(self.num_steps)
        ])

        return np.array(counterfactuals)

# starter code from: https://github.com/aditya-grover/uae
from utils import *
import tensorflow as tf 
import numpy as np
import time
import sys
from math import log, exp
from scipy.special import expit
from datasource import Datasource
from tensorflow.contrib.framework.python.framework import checkpoint_utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import itertools
import tensorflow.contrib.distributions as tfd
from tensorflow.contrib.distributions import Bernoulli, Categorical, RelaxedBernoulli
import pickle
from itertools import product, chain
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt

import pdb
# """
# Add adversarial flip framework
# fixed
# """

FLAGS = flags.FLAGS
# tf.enable_eager_execution()

class NECST():
	def __init__(self, sess, datasource):

		self.seed = FLAGS.seed
		tf.set_random_seed(self.seed)
		np.random.seed(self.seed)

		self.sess = sess
		self.datasource = datasource
		self.input_dim = self.datasource.input_dim
		if self.input_dim == 784:
			self.img_dim = 28
		elif self.input_dim == 100:
			self.img_dim = 100
		elif self.input_dim == 7840 or self.input_dim == 3920:
			self.img_dim = 28
		elif self.input_dim == (32 * 32 * 3):
			self.img_dim = 32
		else:  # celebA
			self.img_dim = 64
		self.z_dim = FLAGS.n_bits
		self.dec_layers = [self.input_dim] + FLAGS.dec_arch
		self.enc_layers = FLAGS.enc_arch + [self.z_dim]
		self.enabel_adv = FLAGS.adv

		self.last_layer_act = tf.nn.sigmoid if FLAGS.non_linear_act else None

		# perturbation experiment
		self.noisy_mnist = FLAGS.noisy_mnist

		# for vimco
		self.is_binary = FLAGS.is_binary
		self.vimco_samples = FLAGS.vimco_samples
		self.flip_samples = FLAGS.flip_samples
		self.discrete_relax = FLAGS.discrete_relax

		# other params
		self.activation = FLAGS.activation
		self.lr = FLAGS.lr
		self.wadv = FLAGS.wadv
		self.cew = FLAGS.cew
		self.klw = FLAGS.klw
		self.lpw = FLAGS.lpw
		self.miw = FLAGS.miw
		self.denw = FLAGS.denw
		self.tcw = FLAGS.miw
		self.without_noise = FLAGS.without_noise
		# if need to use REINFORCE-like optimization scheme
		if not self.discrete_relax:
			self.theta_optimizer = FLAGS.optimizer(learning_rate=self.lr)
			self.phi_optimizer = FLAGS.optimizer(learning_rate=self.lr)
			self.disc_optimizer = FLAGS.optimizer(learning_rate=self.lr)
		else:
			# gumbel-softmax doesn't require 2 optimizers
			self.optimizer = FLAGS.optimizer
		self.training = True

		# noise levels
		self.channel_model = FLAGS.channel_model
		self.noise = FLAGS.noise
		self.test_noise = FLAGS.test_noise

		# TODO: hacky - fix later
		if self.img_dim == 64:
			self.x = tf.placeholder(self.datasource.dtype, shape=[None, self.img_dim, self.img_dim, 3], name='necst_input')
		elif self.img_dim == 28:
			self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='necst_input')
		else:
			# svhn and cifar10
			self.x = tf.placeholder(tf.float32, shape=[None, self.img_dim, self.img_dim, 3], name='necst_input')

		# CS settings
		self.reg_param = tf.placeholder_with_default(FLAGS.reg_param, shape=(), name='reg_param')

		# gumbel-softmax and vimco-compatible; only discrete bits
		if self.img_dim == 64:
			self.mean_, self.z, self.classif_z, self.q, self.x_reconstr_logits, self.mean= self.celebA_create_collapsed_computation_graph(self.x)
		else:
			# MNIST
			if self.channel_model == 'bsc':
				self.mean_, self.z, self.classif_z, self.q, self.x_reconstr_logits, self.mean = self.create_collapsed_computation_graph(self.x)
				#####EXPLAINS for mean,z,classif_Z,q,RECONSTR
				###
			else:
				self.mean, self.z, self.q, self.x_reconstr_logits = self.create_erasure_collapsed_computation_graph(self.x)

		self.mask_z = tf.bitwise
		if self.channel_model == 'bsc':
			self.test_mean, self.test_z, self.test_classif_z, self.test_q, self.test_x_reconstr_logits = self.get_collapsed_stochastic_test_sample(self.x)
		else:
			self.test_mean, self.test_z, self.test_q, self.test_x_reconstr_logits = self.get_collapsed_erasure_stochastic_test_sample(self.x)
		# self.adv_z, self.adv_loss, self.mask_adv = self.perturb_latent_code(self.x, self.z)
		if not self.discrete_relax:
			print('using vimco loss...')
			if self.noisy_mnist:
				print('training with noisy MNIST, using true x values for vimco loss...')
				# if self.wadv == 0:
				self.theta_loss, self.phi_loss, self.reconstr_loss = self.vimco_loss(
				self.true_x, self.x_reconstr_logits)
				# else:
				# 	self.theta_loss, self.phi_loss, self.reconstr_loss = self.vimco_loss(
				# 		self.true_x, self.x_reconstr_logits)

			else:
				self.theta_loss, self.phi_loss, self.reconstr_loss = self.vimco_loss(self.x, self.x_reconstr_logits)
		else:
			self.loss, self.reconstr_loss = self.get_loss(self.x, self.x_reconstr_logits)





		self.conditional_entropy = -tf.reduce_mean(tf.reduce_sum(self.mean*tf.log(self.mean+ 1e-8)+(1-self.mean)*tf.log(1-self.mean+1e-8),axis=1))
			# self.get_conditional_entropy(self.mean)
		self.kl_loss = self.kl_in_batch(self.mean)
		# self.v_loss = self.vat_loss(self.x,self.mean)
		self.denoising_loss = self.denoising_decoder(self.z,self.mean)
		self.denoising_loss_v2 = self.denoising_decoder_v2(self.x,self.mean)
		#add 1e -8  to stable the numerical
		self.theta_loss = self.theta_loss +  self.denw * self.denoising_loss_v2


		self.d_z = self.discriminator(self.mean,reuse=tf.AUTO_REUSE)
		self.per_z = self.perturb_dims(self.mean)
		self.d_z_perm = self.discriminator(self.per_z)
		self.tc_loss = tf.reduce_mean(self.d_z[:,0]-self.d_z[:,1])
		self.phi_loss = self.phi_loss + self.miw * (self.conditional_entropy+ self.kl_loss+self.tc_loss)
						# self.v_loss * self.lpw #calculate encoder loss
		# ones = torch.ones(half_batch_size, dtype=torch.long, device=self.device)
		# zeros = torch.zeros_like(ones)
		# d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones))
		self.ones = tf.ones(tf.shape(self.mean)[0])
		self.ones = tf.expand_dims(self.ones, axis=1)
		self.zeros = tf.zeros_like(self.ones)
		# self.zeros = tf.expand_dims(self.zeros, axis=0)
		self.z_label = tf.concat([self.ones,self.zeros],1)
		self.z_perm_label = tf.concat([self.zeros,self.ones],1)
		self.D_loss =tf.reduce_mean(0.5 *(tf.nn.softmax_cross_entropy_with_logits(logits=self.d_z,labels=self.z_label)+tf.nn.softmax_cross_entropy_with_logits(logits=self.d_z_perm,labels=self.z_perm_label)))
		_,_,self.mask_z = self.perturb_latent_code(self.x, self.classif_z)







		# loss calculation
		if self.noisy_mnist:
			print('training with noisy MNIST, using true x values for vimco loss...')
			self.test_loss = self.get_test_loss(self.true_x, self.test_x_reconstr_logits)
		else:
			self.test_loss = self.get_test_loss(self.x, self.test_x_reconstr_logits)

		# session ops
		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		# set up optimization op
		if not self.discrete_relax:
			print('SETUP: using mutliple train ops due to discrete latent variable')
			# get decoder and encoder variables
			theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/decoder')
			self.theta_vars = theta_vars
			self.theta_grads, variables = zip(*self.theta_optimizer.compute_gradients(self.theta_loss, var_list=theta_vars))
			self.discrete_train_op1 = self.theta_optimizer.minimize(self.theta_loss, global_step=self.global_step, var_list=theta_vars)

			# encoder variables
			phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/encoder')
			self.phi_vars = phi_vars
			self.phi_grads, variables = zip(*self.phi_optimizer.compute_gradients(self.phi_loss, var_list=phi_vars))
			self.discrete_train_op2 = self.phi_optimizer.minimize(self.phi_loss, global_step=self.global_step, var_list=phi_vars)

			disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/discriminator')
			print (disc_vars)
			self.disc_vars = disc_vars
			self.disc_grads, variables = zip(*self.disc_optimizer.compute_gradients(self.D_loss, var_list=disc_vars))
			self.discrete_train_op3 = self.disc_optimizer.minimize(self.D_loss, global_step=self.global_step,
															  var_list=disc_vars)
		else:
			# gumbel-softmax
			train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			self.train_op = self.optimizer(learning_rate=self.lr).minimize(self.loss, 
				global_step=self.global_step, var_list=train_vars)

		# summary ops
		self.summary_op = tf.summary.merge_all()

		# session ops
		self.init_op = tf.global_variables_initializer()
		self.saver = tf.train.Saver(max_to_keep=None)

	def denoising_decoder(self,z,z_uncorrupted,reuse=True):
		dset_name = self.datasource.target_dataset
		z_uncorrupted = tf.expand_dims(z_uncorrupted,axis=0)
		z_uncorrupted = tf.tile(z_uncorrupted, [self.vimco_samples, 1, 1])
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			x_reconstr_logits = self.decoder(z, reuse=reuse)
			x_reconstr_logits_target = self.decoder(z_uncorrupted, reuse=reuse)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.cifar10_convolutional_decoder(z, reuse=reuse)
			x_reconstr_logits_target = self.cifar10_convolutional_decoder(z_uncorrupted, reuse=reuse)
		elif dset_name == 'svhn':
			x_reconstr_logits = self.convolutional_32_decoder(z, reuse=reuse)
			x_reconstr_logits_target = self.convolutional_32_decoder(z_uncorrupted, reuse=reuse)
		elif dset_name == 'celebA':
			x_reconstr_logits = self.complex_decoder(z, reuse=reuse)
			x_reconstr_logits_target = self.complex_decoder(z_uncorrupted, reuse=reuse)
		else:
			print('dataset {} is not implemented'.format(dset_name))
			raise NotImplementedError
		# reg_loss = tf.losses.get_regularization_loss()
		if self.is_binary:
			# TODO: DOUBLE CHECK THIS
			reconstr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstr_logits, labels=tf.stop_gradient(tf.nn.sigmoid(x_reconstr_logits_target))))
		else:
			if self.img_dim == 64:
				reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(tf.stop_gradient(x_reconstr_logits_target), x_reconstr_logits), axis=[1, 2, 3]))
			else:
				reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(tf.stop_gradient(x_reconstr_logits_target), x_reconstr_logits), axis=1))

		return reconstr_loss

	def denoising_decoder_v2(self,x,z_uncorrupted,reuse=True):
		dset_name = self.datasource.target_dataset
		z_uncorrupted = tf.expand_dims(z_uncorrupted,axis=0)
		z_uncorrupted = tf.tile(z_uncorrupted, [self.vimco_samples, 1, 1])
		if self.is_binary:
			x = tf.expand_dims(x, axis=0)
			x = tf.tile(x, [self.vimco_samples, 1, 1])
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			# x_reconstr_logits = self.decoder(z, reuse=reuse)
			x_reconstr_logits_target = self.decoder(z_uncorrupted, reuse=reuse)
		elif dset_name == 'cifar10':
			# x_reconstr_logits = self.cifar10_convolutional_decoder(z, reuse=reuse)
			x_reconstr_logits_target = self.cifar10_convolutional_decoder(z_uncorrupted, reuse=reuse)
		elif dset_name == 'svhn':
			# x_reconstr_logits = self.convolutional_32_decoder(z, reuse=reuse)
			x_reconstr_logits_target = self.convolutional_32_decoder(z_uncorrupted, reuse=reuse)
		elif dset_name == 'celebA':
			# x_reconstr_logits = self.complex_decoder(z, reuse=reuse)
			x_reconstr_logits_target = self.complex_decoder(z_uncorrupted, reuse=reuse)
		else:
			print('dataset {} is not implemented'.format(dset_name))
			raise NotImplementedError
		# reg_loss = tf.losses.get_regularization_loss()
		if self.is_binary:
			# TODO: DOUBLE CHECK THIS
			reconstr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstr_logits_target, labels=x))
		else:
			if self.img_dim == 64:
				reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(x_reconstr_logits_target, x), axis=[1, 2, 3]))
			else:
				reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(x_reconstr_logits_target, x), axis=1))

		return reconstr_loss



	def perturb_latent_code(self, x, z, reuse=tf.AUTO_REUSE):
		"""
		:return: get the adversarial codes for flipping operation.
		And update the generative parts
		"""

		dset_name = self.datasource.target_dataset
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			x_reconstr_logits = self.decoder(z, reuse=reuse)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.cifar10_convolutional_decoder(z, reuse=reuse)
		elif dset_name == 'svhn':
			x_reconstr_logits = self.convolutional_32_decoder(z, reuse=reuse)
		elif dset_name == 'celebA':
			x_reconstr_logits = self.complex_decoder(z, reuse=reuse)
		else:
			print('dataset {} is not implemented'.format(dset_name))
			raise NotImplementedError
		# reg_loss = tf.losses.get_regularization_loss()
		if self.is_binary:
			# TODO: DOUBLE CHECK THIS
			# x = tf.expand_dims(x, axis=0)
			# x = tf.tile(x, [self.vimco_samples, 1, 1])
			# pdb.set_trace()
			reconstr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstr_logits, labels=x))
		else:
			if self.img_dim == 64:
				reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[1, 2, 3]))
			else:
				reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=1))
		g = tf.gradients(reconstr_loss,[z])[0] #5*64*100
		g_norm = g**2
		# print (g_norm.get_shape())
		dim_nums = \
			int(self.noise * self.z_dim)

		dim_nums = self.flip_samples
		# dim_nums = 3
		# pdb.set_trace()
		g_prob = tf.nn.softmax(g_norm,axis=-1)
		z = -tf.log(-tf.log(tf.random_uniform(tf.shape(g_prob), 0, 1)))
		sample_matrix = tf.log(g_prob) + z
		value, dims = tf.nn.top_k(sample_matrix, k=dim_nums, sorted=False)
		# value, dims = tf.nn.top_k(g_norm,k=dim_nums,sorted=False)
		zeros = tf.zeros_like(g)
		ones = tf.ones_like(g)
		pos_idx = tf.cast(tf.greater_equal(g, zeros),tf.int32) #where the gradient larger than 0 is 1 and where the gradient less than 0 is 0
		neg_idx = tf.cast(tf.less(g, zeros),tf.int32)
		zero_idx_z = tf.cast(tf.less(z, ones),tf.int32)
		one_idx_z = tf.cast(tf.greater_equal(z, ones),tf.int32)
		"""Beta version
		"""
		"""test with min value
		"""
		kth = tf.reduce_min(value, axis = -1, keepdims=True)
		mask_stable = tf.cast(tf.greater_equal(sample_matrix, kth),tf.int32)
		"""OLDER version"""
		# mask = tf.cast(mask,tf.int32)
		z = tf.cast(z,tf.int32)
		# mask = tf.bitwise.bitwise_xor(pos_idx,mask) # this will be flipped by 1->1 0->1 or
		mask1 = tf.bitwise.bitwise_and(pos_idx,mask_stable)
		mask1 = tf.bitwise.bitwise_and(mask1,zero_idx_z)
		mask2 =  tf.bitwise.bitwise_and(neg_idx,mask_stable)
		mask2 = tf.bitwise.bitwise_and(mask2,one_idx_z)
		mask = tf.bitwise.bitwise_or(mask1,mask2)

		adv_z = tf.bitwise.bitwise_xor(z,tf.stop_gradient(mask))
		adv_z = tf.cast(adv_z,tf.float32)

		#

		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			ad_x_reconstr_logits = self.decoder(adv_z, reuse=reuse)
		elif dset_name == 'cifar10':
			ad_x_reconstr_logits = self.cifar10_convolutional_decoder(adv_z, reuse=reuse)
		elif dset_name == 'svhn':
			ad_x_reconstr_logits = self.convolutional_32_decoder(adv_z, reuse=reuse)
		elif dset_name == 'celebA':
			ad_x_reconstr_logits = self.complex_decoder(adv_z, reuse=reuse)
		else:
			print('dataset {} is not implemented'.format(dset_name))
			raise NotImplementedError

		if self.is_binary:
			# TODO: DOUBLE CHECK THIS
			ad_reconstr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=ad_x_reconstr_logits, labels=tf.stop_gradient(tf.nn.sigmoid(x_reconstr_logits))))
		else:
			if self.img_dim == 64:
				ad_reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(tf.stop_gradient(x_reconstr_logits), ad_x_reconstr_logits), axis=[1, 2, 3]))
			else:
				ad_reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(tf.stop_gradient(x_reconstr_logits), ad_x_reconstr_logits), axis=1))
		return adv_z,ad_reconstr_loss,mask


	# def vat_loss(self,z):
	def normalize_perturbation(self, d):
		# with tf.name_scope(scope, 'norm_pert'):
		# 	pdb.set_trace()
		output = tf.nn.l2_normalize(d, axis=list(range(1, len(d.shape))))
		# pdb.set_trace()

		return output


	def perturb_image(self,x, p):
		# with tf.name_scope(scope, 'perturb_image'):
		eps = 1e-6 * self.normalize_perturbation(tf.random_normal(shape=tf.shape(x)))


		dset_name = self.datasource.target_dataset
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			eps_p = self.encoder(x + eps, reuse=True)
		elif dset_name == 'cifar10':
			eps_p = self.cifar10_convolutional_encoder(x + eps, reuse=True)
		elif dset_name == 'svhn':
			eps_p = self.convolutional_32_encoder(x + eps, reuse=True)
		elif dset_name == 'celebA':
			eps_p = self.complex_encoder(x + eps, reuse=True)
		else:
			print('dataset {} is not implemented'.format(dset_name))
			raise NotImplementedError

		# loss = softmax_xent_two(labels=p, logits=eps_p)
		loss = -tf.reduce_sum(p * tf.log(eps_p + 1e-8) + (1 - p) * tf.log(1 - eps_p+ 1e-8))
		# Based on perturbed image, get direction of greatest error
		eps_adv = tf.gradients(loss, [eps], aggregation_method=2)[0]
		# Use that direction as adversarial perturbation
		eps_adv = self.normalize_perturbation(eps_adv)
		x_adv = tf.stop_gradient(x + FLAGS.radius * eps_adv)
		return x_adv

	def vat_loss(self, x, p):
		# with tf.name_scope(scope, 'smoothing_loss'):
		dset_name = self.datasource.target_dataset
		# pdb.set_trace()
		x_adv = self.perturb_image(x, p)
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			p_adv = self.encoder(x_adv, reuse=True)
		elif dset_name == 'cifar10':
			p_adv = self.cifar10_convolutional_encoder(x_adv, reuse=True)
		elif dset_name == 'svhn':
			p_adv = self.convolutional_32_encoder(x_adv, reuse=True)
		elif dset_name == 'celebA':
			p_adv = self.complex_encoder(x_adv, reuse=True)
		else:
			print('dataset {} is not implemented'.format(dset_name))
			raise NotImplementedError
		#TODO yuxuan cross entropy
		loss = -tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(p)*tf.log(p_adv+1e-8)+(1-tf.stop_gradient(p))*tf.log(1-p_adv+1e-8),axis=-1))
		return loss





	def kl_in_batch(self,z):
		"""
		:param z: a batch of latent mean, sum(p(y|x)) for all x
		:return: the kl loss of the discrete variable z and the uniform distribution(AN ESTIMATED H(y))
		"""
		#(64,100)
		mariginal = tf.reduce_mean(z,axis=0,keep_dims=False)
		print (mariginal.get_shape())
		# pdb.set_trace()
		kl_loss = tf.reduce_sum(mariginal*tf.log(mariginal+1e-8)+(1-mariginal)*tf.log(1-mariginal+1e-8))
		return kl_loss

	def encoder(self, x, reuse=True):
		"""
		Specifies the parameters for the mean and variance of p(y|x)
		"""
		e = x
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				for layer_idx, layer_dim in enumerate(enc_layers[:-1]):
					e = tf.layers.dense(e, layer_dim, activation=tf.nn.leaky_relu, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(layer_idx))
				if self.channel_model == 'bsc':
					z_mean = tf.layers.dense(e, self.z_dim, activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(len(enc_layers)-1))
				else:
					# N x D x 2 for erasure channel
					z_mean = tf.layers.dense(e, self.z_dim * 2, activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(len(enc_layers)-1))
					z_mean = tf.reshape(z_mean, (-1, self.z_dim, 2))
		return z_mean

	def discriminator(self,z,reuse=True):
		e = z
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('discriminator', reuse=reuse):
				for layer_idx, layer_dim in enumerate(enc_layers[:-1]):
					e = tf.layers.dense(e, layer_dim, activation=tf.nn.leaky_relu, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(layer_idx))
				output = tf.layers.dense(e, 2, activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(len(enc_layers)-1))
		return output


	def perturb_dims(self,latent):
		perm = latent
		b, dim_z = perm.get_shape()
		for z in range(dim_z):
			 tf.random.shuffle(perm[:,z])
			# pi = torch.randperm(batch_size).to(latent_sample.device)
			# perm[:, z] = latent_sample[pi, z]
		return perm


		# perm = torch.zeros_like(latent_sample)
		# batch_size, dim_z = perm.size()
		#
		# for z in range(dim_z):
		# 	pi = torch.randperm(batch_size).to(latent_sample.device)
		# 	perm[:, z] = latent_sample[pi, z]
		#
		# return perm

	def complex_encoder(self, x, reuse=True):
		"""
		more complex encoder architecture for images with more than 1 color channel
		""" 
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				conv1 = tf.layers.conv2d(x, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='conv1')
				conv2 = tf.layers.conv2d(conv1, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='conv2')
				conv3 = tf.layers.conv2d(conv2, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='conv3')
				conv4 = tf.layers.conv2d(conv3, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='conv4')
				conv5 = tf.layers.conv2d(conv4, 256, 4, padding="VALID", activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='conv5')
				flattened = tf.reshape(conv5, (-1, 256*1*1))
				z_mean = tf.layers.dense(flattened, enc_layers[-1], activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-final')
		return z_mean


	def convolutional_32_encoder(self, x, reuse=True):
		"""
		more complex encoder architecture for images with more than 1 color channel
		""" 
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				conv1 = tf.layers.conv2d(x, 128, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv1')
				conv1 = tf.layers.batch_normalization(conv1)

				conv2 = tf.layers.conv2d(conv1, 256, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv2')
				conv2 = tf.layers.batch_normalization(conv2)

				conv3 = tf.layers.conv2d(conv2, 512, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv3')
				conv3 = tf.layers.batch_normalization(conv3)

				flattened = tf.contrib.layers.flatten(conv3)
				z_mean = tf.layers.dense(flattened, enc_layers[-1], activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-final')
		return z_mean





	def convolutional_32_decoder(self, z, reuse=True):
		"""
		more complex decoder architecture for images with more than 1 color channel (e.g. celebA)
		"""
		z = tf.convert_to_tensor(z)
		reuse=tf.AUTO_REUSE

		if self.vimco_samples > 1:
			samples = []

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				if len(z.get_shape().as_list()) == 2:
					# test
					d = tf.layers.dense(z, 4*4*512, activation=tf.nn.relu, use_bias=False, reuse=reuse, name='fc1')	
					d = tf.reshape(d, (-1, 4, 4, 512))
					deconv1 = tf.layers.conv2d_transpose(d, 512, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv1')
					deconv1 = tf.layers.batch_normalization(deconv1)
					deconv2 = tf.layers.conv2d_transpose(deconv1, 256, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv2')
					deconv2 = tf.layers.batch_normalization(deconv2)
					deconv3 = tf.layers.conv2d_transpose(deconv2, 128, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv3')
					deconv3 = tf.layers.batch_normalization(deconv3)
					deconv4 = tf.layers.conv2d(deconv3, 3, 1, strides=(1,1), padding="VALID", activation=self.last_layer_act, reuse=reuse, name='deconv4')
					return deconv4
				else:
					# train
					for i in range(self.vimco_samples):
						# iterate through one vimco sample at a time
						z_sample = z[i]
						d = tf.layers.dense(z_sample, 4*4*512, activation=tf.nn.relu, use_bias=False, reuse=reuse, name='fc1')	
						d = tf.reshape(d, (-1, 4, 4, 512))
						deconv1 = tf.layers.conv2d_transpose(d, 512, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv1')
						deconv1 = tf.layers.batch_normalization(deconv1)
						deconv2 = tf.layers.conv2d_transpose(deconv1, 256, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv2')
						deconv2 = tf.layers.batch_normalization(deconv2)
						deconv3 = tf.layers.conv2d_transpose(deconv2, 128, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv3')
						deconv3 = tf.layers.batch_normalization(deconv3)
						deconv4 = tf.layers.conv2d(deconv3, 3, 1, strides=(1,1), padding="VALID", activation=tf.nn.sigmoid, reuse=reuse, name='deconv4')
						samples.append(deconv4)
		x_reconstr_logits = tf.stack(samples, axis=0)
		print(x_reconstr_logits.get_shape())
		return x_reconstr_logits	


	def cifar10_convolutional_encoder(self, x, reuse=True):
		"""
		more complex encoder architecture for images with more than 1 color channel
		--> architecture specifically for cifar10!
		""" 
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				conv1 = tf.layers.conv2d(x, 64, (3,3), padding="SAME", activation=None, kernel_regularizer=regularizer, reuse=reuse, name='conv1')
				bn1 = tf.layers.batch_normalization(conv1)
				relu1 = tf.nn.relu(bn1)
				conv1_out = tf.layers.max_pooling2d(relu1, (2,2), (2,2), padding='same')
				# 2nd convolutional layer
				conv2 = tf.layers.conv2d(conv1_out, 32, (3,3), padding="SAME", activation=None, kernel_regularizer=regularizer, reuse=reuse, name='conv2')
				bn2 = tf.layers.batch_normalization(conv2)
				relu2 = tf.nn.relu(bn2)
				conv2_out = tf.layers.max_pooling2d(relu2, (2,2), (2,2), padding='same')
				# 3rd convolutional layer
				conv3 = tf.layers.conv2d(conv2_out, 16, (3,3), padding="SAME", activation=None, kernel_regularizer=regularizer, reuse=reuse, name='conv3')
				bn3 = tf.layers.batch_normalization(conv3)
				relu3 = tf.nn.relu(bn3)
				conv3_out = tf.layers.max_pooling2d(relu3, (2,2), (2,2), padding='same')
				flattened = tf.reshape(conv3_out, (-1, 4*4*16))
				z_mean = tf.layers.dense(flattened, enc_layers[-1], activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-final')
		return z_mean


	def cifar10_convolutional_decoder(self, z, reuse=True):
		"""
		more complex decoder architecture for images with more than 1 color channel
		--> NOTE: this architecture is specifically tailored for CIFAR10!
		"""
		z = tf.convert_to_tensor(z)
		reuse=tf.AUTO_REUSE

		if self.vimco_samples > 1:
			samples = []

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				if len(z.get_shape().as_list()) == 2:
					# reshape input properly for deconvolution
					d = tf.layers.dense(z, 4*4*16, activation=None, use_bias=False, reuse=reuse, name='fc1')	
					d = tf.reshape(d, (-1, 4, 4, 16))
					# start deconvolution process
					deconv1 = tf.layers.conv2d(d, 16, (3,3), padding="SAME", activation=None, 
						reuse=reuse, name='deconv1')
					bn1 = tf.layers.batch_normalization(deconv1)
					relu1 = tf.nn.relu(bn1)
					deconv1_out = tf.keras.layers.UpSampling2D((2,2))(relu1)
					# 2nd deconvolutional layer
					deconv2 = tf.layers.conv2d(deconv1_out, 32, (3,3), padding="SAME", activation=None, 
						reuse=reuse, name='deconv2')
					bn2 = tf.layers.batch_normalization(deconv2)
					relu2 = tf.nn.relu(bn2)
					deconv2_out = tf.keras.layers.UpSampling2D((2,2))(relu2)
					# 3rd convolutional layer
					deconv3 = tf.layers.conv2d(deconv2_out, 64, (3,3), padding="SAME", activation=None, 
						reuse=reuse, name='deconv3')
					bn3 = tf.layers.batch_normalization(deconv3)
					relu3 = tf.nn.relu(bn3)
					out = tf.keras.layers.UpSampling2D((2,2))(relu3)
					deconv3_out = tf.layers.conv2d(out, 3, (3, 3), padding="SAME", activation=None)
					deconv3_out = tf.layers.batch_normalization(deconv3_out)

					deconv3_out = tf.nn.sigmoid(deconv3_out)
					return deconv3_out
				else:
					# train
					for i in range(self.vimco_samples):
						# iterate through one vimco sample at a time
						z_sample = z[i]
						# reshape input properly for deconvolution
						d = tf.layers.dense(z_sample, 4*4*16, activation=None, use_bias=False, reuse=reuse, name='fc1')	
						d = tf.reshape(d, (-1, 4, 4, 16))
						# start deconvolution process
						deconv1 = tf.layers.conv2d(d, 16, (3,3), padding="SAME", activation=None, reuse=reuse, name='deconv1')
						bn1 = tf.layers.batch_normalization(deconv1)
						relu1 = tf.nn.relu(bn1)
						deconv1_out = tf.keras.layers.UpSampling2D((2,2))(relu1)
						# 2nd deconvolutional layer
						deconv2 = tf.layers.conv2d(deconv1_out, 32, (3,3), padding="SAME", activation=None, 
							reuse=reuse, name='deconv2')
						bn2 = tf.layers.batch_normalization(deconv2)
						relu2 = tf.nn.relu(bn2)
						deconv2_out = tf.keras.layers.UpSampling2D((2,2))(relu2)
						# 3rd convolutional layer
						deconv3 = tf.layers.conv2d(deconv2_out, 64, (3,3), padding="SAME", activation=None, 
							reuse=reuse, name='deconv3')
						bn3 = tf.layers.batch_normalization(deconv3)
						relu3 = tf.nn.relu(bn3)
						out = tf.keras.layers.UpSampling2D((2,2))(relu3)
						deconv3_out = tf.layers.conv2d(out, 3, (3, 3), padding="SAME", activation=None)
						deconv3_out = tf.layers.batch_normalization(deconv3_out)
						deconv3_out = tf.nn.sigmoid(deconv3_out)
						samples.append(deconv3_out)
		x_reconstr_logits = tf.stack(samples, axis=0)
		return x_reconstr_logits


	def convolutional_decoder(self, z, reuse=True):
		"""
		more complex decoder architecture for images with more than 1 color channel (e.g. celebA)
		"""
		z = tf.convert_to_tensor(z)
		reuse=tf.AUTO_REUSE

		if self.vimco_samples > 1:
			samples = []

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				if len(z.get_shape().as_list()) == 2:
					# test
					d = tf.layers.dense(z, 4*4*32, activation=tf.nn.elu, use_bias=False, reuse=reuse, name='fc1')	
					d = tf.reshape(d, (-1, 4, 4, 32))
					deconv1 = tf.layers.conv2d_transpose(d, 32, 1, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv1')
					deconv2 = tf.layers.conv2d_transpose(deconv1, 32, 5, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv2')
					deconv3 = tf.layers.conv2d_transpose(deconv2, 1, 5, strides=(2,2), padding="SAME", activation=tf.nn.sigmoid, reuse=reuse, name='deconv3')
					return deconv3
				else:
					# train
					for i in range(self.vimco_samples):
						# iterate through one vimco sample at a time
						z_sample = z[i]
						d = tf.layers.dense(z, 4*4*32, activation=tf.nn.elu, use_bias=False, reuse=reuse, name='fc1')	
						d = tf.reshape(d, (-1, 4, 4, 32))
						deconv1 = tf.layers.conv2d_transpose(d, 32, 1, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv1')
						deconv2 = tf.layers.conv2d_transpose(deconv1, 32, 5, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv2')
						deconv3 = tf.layers.conv2d_transpose(deconv2, 1, 5, strides=(2,2), padding="SAME", activation=tf.nn.sigmoid, reuse=reuse, name='deconv3')
						samples.append(deconv3)
		x_reconstr_logits = tf.stack(samples, axis=0)
		print(x_reconstr_logits.get_shape())
		return x_reconstr_logits


	def complex_decoder(self, z, reuse=True):
		"""
		more complex decoder architecture for images with more than 1 color channel (e.g. celebA)
		"""
		z = tf.convert_to_tensor(z)
		reuse=tf.AUTO_REUSE

		if self.vimco_samples > 1:
			samples = []

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				if len(z.get_shape().as_list()) == 2:
					# test
					d = tf.layers.dense(z, 256, activation=tf.nn.elu, use_bias=False, reuse=reuse, name='fc1')		
					d = tf.reshape(d, (-1, 1, 1, 256))
					deconv1 = tf.layers.conv2d_transpose(d, 256, 4, padding="VALID", activation=tf.nn.elu, reuse=reuse, name='deconv1')
					deconv2 = tf.layers.conv2d_transpose(deconv1, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv2')
					deconv3 = tf.layers.conv2d_transpose(deconv2, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv3')
					deconv4 = tf.layers.conv2d_transpose(deconv3, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv4')
					# output channel = 3
					deconv5 = tf.layers.conv2d_transpose(deconv4, 3, 4, strides=(2,2), padding="SAME", activation=tf.nn.sigmoid, reuse=reuse, name='deconv5')
					return deconv5
				else:
					# train; iterate through one vimco sample at a time
					for i in range(self.vimco_samples):
						z_sample = z[i]
						d = tf.layers.dense(z_sample, 256, activation=tf.nn.elu, use_bias=False, reuse=reuse, name='fc1')		
						d = tf.reshape(d, (-1, 1, 1, 256))
						deconv1 = tf.layers.conv2d_transpose(d, 256, 4, padding="VALID", activation=tf.nn.elu, reuse=reuse, name='deconv1')
						deconv2 = tf.layers.conv2d_transpose(deconv1, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv2')
						deconv3 = tf.layers.conv2d_transpose(deconv2, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv3')
						deconv4 = tf.layers.conv2d_transpose(deconv3, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv4')
						# output channel = 3
						deconv5 = tf.layers.conv2d_transpose(deconv4, 3, 4, strides=(2,2), padding="SAME", activation=tf.nn.sigmoid, reuse=reuse, name='deconv5')
						samples.append(deconv5)
		x_reconstr_logits = tf.stack(samples, axis=0)
		print(x_reconstr_logits.get_shape())
		return x_reconstr_logits


	def decoder(self, z, reuse=True, use_bias=False):
		# revert to original decoder for now!!

		d = tf.convert_to_tensor(z)
		dec_layers = self.dec_layers

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				for layer_idx, layer_dim in list(reversed(list(enumerate(dec_layers))))[:-1]:
					d = tf.layers.dense(d, layer_dim, activation=tf.nn.leaky_relu, reuse=reuse, name='fc-' + str(layer_idx), use_bias=use_bias)
				if self.is_binary:  # directly return logits
					x_reconstr_logits = tf.layers.dense(d, dec_layers[0], activation=None, reuse=reuse, name='fc-0', use_bias=use_bias)
				else:  # gaussian decoder
					x_reconstr_logits = tf.layers.dense(d, dec_layers[0], activation=self.last_layer_act, reuse=reuse, name='fc-0', use_bias=use_bias) # clip values between 0 and 1

		return x_reconstr_logits


	def get_loss(self, x, x_reconstr_logits):

		reg_loss = tf.losses.get_regularization_loss()
		if self.is_binary:
			# TODO: DOUBLE CHECK THIS
			x = tf.expand_dims(x, axis=0)
			x = tf.tile(x, [self.vimco_samples, 1, 1])
			reconstr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstr_logits, labels=x))
		else:
			if self.img_dim == 64:
				reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[1,2,3]))
			else:
				reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=1))
		tf.summary.scalar('reconstruction loss', reconstr_loss)
		total_loss = reconstr_loss + reg_loss
		
		return total_loss, reconstr_loss


	def build_vimco_loss(self, l):
	    """Builds VIMCO baseline as in https://arxiv.org/abs/1602.06725
	    Args:
	    l: Per-sample learning signal. shape [k, b] or
	        [number of samples, batch_size]
	    log_q_h: Sum of log q(h^l) over layers
	    Returns:
	    baseline to subtract from l
	    - implementation from: https://github.com/altosaar/vimco_tf
	    """
	    # compute the multi-sample stochastic bound
	    k, b = l.get_shape().as_list()
	    if b is None: b=FLAGS.batch_size
	    kf = tf.cast(k, tf.float32)

	    l_logsumexp = tf.reduce_logsumexp(l, [0], keepdims=True)
	    # L_hat is the multi-sample stochastic bound
	    L_hat = l_logsumexp - tf.log(kf)

	    # precompute the sum of log f
	    s = tf.reduce_sum(l, 0, keepdims=True)
	    # compute baseline for each sample
	    diag_mask = tf.expand_dims(tf.diag(tf.ones([k], dtype=tf.float32)), -1)
	    off_diag_mask = 1. - diag_mask

	    diff = tf.expand_dims(s - l, 0)  # expand for proper broadcasting
	    l_i_diag = 1. / (kf - 1.) * diff * diag_mask
	    l_i_off_diag = off_diag_mask * tf.stack([l] * k)
	    l_i = l_i_diag + l_i_off_diag
	    L_hat_minus_i = tf.reduce_logsumexp(l_i, [1]) - tf.log(kf)
	    
	    # compute the importance weights
	    w = tf.stop_gradient(tf.exp((l - l_logsumexp)))
	    
		# compute gradient contributions
	    local_l = tf.stop_gradient(L_hat - L_hat_minus_i)
	    
	    return local_l, w, L_hat[0, :]

	# def vat(self):

	def generated_mask(self):
		# if self.mask_ADV == False:
		noise_para = self.noise * tf.ones_like(self.mean)
		q = Bernoulli(logits=noise_para)
		mask = tf.cast(q.sample(self.vimco_samples), tf.int32)
		return mask
		# else:



	def vimco_loss(self, x, x_reconstr_logits):
		
		reg_loss = tf.losses.get_regularization_loss()
		if self.is_binary:  # match dimensions with vimco samples
			x = tf.expand_dims(x, axis=0)
			x = tf.tile(x, [self.vimco_samples, 1, 1])
			reconstr_loss = tf.reduce_sum(
				tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstr_logits, labels=x), axis=-1)
		else:
			if self.img_dim == 64:
				reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[2,3,4])
			elif self.img_dim == 32 and self.datasource.target_dataset in ['cifar10', 'svhn']:
				reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[2,3,4])
			else:
				reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=-1)

		# define your distribution q as a bernoulli, get multiple samples for VIMCO

		log_q_h_list = self.q.log_prob(self.z)
		log_q_h = tf.reduce_sum(log_q_h_list, axis=-1)
		
		# to be able to look at the log probabilities 
		self.log_q_h = log_q_h
		self.log_q_h_list = log_q_h_list
		loss = reconstr_loss

		# get vimco loss
		local_l, w, full_loss = self.build_vimco_loss(loss)

		# get appropriate losses for theta and phi respectively
		self.local_l = local_l
		theta_loss = (w * reconstr_loss) # shapes are both (5, batch_size)
		phi_loss = (local_l * log_q_h) + theta_loss

		# first sum over each sample, then average over minibatch
		theta_loss = tf.reduce_mean(tf.reduce_sum(theta_loss, axis=0))
		phi_loss = tf.reduce_mean(tf.reduce_sum(phi_loss, axis=0)) + reg_loss
		full_loss = tf.reduce_mean(full_loss)

		tf.summary.scalar('vimco (no gradient reduction) loss', full_loss)
		return theta_loss, phi_loss, full_loss

	def vimco_loss_ADV(self, x, x_reconstr_logits):

		reg_loss = tf.losses.get_regularization_loss()
		if self.is_binary:  # match dimensions with vimco samples
			x = tf.expand_dims(x, axis=0)
			x = tf.tile(x, [self.vimco_samples, 1, 1])
			reconstr_loss = tf.reduce_sum(
				tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstr_logits, labels=x), axis=-1)
		else:
			if self.img_dim == 64:
				reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[2, 3, 4])
			elif self.img_dim == 32 and self.datasource.target_dataset in ['cifar10', 'svhn']:
				reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[2, 3, 4])
			else:
				reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=-1)

		# define your distribution q as a bernoulli, get multiple samples for VIMCO

		log_q_h_list = self.q.log_prob(self.z)
		log_q_h = tf.reduce_sum(log_q_h_list, axis=-1)

		# to be able to look at the log probabilities
		self.log_q_h = log_q_h
		self.log_q_h_list = log_q_h_list
		loss = reconstr_loss

		# get vimco loss
		local_l, w, full_loss = self.build_vimco_loss(loss)

		# get appropriate losses for theta and phi respectively
		self.local_l = local_l
		theta_loss = (w * reconstr_loss)  # shapes are both (5, batch_size)
		phi_loss = (local_l * log_q_h) + theta_loss

		# first sum over each sample, then average over minibatch
		theta_loss = tf.reduce_mean(tf.reduce_sum(theta_loss, axis=0))
		phi_loss = tf.reduce_mean(tf.reduce_sum(phi_loss, axis=0)) + reg_loss
		full_loss = tf.reduce_mean(full_loss)

		tf.summary.scalar('vimco (no gradient reduction) loss', full_loss)
		return theta_loss, phi_loss, full_loss




	def get_test_loss(self, x, x_reconstr_logits):

		# reconstruction loss only, no regularization 
		if self.is_binary:
			reconstr_loss = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstr_logits, labels=x))
		else:
			if self.img_dim == 64 or self.img_dim == 32:  # RGB
				reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[1,2,3]))
			else:  # grayscale
				reconstr_loss = tf.reduce_mean(
					tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=1))

		return reconstr_loss




	def create_collapsed_computation_graph(self, x, reuse=tf.AUTO_REUSE):
		"""
		this models both (Y_i|X) and N as Bernoullis,
		so you get Y_i|X ~ Bern(sigmoid(WX) - 2*sigmoid(WX)*p + p)
		"""
		print('TRAIN: implicitly flipping individual bits with probability {}'.format(self.noise))
		dset_name = self.datasource.target_dataset
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			mean = self.encoder(x, reuse=reuse)
		elif dset_name == 'cifar10':
			mean = self.cifar10_convolutional_encoder(x, reuse=reuse)
		elif dset_name == 'svhn':
			mean = self.convolutional_32_encoder(x, reuse=reuse)
		elif dset_name == 'celebA':
			mean = self.complex_encoder(x, reuse=reuse)
		else:
			print('dataset {} is not implemented'.format(dset_name))
			raise NotImplementedError

		# for downstream classification
		classif_q = Bernoulli(logits=mean)
		classif_y = tf.cast(classif_q.sample(), tf.float32)

		
		# if self.noise == 0, then you have to feed in logits for the Bernoulli to avoid NaNs
		if self.noise != 0 and self.without_noise == False:
			print('##################')
			y_hat_prob = tf.nn.sigmoid(mean)
			_, _, mask_z = self.perturb_latent_code(x,classif_y)
			mask_z = tf.cast(mask_z,tf.float32)
			# tf.clip_by_value(y_hat_prob, 1e-7, 1. - 1e-7)
			# mask_z = tf.cast(tf.ones_like(y_hat_prob),tf.float32) * 0.01
			print (mask_z.get_shape())
			total_prob =tf.clip_by_value(y_hat_prob - (2 * y_hat_prob * mask_z) + mask_z, 1e-7, 1. - 1e-7)
			q = Bernoulli(probs=total_prob)
		else:
			print('no additional channel noise; feeding in logits for latent q_phi(z|x) to avoid numerical issues')
			total_prob = tf.nn.sigmoid(mean)
			q = Bernoulli(logits=mean)
			y_hat_prob = tf.nn.sigmoid(mean)


		# use VIMCO if self.vimco_samples > 1, else just one sample
		y = tf.cast(q.sample(self.vimco_samples), tf.float32)
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			x_reconstr_logits = self.decoder(y, reuse=reuse)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.cifar10_convolutional_decoder(y, reuse=reuse)
		elif dset_name == 'svhn':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=reuse)
		elif dset_name == 'celebA':
			x_reconstr_logits = self.complex_decoder(y, reuse=reuse)
		else:
			print('dataset {} is not implemented'.format(dset_name))
			raise NotImplementedError
		return total_prob, y, classif_y, q, x_reconstr_logits, y_hat_prob


	def create_erasure_collapsed_computation_graph(self, x, reuse=False):
		"""
		this models both (Y_i|X) and N as Bernoullis,
		so you get Y_i|X ~ Bern(sigmoid(WX) - 2*sigmoid(WX)*p + p)
		"""
		print('TRAIN: implicitly erasing individual bits with probability {}'.format(self.noise))
		dset_name = self.datasource.target_dataset
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			mean = self.encoder(x, reuse=reuse)
		elif dset_name == 'cifar10':
			mean = self.cifar10_convolutional_encoder(x, reuse=reuse)
		elif dset_name == 'svhn':
			mean = self.convolutional_32_encoder(x, reuse=reuse)
		elif dset_name == 'celebA':
			mean = self.complex_encoder(x, reuse=reuse)
		else:
			print('dataset {} is not implemented!'.format(dset_name))
			raise NotImplementedError
		
		# if self.noise == 0, then you have to feed in logits for the Bernoulli to avoid NaNs
		if self.noise != 0:
			print('computing probabilities for erasure channel!')
			# TODO
			y_hat_prob = tf.nn.softmax(mean)
			y_hat_prob = tf.clip_by_value(y_hat_prob, 1e-7, 1.-1e-7)

			# construct mask for erasure channel
			mask = np.zeros((2,3))
			mask[0,0] = 1 - self.noise
			mask[0,2] = self.noise
			mask[1,1] = 1 - self.noise
			mask[1,2] = self.noise

			total_prob = tf.reshape(tf.reshape(y_hat_prob, [-1, 2])@ mask, [-1, self.z_dim, 3])
			total_prob = tf.clip_by_value(total_prob, 1e-7, 1.-1e-7)
			q = Categorical(probs=total_prob)
		else:
			print('use BSC channel if you want to run for noise=0!')
			raise NotImplementedError	

		# use VIMCO if self.vimco_samples > 1, else just one sample
		y = tf.cast(q.sample(self.vimco_samples), tf.float32)
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			x_reconstr_logits = self.decoder(y, reuse=reuse)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.cifar10_convolutional_decoder(y, reuse=reuse)
		elif dset_name == 'svhn':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=reuse)
		elif dset_name == 'celebA':
			x_reconstr_logits = self.complex_decoder(y, reuse=reuse)
		else:
			print('dataset {} is not implemented'.format(dset_name))
			raise NotImplementedError

		return mean, y, q, x_reconstr_logits


	# TODO: vanilla beta-VAE for celebA
	def celebA_create_collapsed_computation_graph(self, x, reuse=False):
		"""
		this models both (Y_i|X) and N as Bernoullis,
		so you get Y_i|X ~ Bern(sigmoid(WX) - 2*sigmoid(WX)*p + p)
		"""
		print('TRAIN: implicitly flipping individual bits with probability {}'.format(self.noise))
		mean = self.complex_encoder(x, reuse=reuse)

		# classif_y
		classif_y = tf.cast(Bernoulli(logits=mean).sample(), tf.float32)

		# if self.noise == 0, then you have to feed in logits for the Bernoulli to avoid NaNs
		if self.noise != 0:
			y_hat_prob = tf.nn.sigmoid(mean)
			_, _, mask_z = self.perturb_latent_code(x, classif_y)
			mask_z = tf.cast(mask_z, tf.float32)
			# tf.clip_by_value(y_hat_prob, 1e-7, 1. - 1e-7)
			# mask_z = tf.cast(tf.ones_like(y_hat_prob),tf.float32) * 0.01
			# print (mask_z.get_shape())
			total_prob = tf.clip_by_value(y_hat_prob - (2 * y_hat_prob * mask_z) + mask_z, 1e-4, 1. - 1e-4)
			#"NUMERICAL issue"
			q = Bernoulli(probs=total_prob)
		else:
			print('no additional channel noise; feeding in logits for latent q_phi(z|x) to avoid numerical issues')
			total_prob = tf.nn.sigmoid(mean)
			q = Bernoulli(logits=mean)
			y_hat_prob = tf.nn.sigmoid(mean)
		y = tf.cast(q.sample(self.vimco_samples), tf.float32)
		x_reconstr_logits = self.complex_decoder(y, reuse=reuse)

		return total_prob, y, classif_y, q, x_reconstr_logits, y_hat_prob

	def get_conditional_entropy(self,z):
		conditional_entropy = -tf.reduce_sum(z * tf.log(z+ 1e-8) + (1 - z) * tf.log(1 - z + 1e-8))
		return  conditional_entropy



	def  get_collapsed_stochastic_test_sample(self, x, reuse=False):
		"""
		use collapsed Bernoulli at test time as well
		"""
		print('TEST: implicitly flipping individual bits with probability {}'.format(self.test_noise))
		dset_name = self.datasource.target_dataset
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			mean = self.encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'cifar10':
			mean = self.cifar10_convolutional_encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'svhn':
			mean = self.convolutional_32_encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'celebA':
			mean = self.complex_encoder(x, reuse=tf.AUTO_REUSE)
		else:
			print('dataset {} is not supported!'.format(dset_name))
			raise NotImplementedError

		# for downstream classification
		classif_q = Bernoulli(logits=mean)
		classif_y = tf.cast(classif_q.sample(), tf.float32)

		# test BSC
		if self.noise != 0:
			y_hat_prob = tf.nn.sigmoid(mean)
			total_prob = y_hat_prob - (2 * y_hat_prob * self.test_noise) + self.test_noise
			q = Bernoulli(probs=total_prob)
		else:
			print('no additional channel noise; feeding in logits for latent q_phi(z|x) to avoid numerical issues')
			total_prob = tf.nn.sigmoid(mean)
			q = Bernoulli(logits=mean)

		y = tf.cast(q.sample(), tf.float32)
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			x_reconstr_logits = self.decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.cifar10_convolutional_decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'svhn':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'celebA':
			x_reconstr_logits = self.complex_decoder(y, reuse=tf.AUTO_REUSE)
		else:
			print('dataset {} is not supported!'.format(dset_name))
			raise NotImplementedError

		return total_prob, y, classif_y, q, x_reconstr_logits

	def test_collapsed_sample_without_noise(self, x, reuse=False):
		"""
		use collapsed Bernoulli at test time as well
		"""
		print('TEST: implicitly flipping individual bits with probability {}'.format(0))
		dset_name = self.datasource.target_dataset
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			mean = self.encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'cifar10':
			mean = self.cifar10_convolutional_encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'svhn':
			mean = self.convolutional_32_encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'celebA':
			mean = self.complex_encoder(x, reuse=tf.AUTO_REUSE)
		else:
			print('dataset {} is not supported!'.format(dset_name))
			raise NotImplementedError

		# for downstream classification
		classif_q = Bernoulli(logits=mean)
		classif_y = tf.cast(classif_q.sample(), tf.float32)
		# test BSC
		# if self.noise != 0:
		# 	y_hat_prob = tf.nn.sigmoid(mean)
		# 	total_prob = y_hat_prob - (2 * y_hat_prob * self.test_noise) + self.test_noise
		# 	q = Bernoulli(probs=total_prob)
		# else:
		print('no additional channel noise; feeding in logits for latent q_phi(z|x) to avoid numerical issues')
		total_prob = tf.nn.sigmoid(mean)
		q = Bernoulli(logits=mean)
		y = tf.cast(q.sample(), tf.float32)
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			x_reconstr_logits = self.decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.cifar10_convolutional_decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'svhn':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'celebA':
			x_reconstr_logits = self.complex_decoder(y, reuse=tf.AUTO_REUSE)
		else:
			print('dataset {} is not supported!'.format(dset_name))
			raise NotImplementedError

		return total_prob, y, classif_y, q, x_reconstr_logits


	def get_collapsed_erasure_stochastic_test_sample(self, x, reuse=False):
		"""
		use collapsed Bernoulli at test time as well
		"""
		print('TEST: implicitly flipping individual bits with probability {}'.format(self.test_noise))
		dset_name = self.datasource.target_dataset
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			mean = self.encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'cifar10':
			mean = self.cifar10_convolutional_encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'svhn':
			mean = self.convolutional_32_encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'celebA':
			mean = self.complex_encoder(x, reuse=tf.AUTO_REUSE)
		else:
			print('dataset {} is not supported!'.format(dset_name))
			raise NotImplementedError

		# test BEC
		if self.noise != 0:
			print('computing probabilities for erasure channel! (test)')
			y_hat_prob = tf.nn.softmax(mean)
			y_hat_prob = tf.clip_by_value(y_hat_prob, 1e-7, 1.-1e-7)

			# construct mask for erasure channel
			mask = np.zeros((2,3))
			mask[0,0] = 1 - self.test_noise
			mask[0,2] = self.test_noise
			mask[1,1] = 1 - self.test_noise
			mask[1,2] = self.test_noise

			total_prob = tf.reshape(tf.reshape(y_hat_prob, [-1, 2]) @ mask, [-1, self.z_dim, 3])
			total_prob = tf.clip_by_value(total_prob, 1e-7, 1.-1e-7)
			q = Categorical(probs=total_prob)
		else:
			print('Use BSC if there is no channel noise!')
			raise NotImplementedError

		y = tf.cast(q.sample(), tf.float32)

		# decoder
		if dset_name in ['mnist', 'BinaryMNIST', 'omniglot', 'random']:
			x_reconstr_logits = self.decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.cifar10_convolutional_decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'svhn':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'celebA':
			x_reconstr_logits = self.complex_decoder(y, reuse=tf.AUTO_REUSE)
		else:
			print('dataset {} is not supported!'.format(dset_name))
			raise NotImplementedError

		return total_prob, y, q, x_reconstr_logits

	# def pairwise_info(self):
	# 	"""
	# 	which is a MUTUAL information between the pairwise codes
	# 	as the implement is like the
	# 	:return:
	# 	"""
	# 	# samples = self.mean
	# 	# joint_probability =
	#
	# 	qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:10000].cuda()))
	#
	# 	K, S = qz_samples.size()
	# 	N, _, nparams = qz_params.size()
	# 	assert (nparams == q_dist.nparams)
	# 	assert (K == qz_params.size(1))
	#
	# 	marginal_entropies = torch.zeros(K).cuda()
	# 	joint_entropy = torch.zeros(1).cuda()
	#
	# 	pbar = tqdm(total=S)
	# 	k = 0
	# 	while k < S:
	# 		batch_size = min(10, S - k)
	# 		logqz_i = q_dist.log_density(
	# 			qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
	# 			qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
	# 		k += batch_size
	#
	# 		# computes - log q(z_i) summed over minibatch
	# 		marginal_entropies += (math.log(N) - logsumexp(logqz_i, dim=0, keepdim=False).data).sum(1)
	# 		# computes - log q(z) summed over minibatch
	# 		logqz = logqz_i.sum(1)  # (N, S)
	# 		joint_entropy += (math.log(N) - logsumexp(logqz, dim=0, keepdim=False).data).sum(0)
	# 		pbar.update(batch_size)
	# 	pbar.close()
	#
	# 	marginal_entropies /= S
	# 	joint_entropy /= S
	#
	# 	return marginal_entropies, joint_entropy




	def train(self, ckpt=None, verbose=True):
		"""
		Trains VAE for specified number of epochs.
		"""
		print("train---------")

		sess = self.sess
		datasource = self.datasource

		if FLAGS.resume:
			if ckpt is None:
				ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
			self.saver.restore(sess, ckpt)
		sess.run(self.init_op)

		t0 = time.time()
		train_dataset = datasource.get_dataset('train')
		train_dataset = train_dataset.batch(FLAGS.batch_size)
		train_dataset = train_dataset.shuffle(buffer_size=10000)
		train_iterator = train_dataset.make_initializable_iterator()
		next_train_batch = train_iterator.get_next()

		valid_dataset = datasource.get_dataset('valid')
		valid_dataset = valid_dataset.batch(FLAGS.batch_size)
		valid_iterator = valid_dataset.make_initializable_iterator()
		next_valid_batch = valid_iterator.get_next()

		self.train_writer = tf.summary.FileWriter(FLAGS.outdir + '/train', graph=tf.get_default_graph())
		self.valid_writer = tf.summary.FileWriter(FLAGS.outdir + '/valid', graph=tf.get_default_graph())

		epoch_train_losses = []
		epoch_valid_losses = []
		epoch_save_paths = []

		for epoch in range(FLAGS.n_epochs):
			sess.run(train_iterator.initializer)
			sess.run(valid_iterator.initializer)
			epoch_train_loss = 0.
			estimated_mi = 0.
			total_tc = 0.
			total_dloss = 0.
			num_batches = 0.
			epohy = 0.
			epohy_x = 0.
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			# summary, _ = sess.run([merged, train_step],
			# 					  feed_dict=feed_dict(True),
			# 					  options=run_options,
			# 					  run_metadata=run_metadata)
			self.train_writer.add_run_metadata(run_metadata, 'step%d' % epoch)
			while True:
				try:
					self.training = True
					if (not self.is_binary) and (self.datasource.target_dataset != 'celebA'):
						x = sess.run(next_train_batch)[0]

					else:
						# no labels available for binarized MNIST

						x = sess.run(next_train_batch)
						# pdb.set_trace()
					if self.noisy_mnist:
						# print('training with noisy MNIST...')
						feed_dict = {self.x: (x + np.random.normal(0, 0.5, x.shape)), self.true_x: x}
					else:
						feed_dict = {self.x: x}

					# REINFORCE-style training with VIMCO or vanilla gradient update
					if not self.discrete_relax:
						# if epoch <150:
							# pdb.set_trace()
						hy, hy_x,tc,_,_ = sess.run([self.kl_loss,self.conditional_entropy, self.tc_loss,self.discrete_train_op1,self.discrete_train_op2], feed_dict)
						_, dloss= sess.run([self.discrete_train_op3,self.D_loss],feed_dict)
						# pdb.set_trace()
						# else:
						# 	hy, hy_x,_,_= sess.run([self.kl_loss, self.conditional_entropy, self.discrete_train_op1], feed_dict)
					else:
						# this works for both gumbel-softmax
						sess.run([self.train_op], feed_dict)


					# hy, hy_x,batch_loss, train_summary, gs = sess.run([
					# 	-self.kl_loss,self.conditional_entropy,self.reconstr_loss, self.summary_op, self.global_step], feed_dict)


					# train_writer.add_summary(summary, i)
					# print('Adding run metadata for', i)

					batch_loss, train_summary, gs = sess.run([
						self.reconstr_loss, self.summary_op, self.global_step],
						feed_dict,options=run_options,
										  run_metadata=run_metadata)

					hy = -hy
					epoch_train_loss += batch_loss
					estimated_mi += hy - hy_x
					epohy += hy
					epohy_x += hy_x
					total_tc += tc
					total_dloss += dloss


					# self.train_writer.add_summary(train_summary, gs)
					num_batches += 1

				except tf.errors.OutOfRangeError:
					break
			# end of training epoch; adjust temperature here if using Gumbel-Softmax
			# if self.discrete_relax:
			# 	if (counter % 1000 == 0) and (counter > 0):
			# 		self.adj_temp = np.maximum(self.tau * np.exp(-self.anneal_rate * counter), self.min_temp)
			# 		print('adjusted temperature to: {}'.format(self.adj_temp))
			# enter validation phase
			if verbose:
				epoch_train_loss /= num_batches
				estimated_mi /= num_batches
				epohy /= num_batches
				epohy_x /=num_batches
				total_tc /= num_batches
				self.training = False
				if (not self.is_binary) and (self.datasource.target_dataset != 'celebA'):
					x = sess.run(next_valid_batch)[0]
				else:
					# no labels available for binarized MNIST and celebA
					x = sess.run(next_valid_batch)
				if self.noisy_mnist:
					# print('training with noisy MNIST...')
					feed_dict = {self.x: (x + np.random.normal(0, 0.5, x.shape)), self.true_x: x}
				else:
					feed_dict = {self.x: x}

				# save run stats
				epoch_valid_loss, valid_summary, gs = sess.run([self.test_loss, self.summary_op, self.global_step], feed_dict=feed_dict)
				if epoch_train_loss < 0:  # note: this only applies to non-binary data since it's L2 loss
					print('Epoch {}, (no sqrt) l2 train loss: {:0.6f}, hy:{:0.6f}, hy_x:{:0.6f},mi:{:0.6f}, l2 valid loss: {:0.6f},total_tc: {:0.6f},time: {}s'. \
				format(epoch+1, epoch_train_loss, epohy, epohy_x, estimated_mi,np.sqrt(epoch_valid_loss), total_tc, int(time.time()-t0)))
				else:
					print (total_dloss)
					print(
						'Epoch {}, l2 train loss: {:0.6f}, hy:{:0.6f}, hy_x:{:0.6f},mi:{:0.6f}, l2 valid loss: {:0.6f}, total_tc: {:0.6f}, dloss:{:0.6f},time: {}s'. \
						format(epoch + 1, epoch_train_loss, epohy, epohy_x, estimated_mi, np.sqrt(epoch_valid_loss), total_tc, total_dloss, int(time.time() - t0)))
				sys.stdout.flush()
				save_path = self.saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'), global_step=gs)
				epoch_train_losses.append(epoch_train_loss)
				epoch_valid_losses.append(epoch_valid_loss)
				epoch_save_paths.append(save_path)
		best_ckpt = None
		if verbose:
			min_idx = epoch_valid_losses.index(min(epoch_valid_losses))
			print('Restoring ckpt at epoch', min_idx+1,'with lowest validation error:', epoch_save_paths[min_idx])
			best_ckpt = epoch_save_paths[min_idx]
		return (epoch_train_losses, epoch_valid_losses), best_ckpt



	def test(self, ckpt=None):
		print ('start------')

		sess = self.sess
		datasource = self.datasource
		self.training = False

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
		
		self.saver.restore(sess, ckpt)

		test_dataset = datasource.get_dataset('test')
		test_dataset = test_dataset.batch(FLAGS.batch_size)
		test_iterator = test_dataset.make_initializable_iterator()
		next_test_batch = test_iterator.get_next()

		test_loss = 0.
		num_batches = 0.
		num_incorrect = 0
		hy_ = 0.
		hy_x_ = 0.

		sess.run(test_iterator.initializer)


		z_list = []

		while True:
			try:
				if not self.is_binary:
					x, y = sess.run(next_test_batch)
				else:
					# no labels available for binarized MNIST
					x = sess.run(next_test_batch)
				# specify whether to train with noise
				if self.noisy_mnist:
					# print('training with noisy MNIST...')
					feed_dict = {self.x: (x + np.random.normal(0, 0.5, x.shape)), self.true_x: x}
				else:
					feed_dict = {self.x: x}


				# what to save and what to not
				if self.img_dim != 64:
					x_reconstr_logits,classif_z = sess.run([self.x_reconstr_logits,self.classif_z], feed_dict)
				else:
					x_reconstr_logits,classif_z = sess.run([self.x_reconstr_logits,self.classif_z], feed_dict)

				hy, hy_x = sess.run(
					[self.kl_loss, self.conditional_entropy],
					feed_dict)
				hy_ += hy
				hy_x_ += hy_x
				z_list.append(classif_z)
				batch_test_loss = sess.run(self.test_loss, feed_dict)
				test_loss += batch_test_loss

				# round output of Gaussian decoder to see how many were incorrectly decoded
				rounded = np.round(x_reconstr_logits)
				wrong = np.sum(~np.equal(x, rounded)) 
				num_incorrect += wrong
				num_batches += 1.
			except tf.errors.OutOfRangeError:
				break

		total_len = len(z_list)
		z_list = np.concatenate(z_list,axis=0)
		distribution_list = z_list.sum(axis=0)*1.0/(total_len*100)
		print (distribution_list)

		n, bins, patches = plt.hist(distribution_list, 100, density=False, facecolor='g', alpha=0.75)
		plt.xlabel('Marginal Probability')
		plt.ylabel('Numbers')
		# plt.title('Histogram of IQ')
		# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
		plt.xlim(0, 1)
		# plt.ylim(0, 0.03)
		# plt.grid(True)
		plt.savefig(os.path.join(FLAGS.outdir, 'distribution.pdf'))

		test_loss /= num_batches
		mi = (-hy_-hy_x_) / num_batches
		print('L2 squared test loss (per image): {:0.6f}'.format(test_loss))
		print('L2 squared test loss (per pixel): {:0.6f}'.format(test_loss/self.input_dim))

		print('L2 test loss (per image): {:0.6f}'.format(np.sqrt(test_loss)))
		print('L2 test loss (per pixel): {:0.6f}'.format(np.sqrt(test_loss)/self.input_dim))
		print ('mutual information', mi)
		# print (hy_,hy_x_)

		return test_loss


	def reconstruct(self, ckpt=None, pkl_file=None):

		sess = self.sess
		datasource = self.datasource

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
		self.saver.restore(sess, ckpt)

		if pkl_file is None:
			test_dataset = datasource.get_dataset('test')
			test_dataset = test_dataset.batch(FLAGS.batch_size)
			test_iterator = test_dataset.make_initializable_iterator()
			next_test_batch = test_iterator.get_next()

			sess.run(test_iterator.initializer)
			if not self.is_binary:
				x = sess.run(next_test_batch)[0]
			else:
				x = sess.run(next_test_batch)
		else:
			with open(pkl_file, 'rb') as f:
				images = pickle.load(f)
			x = np.vstack([images[i] for i in range(10)])
		# grab reconstructions
		if self.noisy_mnist:
			# print('training with noisy MNIST...')
			feed_dict = {self.x: (x + np.random.normal(0, 0.5, x.shape)), self.true_x: x}
		else:
			feed_dict = {self.x: x}
		# grab reconstructions
		x_reconstr_logits = sess.run(self.test_x_reconstr_logits, feed_dict)
		# rounding values here to get hard {0, 1} values
		if self.is_binary:
			x_reconstr_logits = np.round(x_reconstr_logits)
		print(np.max(x_reconstr_logits), np.min(x_reconstr_logits))
		print(np.max(x), np.min(x))
		
		x_reconstr_logits = np.reshape(x_reconstr_logits, (-1, self.input_dim))
		if self.img_dim == 64:
			x = np.reshape(x, (-1, self.input_dim))
			plot(np.vstack((
				x[0:10], x_reconstr_logits[0:10])), m=10, n=2, px=64, title='reconstructions')
		elif self.img_dim == 32:
			x = np.reshape(x, (-1, self.input_dim))
			plot(np.vstack((
				x[0:10], x_reconstr_logits[0:10])), m=10, n=2, px=32, title='reconstructions')
		else:
			# TODO: edited this
			plot(np.vstack((
				x[0:10], x_reconstr_logits[0:10])), m=10, n=2, title='reconstructions')
		
		with open(os.path.join(FLAGS.outdir, 'reconstr.pkl'), 'wb') as f:
			pickle.dump(x_reconstr_logits, f, pickle.HIGHEST_PROTOCOL)
		return x_reconstr_logits

	def markov_chain(self, ckpt=None):

		sess = self.sess
		datasource = self.datasource

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
		self.saver.restore(sess, ckpt)

		print('initializing with samples from test set...')
		test_dataset = datasource.get_dataset('test')
		test_dataset = test_dataset.batch(FLAGS.batch_size)
		test_iterator = test_dataset.make_initializable_iterator()
		next_test_batch = test_iterator.get_next()

		sess.run(test_iterator.initializer)
		if not self.is_binary:
			x_t = sess.run(next_test_batch)[0]
		else:
			x_t = sess.run(next_test_batch)

		# random initialization of 10 samples with noise
		# print('initializing markov chain with random Gaussian noise...')
		# x_t = np.clip(np.random.normal(
		# 	0., 0.01, 10 * self.input_dim).reshape(-1, self.input_dim), 0., 1.)E
		# print('initializing markov chain with random Bernoulli noise...')
		# x_t = np.random.binomial(
			# 1, 0.5, 10 * self.input_dim).reshape(-1, self.input_dim)

		# just get first 10 samples
		samples = [x_t[0:10]]
		for step in range(FLAGS.total_mcmc_steps):
			# whether to train with noise
			if self.noisy_mnist:
				# print('training with noisy MNIST...')
				feed_dict = {self.x: x_t + np.random.normal(0, 0.5, x_t.shape), self.true_x: x_t}
			else:
				feed_dict = {self.x: x_t}

			x_reconstr_mean = sess.run(self.test_x_reconstr_logits, feed_dict)
			x_t_plus_1 = np.clip(np.random.normal(loc=x_reconstr_mean, scale=0.01), 0., 1.)
			x_t = x_t_plus_1

			if (step + 1) % 1000 == 0:
				print('Step', step)
				samples.append(x_t[0:10])

		samples = np.vstack(samples)
		print(samples.shape)
		plot(samples, m=10, n=10, title='markov_chain_samples')

		return samples

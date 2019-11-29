import numpy as np 
import tensorflow as tf 
from utils import *
from datasource import Datasource

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

"""
only change flip samples and miw
"""
# File options
flags.DEFINE_string('datadir', '/mnt/cephfs_hl/mlnlp/yxsong/Codes/necst/data', 'directory for datasets')
flags.DEFINE_string('datasource', 'BinaryMNIST', 'mnist/BinaryMNIST/random/omniglot/binary_omniglot/celebA/svhn/cifar10')
flags.DEFINE_string('logdir', './models/', 'directory to save checkpoints, events files')
flags.DEFINE_string('outdir', './results/', 'directory to save samples, final results')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train.')
flags.DEFINE_bool('test', True, 'True to test.')

flags.DEFINE_string('ckpt', None, 'ckpt to load if resume is True. Defaults (None) to latest ckpt in logdir')
flags.DEFINE_string('exp_id', '082701', 'exp_id appended to logdir and outdir')
flags.DEFINE_string('gpu_id', '0', 'gpu id options')
flags.DEFINE_bool('dump', True, 'Dumps to log.txt if True')

# Probabilistic latent bit representation
flags.DEFINE_bool('is_binary', True, 'True if dataset is binary, false otherwise.')
flags.DEFINE_bool('discrete_relax', False, 'Use gumbel-softmax to sample codes from a RelaxedBernoulli distribution')
flags.DEFINE_bool('without_noise', False, 'whether not use noise')
flags.DEFINE_integer('vimco_samples', 5, 'number of VIMCO samples to use during training')
flags.DEFINE_integer('flip_samples', 10, 'number of flip dims to use during training')


# Noise specifications
flags.DEFINE_bool('noisy_mnist', False, 'specify whether to train necst with noisy MNIST')
flags.DEFINE_string('channel_model', 'bsc', 'bsc/bec')
flags.DEFINE_float('noise', 0.1, 'specify proportion of entires to corrupt in z')
flags.DEFINE_float('test_noise', 0.1, 'specify proportion of entries to corrupt in z at test time.')

# Training options
flags.DEFINE_integer('n_epochs', 200, 'number of training epochs')
flags.DEFINE_integer('batch_size', 100, 'number of datapoints per batch')
flags.DEFINE_float('lr', 0.001, 'learning rate for the model')
flags.DEFINE_float('wadv', 0.01, 'weight for the adversarial regularization term')
flags.DEFINE_float('miw', 0.01, 'Weight for mutual information maximization')
flags.DEFINE_float('cew', 0.0, 'weight for the conditional entropy term')
flags.DEFINE_float('klw', 0.0, 'weight for improve marginal entropy')
flags.DEFINE_float('lpw', 0.0, 'weight for lp penalized')
flags.DEFINE_float('denw', 0.1, 'Weight for denoising term')
flags.DEFINE_float('radius', 3.5, 'weight for marginal kl term')

flags.DEFINE_string('optimizer', 'adam', 'sgd, adam, momentum')
flags.DEFINE_integer('log_interval', 500, 'training steps after which summary and checkpoints dumped')
flags.DEFINE_integer('num_samples', 16, 'number of samples to generate')

# Model options
flags.DEFINE_string('model', 'necst', 'necst')
flags.DEFINE_string('activation', 'relu', 'sigmoid/tanh/softplus/leakyrelu/relu')
flags.DEFINE_integer('seed', 0, 'random seed for initializing model parameters')
flags.DEFINE_string('dec_arch', '500,500', 'comma-separated decoder architecture')
flags.DEFINE_string('enc_arch', '500', 'comma-separated encoder architecture')
flags.DEFINE_integer('n_bits', 100, 'number of measurements')
flags.DEFINE_float('reg_param', 0.0001, 'regularization for encoder')
flags.DEFINE_bool('non_linear_act', True, 'nonlinear activation on final layer of encoder if True')
flags.DEFINE_bool('adv', True, 'whether use the adv permutation to training the model')
# flags.DEFINE_float('adv', True, 'whether use the adv permutation to training the model')
flags.DEFINE_integer('total_mcmc_steps', 9000, 'number of mcmc steps')
flags.DEFINE_string('pkl_file', None, 'pkl file for reconstruction')


def process_flags():
	"""
	processes easy-to-specify cmd line FLAGS to appropriate syntax
	"""
	FLAGS.optimizer = get_optimizer_fn(FLAGS.optimizer)
	FLAGS.activation = get_activation_fn(FLAGS.activation)
	
	if FLAGS.dec_arch == '':
		FLAGS.dec_arch = []
	else:
		FLAGS.dec_arch = list(map(int, FLAGS.dec_arch.split(',')))
	
	if FLAGS.enc_arch == '':
		FLAGS.enc_arch = []
	else:
		FLAGS.enc_arch = list(map(int, FLAGS.enc_arch.split(',')))

	return

def main():
	"""
	run program: preprocess data, train model, validate/test.
	"""
	# print ('a debugging version')
	# print (FLAGS.test)
	# tf.reset_default_graph()
	os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
	# subpath = 'wadv_' + str(FLAGS.wadv) + 'cew_' + str(FLAGS.cew) +'klw_' + str(FLAGS.klw) + 'noise_' + str(FLAGS.noise)
	subpath = 'miw_' + str(FLAGS.miw) + '_flip_' + str(FLAGS.flip_samples) + '_bits_' + str(FLAGS.n_bits) + '_epochs_' + str(FLAGS.n_epochs)
	print(subpath)
	# FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.datasource, subpath, FLAGS.exp_id)
	# FLAGS.outdir = os.path.join(FLAGS.outdir, FLAGS.datasource, subpath, FLAGS.exp_id)
	# subpath = 'noise_' + str(FLAGS.noise)
	FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.datasource, subpath, FLAGS.exp_id)
	FLAGS.outdir = os.path.join(FLAGS.outdir, FLAGS.datasource, subpath, FLAGS.exp_id)
	# FLAGS.outdir = '/mnt/cephfs_hl/arnold/vae/mcmc/run1/tasks/102803/log'
	
	if not os.path.exists(FLAGS.logdir):
		os.makedirs(FLAGS.logdir)
	if not os.path.exists(FLAGS.outdir):
		os.makedirs(FLAGS.outdir)
	# print('---------------------------------1')

	import json
	with open(os.path.join(FLAGS.outdir, 'config.json'), 'w') as fp:
	    json.dump(tf.app.flags.FLAGS.flag_values_dict(), fp, indent=4, separators=(',', ': '))

	if FLAGS.dump:
		import sys
		sys.stdout = open(os.path.join(FLAGS.outdir, 'log.txt'), 'w')

	process_flags()
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(
		gpu_options=gpu_options, allow_soft_placement=True))
	datasource = Datasource(sess)
	model_class = load_dynamic(FLAGS.model.upper(), FLAGS.model)
	model = model_class(sess, datasource)
	# print('---------------------------------2')

	# run computational graph
	best_ckpt = FLAGS.ckpt
	if best_ckpt is not None:
		print('resuming ckpt supplied; restoring model from {}'.format(best_ckpt))
	if FLAGS.train:
		learning_curves, best_ckpt = model.train(ckpt=best_ckpt)
	# print('---------------------------------3'X)
	if FLAGS.test:
		# print('-------------test---------------------4')
		if best_ckpt is None:
			log_file = os.path.join(FLAGS.outdir, 'log.txt')
			if os.path.exists(log_file):
				for line in open(log_file):
					if "Restoring ckpt at epoch" in line:
						best_ckpt = line.split()[-1]
						break
		model.test(ckpt=best_ckpt)
		model.reconstruct(ckpt=best_ckpt, pkl_file=FLAGS.pkl_file)
		model.markov_chain(ckpt=best_ckpt)

if __name__ == "__main__":
	main()
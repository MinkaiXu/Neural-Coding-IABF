# Neural Joint Source-Channel Coding

This repo contains a reference implementation for IABF as described in the paper:
> Infomax Neural Joint Source-Channel Coding via Adversarial Bit Flip </br>
> Yuxuan Song, [Minkai Xu](http://minkaixu.com/), [Lantao Yu](http://lantaoyu.com/), [Hao Zhou](https://zhouh.github.io/), Shuo Shao, [Yong Yu](https://scholar.google.com/citations?user=-84M1m0AAAAJ&hl=en) </br>
> AAAI Conference on Artificial Intelligence (AAAI), 2020. </br>
> Paper: coming soon! </br>


## Requirements
The codebase is implemented in Python 3.6 and Tensorflow. To install the necessary dependencies, run:
```
pip3 install -r requirements.txt
```

## Datasets
A set of scripts for data pre-processing are included in the directory `./data_setup`. Relevant files for 
The IABF model operates over Tensorflow [TFRecords](https://www.tensorflow.org/tutorials/load_data/tf_records). A few points to note:

1. Raw data files for MNIST and BinaryMNIST can be downloaded using `data_setup/download.py`. CelebA files can be downloaded using `data_setup/celebA_download.py`. CIFAR10 can be downloaded (with tfrecords automatically generated) using `data_setup/generate_cifar10_tfrecords.py`. All other data files (Omniglot, SVHN) must be downloaded separately.
2. Omniglot and CelebA should be converted into `.hdf5` format using `data_setup/convert_celebA_h5.py` and `data_setup/convert_omniglot_h5.py` respectively.
3. Random {0,1} bits can be generated using `data_setup/gen_random_bits.py`.
4. After this step, tfrecords must be generated using: `data_setup/convert_to_records.py` before running the model.

## Options
Training the IABF model takes a set of command line arguments in the `main.py` script. The most relevant ones are listed below:
```
--flip_samples: the number of flipped bits for adversarial training
--miw: the weight for mutual information term
--datasource (STRING):    one of [mnist, BinaryMNIST, random, omniglot, celebA, svhn, cifar10]
--is_binary (BOOL):       whether or not the data is binary {0,1}, e.g. BinaryMNIST
--vimco_samples (INT):    number of samples to use for VIMCO
--channel_model (STRING): BSC/BEC
--noise (FLOAT):          channel noise level during training
--test_noise (FLOAT):     channel noise level at TEST time
--n_epochs (INT):         number of training epochs
--batch_size (INT):       size of minibatch
--lr (FLOAT):             learning rate of optimizer
--optimizer (STRING):     one of [adam, sgd]
--dech_arch (STRING):     comma-separated decoder architecture
--enc_arch (STRING):      comma-separated encoder architecture
--reg_param (FLOAT):      regularization for encoder architecture
```

## Examples
Download and Train a 100-bit IABF model with BSC noise = 0.1 on BinaryMNIST:
```
# Download the BinaryMNIST dataset
python3 data_setup/download.py BinaryMNIST

# Generate a tfrecords file corresponding to the dataset
python3 data_setup/convert_to_records.py --dataset=BinaryMNIST

# Train the model
python3 main.py --exp_id="0828" --flip_samples=7 --miw=0.0000001 --noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --test_noise=0.1 --n_bits=100 --is_binary=True
```

## Citing
If you find IABF useful in your research, please consider citing the following paper:

```
@article{song2019iabf,
  title={Infomax Neural Joint Source-Channel Coding via Adversarial Bit Flip},
  author={Song, Yuxuan and Xu, Minkai and Yu, Lantao and Zhou, Hao and Shao, Shuo and Yu, Yong},
  journal={AAAI Conference on Artificial Intelligence (AAAI), 2020.},
  year={2019}
}
```

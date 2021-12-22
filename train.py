"""
Philip Anastassiou (pja2114)
COMS 6998: Fundamentals of Speech Recognition
Professor Homayoon Beigi
Columbia University
Due: December 19th, 2021

Citation for original authors of this file:

@misc{sammutbonnici2021timbre,
      title={Timbre Transfer with Variational Auto Encoding and Cycle-Consistent Adversarial Networks},
      author={Russell Sammut Bonnici and Charalampos Saitis and Martin Benning},
      year={2021},
      eprint={2109.02096},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}

@inproceedings{AlBadawy2020,
  author={Ehab A. AlBadawy and Siwei Lyu},
  title={{Voice Conversion Using Speech-to-Speech Neuro-Style Transfer}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={4726--4730},
  doi={10.21437/Interspeech.2020-3056},
  url={http://dx.doi.org/10.21437/Interspeech.2020-3056}
}

Re-implemented in TensorFlow 2.0 by Philip Anastassiou (Github: @phiana)
For original PyTorch implementations, please refer to their repositories
"""

import argparse
import os
import numpy as np
import itertools
import sys
from tqdm import tqdm

from models import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from data_proc import DataProc

from utils import plot_batch_train
import random

tf.config.run_functions_eagerly(True)

# pja2114: Renamed and rearranged some of these options
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--model_name", type=str, default="pja2114_voice_conversion", help="name of the model")
parser.add_argument("--dataset", type=str, default="data", help="path to dataset for training")
parser.add_argument("--n_spkrs", type=int, default=2, help="number of speakers for conversion, must match that of preprocessing")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--spect_height", type=int, default=128, help="size of image height")
parser.add_argument("--spect_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--plot_interval", type=int, default=1, help="epoch interval between saving plots (disable with -1)")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")

opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
# os.makedirs("saved_models/%s" % opt.model_name, exist_ok=True)
os.makedirs("saved_h5_weights/%s" % opt.model_name, exist_ok=True)
os.makedirs("training_checkpoints/%s" % opt.model_name, exist_ok=True)

# pja2114: Changed for TensorFlow syntax
criterion_GAN = tf.keras.losses.MeanSquaredError()
criterion_pixel = tf.keras.losses.MeanAbsoluteError()

#pja2114: Rewritten with TensorFlow syntax
def compute_kl(mu):
	mu_2 = tf.pow(mu, 2)
	loss = tf.reduce_mean(mu_2)
	return loss

# pja2114: Rewritten to initialize all sub-networks based on my implementations
encoder = encoder()
shared_res_block = res_block()
generators = [generator(shared_block=shared_res_block) for _ in range(opt.n_spkrs)]
discriminators = [discriminator() for _ in range(opt.n_spkrs)]

# pja2114: Build models with expected input sizes for each sub-network
encoder.build((opt.batch_size, 128, 128, 1)) # (4, 128, 128, 1)
generators[0].build((opt.batch_size, 32, 32, 128)) # (4, 32, 32, 128)
generators[1].build((opt.batch_size, 32, 32, 128)) # (4, 32, 32, 128)
discriminators[0].build((opt.batch_size, 128, 128, 1)) # (4, 128, 128, 1)
discriminators[1].build((opt.batch_size, 128, 128, 1)) # (4, 128, 128, 1)

# pja2114: Initialize Adam optmizers for all sub-networks using TensforFlow syntax
encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.b1, beta_2=opt.b2)
generator_1_optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.b1, beta_2=opt.b2)
generator_2_optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.b1, beta_2=opt.b2)
discriminator_1_optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.b1, beta_2=opt.b2)
discriminator_2_optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.b1, beta_2=opt.b2)

# Loss weights
lambda_0 = 10   # Adversarial GAN loss
lambda_1 = 0.1  # KL (encoded spect) loss
lambda_2 = 100  # ID pixel-wise loss
lambda_3 = 0.1  # KL (encoded translated spectrogram) loss
lambda_4 = 100  # Cycle pixel-wise loss
lambda_5 = 10   # Latent space L1 loss

# pja2114: Load and prepare data for training; now uses my custom class
# Please see data_proc.py for new implementational details for TensorFlow
dataloader = DataProc(opt, split='train')

# pja2114: Calculate number of training examples
num_train_examples = len(dataloader.data_dict[0]) + len(dataloader.data_dict[1])
input_shape = (num_train_examples, opt.spect_height, opt.spect_width, opt.channels)


# ---------------------------------------------------------
#  Training (local)
# ---------------------------------------------------------

# pja2114: Method largely modified to abide by TensorFlow syntax
@tf.function
def train_local(i, epoch, batch, id_1, id_2, losses):

    # Create plot output directories if doesn't exist already
    if opt.plot_interval != -1:
        os.makedirs("out_train/%s/plot_%dt%d/" % (opt.model_name, id_1, id_2), exist_ok=True)
        os.makedirs("out_train/%s/plot_%dt%d/" % (opt.model_name, id_2, id_1), exist_ok=True)

    # pja2114: Custom training needed vast changes to work in TensorFlow, including using
    # the tf.GradientTape() functionality to keep track of trainable parameters using Adam optimizers
    with tf.GradientTape(persistent=True) as tape_0:

    	# pja2114: Changed syntax; set model input
        X1 = batch[id_1]
        X2 = batch[id_2]

    	# pja2114: Changed syntax; adversarial ground truths
        valid = tf.convert_to_tensor(np.ones((4, 8, 8, 1)), dtype=tf.float32)
        fake = tf.convert_to_tensor(np.zeros((4, 8, 8, 1)), dtype=tf.float32)


    	# -------------------------------
        #  Train Encoder and Generators
        # -------------------------------

    	# Get shared latent representation
        mu1, Z1 = encoder(X1, training=True)
        mu2, Z2 = encoder(X2, training=True)

        # pja2114: Rewritten for TensorFlow syntax; latent space feats
        feat_1 = tf.reduce_mean(tf.reshape(mu1, [tf.shape(mu1)[0], -1]), axis=0)
        feat_2 = tf.reduce_mean(tf.reshape(mu2, [tf.shape(mu2)[0], -1]), axis=0)

    	# Reconstruct speech
        recon_X1 = generators[id_1](Z1, training=True)
        recon_X2 = generators[id_2](Z2, training=True)

    	# Translate speech
        fake_X1 = generators[id_1](Z2, training=True)
        fake_X2 = generators[id_2](Z1, training=True)

        # Cycle translation
        mu1_, Z1_ = encoder(fake_X1, training=True)
        mu2_, Z2_ = encoder(fake_X2, training=True)
        cycle_X1 = generators[id_1](Z2_, training=True)
        cycle_X2 = generators[id_2](Z1_, training=True)

    	# Losses
        loss_GAN_1 = lambda_0 * criterion_GAN(discriminators[id_1](fake_X1, training=True), valid)
        loss_GAN_2 = lambda_0 * criterion_GAN(discriminators[id_2](fake_X2, training=True), valid)
        loss_KL_1 = lambda_1 * compute_kl(mu1)
        loss_KL_2 = lambda_1 * compute_kl(mu2)
        loss_ID_1 = lambda_2 * criterion_pixel(recon_X1, X1)
        loss_ID_2 = lambda_2 * criterion_pixel(recon_X2, X2)
        loss_KL_1_ = lambda_3 * compute_kl(mu1_)
        loss_KL_2_ = lambda_3 * compute_kl(mu2_)
        loss_cyc_1 = lambda_4 * criterion_pixel(cycle_X1, X1)
        loss_cyc_2 = lambda_4 * criterion_pixel(cycle_X2, X2)
        loss_feat = lambda_5 * criterion_pixel(feat_1, feat_2)

    	# Total loss
        loss_G = (
            loss_KL_1
            + loss_KL_2
            + loss_ID_1
            + loss_ID_2
            + loss_GAN_1
            + loss_GAN_2
            + loss_KL_1_
            + loss_KL_2_
            + loss_cyc_1
            + loss_cyc_2
            + loss_feat
            )

    # pja2114: Added following block of code to calculate gradients w.r.t. final loss
    # for all models and update a single step of gradient descent using Adam optimiziers
    encoder_gradients = tape_0.gradient(loss_G, encoder.trainable_variables)
    generator_1_gradients = tape_0.gradient(loss_G, generators[id_1].trainable_variables)
    generator_2_gradients = tape_0.gradient(loss_G, generators[id_2].trainable_variables)

    encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
    generator_1_optimizer.apply_gradients(zip(generator_1_gradients, generators[id_1].trainable_variables))
    generator_2_optimizer.apply_gradients(zip(generator_2_gradients, generators[id_2].trainable_variables))


    # -----------------------
    #  Train Discriminator 1
    # -----------------------

    # pja2114: Custom training needed vast changes to work in TensorFlow, including using
    # the tf.GradientTape() functionality to keep track of trainable parameters using Adam optimizers
    with tf.GradientTape(persistent=True) as tape_1:
        loss_D1 = criterion_GAN(discriminators[id_1](X1, training=True), valid) + criterion_GAN(discriminators[id_1](tf.stop_gradient(fake_X1)), fake)

    # pja2114: Added lines to calculate gradients w.r.t. discriminator loss (only used for training) and take a single step
    discriminator_1_gradients = tape_1.gradient(loss_D1, discriminators[id_1].trainable_variables)
    discriminator_1_optimizer.apply_gradients(zip(discriminator_1_gradients, discriminators[id_1].trainable_variables))


    # -----------------------
    #  Train Discriminator 2
    # -----------------------

    # pja2114: Custom training needed vast changes to work in TensorFlow, including using
    # the tf.GradientTape() functionality to keep track of trainable parameters using Adam optimizers
    with tf.GradientTape(persistent=True) as tape_2:
        loss_D2 = criterion_GAN(discriminators[id_2](X2, training=True), valid) + criterion_GAN(discriminators[id_2](tf.stop_gradient(fake_X2)), fake)

    # pja2114: Added lines to calculate gradients w.r.t. discriminator loss (only used for training) and take a single step
    discriminator_2_gradients = tape_2.gradient(loss_D2, discriminators[id_2].trainable_variables)
    discriminator_2_optimizer.apply_gradients(zip(discriminator_2_gradients, discriminators[id_2].trainable_variables))

    loss_D = loss_D1 + loss_D2

    # --------------
    #  Log Progress
    # --------------

    # Plot first batch every epoch or few epochs
    if opt.plot_interval != -1 and (epoch+1) % opt.plot_interval == 0 and i == 0:
        plot_batch_train(opt.model_name, 'plot_%dt%d'%(id_1, id_2), epoch, X1, cycle_X1, fake_X2, X2)
        plot_batch_train(opt.model_name, 'plot_%dt%d'%(id_2, id_1), epoch, X2, cycle_X2, fake_X1, X1)

    return loss_G, loss_D # pja2114: Dictionary appending cannot occur for @tf.function, so returning losses instead


# ---------------------------------------------------------
#  Training (global)
# ---------------------------------------------------------

# pja2114: Largely rewrote this function to use my new class objects
def train_global():

    # pja2114: Create checkpoints to save progress over epochs in case failure occurs during training
    checkpoint_dir = "training_checkpoints/%s/" % (opt.model_name)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder_optimizer = encoder_optimizer,
                                     generator_1_optimizer = generator_1_optimizer,
                                     generator_2_optimizer = generator_2_optimizer,
                                     discriminator_1_optimizer = discriminator_1_optimizer,
                                     discriminator_2_optimizer = discriminator_2_optimizer,
                                     encoder = encoder,
                                     generator_1 = generators[0],
                                     generator_2 = generators[1],
                                     discriminator_1 = discriminators[0],
                                     discriminator_2 = discriminators[1])

    # pja2114: Needed to rewrite the custom training loop more explicitly
    num_train = len(dataloader)
    num_batch = num_train//opt.batch_size

    for epoch in range(opt.epoch, opt.n_epochs):

        losses = {'G': [], 'D': []}
        progress = tqdm(range(num_batch), desc='', total=num_batch) # pja2114: Altered for new training procedure

        for i in progress:
            batch = dataloader.prepare_batch(batch_size=opt.batch_size) # pja2114: Uses my new method for preparing a batch

            # pja2114: Only one-to-one training is currently supported
            loss_G_spkr0_to_spkr1, loss_D_spkr0_to_spkr1 = train_local(i, epoch, batch, 0, 1, losses)
            loss_G_spkr1_to_spkr0, loss_D_spkr1_to_spkr0 = train_local(i, epoch, batch, 1, 0, losses)

            # pja2114: Rewrote logging of loss functions for current epoch
            losses['G'].append(loss_G_spkr0_to_spkr1)
            losses['G'].append(loss_G_spkr1_to_spkr0)
            losses['D'].append(loss_D_spkr0_to_spkr1)
            losses['D'].append(loss_D_spkr1_to_spkr0)

            progress.set_description("[Epoch %d/%d] [D loss: %f] [G loss: %f] "
			% (epoch, opt.n_epochs, np.mean(losses['D']), np.mean(losses['G'])))

        # pja2114: Save checkpoints and full models, rewritten for TensorFlow model saving
        if opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            encoder.save_weights("saved_h5_weights/%s/encoder_%02d.h5" % (opt.model_name, epoch))
            for n in range(opt.n_spkrs):
                generators[n].save_weights("saved_h5_weights/%s/G%d_%02d.h5" % (opt.model_name, n+1, epoch))
                discriminators[n].save_weights("saved_h5_weights/%s/D%d_%02d.h5" % (opt.model_name, n+1, epoch))


if __name__ == '__main__':
    # pja2114: Check GPU support, modified for Tensorflow syntax
    num_of_gpus_available = len(tf.config.list_physical_devices('GPU'))
    cuda = True if num_of_gpus_available > 0 else False
    print("Number of GPUs Available:", num_of_gpus_available)
    train_global()

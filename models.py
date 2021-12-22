"""
Philip Anastassiou (pja2114)
COMS 6998: Fundamentals of Speech Recognition
Professor Homayoon Beigi
Columbia University
Due: December 19th, 2021

Citation for original authors of VAE-GAN for speech-to-speech style transfer:

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

Implemented in TensorFlow 2.0 by Philip Anastassiou (Github: @phiana)
For original PyTorch implementations, please refer to their repositories
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import models, layers, callbacks
import tensorflow_addons as tfa
from tensorflow.keras.layers import InputSpec
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--show_summaries", type=int, default=0, help="show model summary")


# 2D reflection padding, as required in the original papers, is not natively supported by TensorFlow yet,
# but Github user @mronta has implemented a great solution that does the trick. Their original code:
# https://github.com/mronta/CycleGAN-in-Keras/blob/master/reflection_padding.py
class reflection_padding_2d(models.Model):  # layers.Layer
    def __init__(self, padding):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(reflection_padding_2d, self).__init__()

    def compute_output_shape(self, input_shape):
        shape = (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3]
        )
        return shape

    def call(self, x, mask=None):
        width_pad, height_pad = self.padding
        return tf.pad(
            x,
            [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]],
            'REFLECT'
        )


# Residual block with skip connection
class res_block(models.Model): # layers.Layer
    def __init__(self):
        super(res_block, self).__init__()

        self.reflect_1 = reflection_padding_2d(padding=(1, 1))
        self.conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3))
        self.in_1 = tfa.layers.InstanceNormalization()
        self.relu_1 = tf.keras.layers.ReLU()
        self.reflect_2 = reflection_padding_2d(padding=(1, 1))
        self.conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3))
        self.in_2 = tfa.layers.InstanceNormalization()

    def call(self, x):

        original_x = x # Store original x value for skip connection

        x = self.reflect_1(x) # 1 x 1 reflection padding
        x = self.conv2_1(x) # 3 x 3 convolutional layer
        x = self.in_1(x) # Instance normalization
        x = self.relu_1(x) # ReLU activation
        x = self.reflect_2(x) # 1 x 1 reflection padding
        x = self.conv2_2(x) # 3 x 3 convolutional layer
        out = self.in_2(x) # Instance normalization

        return original_x + out # Add residual connection to result


# Variational universal encoder for all speakers
class encoder(models.Model):
    def __init__(self): # input_shape
        super(encoder, self).__init__()

        self.reflect_1 = reflection_padding_2d(padding=(3,3))
        self.conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7)) # input_shape=input_shape
        self.in_1 = tfa.layers.InstanceNormalization()
        self.leaky_relu_1 = tf.keras.layers.LeakyReLU(0.2)

        self.conv2_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2))
        self.in_2 = tfa.layers.InstanceNormalization()
        self.relu_2 = tf.keras.layers.ReLU()

        self.conv2_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2))
        self.in_3 = tfa.layers.InstanceNormalization()
        self.relu_3 = tf.keras.layers.ReLU()

        self.res_block_1 = res_block()
        self.res_block_2 = res_block()
        self.res_block_3 = res_block()
        self.res_block_4 = res_block() # Authors use three residual blocks in paper, but four in implementation

    def call(self, x):

        padding = [[0, 0], [1, 1], [1, 1], [0, 0]]

        x = self.reflect_1(x) # 3 x 3 reflection padding
        x = self.conv2_1(x) # 7 x 7 convolutional layer
        x = self.in_1(x) # Instance normalization
        x = self.leaky_relu_1(x) # LeakyReLU activation with 0.2 slope

        x = tf.pad(x, padding) # Equivalent to 'padding=1' in PyTorch when padding is "valid" in TensorFlow
        x = self.conv2_2(x) # 4 x 4 convolutional layer with 2 x 2 stride
        x = self.in_2(x) # Instance normalization
        x = self.relu_2(x) # ReLU activation

        x = tf.pad(x, padding) # Equivalent to 'padding=1' in PyTorch when padding is "valid" in TensorFlow
        x = self.conv2_3(x) # 4 x 4 convolutional layer with 2 x 2 stride
        x = self.in_3(x) # Instance normalization
        x = self.relu_3(x) # ReLU activation

        x = self.res_block_1(x) # Residual block with skip connections
        x = self.res_block_2(x) # Residual block with skip connections
        mu = self.res_block_3(x) # Output of final residual block is mu of latent distribution
        # mu = self.res_block_4(x) # To simplify, I will only use three residual blocks, as originally suggested

        z = mu + np.random.normal(0, 1, mu.shape) # VAE reparameterization trick

        return mu, z # Return variational latent code


# Convolutional generator network for each individual speaker in dataset
class generator(models.Model):
    def __init__(self, shared_block=None): # input_shape
        super(generator, self).__init__()

        self.shared_res_block = shared_block # Expects a res_block() object
        self.res_block_2 = res_block()
        self.res_block_3 = res_block()

        self.conv2trans_1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="valid")
        self.in_1 = tfa.layers.InstanceNormalization()
        self.leaky_relu_1 = tf.keras.layers.LeakyReLU(0.2)

        self.conv2trans_2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), padding="valid")
        self.in_2 = tfa.layers.InstanceNormalization()
        self.leaky_relu_2 = tf.keras.layers.LeakyReLU(0.2)

        self.reflect_3 = reflection_padding_2d(padding=(3, 3)) # Omitted in call()
        self.conv2_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(7, 7), padding="valid", activation='tanh') # Sigmoid not required due to MSE instead of BCE in GAN loss

    def call(self, x):

        # padding = [[0, 0], [2, 2], [2, 2], [0, 0]]

        x = self.shared_res_block(x) # Shared generator residual block
        x = self.res_block_2(x) # Speaker-specific residual block with skip connections
        x = self.res_block_3(x) # Speaker-specific residual block with skip connections

        # x = tf.pad(x, padding) # Removing padding=1
        x = self.conv2trans_1(x) # 4 x 4 upsampling convolutional layer with 2 x 2 stride
        x = self.in_1(x) # Instance normalization
        x = self.leaky_relu_1(x) # LeakyReLU activation with -0.2 slope

        # x = tf.pad(x, padding) # Removing padding=1
        x = self.conv2trans_2(x) # 4 x 4 upsampling convolutional layer with 2 x 2 stride
        x = self.in_2(x) # Instance normalization
        x = self.leaky_relu_2(x) # LeakyReLU activation with -0.2 slope

        # x = self.reflect_3(x) # Removing 3 x 3 reflection padding
        out = self.conv2_3(x) # 7 x 7 convolutional layer with sigmoid due to MSE

        # print("Generator output shape:", out.shape)

        return out


# Convolutional discriminator network for each individual speaker in dataset
class discriminator(models.Model):
    def __init__(self): # input_shape
        super(discriminator, self).__init__()

        self.conv2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="valid") # input_shape=input_shape
        self.leaky_relu_1 = tf.keras.layers.LeakyReLU(0.2)

        self.conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="valid")
        self.in_2 = tfa.layers.InstanceNormalization()
        self.leaky_relu_2 = tf.keras.layers.LeakyReLU(0.2)

        self.conv2_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="valid")
        self.in_3 = tfa.layers.InstanceNormalization()
        self.leaky_relu_3 = tf.keras.layers.LeakyReLU(0.2)

        self.conv2_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="valid")
        self.in_4 = tfa.layers.InstanceNormalization()
        self.leaky_relu_4 = tf.keras.layers.LeakyReLU(0.2)

        self.conv2_5 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding="valid") # Paper says to include strides=(2, 2) here, but their implementation does not

    def call(self, x):

        padding = [[0, 0], [1, 1], [1, 1], [0, 0]]

        x = tf.pad(x, padding) # Equivalent to 'padding=1' in PyTorch when padding is "valid" in TensorFlow
        x = self.conv2_1(x) # 4 x 4 convolutional layer with 2 x 2 stride
        x = self.leaky_relu_1(x)  # LeakyReLU activation with 0.2 slope

        x = tf.pad(x, padding) # Equivalent to 'padding=1' in PyTorch when padding is "valid" in TensorFlow
        x = self.conv2_2(x) # 4 x 4 convolutional layer with 2 x 2 stride
        x = self.in_2(x) # Instance normalization
        x = self.leaky_relu_2(x)  # LeakyReLU activation with 0.2 slope

        x = tf.pad(x, padding) # Equivalent to 'padding=1' in PyTorch when padding is "valid" in TensorFlow
        x = self.conv2_3(x) # 4 x 4 convolutional layer with 2 x 2 stride
        x = self.in_3(x) # Instance normalization
        x = self.leaky_relu_3(x)  # LeakyReLU activation with 0.2 slope

        x = tf.pad(x, padding) # Equivalent to 'padding=1' in PyTorch when padding is "valid" in TensorFlow
        x = self.conv2_4(x) # 4 x 4 convolutional layer with 2 x 2 stride
        x = self.in_4(x) # Instance normalization
        x = self.leaky_relu_4(x)  # LeakyReLU activation with 0.2 slope

        x = tf.pad(x, padding) # Equivalent to 'padding=1' in PyTorch when padding is "valid" in TensorFlow
        out = self.conv2_5(x) # 3 x 3 convolutional layer with 2 x 2 stride

        # print("Discriminator output shape:", out.shape)

        return out # Change back to out, Return final prediction


# pja2114: Build residual block
def create_res_block():
    return res_block()


# pja2114: Build universal variational encoder network
def build_encoder(input_shape):
    model = encoder()
    model.build(input_shape)
    # model.compile(optimizer = opt)
    return model


# pja2114: Build generator network
def build_generator(input_shape, shared_block=None):
    if shared_block == None:
        res_block = res_block()
        model = generator(shared_block=res_block)
    else:
        model = generator(shared_block=shared_block)
    model.build(input_shape)
    # model.compile(optimizer = opt)
    return model


# pja2114: Build discriminator network
def build_discriminator(input_shape):
    model = discriminator()
    model.build(input_shape)
    # model.compile(optimizer = opt)
    return model


def main():
    opt = parser.parse_args()
    show_summaries = opt.show_summaries

    # pja2114: Adding testing funtionalities
    if show_summaries:

        res_block_input_shape = (4, 128, 128, 1) # Dummy 4D tuple representing dataset shape
        res_block = create_res_block(res_block_input_shape)
        res_block.summary() # Prove that compilation of model worked

        encoder_input_shape = (4, 128, 128, 1) # Dummy 4D tuple representing dataset shape
        encoder = create_encoder(encoder_input_shape)
        encoder.summary() # Prove that compilation of model worked

        gen_input_shape = (4, 32, 32, 128) # Dummy 4D tuple representing dataset shape
        generator = create_generator(gen_input_shape)
        generator.summary() # Prove that compilation of model worked

        dis_input_shape = (4, 128, 128, 1) # Dummy 4D tuple representing dataset shape
        discriminator = create_discriminator(dis_input_shape)
        discriminator.summary() # Prove that compilation of model worked


if __name__ == '__main__':
    main()

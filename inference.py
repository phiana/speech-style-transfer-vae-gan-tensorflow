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
import glob
import numpy as np
import itertools
import sys
from tqdm import tqdm

from models import *
import tensorflow as tf
from tensorflow import keras
import pickle

import librosa
from utils import ls, preprocess_wav, melspectrogram, to_numpy, plot_mel_transfer_infer, reconstruct_waveform
from params import sample_rate
import soundfile as sf

# import skimage.metrics
# from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=99, help="saved version based on epoch to test from")
parser.add_argument("--model_name", type=str, default="pja2114_voice_conversion", help="name of the model")
parser.add_argument("--trg_id", type=int, help="id of the generator for the target domain")
parser.add_argument("--src_id", type=int, default=None, help="id of the generator for the source domain (Specify for a recon/cyclic evaluation with SSIM)")
parser.add_argument("--wav", type=str, default=None, help="path to wav file for input to transfer")
parser.add_argument("--wavdir", type=str, default=None, help="path to directory of wav files for input to transfer")
parser.add_argument("--plot", type=int, default=1, help="plot the spectrograms before and after (disable with -1)")
parser.add_argument("--n_overlap", type=int, default=4, help="number of overlaps per slice")
parser.add_argument("--spect_height", type=int, default=128, help="size of image height")
parser.add_argument("--spect_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")

opt = parser.parse_args()
print(opt)

assert opt.wav or opt.wavdir, 'Please specify an input wav file or directory'
assert not opt.wav or not opt.wavdir, 'Cannot specify both wav and wavdir, choose one'

# pja2114: Change GPU checking to TensorFlow formatting
num_of_gpus_available = len(tf.config.list_physical_devices('GPU'))
cuda = True if num_of_gpus_available > 0 else False
print("Number of GPUs Available:", num_of_gpus_available)

assert os.path.exists("saved_h5_weights/%s/encoder_%02d.h5" % (opt.model_name, opt.epoch)), 'Check that trained encoder exists'
assert os.path.exists("saved_h5_weights/%s/G%d_%02d.h5" % (opt.model_name, opt.trg_id, opt.epoch)), 'Check that trained generator exists'

# Prepare directories
root = 'out_infer/%s_%d_G%d'% (opt.model_name, opt.epoch, opt.trg_id)

os.makedirs(root+'/gen/', exist_ok=True)
os.makedirs(root+'/ref/', exist_ok=True)

# pja2114: Initialize encoder and decoder
encoder = encoder()
generator_target = generator(shared_block=res_block())

# pja2114: Build models with expected input sizes for each sub-network
encoder.build((4, 128, 128, 1)) # (4, 128, 128, 1)
generator_target.build((4, 32, 32, 128)) # (4, 32, 32, 128)

# pja2114: Change to TensorFlow syntax; Load pretrained models
encoder.load_weights("saved_h5_weights/%s/encoder_%02d.h5" % (opt.model_name, opt.epoch))
generator_target.load_weights("saved_h5_weights/%s/G%d_%02d.h5" % (opt.model_name, opt.trg_id, opt.epoch))


# ----------------------------------------------------
#  SSIM Computation (Evaluating reconstruction)
# ----------------------------------------------------

def ssim(spect_src, spect_recon):
    return skimage.metrics.structural_similarity(spect_src, spect_recon, data_range=1)


# -----------------------------------------
#  Local Inference and SSIM evaluation
# -----------------------------------------

def infer(S):
    # Takes in a standard sized spectrogram, returns timbre converted version

    # pja2114: Modified for TensorFlow syntax
    S = tf.convert_to_tensor(S)
    X = tf.reshape(S, [1, opt.spect_height, opt.spect_height, 1])
    ret = {} # just stores inference output

    mu, Z = encoder(X)
    fake_X = generator_target(Z)
    ret['fake'] = to_numpy(fake_X)

    return ret

# ------------------------------------------
#  Global Inference (w/ a sliding window)
# ------------------------------------------

def audio_infer(wav):

    # Load audio and preprocess
    sample = preprocess_wav(wav)
    spect_src = melspectrogram(sample)

    spect_src = np.pad(spect_src, ((0,0),(opt.spect_width,opt.spect_width)), 'constant')  # padding for consistent overlap
    spect_trg = np.zeros(spect_src.shape)
    spect_recon = np.zeros(spect_src.shape)
    spect_cyclic = np.zeros(spect_src.shape)

    length = spect_src.shape[1]
    hop = opt.spect_width // opt.n_overlap

    for i in tqdm(range(0, length, hop)):
        x = i + opt.spect_width

        # Get cropped spectro of right dims
        if x <= length:
            S = spect_src[:, i:x]
        else:  # pad sub on right if includes sub segments out of range
            S = spect_src[:, i:]
            S = np.pad(S, ((0,0),(x-length,0)), 'constant')

        ret = infer(S) # perform inference from trained model
        T = ret['fake']

        # Add parts of target spectrogram with an average across overlapping segments
        for j in range(0, opt.spect_width, hop):
            y = j + hop
            if i + y > length: break  # neglect sub segments out of range

            # Select subsegments to consider for overlap
            t = T[:, j:y]

            # Add average element
            spect_trg[:, i+j:i+y] += t/opt.n_overlap

    # Remove initial padding
    spect_src = spect_src[:, opt.spect_width:-opt.spect_width]
    spect_trg = spect_trg[:, opt.spect_width:-opt.spect_width]

    # Prepare file name for saving
    f = wav.split('/')[-1]
    wavname = f.split('.')[0]
    fname = 'G%s_%s' % (opt.trg_id, wavname)

    # Plot transfer if specified
    if opt.plot != -1:
        os.makedirs(root+'/plots/', exist_ok=True)
        plot_mel_transfer_infer(root+'/plots/%s.png' % fname, spect_src, spect_trg)

    # Reconstruct with Griffin Lim (takes a while; optionally: later feed this .wav as input to vocoder)
    print('Reconstructing with Griffin Lim...')
    x = reconstruct_waveform(spect_trg)

    sf.write(root+'/gen/%s_gen.wav'%fname, x, sample_rate)  # generated output
    sf.write(root+'/ref/%s_ref.wav'%fname, sample, sample_rate)  # input reference (for convenience)


if opt.wav:
    audio_infer(opt.wav)

if opt.wavdir:
    audio_files = glob.glob(os.path.join(opt.wavdir, '*.wav'))
    for i, wav in enumerate(audio_files):
        print('[File %d/%d]' % (i+1, len(audio_files)))
        audio_infer(wav)

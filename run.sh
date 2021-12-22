#!/bin/bash
#
# Philip Anastassiou (pja2114)
# COMS 6998: Fundamentals of Speech Recognition
# Professor Homayoon Beigi
# Columbia University
# Due: December 19th, 2021
#
# Citation for original authors of VAE-GAN for speech-to-speech style transfer:
#
# @misc{sammutbonnici2021timbre,
#       title={Timbre Transfer with Variational Auto Encoding and Cycle-Consistent Adversarial Networks},
#       author={Russell Sammut Bonnici and Charalampos Saitis and Martin Benning},
#       year={2021},
#       eprint={2109.02096},
#       archivePrefix={arXiv},
#       primaryClass={cs.SD}
# }
#
# @inproceedings{AlBadawy2020,
#   author={Ehab A. AlBadawy and Siwei Lyu},
#   title={{Voice Conversion Using Speech-to-Speech Neuro-Style Transfer}},
#   year=2020,
#   booktitle={Proc. Interspeech 2020},
#   pages={4726--4730},
#   doi={10.21437/Interspeech.2020-3056},
#   url={http://dx.doi.org/10.21437/Interspeech.2020-3056}
# }
#
# Implemented in TensorFlow 2.0 by Philip Anastassiou (Github: @phiana)
# For original PyTorch implementations, please refer to their repositories
# Bash script written by Philip Anastassiou

echo "A bash script to automatically train a one-to-one VAE-GAN model for speech-to-speech style transfer using TensorFlow"

stage=0

if [ $stage -le 0 ]; then
  echo "Installing pip (assuming you are in a conda environment)..."
  conda install pip
  echo "Done"
fi

if [ $stage -le 1 ]; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
  echo "Done"
fi

if [ $stage -le 2 ]; then
  echo "Downloading dataset..."
  apt-get install wget
  wget https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads/flickr_audio.tar.gz
  echo "Done"
fi

if [ $stage -le 3 ]; then
  echo "Decompressing .tar.gz file..."
  tar -xf flickr_audio.tar.gz
  echo "Done"
fi

if [ $stage -le 4 ]; then
  echo "Moving Flickr corpus to data/ directory..."
  mkdir data
  mv flickr_audio data
  echo "Done"
fi

if [ $stage -le 5 ]; then
  echo "Preparing .wav files for one-to-one training..."
  python3 flickr.py --dataroot "data/flickr_audio/" --outdir "data/"
  echo "Done"
fi

if [ $stage -le 6 ]; then
  echo "Removing original .tar.gz file..."
  rm flickr_audio.tar.gz
  echo "Done"
fi

if [ $stage -le 7 ]; then
  echo "Preprocessing audio files for both speakers..."
  python3 preprocess.py
  echo "Done"
fi

if [ $stage -le 8 ]; then
  echo "Training model from scratch (this may take a while)..."
  python3 train.py
  echo "Done"
fi

if [ $stage -le 9 ]; then
  echo "Running the inference pipeline (perform voice conversion and output result)..."
  python3 inference.py --epoch 15 --wav 'data/spkr_1/2274602044_b3d55df235_3.wav' --trg_id 2
  echo "Done"
fi

echo "Training and inference pipeline complete"

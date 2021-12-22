"""
Philip Anastassiou (pja2114)
COMS 6998: Fundamentals of Speech Recognition
Professor Homayoon Beigi
Columbia University
Due: December 19th, 2021

Citations for original authors of this file:

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
"""

# Originally written by Bonnici, et al. (2021) from the following Github repository:
# https://github.com/RussellSB/tt-vae-gan/blob/main/data_prep/flickr.py

from tqdm import tqdm
import numpy as np
import argparse
import shutil
import os

parser = argparse.ArgumentParser()

# pja2114: Default paths now point to external SSD containing datasets, replace with your own local path using '--dataroot' option
parser.add_argument("--dataroot", type=str, help="data root of flickr")
parser.add_argument("--outdir", type=str, help="output directory")
args = parser.parse_args()
print(args)

# This just gets the filenames with speaker referenes (because it is not clear in the filename otherwise)
textin = args.dataroot + 'wav2spk.txt'
speaker_files = np.genfromtxt(textin, dtype=[('mystring','S27'),('myint','i8')], delimiter=' ')

# Saves wavs belonging to speaker from list of speaker files
def flickr_prep_wavs(outdir, speaker_files, src):
    os.makedirs(outdir, exist_ok=True)

    files = [filename for (filename, spk) in speaker_files if spk == src]
    for filename in tqdm(files, desc="extracting spk %s"%src):
        f = args.dataroot + 'wavs/' + filename.decode()
        shutil.copy(f, outdir)

# Data preparation
flickr_prep_wavs(args.outdir+'spkr_1', speaker_files, 4)
flickr_prep_wavs(args.outdir+'spkr_2', speaker_files, 7)

# pja2114: Removed many-to-many data preparation procedures that were formerly included below

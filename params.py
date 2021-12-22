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

# pja2114: Please note that many of these parameters refer to methods beyond
# the scope of my final project, but for the sake of completeness, I have left
# these values largely untouched

# --- Audio ---
sample_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160 # 1600 ms
# Number of spectrogram frames at inference
inference_n_frames = 80 #  800 ms

# --- Mel-filterbank ---
n_fft = 2048
num_mels = 128
num_samples = 128 # input spect shape num_mels * num_samples
hop_length = int(0.0125*sample_rate) # 12.5ms - in line with Tacotron 2 paper
win_length = int(0.05*sample_rate) # 50ms - same reason as above
fmin = 40
min_level_db = -100
ref_level_db = 20
bits = 9 # Bit depth of signal
mu_law = True # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False

# --- Voice Activation Detection ---
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out.
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 16


# --- Audio volume normalization ---
audio_norm_target_dBFS = -30

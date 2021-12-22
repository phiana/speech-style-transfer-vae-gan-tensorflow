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

This is now a self-contained class that prepares a dictionary of random
samples of log-scaled mel-spectrograms of both speakers. Previously, this
class object was passed into a 'torch.utils.data.DataLoader' object. Being a
TensorFlow implementation, I opted to remove the PyTorch infrastructure and
re-implemented this class to contain all necessary functionality from the
DataLoader object. In the future, 'tf.data' should be used instead.
"""

import numpy as np
import pickle
import random
import tensorflow as tf

from params import num_samples

class DataProc():

    def __init__(self, args, split):
        self.args = args
        self.data_dict = pickle.load(open('%s/data_%s.pickle'%(args.dataset, split),'rb'))
        self.n_spkrs = len(self.data_dict.keys()) # pja2114: Added class attribute


    def __len__(self):
        total_len = 0
        for i in range(len(self.data_dict.keys())):
            tmp = np.sum([j.shape[1] for j in self.data_dict[i]])
            total_len = max(total_len,tmp / 128)
        return int(total_len)


    def __getitem__(self):
        rslt = []

        for i in range(0, self.n_spkrs):
            # Chose random item based on prop distribution (length of each sample)
            tmp_lens = [j.shape[1] for j in self.data_dict[i]]
            item = np.random.choice(len(tmp_lens), p=tmp_lens / np.sum(tmp_lens))
            rslt.append(self.random_sample(i, item))

        # Prepares a random sample per speaker
        samples = {}
        for i in range(0, self.n_spkrs):
            samples[i] = rslt[i] # pja2114: Updated syntax; previous formatting: "samples[i] = np.array(rslt)[i, :]"
        return samples


    # pja2114: Modified to work with TensorFlow Tensor objects (channels-last formatting needed)
    def random_sample(self, i, item):
        n_samples = num_samples # 128
        data = self.data_dict[i][item]
        assert data.shape[1] >= n_samples
        rand_i = random.randint(0,data.shape[1]-n_samples)
        data = np.array(data[:, rand_i:rand_i+n_samples])
        data = data.reshape((data.shape[0], data.shape[1], 1)).astype('float32')
        return data


    # pja2114: A new method I wrote to prepare a batch of random samples for training
    def prepare_batch(self, batch_size):
        batch = {}
        for i in range(batch_size):
            result = self.__getitem__()
            for spkr in range(0, self.n_spkrs):
                if i == 0:
                    batch[spkr] = []
                batch[spkr].append(result[spkr])
        for spkr in range(0, self.n_spkrs):
            batch[spkr] = np.array(batch[spkr])
            batch[spkr] = tf.convert_to_tensor(batch[spkr], dtype=tf.float32)
        return batch


    # Not currently used, but feel free to add for additional robustness in final model
    def augment(self,data,sample_rate=16000, pitch_shift=0.5):
        if pitch_shift == 0 : return data
        return librosa.effects.pitch_shift(data, sample_rate, pitch_shift)

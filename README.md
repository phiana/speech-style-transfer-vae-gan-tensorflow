# Speech-to-Speech Style Transfer with Variational Autoencoder-Generative Adversarial Networks

Neural audio synthesis is the application of deep neural networks to synthesize audio waveforms in a data-driven fashion. The advent of deep learning, coupled with increasingly powerful computational resources, has allowed researchers to train models for this task both on musical and speech signals directly on raw audio waveforms (or related representations in the [time-frequency domain](https://en.wikipedia.org/wiki/Time-frequency_analysis) by applying variants of [integral transforms](https://en.wikipedia.org/wiki/Integral_transform) such as the [Short-Time Fourier Transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)), as opposed to symbolic representations of audio.

This repository contains an implementation of a variational autoencoder-generative adversarial network (VAE-GAN) architecture for speech-to-speech [style transfer](https://en.wikipedia.org/wiki/Neural_Style_Transfer) in [TensorFlow](https://www.tensorflow.org/), originally proposed for voice conversion in _[Voice Conversion Using Speech-to-Speech Neuro-Style Transfer](https://ebadawy.github.io/post/speech_style_transfer/Albadawy_et_al-2020-INTERSPEECH.pdf)_ by AlBadawy, et al. (2020) and expanded upon to include functionality for timbre transfer of musical instruments by _[Timbre Transfer with Variational Auto Encoding and Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/2109.02096.pdf)_ by Bonnici, et al. (2021).

The purpose of the model is to synthesize a new audio signal that retains the linguistic information of an utterance by a source speaker, while applying the timbral characteristics of a target speaker. The architecture does so by training a single universal encoder, but replaces the decoder used in traditional variational autoencoders (VAEs) with unique generator networks for every speaker, each of which competes against adversary discriminator networks used only during the training procedure, enabling style transfer.

Please refer to the original repositories by [AlBadawy, et al.](https://github.com/ebadawy/voice_conversion) or [Bonnici, et al.](https://github.com/RussellSB/tt-vae-gan) for PyTorch implementations of this architecture, the latter of which contains [pre-trained WaveNet vocoder models](https://github.com/RussellSB/tt-vae-gan#pretrained-models) to convert predicted log-scaled mel-spectrograms back into high-fidelity audio signals in the time-domain.

## Training a model

In order to train your own model from scratch using my implementation, please follow these steps:

#### Clone the repository to your machine

Run the following command in your termnal:

```
git clone https://github.com/phiana/speech-style-transfer-vae-gan-tensorflow
```

#### Install dependencies

You can ensure that all required libraries/frameworks are installed in your Python environment by running the following command in your terminal:

```
pip install -r requirements.txt
```

Optionally, you can skip running this command directly, as it is included in the training recipe described below.

#### Run the training recipe

To automatically run the entire training and inference pipeline, including downloading the [Flickr 8k Audio Caption Corpus](https://groups.csail.mit.edu/sls/downloads/flickraudio/index.cgi) (requires at least 4.2 GB of space as a gzip'd tar file), run the provided bash script `run.sh`:

```
./run.sh
```

This will begin the training procedure using default settings (currently only one-to-one training is supported, but many-to-many will be added in the future). 

The script is divided into numbered stages, so you can stop and restart at a particular stage of the pipeline by manually changing the value of the `stage` variable at the top of the file.

Please note that the script assumes you are working within a `conda` virtual environment. The first stage installs `pip` using the `conda` command to ensure that the following installations will run without interruption. However, if you already have `pip` installed in your environment of choice and would prefer not to work in a `conda` environment, you can simply set the `stage` variable from `0` to `1` in `run.sh` prior to running it, which will skip this first step and begin by checking if the dependencies are installed in your environment (which, again, assumes your machine has `pip` installed). In case you do want to work in a `conda` environment, I suggest running `conda update conda` and `conda upgrade pip` before running any other files to ensure that the script will run smoothly. 

After running the script, you will find a directory `data/`, which contains the raw Flickr 8k Audio Caption Corpus used for training a model in the child directory `flickr_audio/`, as well as two prepared directories produced by the `run.sh` recipe (`spkr_1/` and `spkr_2/` containing `.wav` files for two speakers used in one-to-one training). You will also see the preprocessed partitioned datasets for these two speakers (training, test, and validation sets) serialized into `.pickle` files. 

#### Performing inference using a trained model

After training is complete, a directory `out_train/` will be created, in which you will find two additional directories, `plot_0t1` and `plot_1t0`. In the spirit of [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf), this model makes use of a cyclic consistency loss, which encourages the generator networks not to fall into the trap of [mode collapse](https://developers.google.com/machine-learning/gan/problems#mode-collapse) by repeatedly producing a single, realistic output that is guaranteed to deceive the discriminators (this is essentially cheating in the adversarial training process and leads to poor latent embeddings). This outcome is enforced by attempting to reconstruct the original input mel-spectrogram using the generated, style-transferred output in the opposite direction of the pipeline.

The `.png` files contained in these sub-directories are visual representations of the input log-scaled mel-spectrogram (one sample from the current batch) alongside the predicted style-transferred output (top row), and the cyclic reconstruction of the first output alongside the target spectrogram (bottom row).

Finally, to listen to audio samples of the voice conversion inference process produced by a model, please see the directory `out_infer/`. In the sub-directory `ref/`, there will be a `.wav` file of the original, true recording of an utterance by the first speaker. In the sub-directory `gen/`, you may listen to the styled-transferred output that the model produces of the same utterance, as spoken by the second speaker, only using the input audio file as reference. In the sub-directory `plots/`, you will find a side-by-side visual comparison of the original input mel-spectrogram and the style-transferred output mel-spectrogram. 

To actually run the inference pipeline yourself, there are two approaches. First, you may change the `stage` variable in `run.sh` to `9`, and then run the script, which will perform voice conversion on a single audio file as an example, and store the results in `out_infer/`, as explained above. 

Alternatively, you can run the following command on the command line without running the `run.sh` script (the parameters below are simply to demonstrate the syntax of the inference command, but may be changed as needed):

```
python3 inference.py --epoch 15 --wav 'data/spkr_1/2274602044_b3d55df235_3.wav' --trg_id 2
```

## Future work

* Replace basic [Griffin-Lim-based vocoder](https://paperswithcode.com/method/griffin-lim-algorithm) for iterative phase reconstruction and modern [neural WaveNet vocoder](https://arxiv.org/pdf/1609.03499.pdf) with more recent, non-autoregressive [HiFi-GAN vocoder](https://arxiv.org/pdf/2010.05646.pdf) as final step in the inference pipeline
* Train on other datasets, such as the [LibriSpeech ASR Corpus](https://www.openslr.org/12/) or the [TED-LIUM 3 Speaker Adaptation Corpus](https://www.openslr.org/51/), to evaluate architecture's robustness
* Train architecture for musical timbre transfer using [Google Magenta](https://magenta.tensorflow.org/)'s 2017 [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth), or similar datasets
* Re-train model omitting [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) term in the cyclic consistency loss formulation, as suggested by Bonnici, et al.
* Add many-to-many training functionality (currently hard coded as one-to-one)
* Add learning rate scheduling to automatically modify learning rate of Adam optimizers across various epochs of training

## References, related work, and additional reading

1. Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. WaveNet: A generative model for raw audio. arXiv, 2016. URL https://arxiv.org/pdf/1609.03499.pdf.  
1. adityajn105. Flickr8k Audio Caption Dataset. Kaggle, 2020. URL https://www.kaggle.com/adityajn105/flickr8k.  
1. Ben Hayes, Charalampos Saitis, and George Fazekas. Neural waveshaping synthesis. arXiv, 2021. URL https://arxiv.org/pdf/2107.05050.pdf.  
1. Berrak Sisman, Junichi Yamagishi, Simon King, and Haizhou Li. An Overview of Voice Conversion and Its Challenges: From Statistical Modeling to Deep Learning. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2021. URL https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=92620 21.  
1. Daniel Griffin and Jae Lim. Signal estimation from modified short-time fourier transform. IEEE Transactions on Acoustics, Speech, and Signal Processing, 32(2):236–243, 1984.  
1. Diederik P Kingma and Max Welling. Auto-encoding variational bayes. Stat, 1050:1, 2014. URL https://arxiv.org/pdf/1312.6114.pdf.  
1. Ehab A. AlBadawy and Siwei Lyu. Voice Conversion Using Speech-to-Speech Neuro-Style Transfer. In Proc. Interspeech 2020, pages 4726–4730, 2020. URL https://ebadawy.github.io/post/speech_style_transfer/Albadawy_et_al-2020-INTERSPEECH.pdf.  
1. Eric Grinstein, Ngoc Duong, Alexey Ozerov, and Patrick Pérez. Audio style transfer. ICASSP 2018. URL https://hal.archives-ouvertes.fr/hal-01626389/document.  
1. Fadi Biadsy, Ron J. Weiss, Pedro J. Moreno, Dimitri Kanevsky, and Ye Jia. Parrotron: An End-to-End Speech-to- Speech Conversion Model and its Applications to Hearing- Impaired Speech and Speech Separation. Interspeech, 2019. URL https://arxiv.org/pdf/1904.04169.pdf.  
1. François Hernandez, Vincent Nguyen, Sahar Ghannay, Natalia Tomashenko, and Yannick Estève. TED-LIUM 3: twice as much data and corpus repartition for experiments on speaker adaptation. SPECOM, 2018. URL https://arxiv.org/pdf/1805.04699.pdf.  
1. Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Stat, 1050:10, 2014. URL https://arxiv.org/pdf/1406.2661.pdf.  
1. Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck, Karen Simonyan, and Mohammad Norouzi. Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders. arXiv, 2017. URL https://arxiv.org/abs/1704.01279.  
1. Jesse Engel, Kumar Krishna Agrawal, Shuo Chen, Ishaan Gulrajani, Chris Donahue, and Adam Roberts. GANSynth: Adversarial Neural Audio Synthesis. ICLR, 2019. URL https://openreview.net/pdf?id=H1xQVn09FX.  
1. Jesse Engel, Lamtharn (Hanoi) Hantrakul, Chenjie Gu, and Adam Roberts. DDSP: Differentiable Digital Signal Processing. ICLR, 2020. URL https://openreview.net/attachment?id=B1x1ma4tDr&name=original_pdf.  
1. Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, and Yonghui Wu. Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions. ICASSP, 2018. URL https://arxiv.org/pdf/1712.05884.pdf.  
1. Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV, 2017. URL https://arxiv.org/pdf/1703.10593.pdf. 
1. Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis. NeurIPS, 2020. URL https://arxiv.org/pdf/2010.05646.pdf.  
1. Kaizhi Qian, Yang Zhang, Shiyu Chang, Xuesong Yang, and Mark Hasegawa-Johnson. AutoVC: Zero-Shot Voice Style
Transfer with Only Autoencoder Loss. ICML, 2019. URL https://arxiv.org/pdf/1905.05879.pdf.  
1. Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. A Neural Algorithm of Artistic Style. arXiv, 2015. URL https://arxiv.org/pdf/1508.06576.pdf.  
1. Philippe Esling, Axel Chemla–Romeu-Santos, and Adrien Bitton. Bridging audio analysis, perception and synthesis with perceptually-regularized variational timbre spaces. ISMIR, 2018. URL http://ismir2018.ircam.fr/doc/pdfs/219_Paper.pdf.  
1. Russell Sammut Bonnici, Charalampos Saitis, and Martin Benning. Timbre Transfer with Variational Auto Encoding and Cycle-Consistent Adversarial Networks. arXiv, 2021. URL https://arxiv.org/pdf/2109.02096.pdf.  
1. Ryan Prenger, Rafael Valle, and Bryan Catanzaro. WaveGlow: A Flow-based Generative Network for Speech Synthesis. arXiv, 2018. URL https://arxiv.org/pdf/1811.00002.pdf.  
1. Sercan Ö. Arık, Jitong Chen, Kainan Peng, Wei Ping, and Yanqi Zhou. Neural Voice Cloning with a Few Samples. NIPS, 2018. URL https://arxiv.org/pdf/1802.06006.pdf.  
1. Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv, 2015. URL https://arxiv.org/pdf/1502.03167.pdf.  
1. Siyang Yuan, Pengyu Cheng, Ruiyi Zhang, Weituo Hao, Zhe Gan, and Lawrence Carin. Improving Zero-shot Voice Style Transfer via Disentangled Representation Learning. ICLR, 2021. URL https://arxiv.org/pdf/2103.09420.pdf.  
1. Soroush Mehri, Kundan Kumar, Ishaan Gulrajani, Rithesh Kumar, Shubham Jain, Jose Sotelo, Aaron Courville, and Yoshua Bengio. SampleRNN: An Unconditional End-to-End Neural Audio Generation Model. ICLR, 2017.  
URL https://arxiv.org/pdf/1612.07837.pdf.  
1. Stanislav Beliaev, Yurii Rebryk, and Boris Ginsburg. TalkNet: Fully-Convolutional Non-Autoregressive Speech Synthesis Model. arXiv, 2020. URL https://arxiv.org/pdf/2005.05514.pdf.  
1. Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur. LibriSpeech: An ASR corpus based on public domain audio books. ICASSP, 2015. URL http://www.danielpovey.com/files/2015_icassp_librispeech.pdf.  
1. Yi Ren, Jinglin Liu, and Zhou Zhao. PortaSpeech: Portable and High-Quality Generative Text-to-Speech. arXiv, 2021. URL https://arxiv.org/pdf/2109.15166.pdf.
1. Yuxuan Wang, RJ Skerry-Ryan, Daisy Stanton, Yonghui Wu, Ron J. Weiss, Navdeep Jaitly, Zongheng Yang, Ying Xiao, Zhifeng Chen, Samy Bengio, Quoc Le, Yannis Agiomyrgiannakis, Rob Clark, and Rif A. Saurous. TacoTron: Towards end-to-end speech synthesis. arXiv, 2017. URL https://arxiv.org/pdf/1703.10135.pdf.

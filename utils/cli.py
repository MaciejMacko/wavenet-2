import argparse
import os
import numpy as np
import tensorflow as tf
from tensorpack.utils.logger import set_logger_dir

from utils.misc import date_str


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_path',
                      help='Path to train csv.',
                      default='./data/ljspeech-processed/train.txt')
  parser.add_argument('--test_path',
                      help='Path to test csv.',
                      default='./data/ljspeech-processed/test.txt')
  parser.add_argument('--save_dir',
                      help='Directory to save logs and model checkpoints',
                      default=os.path.join('save', 'wavenet', date_str()))
  parser.add_argument('--load_path',
                      help='Path of the model checkpoint to load')
  parser.add_argument('--summary_freq',
                      help='Frequency (in train steps) with which to write tensorboard summaries',
                      default=20,
                      type=int)
  parser.add_argument('--steps_per_epoch',
                      help='Steps per epoch, defaults to the batch size',
                      default=None,
                      type=int)
  parser.add_argument('--skip_inferencing',
                      help='Whether or not to skip inferencing after epochs',
                      action='store_true')
  parser.add_argument('--gpu',
                      help='Which GPU to use')
  parser.add_argument('--n_threads',
                      help='The number of threads to read and process data',
                      default=2,
                      type=int)
  parser.add_argument('--resume_lr',
                      help='Resume the learning rate from the loaded run',
                      action='store_true')

  args = parser.parse_args()

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  set_logger_dir(args.save_dir)

  return args


hp = tf.contrib.training.HParams(
  ##
  # Audio:
  ##

  n_mel_bins=80,
  frame_length_ms=50,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,
  silence_threshold=2,
  fft_size=1024,
  hop_size=256,  # shift can be specified by either hop_size or frame_shift_ms
  frame_shift_ms=None,
  rescaling=True,
  rescaling_max=0.999,
  log_scale_min=float(np.log(1e-14)),  # Mixture of logistic distributions:

  # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
  # happen depends on min_level_db and ref_level_db, causing clipping noise.
  # If False, assertion is added to ensure no clipping happens.o0
  allow_clipping_in_normalization=True,

  # ljspeech
  gc_enable=False,
  global_channel=None,
  global_cardinality=-1,  # speaker num
  sample_rate=22050,

  # cmu_arctic
  # gc_enable=True,
  # global_channel=16,
  # global_cardinality=7,  # speaker num
  # sample_rate=16000,  # LJSpeech


  ##
  # Model:
  ##

  # This should equal to `quantize_channels` if mu-law quantize enabled
  # otherwise num_mixture * 3 (pi, mean, log_scale)
  out_channels=10 * 3,
  filter_width=3,
  initial_filter_width=1,
  layers=24,
  stacks=4,
  residual_channels=512,
  dilation_channels=512,
  skip_channels=256,
  input_type="raw",
  quantize_channels={
      "raw": 2**16,
      "mu-raw": 2**8
  },
  use_biases=True,
  scalar_input=True,
  upsample_conditional_features=True,
  upsample_factor=[4, 4, 4, 4],
  l2_regularization_strength=None,


  ##
  # Training:
  ##

  batch_size=2,
  epochs=100000,

  sample_size=15000,
  clip_thresh=-1,

  dropout_rate=0.5,

  adam_b1=0.9,
  adam_b2=0.9999,
  adam_eps=1e-6,

  lr=1e-4,  # The initial learning rate.
  lr_min=1e-6,  # The minimum learning rate. (Stop decaying once this is reached.)
  lr_decay_factor=0.5,  # The rate of decay
  lr_decay_ratio=0.3
)


def hp_debug_string():
    values = hp.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)

import argparse
import csv
import os
from multiprocessing import cpu_count

import numpy as np
from tqdm import tqdm
from utils.cli import hp

from utils.data import ljspeech
from utils.audio import get_frame_shift_ms


def preprocess_ljspeech(args):
  os.makedirs(args.out_dir, exist_ok=True)
  metadata = ljspeech.build_from_path(args.in_dir, args.out_dir, hp.silence_threshold, hp.fft_size, args.n_workers, tqdm=tqdm)
  write_metadata(metadata, args.out_dir, args.test_frac)


def write_metadata(metadata, out_dir, test_frac):
  np.random.shuffle(metadata)

  test_i = int(len(metadata) * test_frac)
  test_data = metadata[:test_i]
  train_data = metadata[test_i:]

  with open(os.path.join(out_dir, 'test.txt'), 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(test_data)

  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(train_data)


  frame_shift_ms = get_frame_shift_ms()
  frames_total = sum([row[3] for row in metadata])
  hours_total = frames_total * frame_shift_ms / (3600 * 1000)
  frames_train = sum([row[3] for row in train_data])
  hours_train = frames_train * frame_shift_ms / (3600 * 1000)
  frames_test = sum([row[3] for row in test_data])
  hours_test = frames_test * frame_shift_ms / (3600 * 1000)
  print('Wrote %d total utterances, %d frames (%.2f hours)' % (len(metadata), frames_total, hours_total))
  print('Wrote %d train utterances, %d frames (%.2f hours)' % (len(train_data), frames_train, hours_train))
  print('Wrote %d test utterances, %d frames (%.2f hours)' % (len(test_data), frames_test, hours_test))
  print('Max input length:  %d' % max(len(m[2]) for m in metadata))
  print('Max output length: %d' % max(m[3] for m in metadata))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--in_dir', default='data/ljspeech')
  parser.add_argument('--out_dir', default='data/ljspeech-processed')
  parser.add_argument('--output', default='training')
  parser.add_argument('--dataset', default='ljspeech', choices=['ljspeech'])
  parser.add_argument('--n_workers', type=int, default=cpu_count())
  parser.add_argument('--test_frac', type=float, default=0.2)
  args = parser.parse_args()
  if args.dataset == 'ljspeech':
    preprocess_ljspeech(args)


if __name__ == "__main__":
  main()

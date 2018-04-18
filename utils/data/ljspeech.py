import csv
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from nnmnkwii import preprocessing as P

import utils.audio as audio
from utils.cli import hp


def build_from_path(in_dir, out_dir, silence_threshold, fft_size, n_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        n_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=n_workers)
    futures = []
    index = 1

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        csv.register_dialect('pipe-delim', delimiter='|', quoting=csv.QUOTE_NONE)
        reader = csv.reader(f, dialect='pipe-delim')

        for row in reader:
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % row[0])
            text = row[2]
            futures.append(
                executor.submit(partial(_process_utterance, out_dir, index, wav_path, text,
                                        silence_threshold, fft_size)))
            index += 1

    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text, silence_threshold, fft_size):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, text, mel_len) tuple to write to train.txt
    '''
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    if hp.rescaling:
        wav = wav / np.abs(wav).max() * hp.rescaling_max

    if hp.input_type != "raw":
        # Mu-law quantize
        out = P.mulaw_quantize(wav)

        # Trim silences
        start, end = audio.start_and_end_indices(out, silence_threshold)
        out = out[start:end]
        wav = wav[start:end]
        constant_value = P.mulaw_quantize(0, 256)
        out_dtype = np.int16
    else:
        out = wav
        constant_value = 0.
        out_dtype = np.float32

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T

    # lws pads zeros internally before performing stft
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.lws_pad_lr(wav, fft_size, audio.get_hop_size())

    # zero pad for quantized signal
    out = np.pad(out, (l, r), mode="constant", constant_values=constant_value)
    mel_len = mel_spectrogram.shape[0]
    assert len(out) >= mel_len * audio.get_hop_size()

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:mel_len * audio.get_hop_size()]
    assert len(out) % audio.get_hop_size() == 0

    timesteps = len(out)

    wav_id = wav_path.split('/')[-1].split('.')[0]
    # Write the spectrograms to disk:
    audio_path = os.path.join(out_dir, '{}-audio.npy'.format(wav_id))
    mel_path = os.path.join(out_dir, '{}-mel.npy'.format(wav_id))
    np.save(audio_path, out.astype(out_dtype), allow_pickle=False)
    np.save(mel_path, mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return os.path.abspath(audio_path), os.path.abspath(mel_path), text, timesteps

    # # Compute a mel-scale spectrogram from the trimmed wav:
    # # (N, D)
    # mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    # # lws pads zeros internally before performing stft
    # # this is needed to adjust time resolution between audio and mel-spectrogram
    # l, r = audio.lws_pad_lr(wav, fft_size, audio.get_hop_size())
    #
    # # zero pad for quantized signal
    # out = np.pad(out, (l, r), mode="constant", constant_values=constant_value)
    # N = mel_spectrogram.shape[0]
    # assert len(out) >= N * audio.get_hop_size()
    #
    # # time resolution adjustment
    # # ensure length of raw audio is multiple of hop_size so that we can use
    # # transposed convolution to upsample
    # out = out[:N * audio.get_hop_size()]
    # assert len(out) % audio.get_hop_size() == 0
    #
    # timesteps = len(out)
    #
    # wav_id = wav_path.split('/')[-1].split('.')[0]
    # # Write the spectrograms to disk:
    # audio_filename = '{}-audio.npy'.format(wav_id)
    # mel_filename = '{}-mel.npy'.format(wav_id)
    # np.save(os.path.join(out_dir, audio_filename),
    #         out.astype(out_dtype), allow_pickle=False)
    # np.save(os.path.join(out_dir, mel_filename),
    #         mel_spectrogram.astype(np.float32), allow_pickle=False)
    #
    # # Return a tuple describing this training example:
    # return audio_filename, mel_filename, timesteps, text

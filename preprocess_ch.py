# -*- coding:utf-8 -*-

import glob
import os
import librosa
import csv
import data
import re
import scikits.audiolab
import numpy as np
from data_ch import createWordMapFromFile, str2index, str2phoneindex


# data path
_data_path = "asset/data/"


if __name__ == "__main__":
  # createWordMapFromFile()
  wavs = glob.glob(_data_path + "thchs30/data_thchs30/train/*.wav")

  writer = csv.writer(open(_data_path + "thchs30/preprocess/meta/train.csv", 'w'), delimiter=',')
  writer2 = csv.writer(open(_data_path + "thchs30/preprocess/meta/train_phoneme.csv", 'w'), delimiter=',')

  mfcc_path = _data_path + "thchs30/preprocess/mfcc/"
  if not os.path.exists(mfcc_path):
    os.makedirs(mfcc_path)

  for i, wave_file in enumerate(wavs):
    fn = wave_file.split('/')[-1]

    target_filename = mfcc_path + fn + '.npy'
    if os.path.exists(target_filename):
      os.remove(target_filename)
      # continue
    # print info
    print("thchs30 corpus preprocessing (%d / %d) - '%s']" % (i, len(wavs), wave_file))

    # load wave file
    wave, sr = librosa.load(wave_file, mono=True, sr=None)

    # re-sample ( 48K -> 16K )
    # wave = wave[::3]

    # get mfcc feature
    mfcc = librosa.feature.mfcc(wave, sr=sr)

    # get label index
    label = str2index(open(_data_path + "thchs30/data_thchs30/train/" + fn + '.trn').readlines()[0])
    label2 = str2phoneindex(open(_data_path + "thchs30/data_thchs30/train/" + fn + '.trn').readlines()[2])
    # save result ( exclude small mfcc data to prevent ctc loss )
    if len(label) < mfcc.shape[1]:
      # save meta info
      writer.writerow([fn] + label)
      writer2.writerow([fn] + label2)
      # save mfcc
      # np.save(target_filename, mfcc, allow_pickle=False)
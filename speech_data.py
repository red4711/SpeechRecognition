import gzip
import os
import re
import skimage.io # scikit-image
import numpy
import numpy as np
import wave
# import extensions as xx
from random import shuffle
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

CHUNK = 4096
test_fraction=0.1 # 10% of data for test / verification

class Source:  # labels
  NUMBER_WAVES = 'spoken_numbers_wav.tar'
  DIGIT_WAVES = 'spoken_numbers_pcm.tar'
  DIGIT_SPECTROS = 'spoken_numbers_spectros_64x64.tar'  # 64x64  baby data set, works astonishingly well
  NUMBER_IMAGES = 'spoken_numbers.tar'  # width=256 height=256
  TEST_INDEX = 'test_index.txt'
  TRAIN_INDEX = 'train_index.txt'

from enum import Enum
class Target(Enum):  # labels
  digits=1
  speaker=2
  words_per_minute=3
  word_phonemes=4
  word=5#characters=5
  sentence=6
  sentiment=7
  first_letter=8
  
def dense_to_one_hot(batch, batch_size, num_labels):
  sparse_labels = tf.reshape(batch, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  concatenated = tf.concat(1, [indices, sparse_labels])
  concat = tf.concat(0, [[batch_size], [num_labels]])
  output_shape = tf.reshape(concat, [2])
  sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
  return tf.reshape(sparse_to_dense, [batch_size, num_labels])

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  return numpy.eye(num_classes)[labels_dense]

def spectro_batch_generator(batch_size=10,width=64,target=Target.digits):
  path="spoken_words"
  height = width
  batch = []
  labels = []
  if target==Target.digits: num_classes=10
  if target==Target.first_letter: num_classes=32
  files = os.listdir(path)
  # shuffle(files) # todo : split test_fraction batch here!
  # files=files[0:int(len(files)*(1-test_fraction))]
  print("Got %d source data files from %s"%(len(files),path))
  while True:
    # print("shuffling source data files")
    shuffle(files)
    for image_name in files:
      if not "_" in image_name: continue # bad !?!
      image = skimage.io.imread(path + "/" + image_name).astype(numpy.float32)
      data = image / 255.  # 0-1 for Better convergence
      data = data.reshape([width * height])  # tensorflow matmul needs flattened matrices wtf
      batch.append(list(data))
      classe = (ord(image_name[0]) - 48) % 32# -> 0=0  17 for A, 10 for z ;)
      labels.append(dense_to_one_hot(classe,num_classes))
      if len(batch) >= batch_size:
        yield batch, labels
        batch = []  # Reset for next batch
        labels = []
		
def spectro_batch(batch_size=10):
  return spectro_batch_generator(batch_size)
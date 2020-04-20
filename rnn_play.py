# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import txtutil

# these must match what was saved !
ALPHASIZE = txtutil.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

lyrics0 = 'checkpoints/rnn_train_1586612081-0'
lyrics9 = 'checkpoints/rnn_train_1586614351-900000'
lyrics18 = 'checkpoints/rnn_train_1586614351-1800000'
lyrics21 = 'checkpoints/rnn_train_1586614351-2100000'
lyrics30 = 'checkpoints/rnn_train_1586635708-3000000'
lyrics42 = 'checkpoints/rnn_train_1586649298-4200000'

# need a better for getting the latests
author = lyrics42

def main():
  """
  main method
  """
  ncnt = 0
  with tf.Session() as sess:
    checkpoint_dir: str = './checkpoints'
    #new_saver = tf.train.import_meta_graph('checkpoints/rnn_train_1586649298-4200000.meta')
    new_saver = tf.train.import_meta_graph(f'{tf.train.latest_checkpoint(checkpoint_dir)}.meta')
    new_saver.restore(sess, author)
    x = txtutil.convert_from_alphabet(ord("L"))
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)

    output_file = open("generated_output.txt", "w")
    for _ in range(140):
      yo, h = sess.run(['Yo:0', 'H:0'],
                       feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

      # If sampling is be done from the topn most likely characters, the generated text
      # is more credible and more "english". If topn is not set, it defaults to the full
      # distribution (ALPHASIZE)

      # Recommended: topn = 10 for intermediate checkpoints,
      #   topn = 2 or 3 for fully trained checkpoints

      next_char = txtutil.sample_from_probabilities(yo, topn=2)
      y = np.array([[next_char]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
      next_char = chr(txtutil.convert_to_alphabet(next_char))
      print(next_char, end="")
      output_file.write(next_char)

      if next_char == '\n':
        ncnt = 0
      else:
        ncnt += 1
      if ncnt == 100:
        print("")
        output_file.write("")
        ncnt = 0
    output_file.close()

if __name__ == "__main__":
  main()

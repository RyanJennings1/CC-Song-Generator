#!/usr/bin/env python3
"""
  name: song_generator.py
  author: Ryan Jennings
  date: 2020-04-21
"""
import math
import os
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib import rnn

import ccsonggenerator.txtutil as txt

class SongGenerator():
  """
  Song Generator class
  """
  def __init__(self) -> None:
    """
    init method
    """
    tf.set_random_seed(0)
    self.learning_rate: int = 0.001
    self.dropout_pkeep: int = 0.8
    self.lyrics_dir: str = 'lyrics/*.txt'
    self.seqlen: int = 30
    self.batch_size: int = 200
    self.alpha_size: str = txt.ALPHASIZE
    self.internal_size: int = 512
    self.nlayers: int = 3
    self.message_length: int = 140
    self.checkpoint_dir: str = './checkpoints'
    self.output_filename = 'generated_output.txt'

  def train(self) -> None:
    """
    train the generator
    """
    # load data, either the lyrics, or the Python source of Tensorflow itself
    codetext, valitext, bookranges = txt.read_data_files(self.lyrics_dir, validation=False)

    # display some stats on the data
    epoch_size = len(codetext) // (self.batch_size * self.seqlen)
    txt.print_data_stats(len(codetext), len(valitext), epoch_size)

    lr = tf.placeholder(tf.float32, name='lr')  # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
    batchsize = tf.placeholder(tf.int32, name='batchsize')

    # Inputs
    X = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
    Xo = tf.one_hot(X, self.alpha_size, 1.0, 0.0)                 # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
    # Expected outputs = same sequence shifted by 1 since we are trying to
    # predict the next character
    Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
    Yo_ = tf.one_hot(Y_, self.alpha_size, 1.0, 0.0)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
    # Input state
    Hin = tf.placeholder(tf.float32, [None, self.internal_size*self.nlayers], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

    # Using a NLAYERS=3 layers of GRU cells, unrolled SEQLEN=30 times
    # dynamic_rnn infers SEQLEN from the size of the inputs Xo

    # How to properly apply dropout in RNNs: see README.md
    cells = [rnn.GRUCell(self.internal_size) for _ in range(self.nlayers)]
    # "naive dropout" implementation
    dropcells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
    multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
    # Dropout for the softmax layer
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)

    Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
    # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
    # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

    H = tf.identity(H, name='H')  # just to give it a name

    # Softmax layer implementation:
    # Flatten the first two dimension of the output
    # [ BATCHSIZE, SEQLEN, ALPHASIZE ] => [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    # then apply softmax readout layer. This way, the weights and biases are
    # shared across unrolled time steps.
    # From the readout point of view, a value coming from a sequence time
    # step or a minibatch item is the same thing.

    Yflat = tf.reshape(Yr, [-1, self.internal_size])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
    Ylogits = layers.linear(Yflat, self.alpha_size)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Yflat_ = tf.reshape(Yo_, [-1, self.alpha_size])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
    loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
    Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
    Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    # Stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])

    # Init Tensorboard stuff. This will save Tensorboard information into a different
    # folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
    # you can compare training and validation curves visually in Tensorboard.
    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
    validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

    # Init for saving models. They will be saved into a directory named 'checkpoints'.
    # Only the last checkpoint is kept.
    if not os.path.exists("checkpoints"):
      os.mkdir("checkpoints")
    saver = tf.train.Saver(max_to_keep=1000)

    # For display: init the progress bar
    display_freq = 50
    _50_batches = display_freq * self.batch_size * self.seqlen
    progress = txt.Progress(display_freq, size=111+2, msg=f"Training on next {display_freq} batches")

    # init
    istate = np.zeros([self.batch_size, self.internal_size*self.nlayers])  # initial zero input state
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    step = 0

    # training loop
    for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, self.batch_size, self.seqlen, nb_epochs=10):
      # train on one minibatch
      feed_dict = {X: x, Y_: y_, Hin: istate, lr: self.learning_rate,
                   pkeep: self.dropout_pkeep, batchsize: self.batch_size}
      _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

      # Log training data for Tensorboard display a mini-batch of sequences (every 50 batches)
      if step % _50_batches == 0:
        feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: self.batch_size}  # no dropout for validation
        y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch)
        summary_writer.add_summary(smm, step)

      # Run a validation step every 50 batches
      # The validation text should be a single sequence but that's too slow
      # (1s per 1024 chars!),
      # so we cut it up and batch the pieces (slightly inaccurate)
      # tested: validating with 5K sequences instead of 1K is only slightly
      # more accurate, but a lot slower.
      if step % _50_batches == 0 and len(valitext) > 0:
        # Sequence length for validation. State will be wrong at the start of each sequence.
        vali_seqlen = 1*1024
        bsize = len(valitext) // vali_seqlen
        txt.print_validation_header(len(codetext), bookranges)
        # All data in 1 batch
        vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(valitext, bsize, vali_seqlen, 1))
        vali_nullstate = np.zeros([bsize, self.internal_size*self.nlayers])
        # No dropout for validation
        feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate,
                     pkeep: 1.0, batchsize: bsize}
        ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_validation_stats(ls, acc)
        # Save validation data for Tensorboard
        validation_writer.add_summary(smm, step)

      # Display a short text generated with the current weights and biases (every 150 batches)
      if step // 3 % _50_batches == 0:
        txt.print_text_generation_header()
        ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
        rh = np.zeros([1, self.internal_size * self.nlayers])
        for _ in range(1000):
          ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
          rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
          print(chr(txt.convert_to_alphabet(rc)), end="")
          ry = np.array([[rc]])
        txt.print_text_generation_footer()

      # Save a checkpoint (every 500 batches)
      if step // 1 % _50_batches == 0:
        saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
        print("Saved file: " + saved_file)

      # Display progress bar
      progress.step(reset=step % _50_batches == 0)

      # Loop state around
      istate = ostate
      step += self.batch_size * self.seqlen

  def generate(self):
    """
    Generate new text
    """
    ncnt: int = 0
    with tf.Session() as sess:
      latest_checkpoint: str = tf.train.latest_checkpoint(self.checkpoint_dir)
      new_saver = tf.train.import_meta_graph(f'{latest_checkpoint}.meta')
      new_saver.restore(sess, latest_checkpoint)
      x = txt.convert_from_alphabet(ord("L"))
      x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

      # initial values
      y = x
      # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
      h = np.zeros([1, self.internal_size * self.nlayers], dtype=np.float32)

      output_file = open(self.output_filename, "w")
      for _ in range(self.message_length):
        yo, h = sess.run(['Yo:0', 'H:0'],
                         feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        # If sampling is be done from the topn most likely characters, the generated text
        # is more credible and more "english". If topn is not set, it defaults to the full
        # distribution (ALPHASIZE)

        # Recommended: topn = 10 for intermediate checkpoints,
        #   topn = 2 or 3 for fully trained checkpoints

        next_char = txt.sample_from_probabilities(yo, topn=2)
        # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        y = np.array([[next_char]])
        next_char = chr(txt.convert_to_alphabet(next_char))
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

  def get_output_filename(self):
    """
    Return the output file's name
    """
    return self.output_filename

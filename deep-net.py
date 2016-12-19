#! /usr/bin/python35

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import speech_data
from speech_data import Source, Target

n_classes = 32
hm_epochs = 5
batch_size = 64
chunk_size = 512
n_chunks = 512
rnn_size = 256
n_layers = 3
width = 512
n_steps = 256
learning_rate = 0.001

batch=speech_data.spectro_batch_generator(batch_size, width, target=Target.first_letter)

with tf.name_scope('X'):
    x = tf.placeholder('float', [None, n_chunks, chunk_size])
with tf.name_scope('Observed_Values'):
    y = tf.placeholder('float')

def recurrent_neural_network(x):
    with tf.name_scope('Weights'):
        weights = tf.Variable(tf.random_normal([rnn_size, n_classes]))
    with tf.name_scope('Bias'):
        bias = tf.Variable(tf.random_normal([n_classes]))
    
    layer = {'weights': weights, 'biases': bias}

    with tf.name_scope('Input'):
        x = tf.transpose(x, [1,0,2])
        x = tf.reshape(x, [-1, chunk_size])
        x = tf.split(0, n_chunks, x)

    with tf.name_scope('Layers'):
        lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
        lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * n_layers)

        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    
    with tf.name_scope('Prediction'):
        output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    with tf.name_scope('Cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.name_scope('Accuracy'):
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.name_scope('Summaries'):
        tf.scalar_summary("cost", cost)
        tf.scalar_summary("accuracy", accuracy)
    
    summary_op = tf.merge_all_summaries()
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        writer = tf.train.SummaryWriter('./Tensorboard/', sess.graph)

        with tf.name_scope('Testing_Data'):
            test_X, test_Y = next(batch)
            test_X = np.array(test_X)
            test_X = test_X.reshape((-1, n_chunks, chunk_size))
            
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for step in range(n_steps):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                with tf.name_scope('Training_Data'):
                    epoch_x, epoch_y = next(batch)
                    epoch_x = np.array(epoch_x)
                    epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                summary, acc = sess.run([summary_op, optimizer], feed_dict = {x:epoch_x, y:epoch_y},
                                      options=run_options, run_metadata=run_metadata)

                if (epoch * n_steps + step % 50 == 0):
                    writer.add_run_metadata(run_metadata, 'step%d' % (epoch * n_steps + step))
                    writer.add_summary(summary, epoch * n_steps + step)
                
        print('Accuracy', accuracy.eval({x:test_X, y:test_Y}))

if __name__ == '__main__':
    train_neural_network(x)

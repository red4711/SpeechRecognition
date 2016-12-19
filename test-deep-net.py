#! /usr/bin/python35

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
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

batch=speech_data.spectro_batch_generator(batch_size, width, target=Target.first_letter)

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * n_layers)

    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "model.ckpt")

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        avg = 0

        for _ in range(20):
            test_X, test_Y = next(batch)
            test_X = np.array(test_X)
            temp = accuracy.eval({x:test_X.reshape((-1, n_chunks, chunk_size)), y:test_Y})
            print("Accuracy:", temp)
            avg+= temp

        avg /= 20
        print("Average accuracy: ", avg)

if __name__ == '__main__':
    train_neural_network(x)

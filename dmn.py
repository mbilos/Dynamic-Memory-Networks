import os
import argparse
import datetime
import numpy as np
import tensorflow as tf
import babi_util
import random

# filter out unnecessary tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--task', help='', required=False, type=int)
    parser.add_argument('-b', '--batch', help='', required=False, type=int)
    parser.add_argument('-e', '--iterations', help='', required=False, type=int)
    parser.add_argument('-c', '--cellsize', help='', required=False, type=int)

    return vars(parser.parse_args())

args = get_args()

# babi task id, defaults to 1, can be list of tasks
TASKS = args['task'] or 1

# number of samples that are used in single training step
BATCH_SIZE = args['batch'] or 128

# number of training steps
ITERATIONS = args['iterations'] or 1001

# size of hidden states in RNNs
CELL_SIZE = args['cellsize'] or 128

# size of hidden states in attention
HIDDEN_SIZE = 80

# how many passes through episodic memory
PASSES = 3

# regularization for weights
L2 = 0.00001

# learning rate for training with Adam optimizer
LEARNING_RATE = 0.001

# load train, validation and test dataset and word embedding dictionary
stories_train, stories_valid, stories_test, embedding = babi_util.get_data(TASKS)

# how many unique words are in vocabulary
VOCABULARY_SIZE = len(embedding)

# size of vector representation of word
EMBEDDING_SIZE = 80


def add_padding(array):
    """
    Args:
        array: sequence of array of arrays with variable lengths
    Returns:
        sequence of padded items so that they have the same shape
        for input [ [[1], [2]], [[2], [3], [4]], [[5]] ]
        returns [ [[1], [2], [0]], [[2], [3], [4]], [[5], [0], [0]] ]
    """

    lengths = [np.shape(x)[0] for x in array]

    max_length = max(lengths)
    padding_lengths = [max_length - x for x in lengths]

    array = np.array([np.pad(x, ((0, p), (0, 0)), 'constant', constant_values=(0))
                      for x, p in zip(array, padding_lengths)])

    return array

def positional_encoding(shape):
    """
    implements positional encoding as described in
    "End-To-End Memory Networks" (https://arxiv.org/pdf/1503.08895v5.pdf)
    """

    sentence_size, embedding_size = shape
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)

    ls = sentence_size+1
    le = embedding_size+1

    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)

    encoding = 1 + 4 * encoding / embedding_size / sentence_size

    return np.transpose(encoding)

def get_batch(split='train'):
    """
    Args:
        split: train, valid or test
    Returns:
        dictionary feed to dynamic memory network and optional correct label
    """

    # choose data source based on split, if train take random sample
    if split == 'train':
        batch = [stories_train[i] for i in random.sample(range(len(stories_train)), BATCH_SIZE)]
    elif split == 'valid':
        batch = stories_valid
    else:
        batch = stories_test

    # get text embeddings from batch
    text = [x['text_vec'] for x in batch]

    # prepare sentence by applying positional encoding to each sentence
    # get sentence lengths and add padding to each sentence so they are same size
    _sentence = [[np.sum(s * positional_encoding(np.shape(s)), axis=0) for s in t] for t in text]
    _sentence_length = [len(x) for x in _sentence]
    _sentence = add_padding(_sentence)

    # extract question sequences, get lengths and add padding
    _question = [x['question_vec'] for x in batch]
    _question_length = [len(x) for x in _question]
    _question = add_padding(_question)

    # get output word labels
    _y = np.array([x['answer_vec'] for x in batch])

    # store everything into feed dictionary (input to DMN)
    feed = {sentence: _sentence, sentence_len: _sentence_length,
            question: _question, question_len: _question_length,
            y: _y}

    # if training return only feed, else return also label for evaluation
    if split == 'train':
        return feed
    elif split == 'valid':
        return feed, _y
    else:
        return feed, _y, batch

def print_accuracy(label, prediction):
    """
    Args:
        label: true values of answer output (list of ints)
        prediction: predicted values of output with same shape
    Prints how many are correctly predicted and percentage
    """

    total = np.shape(label)[0]
    correct = np.sum(label == prediction)

    print('correct:', correct, '\ttotal:', total)
    print('Accuracy:', correct / total * 100, '%')
    print('\n-----------------------------------')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def dynamic_rnn(inputs, sequence_length, scope):
    cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs,
        sequence_length=sequence_length, dtype=tf.float32, scope=scope)

    return outputs, state

def bidirectional_dynamic_rnn(inputs, sequence_length, scope):
    forward = tf.contrib.rnn.GRUCell(CELL_SIZE)
    backward = tf.contrib.rnn.GRUCell(CELL_SIZE)
    outputs, state = tf.nn.bidirectional_dynamic_rnn(forward, backward, inputs,
        dtype=np.float32, sequence_length=sequence_length)

    return outputs, state


with tf.Graph().as_default():

    with tf.variable_scope('input') as scope:
        # input sentence [BATCH_SIZE, MAX_SENTENCE_SIZE, EMBEDDING_SIZE]
        sentence = tf.placeholder(tf.float32, [None, None, EMBEDDING_SIZE], 'sentence')
        # input sentence lengths [BATCH_SIZE]
        sentence_len = tf.placeholder(tf.int32, [None], 'sentence-length')

        # get facts by feeding sentence sequences into bidirectional GRU
        # which returns forward and backward states: f<->(i) = f->(i) + f<-(i)
        facts, _ = bidirectional_dynamic_rnn(sentence, sentence_len, scope)
        facts = tf.reduce_sum(tf.stack(facts), axis=0)

    with tf.variable_scope('question') as scope:
        # question sequences [BATCH_SIZE, MAX_QUESTION_SIZE, EMBEDDING_SIZE]
        question = tf.placeholder(tf.float32, [None, None, EMBEDDING_SIZE], 'question')
        # input question lengths [BATCH_SIZE]
        question_len = tf.placeholder(tf.int32, [None], 'question-length')

        # question state is final state from GRU that takes question as input
        _, question_state = dynamic_rnn(question, question_len, scope)

    with tf.variable_scope('output'):
        # answer labels [BATCH_SIZE]
        y = tf.placeholder(tf.int32, [None], 'answer')

    with tf.variable_scope('episodic') as scope:
        # get longest sentence size
        max_sentence_size = tf.shape(facts)[1]

        # convert question from shape [BATCH_SIZE, CELL_SIZE] to [BATCH_SIZE, MAX_SENTENCE_SIZE, CELL_SIZE]
        question_tiled = tf.tile(tf.reshape(question_state, [-1, 1, CELL_SIZE]), [1, max_sentence_size, 1])

        # initialize weights for attention
        w1 = weight_variable([1, 4 * CELL_SIZE, HIDDEN_SIZE])
        b1 = bias_variable([1, 1, HIDDEN_SIZE])

        w2 = weight_variable([1, HIDDEN_SIZE, 1])
        b2 = bias_variable([1, 1, 1])

        def attention(memory, reuse):
            with tf.variable_scope("attention", reuse=reuse):
                # extend and tile memory to [BATCH_SIZE, MAX_SENTENCE_SIZE, CELL_SIZE] shape
                memory = tf.tile(tf.reshape(memory, [-1, 1, CELL_SIZE]), [1, max_sentence_size, 1])

                # interactions between facts, memory and question as described in paper
                attending = tf.concat([facts * memory, facts * question_tiled,
                   tf.abs(facts - memory), tf.abs(facts - question_tiled)], 2)

                # get current batch size
                batch = tf.shape(attending)[0]

                # first fully connected layer
                h1 = tf.matmul(attending, tf.tile(w1, [batch, 1, 1]))
                h1 = h1 + tf.tile(b1, [batch, max_sentence_size, 1])
                h1 = tf.nn.tanh(h1)

                # second and final fully connected layer
                h2 = tf.matmul(h1, tf.tile(w2, [batch, 1, 1]))
                h2 = h2 + tf.tile(b2, [batch, max_sentence_size, 1])

                # returns softmax so attention scores are from 0 to 1
                return tf.nn.softmax(h2, 1)

        # first memory state is question state
        memory = question_state
        att = []

        for p in range(PASSES):
            # get attention from memory (and question and facts which are defined before)
            att.append(attention(memory, bool(p)))

            # initialize GRU cell for RNN which returns final episodic memory
            gru = tf.contrib.rnn.GRUCell(num_units=CELL_SIZE, reuse=bool(p))

            # run loop for length of longest sentence
            def valid(state, i):
                return tf.less(i, max_sentence_size)

            # in each step update state with attention (how much to keep from past)
            def body(state, i):
                state = att[-1][:,i,:] * gru(facts[:,i,:], state)[0] + (1 - att[-1][:,i,:]) * state
                return state, i + 1

            # get episode by applying GRU attention
            episode = tf.while_loop(cond=valid, body=body, loop_vars=[memory, 0])[0]

            # initialize weights for updating between episodes
            w_epizode = weight_variable([3 * CELL_SIZE, CELL_SIZE])
            b_epizode = bias_variable([CELL_SIZE])

            m = tf.concat([memory, episode, question_state], 1)
            m = tf.matmul(m, w_epizode) + b_epizode

            # new memory state is dependent on previous, current memory and question
            memory = tf.nn.relu(m)

    with tf.variable_scope('answer'):
        # input into answer module is concatenation of last memory and question state
        concat = tf.concat([memory, question_state], 1)
        concat = tf.nn.dropout(concat, 0.5)

        w1 = weight_variable([CELL_SIZE * 2, VOCABULARY_SIZE])
        b1 = bias_variable([VOCABULARY_SIZE])

        # outputs predictions
        pred = tf.matmul(concat, w1) + b1
        pred_label = tf.argmax(pred, axis=1, name='prediction_label')

        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred)

    with tf.variable_scope('cost'):
        # softmax loss with regularization on weights
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        loss = tf.reduce_mean(softmax) + lossL2 * L2

    with tf.variable_scope('batch'):
        # initialize optimizer
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

        # get gradients and clip them to prevent gradient explosion
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        # add gradient noise for improved training
        gradient_noise = [tf.random_normal(tf.shape(x), stddev=0.001) for x in gradients]
        gradients = [tf.add(x, y) for x, y in zip(gradients, gradient_noise)]

        optimize = optimizer.apply_gradients(zip(gradients, variables))

    tf.summary.scalar("cost_gru_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), loss)
    summary_batch = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./logs')
    writer_train = tf.summary.FileWriter('./logs/dmn/plot_train')
    writer_val = tf.summary.FileWriter('./logs/dmn/plot_val')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)


        for i in range(ITERATIONS):
            feed = get_batch('train')

            # optimization step
            sess.run(optimize, feed)

            # validation every 50 steps
            if i % 50 == 0:
                summary = sess.run(summary_batch, feed)
                writer_train.add_summary(summary, i)

                val_feed, val_label = get_batch('valid')
                summary_val, val_pred = sess.run([summary_batch, pred_label], val_feed)
                writer_val.add_summary(summary_val, i)

                print("Iteration:", i, "\tof", ITERATIONS)
                print_accuracy(val_label, val_pred)

        # after training is complete test on held out set
        test_feed, test_label, test_batch = get_batch('test')
        summary_val, test_pred, attentions = sess.run([summary_batch, pred_label, att], test_feed)
        writer_val.add_summary(summary_val, i)

        print("\n\n\nTEST SET\n")
        print_accuracy(test_label, test_pred)


        """
        print out sample sentences with attention scores through episodes
        """

        test_text = [x['text'] for x in test_batch]
        test_question = [x['question'] for x in test_batch]

        attentions = np.transpose(np.squeeze(attentions), [1,2,0])
        sentence_attentions = [list(zip(x, y)) for x, y in zip(test_text, attentions)]

        sample_indexes = random.sample(range(len(test_text)), 10)
        for i in sample_indexes:
            print('Question:', test_question[i])
            for item in sentence_attentions[i]:
                print(item[0], '\t', item[1])
            print('\n\n')


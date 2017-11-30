#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


def generate_a_batch(UserMatrix, MovieMatrix, ScoreMatrix, batchsize=100):
    # This function must return three matrices, as detailed in the following:
    # Matrix u:
    #   - contains batchsize/2 different vectors representing users randomly selected
    #     among all the users in the training data. But each vector is duplicated, because
    #     we have to add two movies for each user
    # Matrix m:
    #   - each row must contain a vector representing a movie. These movies must be randomly
    #     selected among the set of movies that the user likes/dislikes (depending on the row)
    #     and following a distribution proportional to different novelty functions that must be
    #     implemented (see the PDF paper)
    # Matrix l:
    #   - contains +1 or -1 depending on the row. Thus, in row i there will be a +1 if
    #     the user in row i of matrix u likes the movie in row i of matrix m, and -1 in the
    #     user does not like such movie

    # MATRIX u               |   MATRIX m               |    Matrix l
    #---------------------------------------------------------------------------------------
    # vector for user u_x1   |   vector for movie m_y   |    +1 (because u_x1 likes m_y)
    # vector for user u_x1   |   vector for movie m_z   |    -1 (because u_x1 dislikes m_z)
    # vector for user u_x2   |   vector for movie m_t   |    +1 (because u_x2 likes m_t)
    # vector for user u_x2   |   vector for movie m_r   |    -1 (because u_x2 dislikes m_r)
    # etc...                 |   etc...                 |    etc...

    # todo: implement this function: it is NOT trivial

    return u, m, l


# Let's suppose U is a matrix where each row is a vector codifying
# a different user, with all the additional information available.
# Let's suppose M is the corresponding matrix for movies, where each
# row codifies a different movie.
# And, suppose S a (sparse) matrix with as many rows as different users
# and as may columns as movies, containing in positions (i,j) a label (+1 or -1)
# depending on the score given by the i-th user to the j-th movie.
# These matrices can be built as a pre-processing step, starting from the
# data contained in the MovieLens (or other) database


NUM_USERS = XXXX  # total number of users
NUM_MOVIES = YYYY # total number of movies

USER_VECTOR_SIZE = XXX
MOVIE_VECTOR_SIZE = YYY

learning_rate = 0.001  # for example... but it is better to search for an adequate value
nu = 0.0001  # for example... but it is better to search for an adequate value
K = 500 # for example... this is the dimension of the embedding space, where users and movies are mapped

# Create the TF graph
graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):
    # Define the input
    # User stuff
    user = tf.placeholder(tf.float32, shape=[None, USER_VECTOR_SIZE])

    # Movie stuff
    movie = tf.placeholder(tf.float32, shape=[None, MOVIE_VECTOR_SIZE])

    # Class: +1 / -1
    label = tf.placeholder(tf.float32, shape=[None, 1])

    # Definition of parameters W and A
    W = tf.Variable(initial_value=tf.truncated_normal((K, USER_VECTOR_SIZE),
                                                      stddev=1.0 / np.sqrt(K)))
    A = tf.Variable(initial_value=tf.truncated_normal((K, MOVIE_VECTOR_SIZE),
                                                      stddev=1.0 / np.sqrt(K)))

    # Definition of g (Eq. (14) in the paper g = <Wu, Vi> = u^T * W^T * V * i)
    g = tf.diag_part(tf.transpose(user) @ tf.transpose(W) @ A @ tf.transpose(movie))

    # The previous line should be equivalent to the following three lines
    # f_aux_1 = tf.matmul(user, W, transpose_a=False, transpose_b=True)
    # f_aux = tf.matmul(f_aux_1, A)
    # result = tf.diag_part(tf.matmul(f_aux, movie, transpose_b=True))

    # Loss function
    x = label * g
    loss = tf.reduce_mean(tf.softplus(-x))

    # Regularization
    reg = nu * (tf.nn.l2_loss(W) + tf.nn.l2_loss(A))

    # Loss function with regularization (what we want to minimize)
    loss_to_minimize = loss + reg

    optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss_to_minimize)


# Once the graph is created, let's program the training loop
with tf.Session(graph=graph) as session:
    # mandatory: initialize variables in the graph, i.e. W, A
    session.run(tf.global_variables_initializer())
    avg_loss = 0
    EVERY_N_ITERATIONS = 1000  # print a message every some iterations
    MAX_ITERATIONS = 1e06  # for example
    for iter in range(MAX_ITERATIONS):
        # todo: generate a batch of examples
        users_batch, movies_batch, labels_batch = generate_a_batch(U, M, S)
        feed_dict = {user: users_batch,
                     movie: movies_batch,
                     label: labels_batch}
        _, loss_value = session.run([optimize, loss_to_minimize], feed_dict=feed_dict)
        avg_loss += loss_value
        if iter%EVERY_N_ITERATIONS == 0:
            print('Iteration', iter)
            avg_loss /= EVERY_N_ITERATIONS
            print('Average loss:', avg_loss)
            avg_loss = 0




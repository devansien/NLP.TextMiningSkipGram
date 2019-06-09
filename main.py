import numpy as np
import pandas as pd
import tensorflow as tf

corpus = ['you are kim',
          'you are a developer']
print('\n' + 'original corpus:\n' + str(corpus) + '\n')


# remove stop words
def remove_stop_words(contents):
    stop_words = ['a', 'is', 'be', 'are', 'will']
    modified_contents = []
    for content in contents:
        word_list = content.split(' ')
        for stop_word in stop_words:
            if stop_word in word_list:
                word_list.remove(stop_word)
        modified_contents.append(' '.join(word_list))
    return modified_contents


# one hot encoding
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIMENSION)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding


# pre-process corpus
corpus = remove_stop_words(corpus)
print('modified corpus after removing stop words:\n' + str(corpus) + '\n')

# make a word set
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)
words = set(words)
print('word dictionary without duplicates:\n' + str(words) + '\n')

# word to int
word2int = {}
for i, word in enumerate(words):
    word2int[word] = i
print('word2int mapping:\n' + str(word2int) + '\n')

# get sentences
sentences = []
for sentence in corpus:
    sentences.append(sentence.split())
print('tokenized sentences:\n' + str(sentences) + '\n')

# skip gram mapping
WINDOW_SIZE = 1
data = []
for sentence in sentences:
    for index, word in enumerate(sentence):
        for neighbor in sentence[max(index - WINDOW_SIZE, 0): min(index + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])
print('skip gram data:\n' + str(data) + '\n')

data_frame = pd.DataFrame(data, columns=['input', 'label'])
print(data_frame)

# one hot encoding
ONE_HOT_DIMENSION = len(words)
print('\n' + 'one hot encoding dimension:\n' + str(ONE_HOT_DIMENSION) + '\n')

x = []  # input word
y = []  # target word
zipped = set(zip(data_frame['input'], data_frame['label']))
print('zipped:\n' + str(zipped) + '\n')

for xx, yy in zipped:
    x.append(to_one_hot_encoding(word2int[xx]))
    y.append(to_one_hot_encoding(word2int[yy]))

# encoded strings
x_train = np.asarray(x)
y_train = np.asarray(y)
print('x train:\n' + str(x_train) + '\n')
print('y train:\n' + str(y_train) + '\n')

x_input = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIMENSION))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIMENSION))

# word embedding will be 2 dimension for 2d visualization
EMBEDDING_DIM = 2

# hidden layer: which represents word vector eventually
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIMENSION, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1]))  # bias
hidden_layer = tf.add(tf.matmul(x_input, W1), b1)

# output layer
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIMENSION]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2), b2))

# loss function: cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# training operation
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# Train the NN
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 5000
for i in range(iteration):
    # input is X_train which is one hot encoded word
    # label is Y_train which is one hot encoded neighbor word
    sess.run(train_op, feed_dict={x_input: x_train, y_label: y_train})
    if i % 3000 == 0:
        print('iteration ' + str(i) + ' loss is : ', sess.run(loss, feed_dict={x_input: x_train, y_label: y_train}))

# Now the hidden layer (W1 + b1) is actually the word look up table
vectors = sess.run(W1 + b1)
# print(vectors)

# Print the word vector in a table
w2v_df = pd.DataFrame(vectors, columns=['x1', 'x2'])
w2v_df['word'] = words
w2v_df = w2v_df[['word', 'x1', 'x2']]
print(w2v_df)

corpus = ['my name is bryan',
          'your are a developer']


# remove stop words
def remove_stop_words(corpus):
    stop_words = ['a', 'is', 'be', 'are', 'will']
    modified_corpus = []
    for text in corpus:
        words = text.split(' ')
        for stop_word in stop_words:
            if stop_word in words:
                words.remove(stop_word)
        modified_corpus.append(' '.join(words))

    return modified_corpus


# pre-process corpus
corpus = remove_stop_words(corpus)
print('corpus: ' + str(corpus))

# make a word set
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)

words = set(words)
print('word set: ' + str(words))

# word to int
word2int = {}

for i, word in enumerate(words):
    word2int[word] = i

# get sentences
sentences = []
for sentence in corpus:
    sentences.append(sentence.split())

print(str(sentences))

corpus = ['my name is bryan']


def remove_stop_words(corpus):
    stop_words = ['a', 'is', 'be', 'will']
    modified_corpus = []
    for text in corpus:
        words = text.split(' ')
        for stop_word in stop_words:
            if stop_word in words:
                words.remove(stop_word)
        modified_corpus.append(' '.join(words))

    return modified_corpus


corpus = remove_stop_words(corpus)
print(corpus)

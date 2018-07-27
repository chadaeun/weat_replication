import numpy as np


def word2vec_define_get_word_vectors(args):
    """
    Returns 'get_word_vectors' function for word2vec
    :param args: parsed arguments
    :return: 'get_word_vectors'function
    """
    from gensim.models import KeyedVectors
    import logging

    print('Loading word2vec model...', end='')
    model = KeyedVectors.load_word2vec_format(args.word_embedding_path, binary=True)
    print('DONE')

    def get_word_vectors(words):
        """
        Returns word vectors represent words
        :param words: iterable of words
        :return: (len(words), dim) shaped numpy ndarrary which is word vectors
        """
        words = [w for w in words if w in model]
        return model[words]

    return get_word_vectors


def glove_define_get_word_vectors(args):
    """
    Returns 'get_word_vectors' function for glove
    :param args: parsed arguments
    :return: 'get_word_vectors'function
    """
    import numpy as np

    word_index = dict()
    vectors = []

    with open(args.word_embedding_path) as f:
        for i, line in enumerate(f.readlines()):
            word = line.split()[0]
            vector = np.array(line.split()[1:])
            vector = np.apply_along_axis(float, 1, vector.reshape(-1, 1))

            word_index[word] = i
            vectors.append(vector.reshape(1, -1))

    embeddings = np.concatenate(vectors, axis=0)

    def get_word_vectors(words):
        """
        Returns word vectors represent words
        :param words: iterable of words
        :return: (len(words), dim) shaped numpy ndarrary which is word vectors
        """
        word_ids = [word_index[w] for w in words if w in word_index]
        return embeddings[word_ids]

    return get_word_vectors


def tf_hub_define_get_word_vectors(args):
    """
    Returns 'get_word_vectors' function for nnlm-en-dim50
    :param args: parsed arguments
    :return: 'get_word_vectors'function
    """
    import tensorflow as tf
    import tensorflow_hub as hub

    model = hub.Module(args.tf_hub)

    def get_word_vectors(words):
        """
        Returns word vectors represent words
        :param words: iterable of words
        :return: (len(words), dim) shaped numpy ndarrary which is word vectors
        """
        words = tf.constant(words)
        words = tf.reshape(words, [-1])
        result = model(words)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            output = sess.run(result)

        return output

    return get_word_vectors


def define_get_word_vectors(args):
    """
    Returns 'get_word_vectors' function according to word embedding type
    :param args: parsed arguments
    :return: 'get_word_vectors' function
    :raises ValueError: word embedding type is not available
    """

    if args.word_embedding_type == 'word2vec':
        return word2vec_define_get_word_vectors(args)

    elif args.word_embedding_type == 'glove':
        return glove_define_get_word_vectors(args)

    elif args.word_embedding_type == 'tf_hub':
        return tf_hub_define_get_word_vectors(args)

    else:
        raise ValueError("word_embedding_type must be 'word2vec', 'glove', or 'tf_hub'")


def balance_word_vectors(A, B):
    """
    Balance size of two lists of word vectors by randomly deleting some vectors in larger one.
    If there are words that did not occur in the corpus, some words will ignored in get_word_vectors.
    So result word vectors' size can be unbalanced.
    :param A: (len(words), dim) shaped numpy ndarrary which is word vectors
    :param B: (len(words), dim) shaped numpy ndarrary which is word vectors
    :return: tuple of two balanced word vectors
    """

    diff = len(A) - len(B)

    if diff > 0:
        A = np.delete(A, np.random.choice(len(A), diff, 0), axis=0)
    else:
        B = np.delete(B, np.random.choice(len(B), -diff, 0), axis=0)

    return A, B
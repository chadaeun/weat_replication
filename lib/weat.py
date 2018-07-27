import numpy as np
from sympy.utilities.iterables import multiset_permutations


def unit_vector(vec):
    """
    Returns unit vector
    """
    return vec / np.linalg.norm(vec)


def cos_sim(v1, v2):
    """
    Returns cosine of the angle between two vectors
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)


def weat_association(W, A, B):
    """
    Returns association of the word w in W with the attribute for WEAT score.
    s(w, A, B)
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: (len(W), ) shaped numpy ndarray. each rows represent association of the word w in W
    """
    return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)


def weat_differential_association(X, Y, A, B):
    """
    Returns differential association of two sets of target words with the attribute for WEAT score.
    s(X, Y, A, B)
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: differential association (float value)
    """
    return np.sum(weat_association(X, A, B)) - np.sum(weat_association(Y, A, B))


def weat_p_value(X, Y, A, B):
    """
    Returns one-sided p-value of the permutation test for WEAT score
    CAUTION: this function is not appropriately implemented, so it runs very slowly
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: p-value (float value)
    """
    diff_association = weat_differential_association(X, Y, A, B)
    target_words = np.concatenate((X, Y), axis=0)

    # get all the partitions of X union Y into two sets of equal size.
    idx = np.zeros(len(target_words))
    idx[:len(target_words) // 2] = 1

    partition_diff_association = []
    for i in multiset_permutations(idx):
        i = np.array(i, dtype=np.int32)
        partition_X = target_words[i]
        partition_Y = target_words[1 - i]
        partition_diff_association.append(weat_differential_association(partition_X, partition_Y, A, B))

    partition_diff_association = np.array(partition_diff_association)

    return np.sum(partition_diff_association > diff_association) / len(partition_diff_association)


def weat_score(X, Y, A, B):
    """
    Returns WEAT score
    X, Y, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: this function assumes that there's no intersection word between X and Y
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEAT score
    """

    x_association = weat_association(X, A, B)
    y_association = weat_association(Y, A, B)


    tmp1 = np.mean(x_association, axis=-1) - np.mean(y_association, axis=-1)
    tmp2 = np.std(np.concatenate((x_association, y_association), axis=0))

    return tmp1 / tmp2


def wefat_p_value(W, A, B):
    """
    Returns WEFAT p-value
    W, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: not implemented yet
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEFAT p-value
    """
    pass


def wefat_score(W, A, B):
    """
    Returns WEFAT score
    W, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: this function assumes that there's no intersection word between A and B
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEFAT score
    """
    tmp1 = weat_association(W, A, B)
    tmp2 = np.std(np.concatenate((cos_sim(W, A), cos_sim(W, B)), axis=0))

    return np.mean(tmp1 / tmp2)
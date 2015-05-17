# Copyright (c) 2015, Aleksandar Savkov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of  nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
__author__ = 'Aleksandar Savkov'

import numbers
import warnings
import numpy as np

from math import ceil, floor
from sklearn.utils import check_random_state


class Bootstrap(object):
    """Random sampling with replacement iterator. Each iteration provides a new
    set of indices drawn with replacement from the population. Note that
    contrary to other resampling strategies, bootstrapping will allow some
    observations to occur several times in each resample.

    Parameters
    ----------
    weights : array-like
        List of sample weights.
    n_iter : int (default is 3)
        Number of bootstrapping iterations
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Examples
    --------
    >>>from bootstrapsplit import Bootstrap
    >>>pop = np.array(list('ABDEFGHIJKL'))
    >>>b = Bootstrap(13, n_iter=3)
    >>>print(b)
    Bootstrap(13, n_iter=3, random_state=None)
    >>>print("POPULATION:", pop)
    POPULATION: ['A' 'B' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N']
    >>>for s in b:
    >>>    print("BOOTSTRAP:", pop[s])
    BOOTSTRAP: ['D' 'N' 'G' 'L' 'G' 'D' 'I' 'D' 'K' 'I' 'D' 'M' 'K']
    BOOTSTRAP: ['N' 'H' 'A' 'E' 'B' 'H' 'M' 'B' 'K' 'N' 'N' 'M' 'K']
    BOOTSTRAP: ['G' 'N' 'G' 'L' 'M' 'N' 'I' 'N' 'D' 'M' 'H' 'I' 'G']
    """
    def __init__(self, n, n_iter=3, random_state=None):
        self.n = n
        self.n_iter = n_iter
        self.random_state = random_state

    def __iter__(self):
        rng = check_random_state(self.random_state)
        for i in range(self.n_iter):
            # sample indexes
            sample_idxs = np.arange(self.n)

            # indexes sampled with replacement
            bootstrap_idxs = rng.randint(0, self.n, self.n)
            yield sample_idxs[bootstrap_idxs]

    def __repr__(self):
        return ('%s(%d, n_iter=%d, random_state=%s)' % (
                    self.__class__.__name__,
                    self.n,
                    self.n_iter,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter


class WeightLimitedBootstrap(object):
    """Random sampling with replacement iterator for a population of
    observation weights. Each iteration provides a new set of indices drawn
    with replacement from the population. The sample sizes are measured by the
    aggregate weight of the observations and NOT by their number. The size of
    the samples can be limited to a certain weight, by default this is the
    total weight of the population. Note that contrary to other resampling
    strategies, bootstrapping will allow some observations to occur several
    times in each resample, which will result in varying total weight in each
    resample if sample size is measured in number of observations.

    Parameters
    ----------
    weights : array-like
        List of sample weights.
    n_iter : int (default is 3)
        Number of bootstrapping iterations
    max_weight : int or float (default is 0.5)
        If int, maximum number of observations to include in the training
        split (should be less or equal to the size of the population).
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the population to include in the resampled dataset.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Examples
    --------
    >>>from bootstrapsplit import WeightLimitedBootstrap
    >>>pop = np.array(list('ABDEFGHIJKL'))
    >>>w = np.random.randint(low=1, high=5, size=len(pop))
    >>>wb = WeightLimitedBootstrap(w, n_iter=3, max_weight=len(pop) - 2)
    >>>print(wb)
    3
    >>>print("POPULATION:", pop, "(weight=%s)" % w.sum())
    POPULATION: ['A' 'B' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L'](weight=31)
    >>>for s in wb:
    >>>    print("BOOTSTRAP:", pop[s], "(weight=%s)" % w[s].sum())
    BOOTSTRAP: ['C' 'G' 'F'] (weight=9)
    BOOTSTRAP: ['A' 'B' 'A' 'C' 'A'] (weight=10)
    BOOTSTRAP: ['I' 'D' 'H' 'F'] (weight=11)
    """

    def __init__(self, weights, n_iter=3, max_weight=1.0, random_state=None):
        self.n = len(weights)
        self.weights = np.array(weights)
        self.total_weight = int(self.weights.sum())
        self.n_iter = n_iter
        if isinstance(max_weight, numbers.Integral):
            self.limit = max_weight
            if max_weight in [0, 1]:
                warnings.warn('Limit set to %s not %r. This is '
                              'probably not what you intended.'
                              % (max_weight, float(max_weight)))
        elif (isinstance(max_weight, numbers.Real) and max_weight >= 0.0
                and max_weight <= 1.0):
            self.limit = int(ceil(max_weight * self.total_weight))
        else:
            raise ValueError("Invalid value for limit: %r" %
                             max_weight)
        if self.limit > self.total_weight:
            raise ValueError("limit=%d should not be larger than "
                             "the total_weight=%d" %
                             (self.limit, self.total_weight))
        self.random_state = random_state

    def __iter__(self):
        rng = check_random_state(self.random_state)
        for i in range(self.n_iter):
            # sample indexes
            sample_idxs = np.arange(self.n)

            # indexes sampled with replacement
            bootstrap_idxs = rng.randint(0, self.n, self.n)
            yield sample_idxs[self._sub_item_slice(
                sample_idxs[bootstrap_idxs], max_weight=self.limit)]

    def _sub_item_slice(self, permutation, min_weight=0, max_weight=None):
        return _sub_item_slice(self.weights, permutation, min_weight,
                               max_weight)

    def __repr__(self):
        return ('%s(%d, n_iter=%d, limit=%d random_state=%s)' % (
                    self.__class__.__name__,
                    self.n,
                    self.n_iter,
                    self.limit,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter


class WeightLimitedBootstrapSplit(object):
    """Random sampling with replacement split iterator for a sample of
    sequences of uneven size. Provides train/test indices to split data in
    train test sets while resampling the input n_iter times: each time a new
    random split of the data is performed and then samples are drawn
    (with replacement) on each side of the split to build the training
    and test sets. The sample sizes are measured by the cumulative length of
    the sequences and NOT by the number of sequences.
    Note: contrary to other resampling strategies, bootstrapping
    will allow some sample sequences to occur several times in each split.
    However, a sample that occurs in the train split will never occur in the
    test split and vice-versa.

    Note: sequence bootstrapping corrects sample size deviations when
    re-sampling a dataset of uneven-sized sequences, e.g. sentences.

    Parameters
    ----------
    weights : array-like
        List of sample weights.
    n_iter : int (default is 3)
        Number of bootstrapping iterations
    train_size : int or float (default is 0.5)
        If int, cumulative size of sequence samples to include in the training
        split (should be smaller than the total number of samples passed
        in the dataset).
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.
    test_size : int or float or None (default is None)
        If int, cumulative size of sequence samples to include in the testing
        split (should be smaller than the total number of samples passed
        in the dataset).
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split.
        If None, n_test is set as the complement of n_train.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Examples
    --------
    >>> import numpy as np
    >>> from bootstrapsplit import WeightLimitedBootstrapSplit
    >>> x = np.random.randint(1, 10, 20)
    >>> bs = WeightLimitedBootstrapSplit(x, random_state=0)
    >>> len(bs)
    3
    >>> print(bs)
    Bootstrap(9, n_iter=3, train_size=5, test_size=4, random_state=0)
    >>> for train_index, test_index in bs:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...
    TRAIN: [1 8 7 7 8] TEST: [0 3 0 5]
    TRAIN: [5 4 2 4 2] TEST: [6 7 1 0]
    TRAIN: [4 7 0 1 1] TEST: [5 3 6 5]
    """

    def __init__(self, weights, n_iter=3, train_size=.5, test_size=None,
                 random_state=None):
        self.n = len(weights)
        self.weights = np.array(weights)
        self.total_weight = int(np.sum(weights))
        self.n_iter = n_iter
        if isinstance(train_size, numbers.Integral):
            self.train_size = train_size
            if train_size in [0, 1]:
                warnings.warn('Training set size set to %s not %s. This is '
                              'probably not what you intended.'
                              %(train_size, float(train_size)))
        elif (isinstance(train_size, numbers.Real) and train_size >= 0.0
                and train_size <= 1.0):
            self.train_size = int(ceil(train_size * self.total_weight))
        else:
            raise ValueError("Invalid value for train_size: %r" %
                             train_size)
        if self.train_size > self.total_weight:
            raise ValueError("train_size=%d should not be larger than "
                             "the cumulative sum of subsequences l_seq=%d" %
                             (self.train_size, self.n))

        if isinstance(test_size, numbers.Integral):
            self.test_size = test_size
            if train_size in [0, 1]:
                warnings.warn('Testing set size set to %s not %s. This is '
                              'probably not what you intended.'
                              % (test_size, float(test_size)))
        elif isinstance(test_size, numbers.Real) and 0.0 <= test_size <= 1.0:
            self.test_size = int(floor(test_size * self.total_weight))
        elif test_size is None:
            self.test_size = self.total_weight - self.train_size
        else:
            raise ValueError("Invalid value for test_size: %r" % test_size)
        if self.test_size > self.total_weight - self.train_size:
            raise ValueError(("test_size + train_size=%d, should not be " +
                              "larger than the cumulative sum of " +
                              "subsequences l_seq=%d") %
                             (self.test_size + self.train_size,
                              self.total_weight))

        self.random_state = random_state

    def _sub_item_slice(self, permutation, min_weight=0, max_weight=None):
        return _sub_item_slice(self.weights, permutation, min_weight,
                               max_weight)

    def __iter__(self):
        rng = check_random_state(self.random_state)
        for i in range(self.n_iter):

            # random partition
            permutation = rng.permutation(range(self.n))

            ind_train = self._sub_item_slice(permutation,
                                             max_weight=self.train_size)
            real_train_size = np.sum(self.weights[ind_train])

            ind_test = self._sub_item_slice(permutation,
                                            min_weight=int(real_train_size),
                                            max_weight=real_train_size +
                                            self.test_size)
            real_test_size = np.sum(self.weights[ind_test])

            # bootstrap in each split individually
            train_sample = np.array([], dtype=np.int)
            test_sample = np.array([], dtype=np.int)

            if ind_train != np.array([]):
                train = rng.randint(0, len(ind_train),
                                    size=(len(ind_train)*100,))
                train_sample = self._sub_item_slice(ind_train[train],
                                                    max_weight=real_train_size)
            if ind_test != np.array([]):
                test = rng.randint(0, len(ind_test),
                                   size=(len(ind_test)*100,))
                test_sample = self._sub_item_slice(ind_test[test],
                                                   max_weight=real_test_size)

            yield train_sample, test_sample

    def __repr__(self):
        return ('%s(%d(%d), n_iter=%d, train_size=%d, test_size=%d, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.n,
                    self.total_weight,
                    self.n_iter,
                    self.train_size,
                    self.test_size,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter


def _sub_item_slice(weights, permutation, min_weight=0, max_weight=None):
        """ Returns a slice of the weights sequence based on a permutation
        with weight restrictions.

        Parameters
        ----------
        permutation : array-like
        Index permutation of the original weights sequence.
        min_weight : int or float
        minimum weight
        max_weight : int or float
        maximum weight

        Returns
        -------
        slice : array-like
        Weight-limited slice of permutation
        """
        mnw = min_weight
        mxw = np.sum(weights) if max_weight is None else max_weight
        cs = np.cumsum(weights[permutation])
        slice = permutation[(cs > mnw) & (cs <= mxw)]
        return slice


# The following part of this module was taken from scikit learn before its
# deprecation in version 0.17. I have made slight modifications to fit the
# rest of the module.
#
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>,
#         Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

class BootstrapSplit(object):
    """Random sampling with replacement cross-validation iterator
    Provides train/test indices to split data in train test sets
    while resampling the input n_iter times: each time a new
    random split of the data is performed and then samples are drawn
    (with replacement) on each side of the split to build the training
    and test sets.
    Note: contrary to other cross-validation strategies, bootstrapping
    will allow some samples to occur several times in each splits. However
    a sample that occurs in the train split will never occur in the test
    split and vice-versa.
    If you want each sample to occur at most once you should probably
    use ShuffleSplit cross validation instead.
    Parameters
    ----------
    n : int
        Total number of elements in the dataset.
    n_iter : int (default is 3)
        Number of bootstrapping iterations
    train_size : int or float (default is 0.5)
        If int, number of samples to include in the training split
        (should be smaller than the total number of samples passed
        in the dataset).
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.
    test_size : int or float or None (default is None)
        If int, number of samples to include in the training set
        (should be smaller than the total number of samples passed
        in the dataset).
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split.
        If None, n_test is set as the complement of n_train.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.
    Examples
    --------
    >>> from bootstrapsplit import BootstrapSplit
    >>> bs = BootstrapSplit(9, random_state=0)
    >>> len(bs)
    3
    >>> print(bs)
    Bootstrap(9, n_iter=3, train_size=5, test_size=4, random_state=0)
    >>> for train_index, test_index in bs:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...
    TRAIN: [1 8 7 7 8] TEST: [0 3 0 5]
    TRAIN: [5 4 2 4 2] TEST: [6 7 1 0]
    TRAIN: [4 7 0 1 1] TEST: [5 3 6 5]
    """

    def __init__(self, n, n_iter=3, train_size=.5, test_size=None,
                 random_state=None):
        self.n = n
        self.n_iter = n_iter
        if isinstance(train_size, numbers.Integral):
            self.train_size = train_size
        elif (isinstance(train_size, numbers.Real) and train_size >= 0.0
                and train_size <= 1.0):
            self.train_size = int(ceil(train_size * n))
        else:
            raise ValueError("Invalid value for train_size: %r" %
                             train_size)
        if self.train_size > n:
            raise ValueError("train_size=%d should not be larger than n=%d" %
                             (self.train_size, n))

        if isinstance(test_size, numbers.Integral):
            self.test_size = test_size
        elif isinstance(test_size, numbers.Real) and 0.0 <= test_size <= 1.0:
            self.test_size = int(ceil(test_size * n))
        elif test_size is None:
            self.test_size = self.n - self.train_size
        else:
            raise ValueError("Invalid value for test_size: %r" % test_size)
        if self.test_size > n - self.train_size:
            raise ValueError(("test_size + train_size=%d, should not be " +
                              "larger than n=%d") %
                             (self.test_size + self.train_size, n))

        self.random_state = random_state

    def __iter__(self):
        rng = check_random_state(self.random_state)
        for i in range(self.n_iter):
            # random partition
            permutation = rng.permutation(self.n)
            ind_train = permutation[:self.train_size]
            ind_test = permutation[self.train_size:self.train_size
                                   + self.test_size]

            # bootstrap in each split individually
            train = []
            test = []
            if self.train_size:
                train = rng.randint(0, self.train_size,
                                    size=(self.train_size,))
            if self.test_size:
                test = rng.randint(0, self.test_size,
                                   size=(self.test_size,))
            yield ind_train[train], ind_test[test]

    def __repr__(self):
        return ('%s(%d, n_iter=%d, train_size=%d, test_size=%d, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.n,
                    self.n_iter,
                    self.train_size,
                    self.test_size,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter
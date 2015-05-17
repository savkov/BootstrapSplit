# Bootstrapping

### What is bootstrapping?
> Bootstrapping is the practice of estimating properties of an estimator (such as its variance) by measuring those properties when sampling from an approximating distribution. One standard choice for an approximating distribution is the empirical distribution function of the observed data. In the case where a set of observations can be assumed to be from an independent and identically distributed population, this can be implemented by constructing a number of resamples with replacement, of the observed dataset (and of equal size to the observed datasert [population]). -- [Wikipedia](http://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29 "Bootstrapping (Statistics)")

### What is it used for?
> Bootstrapping allows assigning measures of accuracy (defined in terms of bias, variance, confidence intervals, prediction error or some other such measure) to sample estimates. -- [Effron & Tibshirani, (1993)](https://books.google.co.uk/books/about/An_Introduction_to_the_Bootstrap.html?id=gLlpIUxRntoC&hl=en "An Introduction to the Bootstrap")

### Usage

The `BootstrapSplit` class is a modified version of the `Bootstrap` iterator from the [cross_validation](http://scikit-learn.org/stable/modules/cross_validation.html "Cross-validation") module in [scikit-learn](http://scikit-learn.org/stable/index.html "sklearn"). Provided with the number of observations in the population the iterator returns a tuple containing two lists of index references to the population, representing the two split subsets resampled from the population. The size of the samples and the number of iterations can be controlled with keyword parameters.

```python
import numpy as np
from __future__ import print_function
from bootstrapsplit import BootstrapSplit

# population
pop = np.array(list('ABDEFGHIJKLMN'))
bs = BootstrapSplit(len(pop), random_state=0)
print(bs)
```

    BootstrapSplit(13, n_iter=3, train_size=7, test_size=6, random_state=0)


```python
print('POPULATION:', pop)
for tr_idx, te_idx in bs:
    print("TRAIN:", pop[tr_idx], "TEST:", pop[te_idx])
```

    POPULATION: ['A' 'B' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N']
    TRAIN: ['H' 'D' 'F' 'M' 'B' 'B' 'H'] TEST: ['K' 'N' 'K' 'N' 'I' 'K']
    TRAIN: ['D' 'I' 'I' 'L' 'N' 'J' 'J'] TEST: ['G' 'H' 'M' 'E' 'A' 'A']
    TRAIN: ['J' 'L' 'B' 'B' 'E' 'L' 'L'] TEST: ['K' 'K' 'N' 'N' 'A' 'N']


Contrary to other resampling strategies, bootstrapping will allow some observations to occur several times in each sample. However, an observation that occurs in the train sample will never occur in the test sample and vice-versa.

### What is weight-limited bootstrapping?

In classic bootstrapping, observations are sampled uniformly with replacement until the sample is equal in size to the population. But in some cases we want to use a different limiting criterion. For example, in the well-known knapsack problem items (observations) are selected until their combined weight reaches a threshold. Weight-limited bootstrapping is similar in that each observation is assigned a weight and the total weight of the sample must not exceed a threshold. For example, sentences are made up of different number of words. Suppose we wanted to fill a page with a random sample of sentences from a long document (population). However, in this case if we simply sampled sentences (observations), we may run out of space as sentences are of different length and a page can contain no more than `t` words. The solution is to _weigh_ each sentence (observation) by its word count, and to _limit_ the sample to a maximum weight of `t`.

```python
from bootstrapsplit import WeightLimitedBootstrapSplit

w = np.random.randint(low=1, high=3, size=len(pop))
wb = WeightLimitedBootstrapSplit(w, n_iter=3, train_size=9, test_size=4)
print(wb)
print("POPULATION:", pop, "(weight=%s)" % w.sum())
for tr, te in wb:
    print("TRAIN:", pop[tr], "(weight=%s)" % w[tr].sum(), "TEST:", pop[te], "(weight=%s)" % w[te].sum())
```

    WeightLimitedBootstrapSplit(13(18), n_iter=3, train_size=9, test_size=4, random_state=None)
    POPULATION: ['A' 'B' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N'] (weight=18)
    TRAIN: ['I' 'B' 'I' 'K' 'F' 'E'] (weight=8) TEST: ['D' 'M'] (weight=3)
    TRAIN: ['A' 'A' 'A' 'L'] (weight=8) TEST: ['G' 'N' 'G'] (weight=3)
    TRAIN: ['G' 'A' 'M' 'M' 'B' 'N' 'G' 'G'] (weight=9) TEST: ['D' 'F' 'E'] (weight=4)


The `WeightLimitedBootstrapSplit` class re-implements the `BootstrapSplit` accounting for individual sample weight. In contrast to `WeightLimitedBootstrapSplit` though it only sets a maximum weight for each sample split, which means that the returned sample is of the closest lower weight given a random resampling with replacement. This introduces a small degree of inaccuracy that needs to be kept in mind when working with very small samples. If the weight of a sample split is smaller than that of the first token in the resampled sequence, an empty list is returned.

In the following example we set the weight limit of the test sample to `1` while the lowest word weight is `2`. 

```python
w = np.random.randint(low=2, high=5, size=len(pop))
wb = WeightLimitedBootstrapSplit(w, n_iter=3, train_size=0.8, test_size=1)
print(wb)
for tr_idx, te_idx in wb:
    print("TRAIN:", tr_idx, "(%s)" % w[tr_idx].sum(), "TEST:", te_idx, "(%s)" % w[te_idx].sum())
```

    WeightLimitedBootstrapSplit(13(38), n_iter=3, train_size=31, test_size=1, random_state=None)
    TRAIN: [12 12  0  6  7  1  9  8 10] (28) TEST: [] (0)
    TRAIN: [ 2  7  2 12  7 10 12  7  7] (29) TEST: [] (0)
    TRAIN: [ 4  7  4  0  5  9 12  2  7  6  2] (29) TEST: [] (0)


###Bootstrapping without splitting

The module also contains classes for bootstrapping and weight-limited bootstrapping without splitting the sample. The first is just plain sampling with replacement,

```python
from bootstrapsplit import Bootstrap

b = Bootstrap(len(pop), n_iter=3)
print(b)
print("POPULATION:", pop)
for s in b:
    print("BOOTSTRAP:", pop[s])
```

    Bootstrap(13, n_iter=3, random_state=None)
    POPULATION: ['A' 'B' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N']
    BOOTSTRAP: ['D' 'N' 'G' 'L' 'G' 'D' 'I' 'D' 'K' 'I' 'D' 'M' 'K']
    BOOTSTRAP: ['N' 'H' 'A' 'E' 'B' 'H' 'M' 'B' 'K' 'N' 'N' 'M' 'K']
    BOOTSTRAP: ['G' 'N' 'G' 'L' 'M' 'N' 'I' 'N' 'D' 'M' 'H' 'I' 'G']


the latter offers the option to set the maximum sample weight the same way `WeightLimitedBootstrapSplit` does.

```python
from bootstrapsplit import WeightLimitedBootstrap

w = np.random.randint(low=1, high=5, size=len(pop))
wb = WeightLimitedBootstrap(w, n_iter=3, max_weight=len(pop))
print(wb)
print("POPULATION:", pop, "(weight=%s)" % w.sum())
for s in wb:
    print("BOOTSTRAP:", pop[s], "(weight=%s)" % w[tr].sum())
```

    WeightLimitedBootstrap(13, n_iter=3, limit=13 random_state=None)
    POPULATION: ['A' 'B' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N'] (weight=32)
    BOOTSTRAP: ['K' 'H' 'I' 'K' 'B'] (weight=8)
    BOOTSTRAP: ['M' 'D' 'A' 'M' 'K'] (weight=8)
    BOOTSTRAP: ['B' 'G' 'D' 'N' 'E'] (weight=8)


##See Also
* [jackknife resampling](http://en.wikipedia.org/wiki/Jackknife_resampling "Jackknife resampling")
* [randomisation tests](https://www.uvm.edu/~dhowell/StatPages/Resampling/RandomizationTests.html "Randomisation tests")
* [exact tests](http://en.wikipedia.org/wiki/Exact_test "Exact tests")

## References
* Bradley Efron, and Robert J. Tibshirani. An introduction to the bootstrap, Chapman and Hall, New York, (1993)
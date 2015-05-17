# BootstrapSplit

#### What is bootstrapping?
> Bootstrapping is the practice of estimating properties of an estimator (such as its variance) by measuring those properties when sampling from an approximating distribution. One standard choice for an approximating distribution is the empirical distribution function of the observed data. In the case where a set of observations can be assumed to be from an independent and identically distributed population, this can be implemented by constructing a number of resamples with replacement, of the observed dataset (and of equal size to the observed dataset). -- [Wikipedia](http://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29 "Bootstrapping (Statistics)")

#### What is it used for?
> Bootstrapping allows assigning measures of accuracy (defined in terms of bias, variance, confidence intervals, prediction error or some other such measure) to sample estimates. -- [Effron & Tibshirani, (1993)](https://books.google.co.uk/books/about/An_Introduction_to_the_Bootstrap.html?id=gLlpIUxRntoC&hl=en "An Introduction to the Bootstrap")


```python
from bootstrapsplit import BootstrapSplit
bs = BootstrapSplit(9, random_state=0)
print(bs)
for train_index, test_index in bs:
    print("TRAIN:", train_index, "TEST:", test_index)
```
    >BootstrapSplit(9, n_iter=3, train_size=5, test_size=4, random_state=0)
    >('TRAIN:', array([1, 8, 7, 7, 8]), 'TEST:', array([0, 3, 0, 5]))
    >('TRAIN:', array([5, 4, 2, 4, 2]), 'TEST:', array([6, 7, 1, 0]))
    >('TRAIN:', array([4, 7, 0, 1, 1]), 'TEST:', array([5, 3, 6, 5]))


### Bootstrap of Weighted Samples

#### What is bootstrapping of weighted samples?

Bootstrapping of weighted samples performs bootstrapping in distributions where not all samples are of equal weight, and the size of the subsample is not measured by the number of samples in it, but by their aggregate weight. For example, sentences are made up of different number of tokens, giving them different weight in terms of token quantity.

The `BootstrapSplitWeighted` class re-implements the `BootstrapSplit` accounting for individual sample weight. In contrast to `BootstrapSplitWeighted` though it only sets a maximum weight for each sample split, which means that the returned sample is of the closest lower weight given a random resampling with replacement. This introduces a small degree of inaccuracy that needs to be taken in mind when working with very small samples.

```python
import numpy as np
from bootstrapsplit import BootstrapSplitWeighted
x = np.random.randint(low=1, high=10, size=20)
bw = BootstrapSplitWeighted(x, n_iter=3, train_size=0.8, test_size=0.2)
print bw
for train, test in bw:
    print "TRAIN:", train, "(%s)" % np.sum(x[train]), "TEST:", test, "(%s)" % np.sum(x[test])
```

    >BootstrapSplitWeighted(20(106), n_iter=3, train_size=85, test_size=21, random_state=None)
    >TRAIN: [12  8  4  0 15 15 11  2  4  4 19 13  0 16 12 15] (80) TEST: [5 3] (11)
    >TRAIN: [ 3  2  4  8  5 12  8  2  5  4  5  3  2] (78) TEST: [16  1 16  1 15 15] (16)
    >TRAIN: [11 13 11 17  2 16  2 13  5  6 18 14 15  7] (82) TEST: [ 0  0 19] (19)

If the weight of a sample split is smaller than that of the first token in the resampled sequence, an empty list is returned.

```python
x = np.random.randint(low=1, high=10, size=3)
bw = BootstrapSplitWeighted(x, n_iter=3, train_size=0.8, test_size=0.2)
print bw
for train, test in bw:
    print "TRAIN:", train, "(%s)" % np.sum(x[train]), "TEST:", test, "(%s)" % np.sum(x[test])
```

    >BootstrapSplitWeighted(3(15), n_iter=3, train_size=12, test_size=3, random_state=None)
    >TRAIN: [1] (8) TEST: [] (0)
    >TRAIN: [] (0) TEST: [] (0)
    >TRAIN: [] (0) TEST: [] (0)



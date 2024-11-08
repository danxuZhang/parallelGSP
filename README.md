# parallelGSP
Parallel Implementation of Generalized Sequential Pattern Algorithm

## Installation

``` bash
pip install git+https://github.com/danxuZhang/parallelGSP.git@v1.0
```

## Usage

``` python
sequences = [[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 4, 5]]
gsp = GSP(sequences, min_support=minsup, verbose=False)
_ = gsp.count_support([1, 2])
_ = gsp.find_freq_seq()
```

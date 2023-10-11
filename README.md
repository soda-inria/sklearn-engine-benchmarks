# sklearn-engine-benchmarks
Benchopt suite to compare alternative scikit-learn engines for the same estimators.

This project is an effort at a comprehensive and systematic benchmarking of
implementations of various machine learning algorithms, distributed in the libraries
that are considered to be interfaced as engines with the scikit-learn experimental
[plugin API](https://github.com/scikit-learn/scikit-learn/issues/22438).

It uses [benchopt](https://benchopt.github.io/) to organize, run, collect and compare
all benchmarks that are included in this repository.

## Scope

The main prospect is measuring and comparing the walltime of different solvers that
implement the exact same computations using different backends and running on different
hardware.

Benchmarks are currently available for the following algorithms:
- [k-means](https://github.com/soda-inria/sklearn-engine-benchmarks/tree/main/benchmarks/kmeans)
- [PCA](https://github.com/soda-inria/sklearn-engine-benchmarks/tree/main/benchmarks/pca)

Here is a (non-exhaustive) list of libraries that are compared in the benchmarks:
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [sklearn-numba-dpex](https://github.com/soda-inria/sklearn-numba-dpex)
- [sklearn-pytorch-engine](https://github.com/soda-inria/sklearn-pytorch-engine)
- [scikit-learn-intelex](https://intel.github.io/scikit-learn-intelex/)
- [cuML](https://docs.rapids.ai/api/cuml/stable/)
- [kmeans_dpcpp](https://github.com/oleksandr-pavlyk/kmeans_dpcpp/)

Benchmarks are compared accross all available devices (GPUs from all manufacturers,
iGPUs, CPUs) and for different sizes of input data.

## Browsing results

Results are stored as CSV files in the different benchmark folders of this repository,
but are also easily readable, sortable and filterable in a google spreadsheet available
[here](https://docs.google.com/spreadsheets/d/1te_3jY6vI4wo3-V7xbmWQai5Mdh5plWdLst2ox0wuLM/edit#gid=1392436075).

The google spreadsheet **is always up to date**, when changes are commited to a CSV file
and pushed to the main branch of this repository, a github workflow will automatically
publish the changes to the worksheet accordingly.

Please note that benchmark results should be mostly compared accross rows of the result
tables that share a same value in the `Benchmark id` column. You can use the filter
from the google spreadsheet to select only one particular `Benchmark id`. Rows with
different `Benchmark id` denote benchmark with objective that have different parameters
(e.g. dataset size, different algorithm optimization options,...). In some cases it can
however be relevant to compare values accross rows with different `Benchmark id`
values, for instance to read how the walltime of a given solver scales with the amount
of input data.

## Running the benchmarks

Please refer to benchopt documentation and tutorials that provide step-by-step guides
for running the benchmarks from a benchmark file tree, and refer to the documentation
of the dependencies of the solvers you're interested in running to gather prerequisite
installation instructions.

## Contributing

Please refer to benchopt documentation and tutorials that provide step-by-step guides
if you'd like to add a new implementation for an already existing benchmark.

A contribution should contain the following files:
- a benchopt solver in the `./solvers` folder
- additional rows for benchmark results added to the relevant `results.csv` file using
  the `consolidate_result_csv.py` python script available in the folder of the
  benchmark. (run `python ./consolidate_result_csv.py --help` for more information.)
- If the new candidate implementation can run on CPU, adding relevant instruction in
  the github worklow file that orchestrates test runs on CPUs.

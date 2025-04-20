Overview
========

PyRHE is a unified and efficient Python framework for genomics heritability estimation. 
It provides a modular and extensible platform for implementing various genetic architecture estimation models, while providing computation optimizations for large-scale genomic data.

Computational Efficiency
-----------------------

PyRHE integrates two core optimizations for computational efficiency:

Parallel Processing
~~~~~~~~~~~~~~~~~

- Jackknife subsamples are tiled and distributed across multiple workers
- Parallelizes genotype I/O and matrix computations
- Efficient memory management through shared memory arrays

Tensor Operations
~~~~~~~~~~~~~~~

- Large matrices automatically converted to tensors using PyTorch
- Optimized matrix multiplications for both CPU and CUDA-enabled GPU architectures

Unified Framework
----------------

Base Classes
~~~~~~~~~~~

The framework is built around two core abstract classes:

- ``Base``: Provides common infrastructure for all models
- ``StreamingBase``: Extends Base for memory-efficient processing

Key features of the base classes:

- Abstract interfaces for model-specific implementations
- Shared infrastructure for data processing and estimation
- Built-in support for jackknife resampling
- Efficient multiprocessing and memory management

Model Support and Extensibility
~~~~~~~~~~~~

The framework natively supports multiple models:

- `RHE <https://www.nature.com/articles/s41467-020-17576-9>`__
- `RHE-DOM <https://www.sciencedirect.com/science/article/pii/S0002929721001026>`__
- `GENIE <https://www.nature.com/articles/s41467-020-17576-9>`__

Each model shares common infrastructure while implementing model-specific components through well-defined interfaces. 
In addition, the framework is designed for easy extension to include new models.
# EfficientNNS

# Introduction

Computational fluid dynamics (CFD) has risen to become one of the most
essential tools in engineering design. However, significant progress is
still to be made in high-performance, low-latency post-processing
techniques. For this work, we concentrated on one specific aspect of
particle tracing techniques. Particle tracing techniques are important
for both processing and post-processing steps as both certain physics
models and visualization techniques require it. For the present work, we
focus on accelerating nearest neighbor searches which are required for
efficient, on-demand interpolation at arbitrary locations. We present
two validation domains and begin by assessing the performance of three
approaches. The fastest pair is selected for a more computationally
intensive benchmark simulating the order of magnitude of searches
required for a single particle tracking experiment. The three methods
include our naive approach which assumes a flat array (i.e. a tree with
depth 1) and two octree search approaches, depth-first and best-first
search. Best first search provides the best overall results improving
our existing algorithm by at least 3 orders of magnitude. Given the
fuzzy nature of the problem (we deal with floating point comparisons),
we require an algorithm capable of exploiting spatial locality and
provide a "good enough" solution in a minimal amount of time. The
best-first search algorithm provides this compromise on the types of
problems we deal with.

# Proposed Solution

We propose representing the mesh as an octree (or quad tree if a 2D
representation of the problem is possible). The octree would have a
branching factor of 8 and a fixed maximum depth depending on the largest
dimension, *d* = log<sub>2</sub>(*N*) where *N* is the largest dimension
of the mesh. Consequently, the overhead of this data structure would be
minimal in terms of memory footprint. In terms of algorithms, best-first
search and the depth-first algorithms will be explored. However, it is
likely that given the tree structure, fixed depth, fixed cost and fixed
branch factor that a greedy best-first search would yield optimal and
efficient results.

Currently, our best solution is linear search which at worst is an
*n*<sup>3</sup> algorithm.

# Platform

We will implement the proposed solution using Python 3.7. Given the
relatively small size of the domains, we will execute the benchmarks on
a local machine although the method is easily extensible to a
distributed environment. We leverage the Numpy extensively and avoid
complex indexing to ensure copy elision and reference semantics within
the octree. This should benefit the cache impact on the CPU. All
benchmarks are executed on a single CPU thread. The CPU used for this
test is an Intel 9980HK which is based on Intel’s Coffee Lake Refresh
with 16 MB of L3 cache. Lastly, the system has 64 GB’s of 2667 MHz DDR4
memory.

# Additional Details
## External Links
VIDEO: [Video Submission](https://youtu.be/7vhpBWoc-jA)
CODE: [Code Submission](https://github.com/ChristianLagares/EfficientNNS)

Datasets: [Datasets Submission](https://drive.google.com/drive/folders/1js7Fztupj2vYdCGmRYHZ1hAaV1gS60IX?usp=sharing)


## Dependencies:
    * python 3.7
    * h5py
    * pandas
    * numpy

## Final Remarks
The code as is expects the datasets within the active directory and to vary between the ZPG and complex case the first boolean found within the conditional block at the end of the module must be set. The remainder of the code is automated. Executing the CONCAVE branch is resource intensive and requires a large upfront memory pool (this could be optimized if needed) and total runtime on an Intel 9980HK is roughly 30 minutes.

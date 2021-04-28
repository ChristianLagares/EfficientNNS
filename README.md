# EfficientNNS
Computational fluid dynamics (CFD) has risen to become one of the most essential tools in engineering design. However, significant progress is still to be made in high-performance, low-latency post-processing techniques. For this work, we concentrated on one specific aspect of particle tracing techniques. Particle tracing techniques are important for both processing and post-processing steps as both certain physics models and visualization techniques require it. For the present work, we focus on accelerating nearest neighbor searches which are required for efficient, on-demand interpolation at arbitrary locations. We present two validation domains and begin by assessing the performance of three approaches. The fastest pair is selected for a more computationally intensive benchmark simulating the order of magnitude of searches required for a single particle tracking experiment. The three methods include our naive approach which assumes a flat array (i.e. a tree with depth 1) and two octree search approaches, depth-first and best-first search. Best first search provides the best overall results improving our existing algorithm by at least 3 orders of magnitude. Given the fuzzy nature of the problem (we deal with floating point comparisons), we require an algorithm capable of exploiting spatial locality and provide a "good enough" solution in a minimal amount of time. The best-first search algorithm provides this compromise on the types of problems we deal with.

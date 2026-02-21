# fast-linrec-finder

This repository contains a CUDA/C++ implementation of a **novel algorithm** (to the best of my knowledge) for finding the linear recurrence relation (with least possible order) of a real-valued, noisy 1D sequence that may satisfy it. It features a highly optimized, GPU-accelerated solver that exploits the properties of this problem to reduce the overall time complexity to $O(I \cdot L \log^2 L)$ (where $L$ is the dimension of the Hankel matrix constructed from the given sequence and $I$ is the number of LSQR iterations for solving linear least square as a subproblem).

## Purpose
Given a noisy sequence $S = (s_0, s_1, \dots, s_{2L-1})$, the goal is to find the minimum integer $k$ (also called the linear complexity) and the corresponding coefficients $x = (x_0, x_1, \dots, x_{k-1})$ such that any sequence element $s_{i}$ can be approximated by a linear combination of its $k$ previous terms:

$$ s_{k+i} \approx \sum_{j=0}^{k-1} s_{i+j} x_j $$



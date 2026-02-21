# fast-linrec-finder

This repository contains a CUDA/C++ implementation of a **novel algorithm** (to the best of my knowledge) for finding the linear recurrence relation (with least possible order) of a real-valued, noisy 1D sequence that may satisfy it. It features a highly optimized, GPU-accelerated solver that exploits the properties of this problem to reduce the overall time complexity to $O(I \cdot L \log^2 L)$ (where $L$ is the dimension of the Hankel matrix constructed from the given sequence and $I$ is the number of LSQR iterations for solving linear least square as a subproblem).

## Purpose
Given a noisy sequence $S = (s_0, s_1, \dots, s_{2L-1})$, the goal is to find the minimum integer $k$ (also called the linear complexity) and the corresponding coefficients $x = (x_0, x_1, \dots, x_{k-1})$ such that any sequence element $s_{i}$ can be approximated by a linear combination of its $k$ previous terms:

$$ s_{k+i} \approx \sum_{j=0}^{k-1} s_{i+j} x_j $$

## Mathematical Foundation

### The Hankel Matrix
The given sequence can conceptually be mapped to a square Hankel matrix $H$ of size $L \times L$:

$$ H = \begin{bmatrix} 
s_0 & s_1 & s_2 & \dots & s_{L-1} \\ 
s_1 & s_2 & s_3 & \dots & s_L \\ 
\vdots & \vdots & \vdots & \ddots & \vdots \\ 
s_{L-1} & s_L & s_{L+1} & \dots & s_{2L-2} 
\end{bmatrix} $$

Finding the linear complexity $k$ of the sequence is equivalent to finding the rank of $H$. In other words, we have to find the **first row** in $H$ that is linearly dependent on its preceding rows.

### Dependency Check via LSQR
In order to check if the $k$-th row is dependent on rows $0$ through $k-1$, we can formulate a linear least-squares problem as follows:

$$ A x \approx b $$

Where:
- $b$ is the $k$-th row (transposed as a column vector): $b = [s_k, s_{k+1}, \dots, s_{k+L-1}]^T$
- $A$ is an $L \times k$ matrix whose columns are the first $k$ rows of $H$ (transposed):

$$
A = \begin{bmatrix} 
s_0 & s_1 & \dots & s_{k-1} \\ 
s_1 & s_2 & \dots & s_k \\ 
\vdots & \vdots & \ddots & \vdots \\ 
s_{L-1} & s_L & \dots & s_{k+L-2} 
\end{bmatrix}
$$

We then solve $\min_x ||Ax - b||_2$. If the relative residual $\frac{||Ax - b||_2}{||b||_2}$ is below a defined threshold $\epsilon$, the row can be considered linearly dependent.



## Core Algorithm Steps

1. **Space Optimization (Copy Once):** The $O(L^2)$ matrix $H$ is never explicitly constructed in memory. The 1D sequence $S$ is copied to the GPU only once.
2. **Precompute FFT:** The sequence $S$ is zero-padded to $N$ (the next power of $2$ where $N \ge 2L$). Its Fourier transform, $\mathcal{F}(S)$, is computed once and cached on the device.
3. **Binary Search on Rows:** Perform a binary search over the row indices $k \in [1, L-1]$ to find the first dependent row (with minimal index).
4. **Iterative LSQR Setup:** For a chosen $k$, evaluate dependency using LSQR. Inside LSQR, all dense matrix-vector multiplications ($A x$ and $A^T y$) are replaced with $O(N \log N)$ FFT-based convolutions (actually cross-correlations to be more specific).


## Properties & Optimizations

### 1. Why Binary Search is Correct
In a Hankel matrix which represents a linear recurrence relation, if row $k$ is linearly dependent on its preceding rows, then row $k+1$ is guaranteed to be dependent on its preceding rows as well. Because this property guarantees a single transition point from independent to dependent, we only need to find the first linearly dependent row, which can be done via binary search. Here a binary search can correctly isolate the transition point (the linear complexity $k$) in $O(\log L)$ checks, bypassing the need for a linear $O(L)$ search over the rows.

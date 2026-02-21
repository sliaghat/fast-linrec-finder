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

$$ A = \begin{bmatrix} 
s_0 & s_1 & \dots & s_{k-1} \\ 
s_1 & s_2 & \dots & s_k \\ 
\vdots & \vdots & \ddots & \vdots \\ 
s_{L-1} & s_L & \dots & s_{k+L-2} 
\end{bmatrix} $$
We then solve $\min_x ||Ax - b||_2$. If the relative residual $\frac{||Ax - b||_2}{||b||_2}$ is below a defined threshold $\epsilon$, the row can be considered linearly dependent.

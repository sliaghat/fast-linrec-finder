#include "RecurrenceHankelSolver.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Macro for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


//-------------------------------------------------------------------
// CUDA Kernel: Pointwise Complex Multiplication with Conjugate
// Computes: C[i] = A[i] * conj(B[i])
// Used in cross-correlation via the following formula:
//   cross_correlation(a, b) = IFFT( FFT(a) * conj(FFT(b)) )
//-------------------------------------------------------------------
__global__ void complexMulConjKernel(const cufftDoubleComplex* A, const cufftDoubleComplex* B, cufftDoubleComplex* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        cufftDoubleComplex a = A[i];
        cufftDoubleComplex b = B[i];

        // (a.x + i a.y) * (b.x - i b.y) = (a.x*b.x + a.y*b.y) + i (a.y*b.x - a.x*b.y)
        cufftDoubleComplex res;
        res.x = a.x * b.x + a.y * b.y;
        res.y = a.y * b.x - a.x * b.y;
        C[i] = res;
    }
}

//-------------------------------------------------------------------
// Constructor & Destructor
//-------------------------------------------------------------------
RecurrenceHankelSolver::RecurrenceHankelSolver(const std::vector<double>& sequence, double threshold, int max_iter)
    : seq(sequence), tolerance(threshold) 
{
    // Hankel dimension
    L = (seq.size() + 1) / 2;
    if (L < 2) {
        throw std::invalid_argument("Sequence too short to form a valid Hankel matrix.");
    }

    // Configure LSQR Base parameters to handle Ill-conditioned matrices
    SetMaximumNumberOfIterations(max_iter);
    SetUpperLimitOnConditional(1e8);
    SetDamp(1e-4);
    SetToleranceA(1e-8);
    SetToleranceB(1e-8);
    SetStandardErrorEstimatesFlag(false);

    initGPU();
}

RecurrenceHankelSolver::~RecurrenceHankelSolver() {
    cleanupGPU();
}

//-------------------------------------------------------------------
// GPU Initialization
//-------------------------------------------------------------------
void RecurrenceHankelSolver::initGPU() {
    // Find next power of 2 which is greater than or equal to 2L
    N_FFT = 1;
    while (N_FFT < 2 * L) {
        N_FFT <<= 1;
    }

    int complex_elements = (N_FFT >> 1) + 1;

    // Create CUFFT plans
    if (cufftPlan1d(&planR2C, N_FFT, CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT Plan D2Z failed");
    }
    if (cufftPlan1d(&planC2R, N_FFT, CUFFT_Z2D, 1) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT Plan Z2D failed");
    }

    // Allocate Device Memory
    CUDA_CHECK(cudaMalloc(&d_real_buf, sizeof(double) * N_FFT));
    CUDA_CHECK(cudaMalloc(&d_comp_buf, sizeof(cufftDoubleComplex) * complex_elements));
    CUDA_CHECK(cudaMalloc(&d_S, sizeof(cufftDoubleComplex) * complex_elements));

    // Precompute the FFT of the sequence S ONCE
    CUDA_CHECK(cudaMemset(d_real_buf, 0, sizeof(double) * N_FFT));
    CUDA_CHECK(cudaMemcpy(d_real_buf, seq.data(), sizeof(double) * seq.size(), cudaMemcpyHostToDevice));

    if (cufftExecD2Z(planR2C, d_real_buf, d_S) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT Exec D2Z failed on sequence pre-computation");
    }
}

void RecurrenceHankelSolver::cleanupGPU() {
    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
    cudaFree(d_real_buf);
    cudaFree(d_comp_buf);
    cudaFree(d_S);
}

//-------------------------------------------------------------------
// Aprod1: y = y + A * x (Same as the cross-correlation of s and x)
//-------------------------------------------------------------------
void RecurrenceHankelSolver::Aprod1(unsigned int m, unsigned int n, const double * x, double * y) const {
    // n is k (current row index or the rank assumption), m is L
    CUDA_CHECK(cudaMemset(d_real_buf, 0, sizeof(double) * N_FFT));
    CUDA_CHECK(cudaMemcpy(d_real_buf, x, sizeof(double) * n, cudaMemcpyHostToDevice));

    // X = FFT(x)
    cufftExecD2Z(planR2C, d_real_buf, d_comp_buf);

    // Multiply: C = S * conj(X)
    int complex_elements = N_FFT / 2 + 1;
    int blockSize = 256;
    int numBlocks = (complex_elements + blockSize - 1) / blockSize;
    complexMulConjKernel<<<numBlocks, blockSize>>>(d_S, d_comp_buf, d_comp_buf, complex_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    // p = IFFT(C)
    cufftExecZ2D(planC2R, d_comp_buf, d_real_buf);

    // Bring strictly needed elements back to CPU and accumulate
    std::vector<double> h_temp(m);
    CUDA_CHECK(cudaMemcpy(h_temp.data(), d_real_buf, sizeof(double) * m, cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < m; i++) {
        y[i] += h_temp[i] / N_FFT; // IFFT normalization
    }
}

//-------------------------------------------------------------------
// Aprod2: x = x + A^T * y (Same as the cross-correlation of s and y)
//-------------------------------------------------------------------
void RecurrenceHankelSolver::Aprod2(unsigned int m, unsigned int n, double * x, const double * y) const {
    // n is k (current row index or the rank assumption), m is L
    CUDA_CHECK(cudaMemset(d_real_buf, 0, sizeof(double) * N_FFT));
    CUDA_CHECK(cudaMemcpy(d_real_buf, y, sizeof(double) * m, cudaMemcpyHostToDevice));

    // Y = FFT(y)
    cufftExecD2Z(planR2C, d_real_buf, d_comp_buf);

    // Multiply: C = S * conj(Y)
    int complex_elements = N_FFT / 2 + 1;
    int blockSize = 256;
    int numBlocks = (complex_elements + blockSize - 1) / blockSize;
    complexMulConjKernel<<<numBlocks, blockSize>>>(d_S, d_comp_buf, d_comp_buf, complex_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    // q = IFFT(C)
    cufftExecZ2D(planC2R, d_comp_buf, d_real_buf);

    // Bring strictly needed elements back to CPU and accumulate
    std::vector<double> h_temp(n);
    CUDA_CHECK(cudaMemcpy(h_temp.data(), d_real_buf, sizeof(double) * n, cudaMemcpyDeviceToHost));

    for (unsigned int j = 0; j < n; j++) {
        x[j] += h_temp[j] / N_FFT; // IFFT normalization
    }
}

// ----------------------------------------------------------------------------
// checkDependence
// Target: Checks if the kth row can be formed by a linear combo of rows 0 to k-1.
// ----------------------------------------------------------------------------
bool RecurrenceHankelSolver::checkDependence(int k, std::vector<double>& out_x) {
    // b is the target row (kth row of the Hankel matrix)
    std::vector<double> b(L);
    for (int i = 0; i < L; i++) {
        b[i] = seq[k + i];
    }

    out_x.assign(k, 0.0);

    // Execute LSQR: solve for coefficients mapping rows 0..k-1 -> row k
    // m = L (equations), n = k (unknowns)
    Solve(L, k, b.data(), out_x.data());

    // Validate Residual norm: || A*x - b ||_2 
    std::vector<double> Ax(L, 0.0);
    Aprod1(L, k, out_x.data(), Ax.data()); // Ax = A * x (accumulates into zeroes)

    double res_sq = 0.0;
    double b_sq = 0.0;
    for (int i = 0; i < L; i++) {
        double diff = Ax[i] - b[i];
        res_sq += diff * diff;
        b_sq += b[i] * b[i];
    }

    // Relative error thresholding
    double rel_err = std::sqrt(res_sq) / (std::sqrt(b_sq) + 1e-15);
    return rel_err < tolerance;
}

// ----------------------------------------------------------------------------
// FindLinearComplexity
// Performs binary search to find the minimum row index that is dependent.
// ----------------------------------------------------------------------------
int RecurrenceHankelSolver::FindLinearComplexity(std::vector<double>& out_coefficients) {
    int low = 1;
    int high = L - 1;
    int first_dependent = L; // defaults to full complexity
    std::vector<double> best_x;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        std::vector<double> x_out;
        bool dependent = checkDependence(mid, x_out);
        
        if (dependent) {
            // Row is dependent. Log it, but check if an upper row is also dependent
            first_dependent = mid;
            best_x = x_out;
            high = mid - 1;
        } else {
            // Row is independent, complexity must be strictly greater
            low = mid + 1;
        }
    }

    if (first_dependent < L) {
        out_coefficients = best_x;
    } else {
        out_coefficients.clear(); // No recurrence found within limits
    }
    
    return first_dependent;
}

#ifndef RECURRENCE_HANKEL_SOLVER_CUH
#define RECURRENCE_HANKEL_SOLVER_CUH

#include "lsqrBase.h"
#include <vector>
#include <cufft.h>

/**
 * @class RecurrenceHankelSolver
 * @brief Uses LSQR and cuFFT to find the linear recurrence relation of a noisy sequence.
 */
class RecurrenceHankelSolver : public lsqrBase {
public:
    /**
     * @param sequence Real-valued noisy input sequence. Must contain at least 2L-1 elements.
     * @param threshold The relative residual error threshold to consider a row dependent.
     * @param max_iter Maximum iterations for each LSQR solve call.
     */
    RecurrenceHankelSolver(const std::vector<double>& sequence, double threshold = 1e-3, int max_iter = 1000);
    
    virtual ~RecurrenceHankelSolver();

    // FFT-based implementations for lsqrBase (Matrix-Vector products)
    virtual void Aprod1(unsigned int m, unsigned int n, const double * x, double * y) const override;
    virtual void Aprod2(unsigned int m, unsigned int n, double * x, const double * y) const override;

    /**
     * @brief Performs binary search to find the matrix rank (linear complexity).
     * @param out_coefficients Will hold the recurrence coefficients if a dependence is found.
     * @return The linear complexity of the sequence.
     */
    int FindLinearComplexity(std::vector<double>& out_coefficients);

private:
    void initGPU();
    void cleanupGPU();
    bool checkDependence(int k, std::vector<double>& out_x);

    std::vector<double> seq;
    int L;               // Dimension of the square Hankel matrix
    int N_FFT;           // Padded length for FFT (next power of 2)
    double tolerance;

    // GPU resources. 
    mutable cufftHandle planR2C;
    mutable cufftHandle planC2R;
    
    mutable double* d_real_buf;
    mutable cufftDoubleComplex* d_comp_buf;
    mutable cufftDoubleComplex* d_S;
};

#endif // RECURRENCE_HANKEL_SOLVER_CUH

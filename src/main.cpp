#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "RecurrenceHankelSolver.cuh"


void printSequence(const std::vector<double>& seq, size_t lim = 15) {
    for (size_t i = 0; i < std::min(seq.size(), lim); i++) {
        std::cout << seq[i] << " ";
    }
    if (seq.size() > lim) std::cout << "... ";
    std::cout << std::endl;
}

int main() {
    try {
        std::cout << "==============================" << std::endl;
        std::cout << "=  Linear Recurrence Finder  =" << std::endl;
        std::cout << "==============================" << std::endl;

        // 1. Generate Test Sequence: s[n] = 2*s[n - 1] - 2*s[n - 2] + s[n - 3]
        // This sequence has a linear complexity of 3.
        std::vector<double> seq;
        int length = 50000; // Must be at least 2*L for the solver
        
        std::cout << "Generating test sequence" << std::endl;

        srand(time(0));
        seq.push_back(1.0);
        seq.push_back(0.0);
        seq.push_back(1.0);
        for (int i = 3; i < length; i++) {
            double element = 2 * seq[i - 1] - 2 * seq[i - 2] + seq[i - 3];
            // Add a tiny amount of noise to test robustness
            double noise = (double)rand() * 0.00001;
            seq.push_back(element + noise);
        }


        std::cout << "Input Sequence: ";
        printSequence(seq);

        // 2. Configuration
        // Threshold: 1e-4 is good for this example 
        double threshold = 1e-4; 
        int max_lsqr_iter = 500;

        // 3. Instantiate Solver
        std::cout << "\nInitializing GPU Solver..." << std::endl;
        RecurrenceHankelSolver solver(seq, threshold, max_lsqr_iter);

        // 4. Solve for Linear Complexity
        std::cout << "Starting Binary Search for Linear Complexity..." << std::endl;
        
        std::vector<double> coefficients;
        int rank = solver.FindLinearComplexity(coefficients);

        // 5. Output Results
        if (coefficients.empty()) {
            std::cout << "\nRESULT: No Linear Recurrence Relation Found." << std::endl;
        } else {
            std::cout << "\nRESULT: Linear Recurrence Relation Found!" << std::endl;
            std::cout << "---------------------------------" << std::endl;
            std::cout << "Estimated Linear Complexity (Rank): " << rank << std::endl;
            
            std::cout << "Coefficients Found (x): ";
            std::cout << std::fixed << std::setprecision(4);
            for(double c : coefficients) std::cout << c << " ";
            std::cout << std::endl;

            // Interpret the result:
            // The solver finds x such that: Row_k = sum(x[i] * Row_i) for i=0..k-1
            // This implies: s[n] = x[0]*s[n-k] + x[1]*s[n-(k-1)] + ... + x[k-1]*s[n-1]
            std::cout << "\nLinear Recurrence Relation Formula:" << std::endl;
            std::cout << "s[n] = ";
            for (int i = 1; i <= rank; i++) {
                double coef = coefficients[rank - i];
                // Only print significant terms
                if (std::abs(coef) > 1e-3) {
                    std::cout << "(" << coef << " * s[n-" << i << "]) ";
                    if (i < rank) std::cout << "+ ";
                }
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

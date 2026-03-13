/**
 * @file BatchMandelCalculator.h
 * @author Samuel Kuchta <xkucht11@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 20.10.2024
 */

#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>
#include <cstring>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int * calculateMandelbrot();

private:
    void mandelbrot(const int i, const int j, const int global_j, const int iteration, const float Im0) const;

    int *data;
    float *intermediate_results_real;
    float *intermediate_results_imaginary;
    bool *pointEscaped;
    float *Re0_mem;
    static constexpr int blockSize = 64;
};

#endif



/**
 * @file LineMandelCalculator.h
 * @author Samuel Kuchta <xkucht11@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 20.10.2024
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();
    


private:
    void mandelbrot(const int i, const int j, const int iteration, const float Im0) const;

    int *data;
    float *intermediate_results_real;
    float *intermediate_results_imaginary;
    bool *pointEscaped;
    float *Re0_mem;
};

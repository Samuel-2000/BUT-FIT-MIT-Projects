/**
 * @file LineMandelCalculator.cc
 * @author Samuel Kuchta <xkucht11@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 20.10.2024
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>  // For OpenMP
#include <stdlib.h>
#include "LineMandelCalculator.h"

// Constructor: Allocate memory and initialize the Mandelbrot calculator
LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize, unsigned limit) : BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator") {
	// Allocate memory for the Mandelbrot set (height * width)

	data = (int *) aligned_alloc(64, height * width * sizeof(int));
	if (data == NULL) {
			std::cerr << "Memory allocation failed!" << std::endl;
			exit(1);
	}
	intermediate_results_real = (float *) aligned_alloc(64, width * sizeof(float));
	if (data == NULL) {
			std::cerr << "Memory allocation failed!" << std::endl;
			exit(1);
	}
	intermediate_results_imaginary = (float *) aligned_alloc(64, width * sizeof(float));
	if (data == NULL) {
			std::cerr << "Memory allocation failed!" << std::endl;
			exit(1);
	}

	pointEscaped = (bool *) aligned_alloc(64, width * sizeof(bool));
	if (data == NULL) {
			std::cerr << "Memory allocation failed!" << std::endl;
			exit(1);
	}

	Re0_mem = (float *) aligned_alloc(64, width * sizeof(float));
	if (data == NULL) {
			std::cerr << "Memory allocation failed!" << std::endl;
			exit(1);
	}

	// Prefill memory with initial values (we can initialize them to zero)
	std::fill_n(data, height * width, limit);
}

// Destructor: Free allocated memory
LineMandelCalculator::~LineMandelCalculator() {
	free(data);
	data = NULL;

	free(intermediate_results_real);
	intermediate_results_real = NULL;
	
	free(intermediate_results_imaginary);
	intermediate_results_imaginary = NULL;

	free(pointEscaped);
	pointEscaped = NULL;

	free(Re0_mem);
	Re0_mem = NULL;
}

/* 
// nondivergent variant (~2x slower)
#pragma omp declare simd notinbranch linear(i, j) uniform(this, iteration, Im0)
void LineMandelCalculator::mandelbrot(const int i, const int j, const int iteration, const float Im0) const {
    float Re = intermediate_results_real[j];
    float Im = intermediate_results_imaginary[j];

    float Re2 = Re * Re;
    float Im2 = Im * Im;
    bool escaped = (Re2 + Im2 > 4.0f) && !pointEscaped[j];

    pointEscaped[j] = escaped ? true : pointEscaped[j];
    data[i * width + j] = escaped ? iteration : data[i * width + j];
    data[(height - 1 - i) * width + j] = escaped ? iteration : data[(height - 1 - i) * width + j];

    // Update values unconditionally
    intermediate_results_real[j] = Re2 - Im2 + Re0_mem[j];
    intermediate_results_imaginary[j] = 2.0f * Re * Im + Im0;
}
*/

#pragma omp declare simd notinbranch linear(i, j) uniform(this, iteration, Im0)
void LineMandelCalculator::mandelbrot(const int i, const int j, const int iteration, const float Im0) const {
	if (pointEscaped[j] == false) {
		float Re = intermediate_results_real[j];
		float Im = intermediate_results_imaginary[j];

		float Re2 = Re * Re;
		float Im2 = Im * Im;

		if (Re2 + Im2 > 4.0f) {
			pointEscaped[j] = true;
			data[i * width + j] = iteration;
			data[(height - 1 - i) * width + j] = iteration;  // Mirror the symmetry
		}
		
		intermediate_results_real[j] = Re2 - Im2 + Re0_mem[j];
		intermediate_results_imaginary[j] = 2.0f * Re * Im + Im0;
	}
}

int* LineMandelCalculator::calculateMandelbrot() {
	int halfHeight = height / 2; // Only calculate half of the matrix due to symmetry

	for (int i = 0; i < halfHeight; i++) {
		float Im0 = y_start + i * dy;

		#pragma omp simd aligned(intermediate_results_real: 64) aligned(intermediate_results_imaginary: 64) aligned(Re0_mem: 64) aligned(pointEscaped: 64)
		for (int j = 0; j < width; j++) {
			intermediate_results_imaginary[j] = Im0;
			float re0 = x_start + j * dx;
			intermediate_results_real[j] = re0;
			Re0_mem[j] = re0;
			pointEscaped[j] = false;
		}

		for (int iteration = 0; iteration < limit; iteration++) {
			#pragma omp simd aligned(intermediate_results_real: 64) aligned(intermediate_results_imaginary: 64) aligned(Re0_mem: 64) aligned(pointEscaped: 64) aligned(data: 64)
			for (int j = 0; j < width; j++) {
				mandelbrot(i, j, iteration, Im0);
			}
			
			bool pointEscapedAll = true;
			#pragma omp simd reduction(&:pointEscapedAll) aligned(pointEscaped: 64)
			for (int j = 0; j < width; j++) {
				pointEscapedAll &= pointEscaped[j];
			}
			
			if (pointEscapedAll) {
				break;
			}
		}
	}

	return data;
}


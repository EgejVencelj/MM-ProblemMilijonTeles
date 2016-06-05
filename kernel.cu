#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define	SIZE	(1024)
#define	G		(6.67408e-11f)


__global__ void vectorAdd(float *a, float *b, float *c, int n) {
	int i = threadIdx.x;

	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

__global__ void vectorSubtract(float *a, float *b, float *c, int n) {
	int i = threadIdx.x;

	if (i < n) {
		c[i] = a[i] - b[i];
	}
}

void vectorMultiply(float k, float *a, float *b, int n);

float distanceSquared(float *a, int n);

void matrixCopy(float **a, float **b);

__global__ void matrixAdd(float **a, float **b, float **c, int n, int m) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	c[i][j] = a[i][j] + b[i][j];
}

__global__ void matrixSubtract(float **a, float **b, float **c, int n, int m) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	c[i][j] = a[i][j] - b[i][j];
}

__global__ void matrixMultiply(float k, float **a, float **b, int n, int m) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	a[i][j] = k*a[i][j];
}

// a needs to be a zero matrix before start
void gravity(float **x, float **a, float *mass, int n, int m) {
	float *r;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i != j) {
				vectorSubtract(x[j], x[i], r, n);
				float d = distanceSquared(r, n);
				vectorMultiply(mass[j] / (d*sqrt(d)), r, r, m);
				vectorAdd(a[i], r, a[i], m);
			}
		}
	}
	matrixMultiply(G, a, a, n, m);
}

int main() {
	/*
		1D: cudaMalloc
		2D: cudaMallocPitch
		3D: cudaMalloc3D

		shared memory is faster than local or global
		it's okay to do non-parallel tasks on GPU if it means less transfering
		
		cudaMalloc vs cuMemAlloc ?
		multiply and add in one instruction ?
		math_functions.h device_functions.h

		functions:
			__device__
			executed on device, called from device

			__global__
			"kernel", executed on device, called from host (or device if 3.x), return void, execution configuration, asynchronous

			__host__
			default, executed on host, called from host, can be combined with __device__ for both

			__noinline, __forceinline__

		gridDim, blockIdx, blockDim, threadIdx, warpSize

	*/

	float dt;
	int n; // object count
	int m; // dimension count

	/*
		total memory usage (not counting function locals)
		
		11 * n * m * size + 1 * n * size
		n * size * (11*m + 1)

		for n = 1e6, m = 3, size = 4 (float)

		136MB
	*/
	float **x;
	float **v;
	float *mass;

	float **xWithOffset;

	float **k1x;
	float **k1v;

	float **k2x;
	float **k2v;

	float **k3x;
	float **k3v;

	float **k4x;
	float **k4v;

	while (true) {
		// k1x = v
		matrixCopy(v, k1x);
		// k1v = g(x, mass)
		gravity(x, k1v, mass, n, m);

		// k2x = v + k1v*dt/2
		matrixMultiply(dt / 2, k1x, k2x, n, m);
		matrixAdd(v, k2x, k2x, n, m);
		// k2v = g(x + k1x*dt/2, mass)
		matrixMultiply(dt / 2, k1x, xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);
		gravity(xWithOffset, k2v, mass, n, m);

		// k3x = v + k2v*dt/2
		matrixMultiply(dt / 2, k2x, k3x, n, m);
		matrixAdd(v, k3x, k3x, n, m);
		// k3v = g(x + k2x*dt/2, mass)
		matrixMultiply(dt / 2, k2x, xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);
		gravity(xWithOffset, k3v, mass, n, m);

		// k4x = v + k3v*dt
		matrixMultiply(dt, k3x, k4x, n, m);
		matrixAdd(v, k4x, k4x, n, m);
		// k4v = g(x + k3x*dt, mass)
		matrixMultiply(dt, k3x, xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);
		gravity(xWithOffset, k4v, mass, n, m);

		// x += (k1x + 2*k2x + 2*k3x + k4x)*dt/6;
		matrixMultiply(dt / 6, k1x, k1x, n, m);
		matrixMultiply(dt / 3, k2x, k2x, n, m);
		matrixMultiply(dt / 3, k3x, k3x, n, m);
		matrixMultiply(dt / 6, k4x, k4x, n, m);

		// k1x will hold the sum
		matrixAdd(k1x, k2x, k1x, n, m);
		matrixAdd(k1x, k3x, k1x, n, m);
		matrixAdd(k1x, k4x, k1x, n, m);

		matrixAdd(x, k1x, x, n, m);


		// v += (k1v + 2*k2v + 2*k3v + k4v)*dt/6;
		matrixMultiply(dt / 6, k1v, k1v, n, m);
		matrixMultiply(dt / 3, k2v, k2v, n, m);
		matrixMultiply(dt / 3, k3v, k3v, n, m);
		matrixMultiply(dt / 6, k4v, k4v, n, m);

		// k1v will hold the sum
		matrixAdd(k1v, k2v, k1v, n, m);
		matrixAdd(k1v, k3v, k1v, n, m);
		matrixAdd(k1v, k4v, k1v, n, m);

		matrixAdd(v, k1v, v, n, m);
	}

	return 0;
}
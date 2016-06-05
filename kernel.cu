#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define	BLOCKS          (1)
#define THREADS         (1024)

#define VALS_PER_THREAD (8)

#define	G		        (6.67408e-11f)


__host__ __device__ int ix(int i, int j, int n) { // __forceinline__ ?
	return i*n + j;
}

__device__ void vectorAdd(float *a, float *b, float *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		c[i] = a[i] + b[i];
}

__device__ void vectorSubtract(float *a, float *b, float *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		c[i] = a[i] - b[i];
}

__device__ void vectorSubtract(int ix1, int ix2, float *a, float *b, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m)
		b[ix(i, ix2, n)] = a[ix(i, ix1, n)] - a[ix(i, ix2, n)];
}

__device__ void vectorAdd(float *r, int ix1, float *a, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m)
		a[ix(i, ix1, n)] = a[ix(i, ix1, n)] + r[i];
}

__device__ void vectorMultiply(float k, float *a, float *b, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		b[i] = k * a[i];
}

__device__ float vectorLengthSquared(int ix1, float *a, int n, int m) {
	float result = 0;
	for (int i = 0; i < m; i++) {
		result += a[ix(i, ix1, n)] * a[ix(i, ix1, n)];
	}

	return result;
}


__device__ void matrixCopy(float *a, float *b, int n, int m) { // cudaMemcpy ?
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int len = n*m;
	if (i < len)
		b[i] = a[i];
}

__device__ void matrixAdd(float *a, float *b, float *c, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int len = n * m;
	if (i < len)
		c[i] = a[i] + b[i];
}

__device__ void matrixSubtract(float *a, float *b, float *c, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int len = n*m;
	if (i < len)
		c[i] = a[i] + b[i];
}

__device__ void matrixMultiply(float k, float *a, float *b, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int len = n*m;
	if (i < len)
		b[i] = k * a[i];
}

// TODO: add loops to all functions
__device__ void gravity(float *x, float *a, float *mass, float *r, int n, int m) {
	//cudaMemset(a, 0, n * m * sizeof(float));
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//float *r;
	//cudaMalloc(&r, m * sizeof(float));

	//for (int i = 0; i < n; i++) {
	if (i < n) {
		for (int j = 0; j < m; j++) {
			a[ix(i, j, n)] = 0;
		}
		for (int j = 0; j < n; j++) {
			if (i != j) {
				vectorSubtract(j, i, x, r, n, m);
				float d = vectorLengthSquared(i, r, n, m);
				vectorMultiply(mass[j] / (d*sqrtf(d)), r, r, m);
				vectorAdd(r, i, a, n, m);
			}
		}
	}
	//matrixMultiply(G, a, a, n, m);

	//cudaFree(r);
}

__constant__ float *mass;

__global__ void mainLoop(float *x, float *v, float *mass, float *kx[4], float *kv[4], float *xWithOffset, float *r, float dt, int n, int m) {
	// add more __syncthreads() ? remove some ?

	while (true) {
		// k1x = v
		matrixCopy(v, kx[0], n, m);

		__syncthreads();
		// k1v = g(x, mass)
		gravity(x, kv[0], mass, r, n, m);
		__syncthreads();

		// k2x = v + k1v*dt/2
		matrixMultiply(dt / 2, kx[0], kx[1], n, m);
		matrixAdd(v, kx[1], kx[1], n, m);
		// k2v = g(x + k1x*dt/2, mass)
		matrixMultiply(dt / 2, kx[0], xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);

		__syncthreads();
		gravity(xWithOffset, kv[1], mass, r, n, m);
		__syncthreads();

		// k3x = v + k2v*dt/2
		matrixMultiply(dt / 2, kx[1], kx[2], n, m);
		matrixAdd(v, kx[2], kx[2], n, m);
		// k3v = g(x + k2x*dt/2, mass)
		matrixMultiply(dt / 2, kx[1], xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);

		__syncthreads();
		gravity(xWithOffset, kv[2], mass, r, n, m);
		__syncthreads();

		// k4x = v + k3v*dt
		matrixMultiply(dt, kx[2], kx[3], n, m);
		matrixAdd(v, kx[3], kx[3], n, m);
		// k4v = g(x + k3x*dt, mass)
		matrixMultiply(dt, kx[2], xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);
		__syncthreads();
		gravity(xWithOffset, kv[3], mass, r, n, m);
		__syncthreads();

		// x += (k1x + 2*k2x + 2*k3x + k4x)*dt/6;
		matrixMultiply(dt / 6, kx[0], kx[0], n, m);
		matrixMultiply(dt / 3, kx[1], kx[1], n, m);
		matrixMultiply(dt / 3, kx[2], kx[2], n, m);
		matrixMultiply(dt / 6, kx[3], kx[3], n, m);

		// k1x will hold the sum
		matrixAdd(kx[0], kx[1], kx[0], n, m);
		matrixAdd(kx[0], kx[2], kx[0], n, m);
		matrixAdd(kx[0], kx[3], kx[0], n, m);

		matrixAdd(x, kx[0], x, n, m);


		// v += (k1v + 2*k2v + 2*k3v + k4v)*dt/6;
		matrixMultiply(dt / 6, kv[0], kv[0], n, m);
		matrixMultiply(dt / 3, kv[1], kv[1], n, m);
		matrixMultiply(dt / 3, kv[2], kv[2], n, m);
		matrixMultiply(dt / 6, kv[3], kv[3], n, m);

		// k1v will hold the sum
		matrixAdd(kv[0], kv[1], kv[0], n, m);
		matrixAdd(kv[0], kv[2], kv[0], n, m);
		matrixAdd(kv[0], kv[3], kv[0], n, m);

		matrixAdd(v, kv[0], v, n, m);
	}


}

int main() {
	/*
	shared memory is faster than local or global
	it's okay to do non-parallel tasks on GPU if it means less transfering
	
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

	try managed memory ?
	*/

	float dt;
	int n; // object count
	int m; // dimension count

	float *x;
	cudaMalloc(&x, n*m*sizeof(float));
	float *v;
	cudaMalloc(&v, n*m*sizeof(float));

	cudaMalloc(&mass, n*sizeof(float));

	float *r; // additional memory for computation
	cudaMalloc(&x, n*m*sizeof(float));

	float *xWithOffset;
	cudaMalloc(&xWithOffset, n*m*sizeof(float));

	float *kx[4];
	float *kv[4];

	for (int i = 0; i < 4; i++) {
		cudaMalloc(&kx[i], n*m*sizeof(float));
		cudaMalloc(&kv[i], n*m*sizeof(float));
	}


	mainLoop<<< BLOCKS, THREADS >>>(x, v, mass, kx, kv, xWithOffset, r, dt, n, m);


	for (int i = 0; i < 4; i++) {
		cudaFree(&kx[i]);
		cudaFree(&kv[i]);
	}

	cudaFree(xWithOffset);

	return 0;
}
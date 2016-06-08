#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>

#define _GNU_SOURCE

#define	BLOCKS          (256)
#define THREADS         (1024)

#define VALS_PER_THREAD (8)

#define	G		        (6.67408e-11f)




#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SCREEN_WIDTH 1820
#define SCREEN_HEIGHT 920

float dt;
int n; // object count
int m; // dimension count

float *x;
float *v;
float *d_x;
float *d_v;
float minCoord[2] = { 0.0, 0.0 };
float maxCoord[2] = { 0.0, 0.0 };

float *r; // additional memory for computation
float *xWithOffset;
float *kx[4];
float *kv[4];

float *mass;

float *k1x;
float *k2x;
float *k3x;
float *k4x;

float *k1v;
float *k2v;
float *k3v;
float *k4v;

/* This structure is used by main to communicate with parse_opt. */
struct arguments
{
	char *args[2];            /* ARG1 and ARG2 */
	int verbose;              /* The -v flag */
	char *outfile;            /* Argument for -o */
	char *string1, *string2;  /* Arguments for -a and -b */
};


int dataID = 0;
int runOnCPU = 0;
int visual = 1;
int bench = 0;

__constant__ float *d_mass;

__host__ __device__ int ix(int i, int j, int n) { // __forceinline__ ?
	return i*n + j;
}

__host__ __device__ void vectorSubtract(int ix1, int ix2, float *a, float *b, int n, int m) {
	for (int i = 0; i < m; i++) {
		b[ix(i, ix2, n)] = a[ix(i, ix1, n)] - a[ix(i, ix2, n)];
	}
}

__host__ __device__ void vectorAdd(float *r, int ix1, float *a, int n, int m) {
	for (int i = 0; i < m; i++) {
		a[ix(i, ix1, n)] = a[ix(i, ix1, n)] + r[ix(i, ix1, n)];
	}
}

__host__ __device__ void vectorMultiply(float k, int ix1, float *a, int n, int m) {
	for (int i = 0; i < m; i++)
		a[ix(i, ix1, n)] = k * a[ix(i, ix1, n)];
}

__host__ __device__ float vectorLengthSquared(int ix1, float *a, int n, int m) {
	float result = 0;
	for (int i = 0; i < m; i++) {
		result += a[ix(i, ix1, n)] * a[ix(i, ix1, n)];
	}

	return result;
}

__device__ void matrixCopy(float *a, float *b, int n, int m) { // cudaMemcpy ?
    int i = blockIdx.x * blockDim.x + threadIdx.x;
 
    int len = n*m;
    //if (i < len)
    while (i < len) {
        b[i] = a[i];
        i += blockDim.x * gridDim.x;
    }
}

void matrixCopyCPU(float *a, float *b, int n, int m) { //CPU matrixCopy
	int len = n*m;
	for (size_t i = 0; i < len; i++)
	{
		b[i] = a[i];
	}
}

__device__ void matrixAdd(float *a, float *b, float *c, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int len = n * m;
	//if (i < len
	while (i < len) {
		c[i] = a[i] + b[i];
		i += blockDim.x * gridDim.x;
	}
}

void matrixAddCPU(float *a, float *b, float *c, int n, int m) {
	int len = n * m;
	for (size_t i = 0; i < len; i++)
	{
		c[i] = a[i] + b[i];
	}
}

__device__ void matrixMultiply(float k, float *a, float *b, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int len = n*m;
	//if (i < len)
	while (i < len) {
		b[i] = k * a[i];
		i += blockDim.x * gridDim.x;
	}
}

void matrixMultiplyCPU(float k, float *a, float *b, int n, int m) {
	int len = n*m;
	for (size_t i = 0; i < len; i++)
	{
		b[i] = k * a[i];
	}
}

// TODO: add loops to all functions
__device__ void gravity(float *x, float *a, float *mass, float *r, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//if (i < n) {
	while (i < n) {
		for (int j = 0; j < m; j++) {
			a[ix(j, i, n)] = 0;
		}
		for (int j = 0; j < n; j++) {
			if (i != j) {
				vectorSubtract(j, i, x, r, n, m);
				float d = vectorLengthSquared(i, r, n, m);
				vectorMultiply(G * mass[j] / (d*sqrtf(d)), i, r, n, m);
				vectorAdd(r, i, a, n, m);
			}
		}
		i += blockDim.x * gridDim.x;
	}
}

void gravityCPU(float *x, float *a, float *mass, float *r, int n, int m) {
	for (size_t i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++) {
			a[ix(j, i, n)] = 0;
		}
		for (int j = 0; j < n; j++) {
			if (i != j) {
				vectorSubtract(j, i, x, r, n, m);
				float d = vectorLengthSquared(i, r, n, m);
				vectorMultiply(G * mass[j] / (d*sqrtf(d)), i, r, n, m);
				vectorAdd(r, i, a, n, m);
			}
		}
	}
}



__global__ void mainLoop(float *x, float *v, float *mass, float *kx[4], float *kv[4], float *xWithOffset, float *r, float dt, int n, int m) {
	// add more __syncthreads() ? remove some ?
	while (true) {
		// k1x = v
		matrixCopy(v, kx[0], n, m);
		// k1v = g(x, mass)
		gravity(x, kv[0], mass, r, n, m);

		// k2x = v + k1v*dt/2
		matrixMultiply(dt / 2, kv[0], kx[1], n, m);
		matrixAdd(v, kx[1], kx[1], n, m);
		// k2v = g(x + k1x*dt/2, mass)
		matrixMultiply(dt / 2, kx[0], xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);
		gravity(xWithOffset, kv[1], mass, r, n, m);

		// k3x = v + k2v*dt/2
		matrixMultiply(dt / 2, kv[1], kx[2], n, m);
		matrixAdd(v, kx[2], kx[2], n, m);
		// k3v = g(x + k2x*dt/2, mass)
		matrixMultiply(dt / 2, kx[1], xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);
		gravity(xWithOffset, kv[2], mass, r, n, m);

		// k4x = v + k3v*dt
		matrixMultiply(dt, kv[2], kx[3], n, m);
		matrixAdd(v, kx[3], kx[3], n, m);
		// k4v = g(x + k3x*dt, mass)
		matrixMultiply(dt, kx[2], xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);
		gravity(xWithOffset, kv[3], mass, r, n, m);

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

void stepRKCPU(float *x, float *v, float *mass, float *kx[4], float *kv[4], float *xWithOffset, float *r, float dt, int n, int m) {
	// k1x = v
	matrixCopyCPU(v, kx[0], n, m);
	// k1v = g(x, mass)
	gravityCPU(x, kv[0], mass, r, n, m);

	// k2x = v + k1v*dt/2
	matrixMultiplyCPU(dt / 2, kv[0], kx[1], n, m);
	matrixAddCPU(v, kx[1], kx[1], n, m);
	// k2v = g(x + k1x*dt/2, mass)
	matrixMultiplyCPU(dt / 2, kx[0], xWithOffset, n, m);
	matrixAddCPU(xWithOffset, x, xWithOffset, n, m);
	gravityCPU(xWithOffset, kv[1], mass, r, n, m);

	// k3x = v + k2v*dt/2
	matrixMultiplyCPU(dt / 2, kv[1], kx[2], n, m);
	matrixAddCPU(v, kx[2], kx[2], n, m);
	// k3v = g(x + k2x*dt/2, mass)
	matrixMultiplyCPU(dt / 2, kx[1], xWithOffset, n, m);
	matrixAddCPU(xWithOffset, x, xWithOffset, n, m);
	gravityCPU(xWithOffset, kv[2], mass, r, n, m);

	// k4x = v + k3v*dt
	matrixMultiplyCPU(dt, kv[2], kx[3], n, m);
	matrixAddCPU(v, kx[3], kx[3], n, m);
	// k4v = g(x + k3x*dt, mass)
	matrixMultiplyCPU(dt, kx[2], xWithOffset, n, m);
	matrixAddCPU(xWithOffset, x, xWithOffset, n, m);
	gravityCPU(xWithOffset, kv[3], mass, r, n, m);

	// x += (k1x + 2*k2x + 2*k3x + k4x)*dt/6;
	matrixMultiplyCPU(dt / 6, kx[0], kx[0], n, m);
	matrixMultiplyCPU(dt / 3, kx[1], kx[1], n, m);
	matrixMultiplyCPU(dt / 3, kx[2], kx[2], n, m);
	matrixMultiplyCPU(dt / 6, kx[3], kx[3], n, m);

	// k1x will hold the sum
	matrixAddCPU(kx[0], kx[1], kx[0], n, m);
	matrixAddCPU(kx[0], kx[2], kx[0], n, m);
	matrixAddCPU(kx[0], kx[3], kx[0], n, m);

	matrixAddCPU(x, kx[0], x, n, m);


	// v += (k1v + 2*k2v + 2*k3v + k4v)*dt/6;
	matrixMultiplyCPU(dt / 6, kv[0], kv[0], n, m);
	matrixMultiplyCPU(dt / 3, kv[1], kv[1], n, m);
	matrixMultiplyCPU(dt / 3, kv[2], kv[2], n, m);
	matrixMultiplyCPU(dt / 6, kv[3], kv[3], n, m);

	// k1v will hold the sum
	matrixAddCPU(kv[0], kv[1], kv[0], n, m);
	matrixAddCPU(kv[0], kv[2], kv[0], n, m);
	matrixAddCPU(kv[0], kv[3], kv[0], n, m);

	matrixAddCPU(v, kv[0], v, n, m);
}

__global__ void go(float *x, float *v, float *mass, float *xWithOffset, float *r, float dt, int n, int m,
	float *k1x, float *k2x, float *k3x, float *k4x, float *k1v, float *k2v, float *k3v, float *k4v) {
	// k1x = v
	matrixCopy(v, k1x, n, m);

	// k1v = g(x, mass)
	__syncthreads();
	gravity(x, k1v, mass, r, n, m);
	__syncthreads();

	// k2x = v + k1v*dt/2
	matrixMultiply(dt / 2, k1v, k2x, n, m);
	matrixAdd(v, k2x, k2x, n, m);
	// k2v = g(x + k1x*dt/2, mass)
	matrixMultiply(dt / 2, k1x, xWithOffset, n, m);
	matrixAdd(xWithOffset, x, xWithOffset, n, m);
	__syncthreads();
	gravity(xWithOffset, k2v, mass, r, n, m);
	__syncthreads();



	// k3x = v + k2v*dt/2
	matrixMultiply(dt / 2, k2v, k3x, n, m);
	matrixAdd(v, k3x, k3x, n, m);
	// k3v = g(x + k2x*dt/2, mass)
	matrixMultiply(dt / 2, k2x, xWithOffset, n, m);
	matrixAdd(xWithOffset, x, xWithOffset, n, m);
	__syncthreads();
	gravity(xWithOffset, k3v, mass, r, n, m);
	__syncthreads();

	// k4x = v + k3v*dt
	matrixMultiply(dt, k3v, k4x, n, m);
	matrixAdd(v, k4x, k4x, n, m);
	// k4v = g(x + k3x*dt, mass)
	matrixMultiply(dt, k3x, xWithOffset, n, m);
	matrixAdd(xWithOffset, x, xWithOffset, n, m);
	__syncthreads();
	gravity(xWithOffset, k4v, mass, r, n, m);
	__syncthreads();


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

	//if (threadIdx.x == 1) {
	//	printf("x %.9f %.9f %.9f %.9f\n", x[0], x[1], x[2], x[3]);
	//	printf("v %.9f %.9f %.9f %.9f\n", v[0], v[1], v[2], v[3]);
	//}
}

__global__ void stagedRK(int stage, float *x, float *v, float *mass, float *xWithOffset, float *r, float dt, int n, int m,
	float *k1x, float *k2x, float *k3x, float *k4x, float *k1v, float *k2v, float *k3v, float *k4v) {

	switch (stage) {
	case 0:
		// k1x = v
		matrixCopy(v, k1x, n, m);

		// k1v = g(x, mass)
		//__syncthreads();
		gravity(x, k1v, mass, r, n, m);
		//__syncthreads();
		break;
	case 1:
		// k2x = v + k1v*dt/2
		matrixMultiply(dt / 2, k1v, k2x, n, m);
		matrixAdd(v, k2x, k2x, n, m);
		// k2v = g(x + k1x*dt/2, mass)
		matrixMultiply(dt / 2, k1x, xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);
		break;
	case 2:
		//__syncthreads();
		gravity(xWithOffset, k2v, mass, r, n, m);
		//__syncthreads();
		break;
	case 3:
		// k3x = v + k2v*dt/2
		matrixMultiply(dt / 2, k2v, k3x, n, m);
		matrixAdd(v, k3x, k3x, n, m);
		// k3v = g(x + k2x*dt/2, mass)
		matrixMultiply(dt / 2, k2x, xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);
		break;
	case 4:
		//__syncthreads();
		gravity(xWithOffset, k3v, mass, r, n, m);
		//__syncthreads();
		break;
	case 5:
		// k4x = v + k3v*dt
		matrixMultiply(dt, k3v, k4x, n, m);
		matrixAdd(v, k4x, k4x, n, m);
		// k4v = g(x + k3x*dt, mass)
		matrixMultiply(dt, k3x, xWithOffset, n, m);
		matrixAdd(xWithOffset, x, xWithOffset, n, m);
		break;
	case 6:
		//__syncthreads();
		gravity(xWithOffset, k4v, mass, r, n, m);
		//__syncthreads();
		break;
	case 7:
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
		break;
	}
}

void stagedRK(int blocks, int threads, float *x, float *v, float *mass, float *xWithOffset, float *r, float dt, int n, int m,
	float *k1x, float *k2x, float *k3x, float *k4x, float *k1v, float *k2v, float *k3v, float *k4v) {

	for (int i = 0; i <= 7; i++)
		stagedRK << <blocks, threads >> >(i, x, v, mass, xWithOffset, r, dt, n, m, k1x, k2x, k3x, k4x, k1v, k2v, k3v, k4v);
}

void drawFunc(){
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 1.0, 1.0);

	glEnable(GL_POINT_SMOOTH);
	glPointSize(1);

	if (runOnCPU){
		
		stepRKCPU(x, v, mass, kx, kv, xWithOffset, r, dt, n, m);

		//for (int i = 0; i < n; i++) {
		//	printf("x: %f %f %f\n", x[ix(0, i, n)], x[ix(1, i, n)], x[ix(2, i, n)]);
		//}
	}
	else{
		//go <<< BLOCKS, THREADS >>>(d_x, d_v, d_mass, xWithOffset, r, dt, n, m, k1x, k2x, k3x, k4x, k1v, k2v, k3v, k4v);
		stagedRK(BLOCKS*16, THREADS/8, d_x, d_v, d_mass, xWithOffset, r, dt, n, m, k1x, k2x, k3x, k4x, k1v, k2v, k3v, k4v);

		cudaMemcpy(x, d_x, n*m*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(v, d_v, n*m*sizeof(int), cudaMemcpyDeviceToHost);
	}

	glBegin(GL_POINTS);
	for (size_t i = 0; i < n; i++)
	{
		//printf("%f %f\n", (x[ix(0, i, n)] - minCoord[0]) / maxCoord[0], (x[ix(1, i, n)] - minCoord[1]) / maxCoord[1]);
		glVertex2d((x[ix(0, i, n)] - minCoord[0])/maxCoord[0], (x[ix(1, i, n)]-minCoord[1])/maxCoord[1]);
	}
	glEnd();

	glutSwapBuffers();
}

void GLtimer(int frameTimeMs){
	glutPostRedisplay();
	glutTimerFunc(frameTimeMs, GLtimer, 0);
}

void GLinit(int argc, char* argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Galaxy");

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
}

void GLstart(){
	glutDisplayFunc(drawFunc);

	GLtimer(100);
	glutMainLoop();
}


void loadData(int sampleID){
	FILE* fp;
	char* fname;
	switch (sampleID)
	{
	case 0:
		dt = 0.01;

		n = 128;
		m = 3;
		fname = "tab128";

		break;
	case 1:
		dt = 0.05;

		n = 1024;
		m = 3;
		fname = "tab1024";
		break;
	case 2:
		dt = 0.3;

		n = 8096;
		m = 3;
		fname = "tab8096";
		break;
	default:
		break;
	}

	x = (float*)malloc(n*m*sizeof(float));
	v = (float*)malloc(n*m*sizeof(float));
	mass = (float*)malloc(n*sizeof(float));

	r = (float*)malloc(n*m*sizeof(float)); // additional memory for computation

	xWithOffset = (float*)malloc(n*m*sizeof(float));

	//DATA
	//n,x,y,z,vx,vy,vz

	fp = fopen(fname, "r");

	if (fp == NULL)
		exit(EXIT_FAILURE);
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < 7; j++)
		{
			switch (j)
			{
			case 0:
				fscanf(fp, "%f", &mass[i]);
				break;
			case 1:
			case 2:
			case 3:
				fscanf(fp, "%f", &x[ix(j - 1, i, n)]);
				if (j - 1 < 2){
					if (x[ix(j - 1, i, n)] < minCoord[j - 1]){
						minCoord[j - 1] = x[ix(j - 1, i, n)];
					}
					if (x[ix(j - 1, i, n)] > maxCoord[j - 1]){
						maxCoord[j - 1] = x[ix(j - 1, i, n)];
					}
				}
				break;
			case 4:
			case 5:
			case 6:
				fscanf(fp, "%f", &v[ix(j - 4, i, n)]);
				break;
			}
		}
		//printf("m: %f\nx: %f %f %f\nv: %f %f %f\n\n", mass[i], x[ix(0, i, n)], x[ix(1, i, n)], x[ix(2, i, n)], v[ix(0, i, n)], v[ix(1, i, n)], v[ix(2, i, n)]);
	}
	fclose(fp);
	maxCoord[0] = maxCoord[0] - minCoord[0];
	maxCoord[1] = maxCoord[1] - minCoord[1];
	printf("%f %f\n%f %f\n", minCoord[0], minCoord[1], maxCoord[0], maxCoord[1]);
	//end data load segment
}


int main(int argc, char *argv[]) {
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

	for (int i = 1; i < argc; i++){
		if (argv[i][0] == '-'){
			switch (argv[i][1]){
			case 't':
				dataID = sscanf(argv[i + 1], "%i", &dataID);
				if (dataID < 0 && dataID > 2){
					printf("Test %d not available, default will be tested.", dataID);
					dataID = 0;
				}
				i++;
				break;
			case 'c':
				runOnCPU = 1;
				break;
			case 'v':
				visual = 0;
				break;
			case 'b':
				bench = 1;
				break;
			default:
				goto parseError;
			}
		}
		else {
			parseError:
			printf("Unrecognized command. Avaliable commands and switches:\n-t <N>\t- executes test N\n-c\t- executes on CPU\n-v\t- turns off visualization\n\nRerun with correct parameters.\n");
			return;
		}
	}
	printf("Test: %d\n", dataID);
	printf("Run on %s\n", runOnCPU == 0 ? "GPU" : "CPU");
	printf("Visualization: %s\n", visual == 1 ? "on" : "off");

	loadData(dataID);

	if (runOnCPU){
		for (int i = 0; i < 4; i++) {
			kx[i] = (float*)malloc(n*m*sizeof(float));
			kv[i] = (float*)malloc(n*m*sizeof(float));
		}


		GLinit(NULL, NULL);
		GLstart();

		for (int i = 0; i < 4; i++) {
			free(&kx[i]);
			free(&kv[i]);
		}

		free(&x);
		free(&v);
		free(&mass);
	}
	else{
		cudaError_t cudaStatus;

		//demo 1

		//float mass[] = { 1e-5f, 1e-6f };
		//float x[] = { 0, 0, 0, 1e-6 };
		//float v[] = { 0, 1e-5, 1e-5, 0 };
		//float t = 0.1;
		//float dt = 0.001;
		
		//int m = 2;
		//int n = 2;

		cudaMalloc(&d_x, n*m*sizeof(float));
		cudaMemcpy(d_x, x, n*m*sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc(&d_v, n*m*sizeof(float));
		cudaMemcpy(d_v, v, n*m*sizeof(int), cudaMemcpyHostToDevice);
		//float *d_mass;
		cudaMalloc(&d_mass, n*sizeof(float));
		cudaStatus = cudaMemcpy(d_mass, mass, n*sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}


		cudaMalloc(&r, n*m*sizeof(float));
		cudaMalloc(&xWithOffset, n*m*sizeof(float));



		cudaMalloc(&k1x, n*m*sizeof(float));
		cudaMalloc(&k2x, n*m*sizeof(float));
		cudaMalloc(&k3x, n*m*sizeof(float));
		cudaStatus = cudaMalloc(&k4x, n*m*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		cudaMalloc(&k1v, n*m*sizeof(float));
		cudaMalloc(&k2v, n*m*sizeof(float));
		cudaMalloc(&k3v, n*m*sizeof(float));
		cudaStatus = cudaMalloc(&k4v, n*m*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}


		//mainLoop<<< BLOCKS, THREADS >>>(d_x, d_v, d_mass, kx, kv, xWithOffset, r, dt, n, m);

		GLinit(NULL, NULL);
		GLstart();

		cudaFree(xWithOffset);	
	}
	return 0;
}

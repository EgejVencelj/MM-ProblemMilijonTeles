#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>

#define _GNU_SOURCE

int BLOCKS = 256;
int THREADS = 1024;

#define	G		        (6.67408e-11f)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int SCREEN_WIDTH = 800;
int SCREEN_HEIGHT = 800;

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

__host__ __device__ int ix(int i, int j, int n) {
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

__device__ void matrixCopy(float *a, float *b, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
 
    int len = n*m;
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

__device__ void gravity(float *x, float *a, float *mass, float *r, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void stagedRK(int stage, float *x, float *v, float *mass, float *xWithOffset, float *r, float dt, int n, int m,
	float *k1x, float *k2x, float *k3x, float *k4x, float *k1v, float *k2v, float *k3v, float *k4v) {

	switch (stage) {
	case 0:
		// k1x = v
		matrixCopy(v, k1x, n, m);

		// k1v = g(x, mass)
		gravity(x, k1v, mass, r, n, m);
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
		gravity(xWithOffset, k2v, mass, r, n, m);
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
		gravity(xWithOffset, k3v, mass, r, n, m);
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
		gravity(xWithOffset, k4v, mass, r, n, m);
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

	}
	else{
		stagedRK(BLOCKS, THREADS, d_x, d_v, d_mass, xWithOffset, r, dt, n, m, k1x, k2x, k3x, k4x, k1v, k2v, k3v, k4v);

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


double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		printf("QueryPerformanceFrequency failed!\n");

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
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

	if (fp == NULL){
		printf("ERROR: Missing file data file, exiting...\n");
		exit(EXIT_FAILURE);
	}

		
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
	//printf("%f %f\n%f %f\n", minCoord[0], minCoord[1], maxCoord[0], maxCoord[1]);
	//end data load segment
}


void benchmark(){
	printf("\nAverage time:\n");
	double diff;
	for (size_t i = 0; i < 100; i++)
	{
		n = 2 + 8094 * i / 100;
		StartCounter();
		//GetPerformanceCounter(1); // reset counter to zero 
		for (size_t i = 0; i < 10; i++)
		{
			if (runOnCPU){
				stepRKCPU(x, v, mass, kx, kv, xWithOffset, r, dt, n, m);
			}
			else{
				stagedRK(BLOCKS, THREADS, d_x, d_v, d_mass, xWithOffset, r, dt, n, m, k1x, k2x, k3x, k4x, k1v, k2v, k3v, k4v);

				cudaMemcpy(x, d_x, n*m*sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(v, d_v, n*m*sizeof(int), cudaMemcpyDeviceToHost);
			}
		}
		diff = GetCounter()/10;
		printf("%d\t%f\n", n, diff);
	}
}

void runCompute(){
	if (bench){
		benchmark();
	}
	else{
		while (1){
			if (runOnCPU){
				stepRKCPU(x, v, mass, kx, kv, xWithOffset, r, dt, n, m);
			}
			else{
				stagedRK(BLOCKS, THREADS, d_x, d_v, d_mass, xWithOffset, r, dt, n, m, k1x, k2x, k3x, k4x, k1v, k2v, k3v, k4v);

				cudaMemcpy(x, d_x, n*m*sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(v, d_v, n*m*sizeof(int), cudaMemcpyDeviceToHost);
			}
		}
	}
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




int main(int argc, char *argv[]) {
	for (int i = 1; i < argc; i++){
		if (argv[i][0] == '-'){
			switch (argv[i][1]){
			case 't':
				sscanf(argv[i + 1], "%i", &dataID);
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
				visual = 0;
				bench = 1;
				dataID = 2;
				break;
			case 'B':
				sscanf(argv[i + 1], "%i", &BLOCKS);
				i++;
				break;
			case 'T':
				sscanf(argv[i + 1], "%i", &THREADS);
				i++;
				break;
			case 'w':
				sscanf(argv[i + 1], "%i", &SCREEN_WIDTH);
				sscanf(argv[i + 2], "%i", &SCREEN_HEIGHT);
				i += 2;
				break;
			default:
				goto parseError;
			}
		}
		else {
			parseError:
			printf("Unrecognized command. Avaliable commands and switches:\
				   \n-t <N>\t- executes test N\
				   \n-c\t- executes on CPU\
				   \n-v\t- turns off visualization\
				   \n-b\t- benchmark preformance\
				   \n-T <N>\t- sets the number N of GPU threads (default = 1024)\
				   \n-B <N>\t- sets the number N of GPU blocks (default = 256)\
				   \n-w <W><H>\t-t sets window size W x H\
				   \n\n\nRerun with correct parameters.\n");
			return;
		}
	}
	printf("Test: %d\n", dataID);
	printf("Run on %s\n", runOnCPU == 0 ? "GPU" : "CPU");
	printf("Visualization: %s\n", visual == 1 ? "on" : "off");
	if (runOnCPU == 0){
		printf("Threads: %d\nBlocks: %d\n", THREADS, BLOCKS);
	}
	

	loadData(dataID);

	if (runOnCPU){
		for (int i = 0; i < 4; i++) {
			kx[i] = (float*)malloc(n*m*sizeof(float));
			kv[i] = (float*)malloc(n*m*sizeof(float));
		}

		if (visual){
			GLinit(NULL, NULL);
			GLstart();
		}
		else{
			runCompute();
		}

		for (int i = 0; i < 4; i++) {
			free(&kx[i]);
			free(&kv[i]);
		}

		free(&x);
		free(&v);
		free(&mass);
	}
	else{
		cudaMalloc(&d_x, n*m*sizeof(float));
		cudaMemcpy(d_x, x, n*m*sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc(&d_v, n*m*sizeof(float));
		cudaMemcpy(d_v, v, n*m*sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc(&d_mass, n*sizeof(float));
		cudaMemcpy(d_mass, mass, n*sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc(&r, n*m*sizeof(float));
		cudaMalloc(&xWithOffset, n*m*sizeof(float));

		cudaMalloc(&k1x, n*m*sizeof(float));
		cudaMalloc(&k2x, n*m*sizeof(float));
		cudaMalloc(&k3x, n*m*sizeof(float));
		cudaMalloc(&k4x, n*m*sizeof(float));

		cudaMalloc(&k1v, n*m*sizeof(float));
		cudaMalloc(&k2v, n*m*sizeof(float));
		cudaMalloc(&k3v, n*m*sizeof(float));
		cudaMalloc(&k4v, n*m*sizeof(float));


		if (visual){
			GLinit(NULL, NULL);
			GLstart();
		}
		else{
			runCompute();
		}

		cudaFree(xWithOffset);	
	}
	return 0;
}

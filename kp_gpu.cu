#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#pragma comment(lib, "glew32.lib")
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <chrono>

using namespace std;

#define CSC(call) {							\
    cudaError err = call;						\
    if(err != cudaSuccess) {						\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));		\
        exit(1);							\
    }									\
} while (0)

#define sqr(x) ((x)*(x))

struct particle
{
	double2 x;
	double2 v;
	double2 p_best;
	double2 f;
};

const int width = 1024;
const int height = 768;

dim3 blocks(32, 32), threads(16, 16);

double xc = 0.0f, yc = 0.0f;
double dt, k;

__device__ double dev_xc = 0;
__device__ double dev_yc = 0;
__device__ double dev_minf;
__device__ double dev_maxf;

curandState* devStates;

__device__ double data_raw[height * width];
__device__ double2 g_best;

double2 *g_best_dev;
double2 *g_best_host;

struct cudaGraphicsResource *res;

particle *p_arr_dev;
double w, a1, a2;
double scale_x = 5;
double scale_y = scale_x * height / width;
int p_number;

const int reduce_threads_count = 1024; //should be power of 2
const int reduce_blocks_count = width * height / reduce_threads_count + 1;
double *partial_max_dev;
double *partial_min_dev;

double2 *partial_g_best;

__device__ double fun(double2 arg) {
	return sqr((1 - arg.x)) + 100 * sqr((arg.y - sqr(arg.x)));
}

__device__ double fun(double x, double y) {
	return sqr((1 - x)) + 100 * sqr((y - sqr(x)));
}

__device__ double fun(int i, int j, double scale_x, double scale_y)  {
	double x = 2.0f * i / (double)(width - 1) - 1.0f;
	double y = 2.0f * j / (double)(height - 1) - 1.0f;	
	return fun(x * scale_x + dev_xc, -y * scale_y + dev_yc);
}

__global__ void kernel_data_raw(double scale_x, double scale_y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	for (j = idy; j < height; j += offsety)
	{
		for (i = idx; i < width; i += offsetx)
		{
			data_raw[j * width + i] = fun(i, j, scale_x, scale_y);
		}
	}
}

__global__ void kernel_find_max(double *partial_max)
{
	__shared__ double seq[reduce_threads_count];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < width * height)
	{
		seq[threadIdx.x] = data_raw[tid];
	}
	else
	{
		seq[threadIdx.x] = -1; // fun() always >= 0; 
	}
	__syncthreads();

	int pow = 2;
	while (pow <= reduce_threads_count)
	{
		if (threadIdx.x * pow + pow - 1 < reduce_threads_count)
		{
			seq[threadIdx.x * pow + pow - 1] = (seq[threadIdx.x * pow + pow - 1] > seq[threadIdx.x * pow + pow - pow / 2 - 1]) ? seq[threadIdx.x * pow + pow - 1] : seq[threadIdx.x * pow + pow - pow / 2 - 1];
		}
		__syncthreads();
		pow *= 2;
	}
	if (threadIdx.x == 0)
	{
		partial_max[blockIdx.x] = seq[reduce_threads_count - 1];
	}
}

__global__ void kernel_find_max_final(double *partial_max, int size)
{
	double max = partial_max[0];
	for (int i = 1; i < size; i++)
	{
		if (partial_max[i] > partial_max[i - 1])
			max = partial_max[i];
	}
	dev_maxf = max;
}

__global__ void kernel_find_min(double *partial_min)
{
	__shared__ double seq[reduce_threads_count];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < width * height)
	{
		seq[threadIdx.x] = data_raw[tid];
	}
	else
	{
		seq[threadIdx.x] = INFINITY; 
	}
	__syncthreads();

	int pow = 2;
	while (pow <= reduce_threads_count)
	{
		if (threadIdx.x * pow + pow - 1 < reduce_threads_count)
		{
			seq[threadIdx.x * pow + pow - 1] = (seq[threadIdx.x * pow + pow - 1] < seq[threadIdx.x * pow + pow - pow / 2 - 1]) ? seq[threadIdx.x * pow + pow - 1] : seq[threadIdx.x * pow + pow - pow / 2 - 1];
		}
		__syncthreads();
		pow *= 2;
	}
	if (threadIdx.x == 0)
	{
		partial_min[blockIdx.x] = seq[reduce_threads_count - 1];
	}
}

__global__ void kernel_find_min_final(double *partial_min, int size)
{
	double min = partial_min[0];
	for (int i = 1; i < size; i++)
	{
		if (partial_min[i] < partial_min[i - 1])
			min = partial_min[i];
	}
	dev_minf = min;
}

__global__ void kernel_data(uchar4* data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	for (j = idy; j < height; j += offsety)
	{
		for (i = idx; i < width; i += offsetx)
		{
			double f = (data_raw[j * width + i] - dev_minf) / (dev_maxf - dev_minf);
			data[j * width + i] = make_uchar4(0, (int)(f * 255), 0, 255);
		}
	}
}

__device__ void calc(particle & p, double w, double a1, double a2, double dt, double k, curandState * state)
{
	p.v.x = w * p.v.x + (a1 * curand_uniform(state) * (p.p_best.x - p.x.x) + a2 * curand_uniform(state) * (g_best.x - p.x.x) + k * p.f.x) * dt;
	p.v.y = w * p.v.y + (a1 * curand_uniform(state) * (p.p_best.y - p.x.y) + a2 * curand_uniform(state) * (g_best.y - p.x.y) + k * p.f.y) * dt;
	p.x.x += p.v.x * dt;
	p.x.y += p.v.y * dt;
	if (fun(p.x.x, p.x.y) < fun(p.p_best.x, p.p_best.y))
	{
		p.p_best.x = p.x.x;
		p.p_best.y = p.x.y;
	}
}

__global__ void kernel_main(double w, double a1, double a2, double dt, double k, particle *p_arr, int p_number, uchar4* data, double scale_x, double scale_y, curandState * state)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < p_number)
	{
		calc(p_arr[tid], w, a1, a2, dt, k, &state[tid]);
		
		double2 tmp = p_arr[tid].x;
		tmp.x -= dev_xc;
		tmp.x /= scale_x;
		tmp.x += 1;
		tmp.x *= (double)(width - 1);
		tmp.x /= 2;
		tmp.y -= dev_yc;
		tmp.y /= scale_y;
		tmp.y *= -1;
		tmp.y += 1;
		tmp.y *= (double)(height - 1);
		tmp.y /= 2;
		
		int2 tmp2;
		tmp2.x = (int)tmp.x;
		tmp2.y = (int)tmp.y;

		if (tmp2.x > 0 && tmp2.x < width && tmp2.y > 0 && tmp2.y < height)
		{
			data[tmp2.y * width + tmp2.x] = make_uchar4(255, 255, 255, 255);
		}
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_calc_force(particle *p_arr, int p_number)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < p_number)
	{
		p_arr[tid].f = double2();
		for (int i = 0; i < p_number; i++)
		{
			if (i == tid)
				continue;
			double d2 = sqr(p_arr[i].x.x - p_arr[tid].x.x) + sqr(p_arr[i].x.y - p_arr[tid].x.y);
			p_arr[tid].f.x -= (p_arr[i].x.x - p_arr[tid].x.x) / (sqr(d2) + 1e-3);
			p_arr[tid].f.y -= (p_arr[i].x.y - p_arr[tid].x.y) / (sqr(d2) + 1e-3);
		}
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_mass_center(particle *p_arr, int p_number)
{
	double2 tmp = double2();
	for (int i = 0; i < p_number; i++)
	{
		tmp.x += p_arr[i].x.x;
		tmp.y += p_arr[i].x.y;
	}
	tmp.x /= p_number;
	tmp.y /= p_number;
	dev_xc = tmp.x;
	dev_yc = tmp.y;
}

__global__ void kernel_g_best_find(particle *p_arr, double2 *partial_g_best, int p_number)
{
	__shared__ double2 seq[32];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	seq[threadIdx.x] = p_arr[tid].x;
	__syncthreads();

	int pow = 2;
	while (pow <= 32)
	{
		if (threadIdx.x * pow + pow - 1 < 32)
		{
			seq[threadIdx.x * pow + pow - 1] = (fun(seq[threadIdx.x * pow + pow - 1]) < fun(seq[threadIdx.x * pow + pow - pow / 2 - 1])) ? seq[threadIdx.x * pow + pow - 1] : seq[threadIdx.x * pow + pow - pow / 2 - 1];
		}
		__syncthreads();
		pow *= 2;
	}
	if (threadIdx.x == 0)
	{
		partial_g_best[blockIdx.x] = seq[31];
	}
}

__global__ void kernel_g_best_find_final(particle *p_arr, double2 *partial_g_best, int size, int p_number)
{
	double2 max;
	if (size > 0)
	{
		max = partial_g_best[0];
	}
	else
	{
		max = p_arr[0].x;
	}
	for (int i = 1; i < size; i++)
	{
		if (fun(partial_g_best[i]) < fun(partial_g_best[i - 1]))
			max = partial_g_best[i];
	}
	for (int i = 32 * size; i < p_number; i++)
	{
		if (fun(p_arr[i].x) < fun(max))
			max = p_arr[i].x;
	}
	if(fun(max) < fun(g_best))
		g_best = max;
}

__global__ void kernel_g_best_find_final_init(particle *p_arr, double2 *partial_g_best, int size, int p_number)
{
	double2 max;
	if (size > 0)
	{
		max = partial_g_best[0];
	}
	else
	{
		max = p_arr[0].x;
	}
	for (int i = 1; i < size; i++)
	{
		if (fun(partial_g_best[i]) < fun(partial_g_best[i - 1]))
			max = partial_g_best[i];
	}
	for (int i = 32 * size; i < p_number; i++)
	{
		if (fun(p_arr[i].x) < fun(max))
			max = p_arr[i].x;
	}
	g_best = max;
}

__global__ void kernel_particles_init(particle *p_arr, int p_number, curandState * state, unsigned long seed)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < p_number)
	{
		p_arr[tid].v = double2();
		p_arr[tid].f = double2();

		curand_init(seed, tid, 0, &state[tid]); // U(0,1)

		p_arr[tid].x.x = curand_uniform(&state[tid]) * 100;
		p_arr[tid].x.y = curand_uniform(&state[tid]) * 100;

		p_arr[tid].p_best = p_arr[tid].x;

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_printf(double2 *g_best_to_host)
{
	*g_best_to_host = g_best;
}

void update() {
	uchar4* dev_data;
	size_t size;
	auto start_time = chrono::high_resolution_clock::now();
	CSC(cudaGraphicsMapResources(1, &res, 0));
	CSC(cudaGraphicsResourceGetMappedPointer((void**) &dev_data, &size, res));

	kernel_mass_center << <1, 1 >> >(p_arr_dev, p_number);
	kernel_data_raw << <blocks, threads >> >(scale_x, scale_y);
	kernel_find_max << <reduce_blocks_count, reduce_threads_count >> >(partial_max_dev);
	kernel_find_max_final << <1, 1 >> >(partial_max_dev, reduce_blocks_count);
	kernel_find_min << <reduce_blocks_count, reduce_threads_count >> >(partial_min_dev);
	kernel_find_min_final << <1, 1 >> >(partial_min_dev, reduce_blocks_count);
	kernel_data<<<blocks, threads>>>(dev_data);
	kernel_calc_force << <64, 64 >> >(p_arr_dev, p_number);
	kernel_main << <32, 32 >> >(w, a1, a2, dt, k, p_arr_dev, p_number, dev_data, scale_x, scale_y, devStates);
	kernel_g_best_find << <p_number / 32, 32 >> >(p_arr_dev, partial_g_best, p_number);
	kernel_g_best_find_final << <1, 1 >> >(p_arr_dev, partial_g_best, p_number / 32, p_number);
	kernel_printf << <1, 1 >> >(g_best_dev);

	CSC(cudaDeviceSynchronize());
	CSC(cudaGraphicsUnmapResources(1, &res, 0));

	CSC(cudaMemcpy(g_best_host, g_best_dev, sizeof(double2), cudaMemcpyDeviceToHost));
	auto end_time = chrono::high_resolution_clock::now();
	printf("%f %f\n", (*g_best_host).x, (*g_best_host).y);
	cout << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "\n";

	glutPostRedisplay();
}

void display() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);	
	glutSwapBuffers();
}

void MyKeyboardFunc(unsigned char Key, int x, int y)
{
	switch (Key)
	{
	case '-':
		scale_x += 0.5;
		scale_y = scale_x * height / width;
		break;
	case '+':
		if (scale_x > 1)
		{
			scale_x -= 0.5;
			scale_y = scale_x * height / width;
		}
		break;
	};
}

int main(int argc, char** argv)
{
	cout << "Enter number of particles: ";
	cin >> p_number;
	cout << "w = ";
	cin >> w;
	cout << "a1 = ";
	cin >> a1;
	cout << "a2 = ";
	cin >> a2;
	cout << "k = ";
	cin >> k;
	cout << "dt = ";
	cin >> dt;

	CSC(cudaMalloc(&g_best_dev, sizeof(double2)));
	g_best_host = (double2 *)malloc(sizeof(double2));

	CSC(cudaMalloc(&p_arr_dev, sizeof(particle) * p_number));
	CSC(cudaMalloc(&devStates, sizeof(curandState) * p_number));
	kernel_particles_init << <16, 16 >> >(p_arr_dev, p_number, devStates, (unsigned long)time(NULL));

	CSC(cudaMalloc(&partial_g_best, sizeof(double2) * (p_number / 32)));
	kernel_g_best_find << <p_number / 32, 32 >> >(p_arr_dev, partial_g_best, p_number);
	kernel_g_best_find_final_init << <1, 1 >> >(p_arr_dev, partial_g_best, p_number / 32, p_number);

	CSC(cudaMalloc(&partial_max_dev, sizeof(double) * reduce_blocks_count));
	CSC(cudaMalloc(&partial_min_dev, sizeof(double) * reduce_blocks_count));

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("Hot map");
	
	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(MyKeyboardFunc);
	//glutMouseFunc(MyMouseFunc);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble) width, 0.0, (GLdouble) height);

	glewInit();

	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

	CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));

	glutMainLoop();

	CSC(cudaGraphicsUnregisterResource(res));

	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);

	return 0;
}

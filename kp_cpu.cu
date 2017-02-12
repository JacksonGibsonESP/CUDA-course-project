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

double xc = 0.0f, yc = 0.0f;
double scale_x = 5;
double scale_y = scale_x * height / width;

double w, a1, a2;
double dt, k;
double minf;
double maxf;

double data_raw[height * width];
uchar4 data[width * height];
double2 g_best;

particle *p_arr;
int p_number;

double fun(double2 arg) {
	return sqr((1 - arg.x)) + 100 * sqr((arg.y - sqr(arg.x)));
}

double fun(double x, double y) {
	return sqr((1 - x)) + 100 * sqr((y - sqr(x)));
}

double fun(int i, int j, double scale_x, double scale_y)  {
	double x = 2.0f * i / (double)(width - 1) - 1.0f;
	double y = 2.0f * j / (double)(height - 1) - 1.0f;	
	return fun(x * scale_x + xc, -y * scale_y + yc);
}

void _data_raw()
{
	maxf = -1;
	minf = -1;
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			data_raw[j * width + i] = fun(i, j, scale_x, scale_y);
			if (data_raw[j * width + i] > maxf)
			{
				maxf = data_raw[j * width + i];
			}
			if (data_raw[j * width + i] < minf)
			{
				minf = data_raw[j * width + i];
			}
		}
	}
}

void _data(uchar4* data)
{
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			double f = (data_raw[j * width + i] - minf) / (maxf - minf);
			data[j * width + i] = make_uchar4(0, (int)(f * 255), 0, 255);
		}
	}
}

void calc(particle & p)
{
	p.v.x = w * p.v.x + (a1 * (double)rand() / (double)RAND_MAX * (p.p_best.x - p.x.x) + a2 * (double)rand() / (double)RAND_MAX * (g_best.x - p.x.x) + k * p.f.x) * dt;
	p.v.y = w * p.v.y + (a1 * (double)rand() / (double)RAND_MAX * (p.p_best.y - p.x.y) + a2 * (double)rand() / (double)RAND_MAX * (g_best.y - p.x.y) + k * p.f.y) * dt;
	p.x.x += p.v.x * dt;
	p.x.y += p.v.y * dt;
	if (fun(p.x.x, p.x.y) < fun(p.p_best.x, p.p_best.y))
	{
		p.p_best.x = p.x.x;
		p.p_best.y = p.x.y;
	}
}

void _main(uchar4* data)
{
	for (int i = 0; i < p_number; i++)
	{
		calc(p_arr[i]);
		double2 tmp = p_arr[i].x;
		tmp.x -= xc;
		tmp.x /= scale_x;
		tmp.x += 1;
		tmp.x *= (double)(width - 1);
		tmp.x /= 2;
		tmp.y -= yc;
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
	}
}

void calc_force()
{
	for (int j = 0; j < p_number; j++)
	{
		p_arr[j].f = double2();
		for (int i = 0; i < p_number; i++)
		{
			if (i == j)
				continue;
			double d2 = sqr(p_arr[i].x.x - p_arr[j].x.x) + sqr(p_arr[i].x.y - p_arr[j].x.y);
			p_arr[j].f.x -= (p_arr[i].x.x - p_arr[j].x.x) / (sqr(d2) + 1e-3);
			p_arr[j].f.y -= (p_arr[i].x.y - p_arr[j].x.y) / (sqr(d2) + 1e-3);
		}
	}
}

void mass_center()
{
	double2 tmp = double2();
	for (int i = 0; i < p_number; i++)
	{
		tmp.x += p_arr[i].x.x;
		tmp.y += p_arr[i].x.y;
	}
	tmp.x /= p_number;
	tmp.y /= p_number;
	xc = tmp.x;
	yc = tmp.y;
}

void g_best_find()
{
	for (int i = 0; i < p_number; i++)
	{
		if (fun(p_arr[i].x) < fun(g_best))
			g_best = p_arr[i].x;
	}
}

void g_best_init()
{
	g_best = p_arr[0].x;
	for (int i = 1; i < p_number; i++)
	{
		if (fun(p_arr[i].x) < fun(g_best))
			g_best = p_arr[i].x;
	}
}

void particles_init()
{
	for (int i = 0; i < p_number; i++)
	{
		p_arr[i].v = double2();
		p_arr[i].f = double2();

		p_arr[i].x.x = (double)rand() / (double)RAND_MAX * 100;
		p_arr[i].x.y = (double)rand() / (double)RAND_MAX * 100;

		p_arr[i].p_best = p_arr[i].x;
	}
}

void update() {
	
	auto start_time = chrono::high_resolution_clock::now();
	mass_center();
	_data_raw();
	_data(data);
	calc_force();
	_main(data);
	g_best_find();
	auto end_time = chrono::high_resolution_clock::now();

	printf("%f %f\n", g_best.x, g_best.y);
	cout << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "\n";


	glutPostRedisplay();
}

void display() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);	
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

	p_arr = (particle *)malloc(sizeof(particle) * p_number);
	particles_init();

	g_best_init();

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("Hot map");
	
	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(MyKeyboardFunc);
	//glutMouseFunc(MyMouseFunc);

	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	//gluOrtho2D(0.0, (GLdouble) width, 0.0, (GLdouble) height);

	//glewInit();

	//GLuint vbo;
	//glGenBuffers(1, &vbo);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
	//glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

	//CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));

	glutMainLoop();

	//CSC(cudaGraphicsUnregisterResource(res));

	//glBindBuffer(1, vbo);
	//glDeleteBuffers(1, &vbo);

	return 0;
}

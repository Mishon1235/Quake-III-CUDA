
// System includes
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include "fun.h"
	// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

	// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

int N=10000000;

int main() {
float *A  = (float *)malloc( N * sizeof(  float  ) ); 
float *B  = (float *)malloc( N * sizeof(  float  ) ); 
float *C  = (float *)malloc( N * sizeof(  float  ) ); 
float *D_F  = (float *)malloc( N * sizeof(  float  ) ); 
float *D_CPU_N  = (float *)malloc( N * sizeof(  float  ) ); 
float *D_CPU_Q  = (float *)malloc( N * sizeof(  float  ) ); 
float *D_CPU_NEWTON  = (float *)malloc( N * sizeof(  float  ) );

size_t size = N * sizeof(float);


float *D_GPU_NEWTON;
cudaMallocManaged(&D_GPU_NEWTON, size); 
float *D_GPU_Q;
cudaMallocManaged(&D_GPU_Q, size); 
float *D_GPU_N;
cudaMallocManaged(&D_GPU_N, size); 
 float *A_GPU;
cudaMallocManaged(&A_GPU, size); 
  float *B_GPU;
cudaMallocManaged(&B_GPU, size); 
 float *C_GPU;
cudaMallocManaged(&C_GPU, size); 

float *a, *b, *c, *d;
cudaMalloc((void **)&a, size);
cudaMalloc((void **)&b, size);
cudaMalloc((void **)&c, size);
cudaMalloc((void **)&d, size);
 
			//cudaMemcpy(d, a, size, cudaMemcpyHostToDevice);







int block=32, threads=32;//do zrobienia
init(A,N);
init(B,N);
init(C,N);
for(int i=0; i<N; i++)
{
A_GPU[i]=A[i];
B_GPU[i]=B[i];
C_GPU[i]=C[i];

}
	//NAIVE
clock_t pocz = clock();
insqrt(A,B,C,D_CPU_N,N);
clock_t kon = clock();
double msec1 = double(kon-pocz)*1000/CLOCKS_PER_SEC;

	//QUAKE
pocz = clock();
Quake(A,B,C,D_CPU_Q,N);
kon = clock();
double msec2 = double(kon-pocz)*1000/CLOCKS_PER_SEC;
	
	//NEWTON
pocz = clock();
Newton(A,B,C,D_CPU_NEWTON,N);
kon = clock();
double msec5 = double(kon-pocz)*1000/CLOCKS_PER_SEC;


/*--------------------------------------------------GPU---------------------------------------------------*/

	//NAIVE
pocz = clock();
insqrtCUDA<<< block, threads>>>(A_GPU,B_GPU,C_GPU,D_GPU_N,N);
cudaDeviceSynchronize();
kon = clock();
double msec3 = double(kon-pocz)*1000/CLOCKS_PER_SEC;

	//QUAKE
pocz = clock();
insqrtCUDAQ<<< block, threads>>>(A_GPU,B_GPU,C_GPU,D_GPU_Q,N);
cudaDeviceSynchronize();
kon = clock();
double msec4 = double(kon-pocz)*1000/CLOCKS_PER_SEC;

	//NEWTON
pocz = clock();
insqrtCUDAN<<< block, threads>>>(A_GPU,B_GPU,C_GPU,D_GPU_NEWTON,N);
cudaDeviceSynchronize();
kon = clock();
double msec6 = double(kon-pocz)*1000/CLOCKS_PER_SEC;



/*----------------------------------------Co to jest to fast?-Sonic---------------------------------------------*/
//////Fast//////
cudaMemcpy(a, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b, B, size, cudaMemcpyHostToDevice);
		cudaMemcpy(c, C, size, cudaMemcpyHostToDevice);
pocz = clock();
insqrtCUDAQ<<< block, threads>>>(a,b,c,d,N);
cudaMemcpy(D_F, d, size, cudaMemcpyDeviceToHost);
kon = clock();

double msecf = double(kon-pocz)*1000/CLOCKS_PER_SEC;
/*----------------------------------------------------------------------------------------------------------*/


float t1= (float)msec1;			//cpu naive
float t2= (float)msec2;			//cpu quake
float t5= (float)msec5;			//cpu newton
float t3= (float)msec3;			//gpu naive			
float t4= (float)msec4;			//gpu quake
float t6= (float)msec6;			//gpu newton
	float t7= (float)msecf;
float e1=res(D_CPU_N,D_CPU_N,N);
float e2=res(D_CPU_N,D_GPU_N,N);
float e3=res(D_CPU_N,D_CPU_Q,N); 
float e4=res(D_CPU_N,D_GPU_Q,N);
float e5=res(D_CPU_N,D_CPU_NEWTON,N);
float e6=res(D_CPU_N,D_GPU_NEWTON,N);

	float e7=res(D_CPU_N,D_F,N);



//float fu=0;
Tab(e1,t1,e2,t2,
		e5,t5,e6,t6,
			    e3,t3,e4,t4);

for(int i=0; i<=10;++i){printf("CPU_Naive[%d]=%f,	GPU_Naive[%d]=%f,		CPU_Quake[%d]=%f,		GPU_Quake[%d]=%f\n,		CPU_Newton[%d]=%f\n		GPU_Newton[%d]=%f\n",
			i,D_CPU_N[i],		i,D_GPU_N[i],			i,D_CPU_Q[i],			i,D_GPU_Q[i],			i,D_CPU_NEWTON[i],		i,D_GPU_NEWTON[i]);}

printf("res=%f czas=%f \n",e7,t7);

free(A);
free(B);
free(C);
free(D_F);
free(D_CPU_N);
free(D_CPU_Q);
free(D_CPU_NEWTON);
cudaFree(D_GPU_N);
cudaFree(D_GPU_Q);
cudaFree(D_GPU_NEWTON);
cudaFree(A_GPU);
cudaFree(B_GPU);
cudaFree(C_GPU);  
cudaFree(a);
cudaFree(b);
cudaFree(c);  
cudaFree(d);  

  }

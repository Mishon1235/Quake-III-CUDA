float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the ****? 
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
		//y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed
	return y;
}
/*-----------------------------------------------------------------------------------------------------*/
//NEWTON ITERATION
void Newton(float *A,  float *B,  float *C, float *D, int N){
  int i=0;
  float x;
				int set= 12; //how good is approximation
  while(i<N){
x=A[i]+B[i]+C[i];
		
  float z = 1;

  int j=0;
  while (set > j) {
    z = 0.5f * (z + x / z);

   // printf("%f\n",z);
    j++;

       }D[i]=1/z;i++;}
}
/*-----------------------------------------------------------------------------------------------------*/

__global__
 void insqrtCUDAN( float *A,  float *B, float *C, float *D, int N) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	int x;
	for (int i = idx; i < N; i += stride)	
	{
	    x=A[i]+B[i]+C[i];
		
  float z = 1;

  int j=0;
  while (15 > j) {
    z = 0.5f * (z + x / z);

    //printf("%f\n",z);
    j++;

       }D[i]=1/z;	}
}
/*-----------------------------------------------------------------------------------------------------*/
void Tab(float cpu_naive_err, float cpu_naive_time, float gpu_naive_err,float gpu_naive_time,float cpu_newton_err,float cpu_newton_time,
float gpu_newton_err,float gpu_newton_time,float cpu_quake_err,float cpu_quake_time,float gpu_quake_err,float gpu_quake_time){
    printf("*****************************************************************\n");
    printf("*              *          CPU          **          GPU          *\n");
    printf("*****************************************************************\n");
    printf("*              *  Errors  *    Time    **  Errors  *    Time    *\n");
    printf("*****************************************************************\n");
    printf("* Naive        * %.6f * %7.3f ** %1.6f * %7.3f *\n",cpu_naive_err,cpu_naive_time,gpu_naive_err,gpu_naive_time);
    printf("* function     *          *            **          *            *\n");
    printf("*****************************************************************\n");
    printf("* Newton       * %.6f * %7.3f ** %1.6f * %7.3f *\n",cpu_newton_err,cpu_newton_time,gpu_newton_err,gpu_newton_time);
    printf("* iteration    *          *            **          *            *\n");
    printf("*****************************************************************\n");
    printf("* Quake        * %.6f * %7.3f ** %1.6f * %7.3f *\n",cpu_quake_err,cpu_quake_time,gpu_quake_err,gpu_quake_time);
    printf("* algorithm    *          *            **          *            *\n");
    printf("*****************************************************************\n");
    
}
/*-----------------------------------------------------------------------------------------------------*/

void Quake( float *A,  float *B,  float *C, float *D, int N ){
float number=0;
for(int in=0; in<N;in++){
number = A[in]+B[in]+C[in];
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the ****? 
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
		//y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	D[in]=abs(y);	}
}
/*-----------------------------------------------------------------------------------------------------*/

float res(float *A,  float *B, int N){
	float sum=0,summ=0;
	for(int i=0;i<N;i++){
		sum+=abs(A[i]-B[i]);
		if(abs(A[i]-B[i])>=0.001){
			printf("A=%f     dif=%f \nB=%f\n",A[i],abs(A[i]-B[i]),B[i]);
}
		summ+=A[i];}
	float x=sum/summ;
	return x;
}
/*-----------------------------------------------------------------------------------------------------*/

void insqrt ( float *A,  float *B,  float *C, float *D, int N ){
	for(int i=0;i<N;i++){
	D[i]=1/sqrt(A[i]+B[i]+C[i]);}
}
/*-----------------------------------------------------------------------------------------------------*/
__global__ void insqrtCUDAQ( float *A,  float *B, float *C, float *D, int N){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = idx; i < N; i += stride){
		long in;
	float sum=A[i]+B[i]+C[i];
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = sum * 0.5F;
	y  = sum;
	in  = * ( long * ) &y;                       // evil floating point bit level hacking
	in  = 0x5f3759df - ( in >> 1 );               // what the ****? 
	y  = * ( float * ) &in;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
		//y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed
	D[i]=y;}
}
/*-----------------------------------------------------------------------------------------------------*/
__global__
 void insqrtCUDA( float *A,  float *B, float *C, float *D, int N) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
		//printf("CZEMU NIE DZIALA, a dziala w ogole liczby %f???\n", A[idx]);
	int stride = gridDim.x * blockDim.x;
	for (int i = idx; i < N; i += stride)	
	{D[i]=1/sqrt(A[i]+B[i]+C[i]);}
		//printf("GPU=%f indx=%i\n",D[idx],idx);
}
/*-----------------------------------------------------------------------------------------------------*/
void init( float *A, int N){
	for(int i=0; i<N; i++){
	A[i]=(float)rand()/(float)(RAND_MAX);}
	for(int i=0; i<N; i++){
		A[i]+=rand()/10000;}
}
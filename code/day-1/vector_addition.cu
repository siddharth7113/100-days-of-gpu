#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include<math.h>
#include<chrono> // For cpu timing

__global__
void VecAddKernel(float *A, float* B, float *C, int n){
  int i = threadIdx.x + (blockIdx.x * blockDim.x);
  if( i < n){
    C[i] = A[i] + B[i];
  }
}

//Cuda Vector Addition
void vector_add_cuda(float* A_h, float* B_h, float* C_h,int n){
  float *A_d,*B_d, *C_d;
  int size = n * sizeof(float);

  // Allocating memory in cuda device
  cudaMalloc((void**) &A_d, size);
  cudaMalloc((void**) &B_d, size);
  cudaMalloc((void**) &C_d, size);

  // Copying data from host to device
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
  
  //Recording events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // Calling kernel operation
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // Ceiling calculation
  VecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
  
  cudaEventRecord(stop);

  //Event finish
  cudaEventSynchronize(stop);

  // Calculate time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Cuda Time : %f ms\n", milliseconds);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

  // Copying from device back to host
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
  
  // Freeing the memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}


//CPU Vector Addtion


void  vector_add_cpu(float *A, float *B, float*C ,int n){
  //Start timing
  auto start = std::chrono::high_resolution_clock::now();

  for(int i =0; i< n; i++){
    C[i] = A[i] + B[i];
  }

  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration=stop - start;
  printf("CPU time %f ms \n", duration.count());

}
int main(){
  // Size of the vector
  int n = 100000;

  //Host memmory
  float *A_h = (float*) malloc(n * sizeof(float));
  float *B_h = (float*) malloc(n * sizeof(float));
  float *C_h = (float*) malloc(n * sizeof(float));
  float *C_cpu = (float*) malloc(n * sizeof(float));


  for(int i=0; i < n; i++){
    A_h[i] = i * 1.0f;
    B_h[i] = i * 2.0f;
  }
  
  vector_add_cuda(A_h, B_h, C_h, n);
  vector_add_cpu(A_h,B_h, C_cpu, n );

  //Verification
  for(int i =0; i < 10; i++){
    if(C_h[i] != C_cpu[i]){
      printf("mismatch at index %d: GPU %f , CPU%f \n",i , C_h[i], C_cpu[i]);
    }
  }

  free(A_h);
  free(B_h);
  free(C_h);
  free(C_cpu);

  return 0;
}

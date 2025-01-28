#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include<chrono> // For cpu timing

__global__
void colorToGrayscaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check boundary conditions
    if ( col < width && row < height){
        int grayOffset = row * width + col; // 1d index for grayscale
        int rgbOffset = grayOffset * 3; // 1D index for rgb channels

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = 0.21f * r + 0.72f * g + 0.07 * b;
    
    }
}

void grayScaleConverter(unsigned char *Pin_h, unsigned char *Pout_h, int width, int height){
    unsigned char * Pin_d,*Pout_d;
    int size = width * height * sizeof(unsigned char);

    cudaMalloc((void**)&Pin_d, 3 * size);
    cudaMalloc((void**)&Pout_d,size);

    cudaMemcpy(Pin_d,Pin_h,3 * size, cudaMemcpyHostToDevice);

    //Recording events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 dimBlock(16,16,1);
    dim3 dimGrid(ceil(width/16.0),ceil(height/16.0), 1);
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(Pout_d,Pin_d, width, height);
    cudaEventRecord(stop);

    //Event finish
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("Cuda Time: %f ms\n", milliseconds);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("CUDA error %s\n",cudaGetErrorString(err));
    }

    cudaMemcpy(Pout_h, Pout_d,size,cudaMemcpyDeviceToHost);

    cudaFree(Pin_d);
    cudaFree(Pout_d);
}

int main(){
    int width = 512;
    int height = 512;

    unsigned char *Pin_h =(unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *Pout_h =(unsigned char*)malloc(width * height * sizeof(unsigned char));

    grayScaleConverter(Pin_h,Pout_h,width,height);

    free(Pin_h);
    free(Pout_h);

    return 0;
}
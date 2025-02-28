#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include<chrono> // For cpu timing
#define BLUR_SIZE 10

__global__
void blurKernel(unsigned char *in, unsigned char *out,int w, int h){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < w && row < h){
        int pixVal = 0;
        int pixels = 0;

        // Get average of surronding BLUR_SIZE * BLUR_BOX
        for(int blurRow=BLUR_SIZE;blurRow<BLUR_SIZE+1; ++blurRow){
            for(int blurCol=-BLUR_SIZE; blurCol<BLUR_SIZE+1;++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                    // Verify if we have valid pixels
                if(curRow>=0 && curRow<h && curCol>=0 && curCol<w){
                    pixVal += in[curRow*w + curCol];
                    ++pixels; // Keep track of number of pixels in avg
                }
            }
        }

        //Write out new pixel value
        out[row*w + col] = (unsigned char)(pixVal/pixels);
    }
}

void imageBlur(unsigned char *in_h, unsigned char *out_h, int w, int h){
    unsigned char *in_d, *out_d;
    int size = w * h * sizeof(unsigned char);
    cudaMalloc((void**)&in_d,size);
    cudaMalloc((void**)&out_d,size);

    cudaMemcpy(in_d, in_h ,size,cudaMemcpyHostToDevice);

    //Recording events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 dimBlock(16,16,1);
    dim3 dimGrid(ceil(w/16.0),ceil(h/16.0), 1);
    blurKernel<<<dimGrid,dimBlock>>>(in_d,out_d,w,h);
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

    cudaMemcpy(out_h, out_d,size,cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);
}    
    

int main(){

    const int w = 1024;
    const int h = 1024;

    unsigned char *in_h = (unsigned char *)malloc(w * h * sizeof(unsigned char));
    unsigned char  *out_h = (unsigned char *)malloc(w * h * sizeof(unsigned char));

    // Initialize input image with some pattern (e.g., a gradient or random values)
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            in_h[i * w + j] = (unsigned char)((i + j) % 256);  // Example gradient pattern
        }
    }

    imageBlur(in_h,out_h,w,h);

    printf("Blurred image first few values:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", out_h[i]);
    }
    printf("\n");

        // Free host memory
        free(in_h);
        free(out_h);
    
        return 0;


}
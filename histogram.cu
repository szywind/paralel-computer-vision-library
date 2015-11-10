/*
 * Zhenyuan Shen created 2015/11/09
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <cassert>
#include <fstream>
#include <cstring>
#include <ccv_image.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

// convert data from float to unsigned char
__global__ void cvtFltToUchar(float* inputImg, unsigned char* outputImg, int len)
{
    int t = blockIdx.x*blockDim.x+threadIdx.x;
    if(t<len)
    {
        outputImg[t] = (unsigned char)(255*inputImg[t]);
    }
    /* test code 1
    if (t/3<10)
        printf("%d-th thread: %d\n", t, outputImg[t]);
    */
    
}

// convert color image to gray image
__global__ void cvtClrToGray(unsigned char* outputImg, unsigned char* inputImg, int width, int height, int channels)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int idx = row*width+col;
    if(row < height && col < width && channels == 3)
    {
        outputImg[idx] = 0.21*inputImg[idx*3] + 0.71*inputImg[idx*3+1] + 0.07*inputImg[idx*3+2];
    }
    /* test code 2
    if(idx<10)
    {
        printf("=%d\n",outputImg[idx]);
    }
    */  
}

// compute the histogram
__global__ void histo_kernel(unsigned char* buffer, long size, unsigned int* histo)
{
    __shared__ unsigned int histo_private[256];
    if(threadIdx.x < 256)
        histo_private[threadIdx.x] = 0;
    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;
    
    while(i < size)
    {
        atomicAdd(&(histo_private[buffer[i]]), 1);
        i += stride;
    }
    
    // wait for all other threads in the block to finish
    __syncthreads();
    
    if(threadIdx.x < 256)
    {
        atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
    }
}

// 
__device__ float p(unsigned int x, int width, int height)
{
    return float(x) / (width * height);
}

__global__ void scan(unsigned int * input, float * output, int len, int width, int height) 
{    
    __shared__ int XY[2*BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockDim.x*blockIdx.x;
    assert(BLOCK_SIZE == blockDim.x);
    
    XY[t] = (start+t < len)? input[start + t] : 0;
    XY[t+blockDim.x] = (start+blockDim.x+t < len)? input[start + blockDim.x + t] : 0;
    
    __syncthreads();
    // Reduction Phase
    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int index = (threadIdx.x+1)*stride*2 - 1;
        assert(index >= stride);  // SZY add
        
        if(index < 2*BLOCK_SIZE)
        XY[index] += XY[index-stride];
        __syncthreads();
    }
    
    // Post Reduction Phase
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index+stride < 2*BLOCK_SIZE) 
        {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    //if (i < len) output[i] = XY[threadIdx.x];
    
    // Recording the values to output

    if(start < len){
        output[start+t] = p(XY[t], width, height);
    }
    if((start + BLOCK_SIZE) < len){
        output[start + BLOCK_SIZE + t] = p(XY[t + BLOCK_SIZE], width, height);
    }
}


/* Redundant for this case, since the minimum of CDF(monotonically increasing) is the first element input[0] */
/*
__global__ void minInCDF(float * input, float * output, int len) {
    
    __shared__ int partialMin[2*BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockDim.x*blockIdx.x;
    assert(BLOCK_SIZE == blockDim.x);
    
    partialMin[t] = (start+t < len)? input[start + t] : 0;
    partialMin[t+blockDim.x] = (start+blockDim.x+t < len)? input[start + blockDim.x + t] : 0;
    
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
            partialMin[t] = (partialMin[t] < partialMin[t+stride]) ? partialMin[t] : partialMin[t+stride];
    }
    output[blockIdx.x] = partialMin[0];
    
}
*/
__device__ float clamp(float x, float start, float end)
{
    return min(max(x, start), end);
}

__global__ void histEqual(float * outputCDF, int len)
{
    int t = blockIdx.x*blockDim.x+threadIdx.x;
    float minCDF = outputCDF[0];   // compute the minimum of outputCDF
    if(t<len)
    {
        //printf("Before: %d-th thread: %f\n", t, outputCDF[t]);
        outputCDF[t] = clamp(255*(outputCDF[t]-minCDF)/(1-minCDF), 0.0, 255.0);
        //printf("After: %d-th thread: %f\n", t, outputCDF[t]);
    }
    __syncthreads();
    outputCDF[0] = 0.0f;
    
    // test code 6
    /*
    if (t<len)
        printf("%d-th thread: %f\n", t, outputCDF[t]);
    printf("The minimum of CDF is %f\n", minCDF);
    */
    
    
}

__global__ void applyEqualAndCastBack(float* outputImg, unsigned char* inputImg, float* inputCDF, int len)
{
    int t = blockIdx.x*blockDim.x+threadIdx.x;
    if(t<len)
    {
        assert(inputImg[t] < HISTOGRAM_LENGTH);
        outputImg[t] = (float)(inputCDF[inputImg[t]]/255.0);
    }
}
    
//------------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    if(argc != )

    int imageWidth;
    int imageHeight;
    int imageChannels;
    ccvImage inputImage;
    ccvImage outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    
    
    for(int i=0; i<imageWidth; i++)
        for(int j=0; j<imageHeight; j++)
            for(int c=0; c<imageChannels; c++)
    {
        if ((i*imageWidth+j)<10)
            wbLog(TRACE, "The value of C at  i= ",i,", j= ",j, " is: ",hostInputImageData[(i*imageWidth+j)*3+c]);
    }
    
    // 1. Cast the image from float to unsigned char
    dim3 dimBlock(BLOCK_SIZE,1);
    dim3 dimGrid((imageWidth*imageHeight*imageChannels-1)/BLOCK_SIZE+1,1);
    
    float * deviceInputImageData;
    unsigned char * deviceOutputImageData;
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    
    cvtFltToUchar<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth*imageHeight*imageChannels);
    cudaDeviceSynchronize();
    
    /*
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    */
    
    
    /* 
    // test code 1
    unsigned char *testOutput1 = new unsigned char[imageWidth * imageHeight * imageChannels];
    cudaMemcpy(testOutput1,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    for(int i=0; i<imageWidth; i++)
        for(int j=0; j<imageHeight; j++)
            for(int c=0; c<imageChannels; c++)
    {
        if ((i*imageWidth+j)<10)
            printf("The value of C at  i= %d, j= %d is: %d\n",i,j,testOutput1[(i*imageWidth+j)*3+c]);
    }
    
    delete[] testOutput1;
    // end of test code 1
    */
    
    //--------------------------------------------------------------------------------
    // 2. Convert the image from RGB to GrayScale
    dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
    dimGrid = dim3((imageWidth-1)/BLOCK_SIZE+1,(imageHeight-1)/BLOCK_SIZE+1);
    
    unsigned char * deviceOutputImgGray;
    cudaMalloc((void **) &deviceOutputImgGray, imageWidth * imageHeight * sizeof(unsigned char));
    
    cvtClrToGray<<<dimGrid, dimBlock>>>(deviceOutputImgGray, deviceOutputImageData, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();
    
    /*   
    // test code 2
    unsigned char *testOutput2 = new unsigned char[imageWidth * imageHeight];
    cudaMemcpy(testOutput2,
               deviceOutputImgGray,
               imageWidth * imageHeight * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    for(int i=0; i<imageWidth; i++)
        for(int j=0; j<imageHeight; j++)
    {
        if ((i*imageWidth+j)<10)
            printf("The value of pixel at  i= %d, j= %d is: %d\n",i,j,testOutput2[i*imageWidth+j]);
    }
    
    delete[] testOutput2;
    // end of test code 2
    */
    
    //--------------------------------------------------------------------------------
    // 3. Compute the histogram of grayImage
    unsigned int* deviceHist;
    cudaMalloc((void **) &deviceHist, HISTOGRAM_LENGTH*sizeof(unsigned int));
    
    dimBlock = dim3(HISTOGRAM_LENGTH);
    dimGrid = dim3((imageWidth*imageHeight-1)/HISTOGRAM_LENGTH+1);
    
    histo_kernel<<<dimGrid, dimBlock>>>(deviceOutputImgGray, imageWidth*imageHeight, deviceHist);
    cudaDeviceSynchronize();    
    
    /*
    // test code 3
    unsigned int hostHist[HISTOGRAM_LENGTH];
    cudaMemcpy(hostHist,
               deviceHist,
               HISTOGRAM_LENGTH * sizeof(unsigned int),
               cudaMemcpyDeviceToHost); 
    for(int i=0; i<HISTOGRAM_LENGTH; i++)
        printf("The %d-th value of the hist is: %d\n", i, hostHist[i]);
    // end of test code 3
    */
    
    //--------------------------------------------------------------------------------
    // 4. Compute the Cumulative Distribution Function of histogram
    dimBlock = dim3(BLOCK_SIZE,1);
    dimGrid = dim3((HISTOGRAM_LENGTH-1)/(2*BLOCK_SIZE)+1,1); // dimGrid = (1,1) 
    // Thus, no need to combine the scan results of each thread block (covering 2*BLOCK_SIZE outputs)
    float * deviceCDF;
    cudaMalloc((void **) &deviceCDF, HISTOGRAM_LENGTH*sizeof(float));   
    scan<<<dimGrid, dimBlock>>>(deviceHist, deviceCDF, HISTOGRAM_LENGTH, imageWidth, imageHeight);
    cudaDeviceSynchronize();
    
    /*
    // test code 4
    float hostCDF[HISTOGRAM_LENGTH];
    cudaMemcpy(hostCDF,
               deviceCDF,
               HISTOGRAM_LENGTH * sizeof(float),
               cudaMemcpyDeviceToHost); 
    for(int i=0; i<HISTOGRAM_LENGTH; i++)
        printf("The %d-th value of the hist is: %f\n", i, hostCDF[i]);
    // end of test code 4
    */
    
    //--------------------------------------------------------------------------------
    // 5. Compute the minimum value of the CDF
    
    /* Redundant but ok */
    /*
    int numInputElements = HISTOGRAM_LENGTH;
    int numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    
    dimBlock = dim3(BLOCK_SIZE,1);
    dimGrid = dim3(numOutputElements,1);

    float * devMinCDF;
    float * hostMinCDF;
    cudaMalloc((void **) &devMinCDF, numOutputElements*sizeof(float));
    hostMinCDF = (float*) malloc(numOutputElements * sizeof(float));
    
    minInCDF<<<dimGrid, dimBlock>>>(deviceCDF, devMinCDF, numInputElements);
    cudaDeviceSynchronize();
    
    cudaMemcpy(hostMinCDF, devMinCDF, numOutputElements*sizeof(float), cudaMemcpyDeviceToHost);
    for (int ii = 1; ii < numOutputElements; ii++) {
        if (hostMinCDF[0] == 0.0f)
            break;
        hostMinCDF[0] = (hostMinCDF[0]<hostMinCDF[ii]) ? hostMinCDF[0] : hostMinCDF[ii];
    }
    
    // test code 5
    printf("The minimum of CDF is %f\n", hostMinCDF[0]);
    // end of test code 5
    */
    
    //--------------------------------------------------------------------------------
    // 6. Define the histogram equalization function
    dimBlock = dim3(HISTOGRAM_LENGTH,1);
    dimGrid = dim3(1,1);  
    
    histEqual<<<dimGrid, dimBlock>>>(deviceCDF, HISTOGRAM_LENGTH);
    cudaDeviceSynchronize();    
    
    //--------------------------------------------------------------------------------
    // 7. Apply the histogram equalization function and Cast back to float
    dimBlock = dim3(BLOCK_SIZE,1);
    dimGrid = dim3((imageWidth*imageHeight*imageChannels-1)/BLOCK_SIZE+1,1);
    float * deviceOutputImgClr;
    cudaMalloc((void **) &deviceOutputImgClr, imageWidth*imageHeight*imageChannels*sizeof(float));
    applyEqualAndCastBack<<<dimGrid, dimBlock>>>(deviceOutputImgClr, deviceOutputImageData, deviceCDF, imageWidth*imageHeight*imageChannels);
    cudaDeviceSynchronize();
    
    cudaMemcpy(hostOutputImageData,
               deviceOutputImgClr,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    

    wbSolution(args, outputImage);
    
    
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceOutputImgGray);
    cudaFree(deviceHist);
    cudaFree(deviceCDF);
    cudaFree(deviceOutputImgClr);

    return 0;
}


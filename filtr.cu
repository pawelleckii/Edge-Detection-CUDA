#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>

#define IN_FILE_NAME "lena.png"
#define GRAY_FILE_NAME "CUDA_GRAY.bmp"
#define XFILTERED_FILE_NAME "CUDA_XFILTERED.bmp"
#define YFILTERED_FILE_NAME "CUDA_YFILTERED.bmp"
#define RES_FILE_NAME "CUDA_ZRESULT.bmp"

#define MASK_DIM 3
#define MASK_LENGTH MASK_DIM*MASK_DIM
#define MASK_RADIUS MASK_DIM/2
const char sobelx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
const char sobely[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };


//returns brightness of a pixel. If number is out of bounds, returns value of border pixel.
int getBrightness(int x, int y, unsigned char *img, int imgWidth, int imgHeight){
	if (x < 0) x = 0;
	else if (x >= imgWidth) x = imgWidth - 1;
	if (y < 0) y = 0;
	else if (y >= imgHeight) y = imgHeight - 1;
	return img[y*imgWidth + x];
}

__global__ void
copyArrays(const unsigned char *imgDev, unsigned char *resultDev, int width, int height){
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < width*height)
	{
		resultDev[i] = imgDev[i];
	}
	__syncthreads();
}

__global__ void

copyArrays2(const unsigned char *imgDev, unsigned char *resultDev, int width, int height){
	
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int x = index%width;
	int y = index/width;
	for (int y = 0; y < width; y++){
			for (int x = 0; x < height; x++){
				resultDev[y*width+x] = 255-imgDev[y*width+x];
			}
		}
	__syncthreads();
}

__global__ void
filter(const unsigned char *imgDev, unsigned char *resultDev, int width, int height, const char* mask, int sizeOfMask){

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int x = index%width;
	int y = index/width;

	int res = 0;
	int maskIndex = 0;
	int dx = 0;
	int dy = 0;
	if(index < width*height){
		for (int yi = y - MASK_RADIUS; yi <= y + MASK_RADIUS; yi++){
			for (int xi = x - MASK_RADIUS; xi <= x + MASK_RADIUS; xi++){
				dx = xi;
				dy = yi;
				dx = max(0, dx);
				dx = min(dx, width - 1);
				dy = max(0, dy);
				dy = min(dy, height - 1);
				res += mask[maskIndex] * imgDev[dy*width + dx];
				maskIndex++;
			}
		}
		if (res < 0) res = -res;
		resultDev[index] = (unsigned char)res;
	}
	__syncthreads();
}

void filterSeq(int x, int y, const unsigned char *img, unsigned char **result, int width, int height, const char* mask, int sizeOfMask){

	//load image?

	int i = y*width + x;

	int res = 0;
	int maskIndex = 0;
	int dx = 0;
	int dy = 0;
	for (int yi = y - MASK_RADIUS; yi <= y + MASK_RADIUS; yi++){
		for (int xi = x - MASK_RADIUS; xi <= x + MASK_RADIUS; xi++){
			dx = xi;
			dy = yi;
			dx = max(0, dx);
			dx = min(dx, width - 1);
			dy = max(0, dy);
			dy = min(dy, height - 1);
			res += mask[maskIndex] * img[dy*width + dx];
			maskIndex++;
		}
	}
	if (res < 0) res = -res;
	(*result)[i] = (unsigned char)res;
	
	return;
}

void countGradient(int x, int y, const unsigned char *xfiltered, const unsigned char *yfiltered, unsigned char **result, int width, int height){
	
	int i = y*width + x;
	int res = 0;
	
	if (xfiltered[i] > yfiltered[i]){
		res = xfiltered[i] - yfiltered[i];
	}else{
		res = yfiltered[i] - xfiltered[i];
	}
	
	(*result)[i] = res;
	//(*result)[i] = res > threshold ? 255 : 0;
	return;
}

int main(int argc, char *argv[]) {

	cudaError_t err = cudaSuccess;

	printf("MASK_RADIUS = %d\n", MASK_RADIUS);

	int width, height, bpp;
	printf("Loading image file\n");
	unsigned char* img = stbi_load(IN_FILE_NAME, &width, &height, &bpp, STBI_grey);
	if (img == 0){
		printf("Failed to load image\n");
		system("PAUSE");
		return 0;
	}
	else{
		printf("Load complete!\n");
	}

	int size = width*height;
	int threadsPerBlock = 256;
	int blocksPerGrid = size/threadsPerBlock;
	printf("TOTAL PIXELS  %d!\n", width*height);
	printf("TOTAL THREADS %d!\n", blocksPerGrid*threadsPerBlock);

	size_t allocSize = size*sizeof(unsigned char);
	printf("I want to alloc memes!\n");
	unsigned char* xfiltered = (unsigned char*)malloc(allocSize);
	printf("X filtered mem aloced!\n");
	unsigned char* yfiltered = (unsigned char*)malloc(allocSize);
	printf("Y filtered mem aloced!\n");
	unsigned char* result = (unsigned char*)malloc(allocSize);
	printf("Result mem aloced!\n");
	//----------------------------------------------------------------------CUDA MALLOC
	unsigned char *imgDev = NULL;
	unsigned char *xfilteredDev = NULL;
	unsigned char *yfilteredDev = NULL;
	unsigned char *resultDev = NULL;
	err = cudaMalloc((void **)&imgDev, allocSize);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device image memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("device image mem aloced!\n");
	err = cudaMalloc((void **)&xfilteredDev, allocSize);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device xfiltered memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("device xfiltered mem aloced!\n");
	err = cudaMalloc((void **)&yfilteredDev, allocSize);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device yfiltered memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("device yfiltered mem aloced!\n");
	err = cudaMalloc((void **)&resultDev, allocSize);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device image memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("device result mem aloced!\n");
	//----------------------------------------------------------------------
	printf("Copying input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(imgDev, img, allocSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy image from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}else{
		printf("Image copied to device!\n");
	}

	//***********************************************************************************************************************************************************
	/*filter<<<blocksPerGrid, threadsPerBlock>>>(imgDev, xfilteredDev, width, height, sobelx, MASK_LENGTH);
	err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	filter<<<blocksPerGrid, threadsPerBlock>>>(imgDev, yfilteredDev, width, height, sobely, MASK_LENGTH);
	err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}*/
	copyArrays2<<<blocksPerGrid, threadsPerBlock>>>(imgDev, resultDev, width, height);
	err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//SEQUENTIAL CPU CODE HERE
	/*for (int y = 0; y < height ; y++){
		for (int x = 0; x < width; x++){
			//if (x == 0) printf("|\n");
			//printf("%d ", getBrightness(x, y, img, width, height));
			filterSeq(x, y, img, &xfiltered, width, height, sobelx, MASK_LENGTH);	
			filterSeq(x, y, img, &yfiltered, width, height, sobely, MASK_LENGTH);
			countGradient(x, y, xfiltered, yfiltered, &result, width, height);
		}
		if (y % 100 == 0) printf("\rFiltering %.2f %%", 100.0*y / height);
	}
	printf("\rFiltering 100.00%%");
	*/

	//----------------------------------------------------------------------------------------CUDA MEM COPY DEVICE TO HOST
	err = cudaMemcpy(result, resultDev, allocSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}else{
		printf("result copied to device!\n");
	}
	err = cudaMemcpy(xfiltered, xfilteredDev, allocSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy xfiltered from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}else{
		printf("xfiltered copied to device!\n");
	}
	err = cudaMemcpy(yfiltered, yfilteredDev, allocSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy yfiltered from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}else{
		printf("yfiltered copied to device!\n");
	}
	

	//saving images to HDD
	printf("\nCreating gray file\n");
	stbi_write_bmp(GRAY_FILE_NAME, width, height, STBI_grey, img);
	printf("\nCreating X filtered file");
	stbi_write_bmp(XFILTERED_FILE_NAME, width, height, STBI_grey, xfiltered);
	printf("\nCreating Y filtered file");
	stbi_write_bmp(YFILTERED_FILE_NAME, width, height, STBI_grey, yfiltered);
	printf("\nCreating result file");
	stbi_write_bmp(RES_FILE_NAME, width, height, STBI_grey, result);
	
	err = cudaFree(xfilteredDev);
	err = cudaFree(yfilteredDev);
	err = cudaFree(resultDev);
	err = cudaDeviceReset();
	printf("\nEnd of program!\n");
	stbi_image_free(img);	
	system("PAUSE");
	return 0;
}

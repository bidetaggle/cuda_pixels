
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 16
#define KRANGE 8
#define TLEVEL 3

void prompt(int array[N][N]) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if(array[i][j] < 10)
				printf("%i  ", array[i][j]);
			else if (array[i][j] < 100)
				printf("%i ", array[i][j]);
			else if (array[i][j] < 1000)
				printf("%i", array[i][j]);
			printf(" ");
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void quantization(int A[N][N], int quantizationRange, int greyscaleRange) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N) {
		for (int t = 0; t<greyscaleRange / quantizationRange; t++)
			if (A[i][j] >= t*quantizationRange && A[i][j] < (t + 1)*(quantizationRange)) {
				A[i][j] = t;
				break;
			}
	}
}

__global__ void neigborsCount(int A[N][N]) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = 0;
	if (i < N && j < N) {
		/* edges */
		if (A[i - 1][j] != A[i][j] && (i-1) >= 0)
			k++;
		if (A[i + 1][j] != A[i][j] && (i+1) < N)
			k++;
		if (A[i][j - 1] != A[i][j] && (j-1) >= 0)
			k++;
		if (A[i][j + 1] != A[i][j] && (j+1) < N)
			k++;

		/* corners (diagonals) */
		if (A[i - 1][j - 1] != A[i][j] && (i - 1) >= 0 && (j - 1) >= 0)
			k++;
		if (A[i + 1][j - 1] != A[i][j] && (i + 1) < N && (j - 1) >= 0)
			k++;
		if (A[i - 1][j + 1] != A[i][j] && (i - 1) >= 0 && (j + 1) < N)
			k++;
		if (A[i + 1][j + 1] != A[i][j] && (i + 1) < N && (j + 1) < N)
			k++;

		A[i][j] = k;
	}
}

__global__ void initRand(unsigned int seed, curandState_t** states) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
		i+j, /* the sequence number should be different for each core (unless you want all
					cores to get the same sequence of numbers for some reason - use thread id! */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		states[i][j]);
}

__global__ void randoms(curandState_t* states, int A[N][N]) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	A[i][j] = curand(&states[blockIdx.x]) % 100;
}

int main() {
	int A[N][N];

	curandState_t **states; //random states
	int(*d_A)[N]; //pointers to arrays of dimension N

	int greyscaleRange = int(pow(2, KRANGE));
	int quantizationRange = int(pow(2, KRANGE)) / int(pow(2, TLEVEL));

	printf("grayscale range : %i\n", greyscaleRange);
	printf("quantization range : %i\n", quantizationRange);
	printf("categories : %i\n", greyscaleRange / quantizationRange);

	// Kernel invocation
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

	cudaMalloc((void**)&d_A, (N*N) * sizeof(int));
	cudaMalloc((void**)&states, (N*N) * sizeof(curandState_t));
	initRand << <numBlocks, threadsPerBlock >> >(time(0), states);
	randoms << <numBlocks, threadsPerBlock >> >(states, d_A);
	cudaMemcpy(A, d_A, (N*N) * sizeof(int), cudaMemcpyDeviceToHost);
	
	/*
	//fill the array with random numbers
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i][j] = rand() % greyscaleRange;
		}
	}
	*/

	printf("\ninitial table : \n\n");
	prompt(A);

	//copying from host to device
	cudaMemcpy(d_A, A, (N*N) * sizeof(int), cudaMemcpyHostToDevice);

	/* quantization */
	quantization << <numBlocks, threadsPerBlock >> >(d_A, quantizationRange, greyscaleRange);
	cudaMemcpy(A, (d_A), (N*N) * sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nArray quantization : \n\n");
	prompt(A);

	/* count the neighbors */
	neigborsCount << <numBlocks, threadsPerBlock >> >(d_A);
	cudaMemcpy(A, (d_A), (N*N) * sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nDifferents neightbors count : \n\n");
	prompt(A);

	return 0;
}
#include "hip/hip_runtime.h"
#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#ifdef __CUDA_CC__
#include "cuda.h"
#endif

#include <sys/time.h>

#define HIP_CHECK(call)                                                        \
	do {                                                                     \
		hipError_t error = (call);                                         \
		if (error != hipSuccess) {                                         \
			fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,       \
				__LINE__, hipGetErrorString(error));                    \
			exit(EXIT_FAILURE);                                          \
		}                                                                \
	} while (0)

// includes, kernels
#include "needle_kernel_hip.cu"

#ifdef PROFILING
#include "RDTimer.h"
SimplePerfSerializer* serializeTime;
RDTimerCPU* rdtimercpu;
#endif

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{

  printf("WG size of kernel = %d \n", BLOCK_SIZE);
#ifdef PROFILING
	float overall_cpu_t;
    	RDTimerCPU* rdtimerOverallCpu = new RDTimerCPU();
    	rdtimerOverallCpu->Reset("Overall CPU Time");
    	serializeTime = new SimplePerfSerializer( argv[0] );
    	rdtimercpu = new RDTimerCPU();
    	rdtimerOverallCpu->Start();
#endif
    runTest( argc, argv);
#ifdef PROFILING
	overall_cpu_t = rdtimerOverallCpu->Stop();
	serializeTime->Serialize(rdtimerOverallCpu);

	//printf("Overall CPU time = %fs\n", overall_cpu_t);

	delete rdtimercpu;
	delete rdtimerOverallCpu;
	delete serializeTime;
 #endif
    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

void runTest( int argc, char** argv)
{
    int max_rows, max_cols, penalty;
	int *input_itemsets, *output_itemsets, *referrence;
	int *sequence_rows, *sequence_cols;
	int *matrix_cuda,  *referrence_cuda;
	int size;


    // the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 3)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
    else{
	usage(argc, argv);
    }

	if(atoi(argv[1])%16!=0){
	fprintf(stderr,"The dimension values must be a multiple of 16\n");
	exit(1);
	}


	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );

	output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	sequence_rows = (int *)malloc((max_rows - 1) * sizeof(int));
	sequence_cols = (int *)malloc((max_cols - 1) * sizeof(int));


	if (!referrence || !input_itemsets || !output_itemsets ||
	    !sequence_rows || !sequence_cols) {
		fprintf(stderr, "error: can not allocate host memory\n");
		exit(EXIT_FAILURE);
	}

    srand ( 7 );


    for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			input_itemsets[i*max_cols+j] = 0;
		}
	}

	printf("Start Needleman-Wunsch\n");

	for( int i=1; i< max_rows ; i++){    //please define your own sequence.
	       input_itemsets[i*max_cols] = rand() % 10 + 1;
	       sequence_rows[i - 1] = input_itemsets[i*max_cols];
	}
	    for( int j=1; j< max_cols ; j++){    //please define your own sequence.
	       input_itemsets[j] = rand() % 10 + 1;
	       sequence_cols[j - 1] = input_itemsets[j];
	}


	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
		referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

    for( int i = 1; i< max_rows ; i++)
    {
       input_itemsets[i*max_cols] = -i * penalty;
    }
	for( int j = 1; j< max_cols ; j++)
	{
       		input_itemsets[j] = -j * penalty;
       	}


    size = max_cols * max_rows;
#ifdef PROFILING
    	rdtimercpu->Reset("Malloc Time");
    	rdtimercpu->Start();
#endif
	HIP_CHECK(hipMalloc((void**)& referrence_cuda, sizeof(int)*size));
	HIP_CHECK(hipMalloc((void**)& matrix_cuda, sizeof(int)*size));
#ifdef PROFILING
    rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("CPU to GPU Transfer Time");
    rdtimercpu->Start();
#endif
	HIP_CHECK(hipMemcpy(referrence_cuda, referrence, sizeof(int) * size,
	                    hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size,
	                    hipMemcpyHostToDevice));
#ifdef PROFILING
    rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("COMPUTE:Compute Time");
    rdtimercpu->Start();
#endif
    dim3 dimGrid;
	dim3 dimBlock(BLOCK_SIZE, 1);
	int block_width = ( max_cols - 1 )/BLOCK_SIZE;

	//printf("Processing top-left matrix\n");
	//process top-left matrix
	for( int i = 1 ; i <= block_width ; i++){
		dimGrid.x = i;
		dimGrid.y = 1;
		hipLaunchKernelGGL(needle_cuda_shared_1, dim3(dimGrid), dim3(dimBlock), 0, 0, referrence_cuda, matrix_cuda
		                                      ,max_cols, penalty, i, block_width);
		HIP_CHECK(hipGetLastError());
	}
	//printf("Processing bottom-right matrix\n");
    //process bottom-right matrix
	for( int i = block_width - 1  ; i >= 1 ; i--){
		dimGrid.x = i;
		dimGrid.y = 1;
		hipLaunchKernelGGL(needle_cuda_shared_2, dim3(dimGrid), dim3(dimBlock), 0, 0, referrence_cuda, matrix_cuda
		                                      ,max_cols, penalty, i, block_width);
		HIP_CHECK(hipGetLastError());
	}

#ifdef PROFILING
    rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("GPU to CPU Transfer Time");
    rdtimercpu->Start();
#endif
	    HIP_CHECK(hipMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size,
	                        hipMemcpyDeviceToHost));
#ifdef PROFILING
    rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
#endif
	FILE *fpo = fopen("result.txt", "w");
	if (!fpo) {
		fprintf(stderr, "error: can not open result.txt\n");
		exit(EXIT_FAILURE);
	}
	fprintf(fpo, "dimension %d\n", max_rows - 1);
	fprintf(fpo, "penalty %d\n", penalty);
	fprintf(fpo, "sequence_rows");
	for (int i = 0; i < max_rows - 1; i++)
		fprintf(fpo, " %d", sequence_rows[i]);
	fprintf(fpo, "\nsequence_cols");
	for (int j = 0; j < max_cols - 1; j++)
		fprintf(fpo, " %d", sequence_cols[j]);
	fprintf(fpo, "\nscore %d\n",
	        output_itemsets[(max_rows - 1) * max_cols + max_cols - 1]);
	if (fclose(fpo) != 0) {
		fprintf(stderr, "error: can not close result.txt\n");
		exit(EXIT_FAILURE);
	}

	HIP_CHECK(hipFree(referrence_cuda));
	HIP_CHECK(hipFree(matrix_cuda));

	free(referrence);
		free(input_itemsets);
		free(output_itemsets);
		free(sequence_rows);
		free(sequence_cols);

}

